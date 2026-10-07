from __future__ import annotations

import os
import subprocess
from importlib.util import find_spec

from agent.config import Settings


class VoiceSpeaker:
    def __init__(self, settings: Settings):
        self.enabled = settings.voice_enabled
        self.voice_name = settings.voice_name
        self.rate = settings.voice_rate
        self.volume = settings.voice_volume

    def speak(self, text: str) -> None:
        if not self.enabled or not text.strip():
            return

        env = os.environ.copy()
        env["AGENT_SPEAK_TEXT"] = text
        env["AGENT_SPEAK_VOICE"] = self.voice_name
        env["AGENT_SPEAK_RATE"] = str(self.rate)
        env["AGENT_SPEAK_VOLUME"] = str(self.volume)
        subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                _SPEAK_SCRIPT,
            ],
            env=env,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


class VoiceListener:
    def __init__(self, settings: Settings):
        self.enabled = settings.listen_enabled
        self.engine = settings.listen_engine
        self.culture = settings.listen_culture
        self.timeout_seconds = settings.listen_timeout_seconds

    def listen_once(self) -> str:
        if self.engine == "windows":
            return self._listen_windows()
        if self.engine == "speech_recognition":
            return self._listen_speech_recognition()

        try:
            return self._listen_windows()
        except RuntimeError:
            return self._listen_speech_recognition()

    def _listen_windows(self) -> str:
        env = os.environ.copy()
        env["AGENT_LISTEN_CULTURE"] = self.culture
        env["AGENT_LISTEN_TIMEOUT"] = str(self.timeout_seconds)
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                _LISTEN_SCRIPT,
            ],
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(detail or "Speech recognition failed.")
        return result.stdout.strip()

    def _listen_speech_recognition(self) -> str:
        try:
            import speech_recognition as sr
        except ImportError as exc:
            raise RuntimeError(
                "SpeechRecognition is not installed. Run: pip install SpeechRecognition sounddevice numpy"
            ) from exc

        recognizer = sr.Recognizer()
        try:
            audio = self._record_audio_data(sr)
            return recognizer.recognize_google(audio)
        except sr.WaitTimeoutError as exc:
            raise RuntimeError("No speech was heard before the listen timeout.") from exc
        except sr.UnknownValueError as exc:
            raise RuntimeError("Speech was heard, but it could not be understood.") from exc
        except sr.RequestError as exc:
            raise RuntimeError(f"Speech recognition request failed: {exc}") from exc
        except Exception as exc:
            raise RuntimeError(f"Microphone speech recognition failed: {exc}") from exc

    def _record_audio_data(self, sr: object) -> object:
        try:
            microphone = sr.Microphone()
        except (AttributeError, OSError):
            microphone = None

        if microphone is not None:
            recognizer = sr.Recognizer()
            with microphone as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                return recognizer.listen(source, timeout=self.timeout_seconds)

        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError(
                "No microphone capture backend is installed. Run: pip install sounddevice numpy"
            ) from exc

        sample_rate = 16000
        recording = sd.rec(
            int(self.timeout_seconds * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
        )
        sd.wait()
        return sr.AudioData(recording.tobytes(), sample_rate, 2)


def list_voices() -> list[str]:
    result = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            _LIST_VOICES_SCRIPT,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def list_recognizers() -> list[str]:
    result = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            _LIST_RECOGNIZERS_SCRIPT,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def recognizer_diagnostic() -> str:
    result = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            _RECOGNIZER_DIAGNOSTIC_SCRIPT,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    output = (result.stdout or result.stderr).strip()
    return output or "No recognizer diagnostic output was returned."


def microphone_diagnostic() -> str:
    windows_status = recognizer_diagnostic()
    speech_recognition_available = find_spec("speech_recognition") is not None
    pyaudio_available = find_spec("pyaudio") is not None
    sounddevice_available = find_spec("sounddevice") is not None

    lines = [
        f"Windows recognizer: {windows_status}",
        f"SpeechRecognition package: {'installed' if speech_recognition_available else 'missing'}",
        f"PyAudio microphone package: {'installed' if pyaudio_available else 'missing'}",
        f"sounddevice microphone package: {'installed' if sounddevice_available else 'missing'}",
    ]
    if not speech_recognition_available or not (pyaudio_available or sounddevice_available):
        lines.append("Fallback install command: pip install SpeechRecognition sounddevice numpy")
    return "\n".join(lines)


_SPEAK_SCRIPT = r"""
Add-Type -AssemblyName System.Speech
$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer
$voice = $env:AGENT_SPEAK_VOICE
if ($voice) {
    try { $speaker.SelectVoice($voice) } catch {}
}
$speaker.Rate = [int]$env:AGENT_SPEAK_RATE
$speaker.Volume = [int]$env:AGENT_SPEAK_VOLUME
$speaker.Speak($env:AGENT_SPEAK_TEXT)
$speaker.Dispose()
"""

_LISTEN_SCRIPT = r"""
Add-Type -AssemblyName System.Speech
$culture = $env:AGENT_LISTEN_CULTURE
$timeout = [int]$env:AGENT_LISTEN_TIMEOUT
try {
    $recognizers = [System.Speech.Recognition.SpeechRecognitionEngine]::InstalledRecognizers()
}
catch {
    [Console]::Error.WriteLine("Windows speech recognition is not available: " + $_.Exception.Message)
    exit 2
}
$recognizerInfo = $recognizers | Where-Object { $_.Culture.Name -eq $culture } | Select-Object -First 1
if ($null -eq $recognizerInfo) {
    $recognizerInfo = $recognizers | Select-Object -First 1
}
if ($null -eq $recognizerInfo) {
    [Console]::Error.WriteLine("No Windows speech recognizer is installed.")
    exit 2
}
$recognizer = New-Object System.Speech.Recognition.SpeechRecognitionEngine($recognizerInfo)
try {
    $recognizer.SetInputToDefaultAudioDevice()
    $recognizer.LoadGrammar((New-Object System.Speech.Recognition.DictationGrammar))
    $result = $recognizer.Recognize((New-TimeSpan -Seconds $timeout))
    if ($null -eq $result -or [string]::IsNullOrWhiteSpace($result.Text)) {
        exit 3
    }
    Write-Output $result.Text
}
finally {
    $recognizer.Dispose()
}
"""

_LIST_RECOGNIZERS_SCRIPT = r"""
Add-Type -AssemblyName System.Speech
try {
    [System.Speech.Recognition.SpeechRecognitionEngine]::InstalledRecognizers() |
        ForEach-Object { "$($_.Name) [$($_.Culture.Name)]" }
}
catch {
    exit 2
}
"""

_RECOGNIZER_DIAGNOSTIC_SCRIPT = r"""
Add-Type -AssemblyName System.Speech
try {
    $recognizers = [System.Speech.Recognition.SpeechRecognitionEngine]::InstalledRecognizers()
    if ($null -eq $recognizers -or $recognizers.Count -eq 0) {
        Write-Output "No Windows speech recognizer is installed."
        exit 0
    }
    $recognizers | ForEach-Object { "$($_.Name) [$($_.Culture.Name)]" }
}
catch {
    Write-Output ("Windows speech recognition is not available: " + $_.Exception.Message)
}
"""

_LIST_VOICES_SCRIPT = r"""
Add-Type -AssemblyName System.Speech
$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer
$speaker.GetInstalledVoices() | ForEach-Object { $_.VoiceInfo.Name }
$speaker.Dispose()
"""
