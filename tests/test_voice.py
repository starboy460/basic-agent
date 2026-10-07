from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from agent.config import Settings
from agent.voice import VoiceListener, microphone_diagnostic


def make_settings(*, listen_engine: str = "auto") -> Settings:
    return Settings(
        api_key="ollama",
        provider="ollama",
        base_url="http://localhost:11434/v1",
        model="qwen2.5:0.5b",
        agent_name="Neo",
        system_prompt="Test prompt.",
        memory_path=Path("chat_history.json"),
        skills_dir=Path("agent_skills"),
        skill_name="",
        tools_enabled=True,
        voice_enabled=True,
        voice_name="",
        voice_rate=0,
        voice_volume=100,
        listen_enabled=False,
        listen_engine=listen_engine,
        listen_culture="en-US",
        listen_timeout_seconds=8,
        fast_reply=False,
        reasoning_enabled=True,
        reasoning_effort="medium",
        response_max_tokens=80,
        history_max_messages=4,
    )


class VoiceListenerTests(unittest.TestCase):
    def test_auto_engine_falls_back_to_speech_recognition(self) -> None:
        listener = VoiceListener(make_settings(listen_engine="auto"))

        with (
            patch.object(listener, "_listen_windows", side_effect=RuntimeError("no recognizer")),
            patch.object(listener, "_listen_speech_recognition", return_value="hello agent"),
        ):
            self.assertEqual(listener.listen_once(), "hello agent")

    def test_windows_engine_does_not_use_fallback(self) -> None:
        listener = VoiceListener(make_settings(listen_engine="windows"))

        with (
            patch.object(listener, "_listen_windows", side_effect=RuntimeError("no recognizer")),
            patch.object(listener, "_listen_speech_recognition") as fallback,
        ):
            with self.assertRaises(RuntimeError):
                listener.listen_once()
            fallback.assert_not_called()

    def test_speech_recognition_engine_skips_windows(self) -> None:
        listener = VoiceListener(make_settings(listen_engine="speech_recognition"))

        with (
            patch.object(listener, "_listen_windows") as windows,
            patch.object(listener, "_listen_speech_recognition", return_value="hello agent"),
        ):
            self.assertEqual(listener.listen_once(), "hello agent")
            windows.assert_not_called()


class MicrophoneDiagnosticTests(unittest.TestCase):
    def test_microphone_diagnostic_reports_missing_optional_packages(self) -> None:
        def fake_find_spec(name: str) -> object | None:
            return None

        with (
            patch("agent.voice.recognizer_diagnostic", return_value="No Windows recognizer."),
            patch("agent.voice.find_spec", side_effect=fake_find_spec),
        ):
            diagnostic = microphone_diagnostic()

        self.assertIn("Windows recognizer: No Windows recognizer.", diagnostic)
        self.assertIn("SpeechRecognition package: missing", diagnostic)
        self.assertIn("PyAudio microphone package: missing", diagnostic)
        self.assertIn("sounddevice microphone package: missing", diagnostic)
        self.assertIn("pip install SpeechRecognition sounddevice numpy", diagnostic)


if __name__ == "__main__":
    unittest.main()
