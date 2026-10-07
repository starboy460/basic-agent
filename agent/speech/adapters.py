from __future__ import annotations
import os
from pathlib import Path
from agent.speech.base import SpeechToTextEngine, TextToSpeechEngine
from agent.speech.normalizer import normalize_for_tts

class WhisperSTTAdapter(SpeechToTextEngine):
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")

    def transcribe(self, audio_bytes: bytes, filename: str = "audio.wav") -> str:
        if not self.api_key or self.api_key == "ollama":
            return "నమస్కారం, వాయిస్ అసిస్టెంట్ సిద్ధంగా ఉంది."
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            temp_path = Path(filename)
            temp_path.write_bytes(audio_bytes)
            with open(temp_path, "rb") as f:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    language="te"
                )
            temp_path.unlink(missing_ok=True)
            return str(getattr(transcript, "text", "")).strip()
        except Exception as exc:
            return f"Transcription error: {exc}"
