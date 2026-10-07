from __future__ import annotations
from abc import ABC, abstractmethod

class SpeechToTextEngine(ABC):
    @abstractmethod
    def transcribe(self, audio_bytes: bytes, filename: str = "audio.wav") -> str:
        pass

class TextToSpeechEngine(ABC):
    @abstractmethod
    def synthesize(self, text: str) -> bytes:
        pass
