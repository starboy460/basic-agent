from __future__ import annotations
import unittest
from agent.speech.normalizer import normalize_for_tts
from agent.speech.adapters import WhisperSTTAdapter

class SpeechTests(unittest.TestCase):
    def test_normalizer_strips_markdown_and_urls(self) -> None:
        raw = "Hello [Google](https://google.com) `code` **bold** ₹500"
        normalized = normalize_for_tts(raw)
        self.assertNotIn("https://google.com", normalized)
        self.assertNotIn("**", normalized)
        self.assertIn("రూపాయలు", normalized)

    def test_whisper_adapter_fallback(self) -> None:
        adapter = WhisperSTTAdapter(api_key="ollama")
        res = adapter.transcribe(b"dummy")
        self.assertIn("తెలుగు", res)

if __name__ == "__main__":
    unittest.main()
