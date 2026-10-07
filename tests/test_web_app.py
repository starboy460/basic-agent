from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from agent.web_app import WebAgent, extract_document_text, generate_svg_image, generate_video_storyboard


class FakeRunner:
    def __init__(self, settings: object) -> None:
        self.history = [
            {"role": "system", "content": "hidden"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "tool", "content": "hidden"},
        ]

    def ask(self, prompt: str) -> str:
        self.history.append({"role": "user", "content": prompt})
        self.history.append({"role": "assistant", "content": "reply"})
        return "reply"

    def ask_with_reasoning(self, prompt: str) -> dict[str, str]:
        reply = self.ask(prompt)
        return {"reply": reply, "reasoning": "Checked the prompt and produced a test reply."}

    def reset_history(self) -> None:
        self.history = [{"role": "system", "content": "hidden"}]


class WebAgentTests(unittest.TestCase):
    def test_history_only_exposes_chat_messages(self) -> None:
        with patch("agent.web_app.AgentRunner", FakeRunner):
            agent = WebAgent()

        self.assertEqual(
            agent.history(),
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ],
        )

    def test_ask_returns_reply(self) -> None:
        with patch("agent.web_app.AgentRunner", FakeRunner):
            agent = WebAgent()

        self.assertEqual(agent.ask("how are you"), "reply")

    def test_ask_with_reasoning_returns_reply_and_summary(self) -> None:
        with patch("agent.web_app.AgentRunner", FakeRunner):
            agent = WebAgent()

        self.assertEqual(
            agent.ask_with_reasoning("how are you"),
            {"reply": "reply", "reasoning": "Checked the prompt and produced a test reply."},
        )

    def test_extract_document_text_handles_html(self) -> None:
        text = extract_document_text(
            path=Path("sample.html"),
            raw=b"<html><body><h1>Title</h1><script>skip()</script><p>Hello</p></body></html>",
        )

        self.assertIn("Title", text)
        self.assertIn("Hello", text)
        self.assertNotIn("skip", text)

    def test_media_fallback_artifacts_are_created(self) -> None:
        image = generate_svg_image("a compact test image")
        video = generate_video_storyboard("a compact test video")

        self.assertTrue(image.exists())
        self.assertEqual(image.suffix, ".svg")
        self.assertTrue(video.exists())
        self.assertEqual(video.suffix, ".html")

    def test_export_chat_markdown(self) -> None:
        with patch("agent.web_app.AgentRunner", FakeRunner):
            agent = WebAgent()

        md = agent.export_chat_markdown()
        self.assertIn("# Neo Conversation Export", md)
        self.assertIn("hello", md)
        self.assertIn("hi", md)


if __name__ == "__main__":
    unittest.main()
