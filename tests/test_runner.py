from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from agent.config import Settings
from agent.runner import AgentRunner, _split_reasoned_response


def make_settings(*, provider: str = "openai", model: str = "gpt-5-mini") -> Settings:
    return Settings(
        api_key="test",
        provider=provider,
        base_url=None,
        model=model,
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
        listen_engine="auto",
        listen_culture="en-US",
        listen_timeout_seconds=8,
        fast_reply=False,
        reasoning_enabled=True,
        reasoning_effort="medium",
        response_max_tokens=160,
        history_max_messages=8,
    )


class RunnerReasoningTests(unittest.TestCase):
    def test_split_reasoned_response_extracts_summary_and_answer(self) -> None:
        parsed = _split_reasoned_response(
            "Reasoning summary: I checked the available context.\n"
            "Answer: Use the web launcher."
        )

        self.assertEqual(parsed["reasoning"], "I checked the available context.")
        self.assertEqual(parsed["reply"], "Use the web launcher.")

    def test_split_reasoned_response_keeps_unformatted_answer(self) -> None:
        parsed = _split_reasoned_response("Plain answer")

        self.assertEqual(parsed, {"reply": "Plain answer", "reasoning": ""})

    def test_reasoning_effort_is_only_used_for_openai_reasoning_models(self) -> None:
        with patch("agent.runner.OpenAI"):
            openai_runner = AgentRunner(make_settings(provider="openai", model="gpt-5-mini"))
            ollama_runner = AgentRunner(make_settings(provider="ollama", model="llama3"))
            standard_runner = AgentRunner(make_settings(provider="openai", model="gpt-4.1-mini"))

        self.assertTrue(openai_runner._supports_reasoning_effort())
        self.assertFalse(ollama_runner._supports_reasoning_effort())
        self.assertFalse(standard_runner._supports_reasoning_effort())

    def test_fast_reply_disables_reasoning_and_tools(self) -> None:
        with patch("agent.runner.OpenAI"):
            runner = AgentRunner(make_settings())

        runner.set_fast_reply(True)
        self.assertFalse(runner._supports_reasoning_effort())
        self.assertFalse(runner._supports_tools())
        self.assertEqual(runner._effective_response_max_tokens(), 96)
        self.assertEqual(runner._effective_history_max_messages(), 4)


if __name__ == "__main__":
    unittest.main()
