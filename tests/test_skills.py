from __future__ import annotations

import unittest
from pathlib import Path
import shutil
from unittest.mock import patch

from agent.config import Settings
from agent.runner import AgentRunner
from agent.skills import SkillCatalog


class SkillTests(unittest.TestCase):
    def test_skill_catalog_loads_markdown_skills(self) -> None:
        root = Path(__file__).resolve().parent / "_tmp_skill_catalog"
        root.mkdir(exist_ok=True)
        try:
            (root / "analysis.md").write_text(
                "---\n"
                "name: analysis\n"
                "description: Analyze requests carefully.\n"
                "---\n"
                "Read the code first.\n",
                encoding="utf-8",
            )

            catalog = SkillCatalog.load(root)
        finally:
            shutil.rmtree(root, ignore_errors=True)
        self.assertEqual(catalog.names(), ["analysis"])
        self.assertEqual(catalog.get("analysis").description, "Analyze requests carefully.")

    def test_runner_applies_startup_skill(self) -> None:
        root = Path(__file__).resolve().parent / "_tmp_skill_runner"
        root.mkdir(exist_ok=True)
        try:
            (root / "analysis.md").write_text(
                "---\n"
                "name: analysis\n"
                "description: Analyze requests carefully.\n"
                "---\n"
                "Read the code first.\n",
                encoding="utf-8",
            )
            settings = Settings(
                api_key="test",
                provider="openai",
                base_url=None,
                model="gpt-5-mini",
                agent_name="Neo",
                system_prompt="Test prompt.",
                memory_path=root / "chat_history.json",
                skills_dir=root,
                skill_name="analysis",
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

            with patch("agent.runner.OpenAI"):
                runner = AgentRunner(settings)
        finally:
            shutil.rmtree(root, ignore_errors=True)
        self.assertIsNotNone(runner.active_skill)
        self.assertEqual(runner.active_skill.name, "analysis")
        self.assertIn("analysis", runner.active_skill_summary())


if __name__ == "__main__":
    unittest.main()
