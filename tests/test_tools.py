from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from agent import tools
from agent.tools import handle_local_command


class ToolTests(unittest.TestCase):
    def test_project_path_rejects_outside_paths(self) -> None:
        with self.assertRaises(ValueError):
            tools.read_file("../outside.txt")

    def test_write_read_and_search_file(self) -> None:
        path = "tests/tmp_tool_file.txt"
        target = tools.PROJECT_ROOT / path
        try:
            self.assertEqual(tools.write_file(path, "hello coding agent\n"), f"Wrote {path}")
            self.assertEqual(tools.read_file(path), "hello coding agent\n")
            self.assertIn("hello coding agent", tools.search_files("coding agent", "tests"))
        finally:
            if target.exists():
                target.unlink()

    def test_run_project_command_rejects_unapproved_command(self) -> None:
        with self.assertRaises(ValueError):
            tools.run_project_command("cmd /c dir")

    def test_run_project_command_reports_output(self) -> None:
        class Result:
            returncode = 0
            stdout = "ok"
            stderr = ""

        with patch("agent.tools.subprocess.run", return_value=Result()) as run:
            output = tools.run_project_command("python --version")

        self.assertIn("Exit code: 0", output)
        self.assertIn("ok", output)
        run.assert_called_once()

    def test_handle_local_command_returns_help(self) -> None:
        result = handle_local_command("/code-help")

        self.assertIsNotNone(result)
        self.assertIn("/files", result or "")

    def test_handle_local_command_ignores_normal_chat(self) -> None:
        self.assertIsNone(handle_local_command("hello"))


if __name__ == "__main__":
    unittest.main()
