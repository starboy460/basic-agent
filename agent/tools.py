from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_project_path(raw_path: str) -> Path:
    candidate = (PROJECT_ROOT / raw_path).resolve()
    if PROJECT_ROOT not in candidate.parents and candidate != PROJECT_ROOT:
        raise ValueError("Path is outside the project root.")
    return candidate


def get_time() -> str:
    return datetime.now().astimezone().isoformat()


def list_files(path: str = ".") -> list[str]:
    target = _resolve_project_path(path)
    if not target.exists():
        raise ValueError(f"Path does not exist: {path}")
    if not target.is_dir():
        raise ValueError(f"Path is not a directory: {path}")
    return sorted(item.name for item in target.iterdir())


def read_file(path: str) -> str:
    target = _resolve_project_path(path)
    if not target.exists():
        raise ValueError(f"File does not exist: {path}")
    if not target.is_file():
        raise ValueError(f"Path is not a file: {path}")
    return target.read_text(encoding="utf-8")


def write_file(path: str, content: str) -> str:
    target = _resolve_project_path(path)
    if target.exists() and not target.is_file():
        raise ValueError(f"Path is not a file: {path}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"Wrote {path}"


def search_files(query: str, path: str = ".") -> str:
    target = _resolve_project_path(path)
    if not target.exists():
        raise ValueError(f"Path does not exist: {path}")
    if not target.is_dir():
        raise ValueError(f"Path is not a directory: {path}")

    matches: list[str] = []
    for file_path in target.rglob("*"):
        if not file_path.is_file() or ".venv" in file_path.parts or "__pycache__" in file_path.parts:
            continue
        try:
            text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            if query.lower() in line.lower():
                relative = file_path.relative_to(PROJECT_ROOT)
                matches.append(f"{relative}:{line_number}: {line.strip()}")
                if len(matches) >= 50:
                    return "\n".join(matches)
    return "\n".join(matches) if matches else "No matches found."


def run_project_command(command: str) -> str:
    allowed_prefixes = (
        "python ",
        ".\\.venv\\Scripts\\python ",
        "pytest ",
        "pip show ",
        "ollama list",
    )
    normalized = command.strip()
    if not normalized.startswith(allowed_prefixes):
        raise ValueError(
            "Command is not allowed. Allowed commands start with: "
            + ", ".join(allowed_prefixes)
        )
    result = subprocess.run(
        normalized,
        cwd=PROJECT_ROOT,
        shell=True,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    output = (result.stdout + result.stderr).strip()
    if len(output) > 6000:
        output = output[:6000] + "\n...[truncated]"
    return f"Exit code: {result.returncode}\n{output}"


TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current local date and time.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files and directories inside the project folder.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative directory path inside the project. Use '.' for the project root.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file inside the project folder.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative file path inside the project.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or replace a UTF-8 text file inside the project folder.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative file path inside the project.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Complete file contents to write.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search project text files for a query and return file:line matches.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Text to search for.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Relative directory path inside the project.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_project_command",
            "description": "Run a limited safe command inside the project folder for coding checks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Command to run. Allowed prefixes include python, .\\.venv\\Scripts\\python, pytest, pip show, and ollama list.",
                    }
                },
                "required": ["command"],
            },
        },
    },
]


def run_tool(name: str, arguments: dict) -> str:
    if name == "get_time":
        return get_time()
    if name == "list_files":
        return "\n".join(list_files(arguments.get("path", ".")))
    if name == "read_file":
        return read_file(arguments["path"])
    if name == "write_file":
        return write_file(arguments["path"], arguments["content"])
    if name == "search_files":
        return search_files(arguments["query"], arguments.get("path", "."))
    if name == "run_project_command":
        return run_project_command(arguments["command"])
    raise ValueError(f"Unknown tool: {name}")


def handle_local_command(user_input: str) -> str | None:
    text = user_input.strip()
    if not text.startswith("/"):
        return None

    command, _, rest = text.partition(" ")
    command = command.lower()
    rest = rest.strip()

    if command in {"/code-help", "/coding"}:
        return (
            "Coding commands:\n"
            "/files [path]\n"
            "/read <path>\n"
            "/search <query>\n"
            "/run <python or test command>\n"
            "/write <path> | <complete file content>\n"
            "Skills:\n"
            "/skills\n"
            "/skill <name>\n"
            "/skill off"
        )
    if command == "/files":
        return "\n".join(list_files(rest or "."))
    if command == "/read":
        if not rest:
            return "Usage: /read <path>"
        return read_file(rest)
    if command == "/search":
        if not rest:
            return "Usage: /search <query>"
        return search_files(rest)
    if command == "/run":
        if not rest:
            return "Usage: /run <python or test command>"
        return run_project_command(rest)
    if command == "/write":
        path, separator, content = rest.partition("|")
        if not separator or not path.strip():
            return "Usage: /write <path> | <complete file content>"
        return write_file(path.strip(), content.lstrip())
    return None
