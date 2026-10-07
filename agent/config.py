from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=True)


@dataclass(frozen=True)
class Settings:
    api_key: str
    provider: str
    base_url: str | None
    model: str
    agent_name: str
    system_prompt: str
    memory_path: Path
    skills_dir: Path
    skill_name: str
    tools_enabled: bool
    voice_enabled: bool
    voice_name: str
    voice_rate: int
    voice_volume: int
    listen_enabled: bool
    listen_engine: str
    listen_culture: str
    listen_timeout_seconds: int
    fast_reply: bool
    reasoning_enabled: bool
    reasoning_effort: str
    response_max_tokens: int
    history_max_messages: int


def get_settings() -> Settings:
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower() or "openai"
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if provider == "openai" and not api_key:
        raise ValueError("OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.")
    voice_rate = _int_env("AGENT_VOICE_RATE", 0)
    voice_volume = _int_env("AGENT_VOICE_VOLUME", 100)
    listen_timeout = _int_env("AGENT_LISTEN_TIMEOUT_SECONDS", 8)
    response_max_tokens = _int_env("AGENT_RESPONSE_MAX_TOKENS", 160)
    history_max_messages = _int_env("AGENT_HISTORY_MAX_MESSAGES", 8)

    return Settings(
        api_key=api_key,
        provider=provider,
        base_url=os.getenv("OPENAI_BASE_URL", "").strip() or None,
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip() or "gpt-4.1-mini",
        agent_name=os.getenv("AGENT_NAME", "Neo").strip() or "Neo",
        system_prompt=(
            os.getenv(
                "AGENT_SYSTEM_PROMPT",
                (
                    "You are a friendly conversational assistant. "
                    "Hold a natural back-and-forth conversation, remember context from the chat, "
                    "ask follow-up questions when useful, and stay concise unless the user wants depth."
                ),
            ).strip()
        ),
        memory_path=PROJECT_ROOT / "chat_history.json",
        skills_dir=Path(os.getenv("AGENT_SKILLS_DIR", str(PROJECT_ROOT / "agent_skills"))).resolve(),
        skill_name=os.getenv("AGENT_SKILL", "").strip(),
        tools_enabled=_bool_env("AGENT_TOOLS_ENABLED", default=True),
        voice_enabled=_bool_env("AGENT_VOICE_ENABLED", default=True),
        voice_name=os.getenv("AGENT_VOICE_NAME", "").strip(),
        voice_rate=max(-10, min(10, voice_rate)),
        voice_volume=max(0, min(100, voice_volume)),
        listen_enabled=_bool_env("AGENT_LISTEN_ENABLED", default=False),
        listen_engine=os.getenv("AGENT_LISTEN_ENGINE", "auto").strip().lower() or "auto",
        listen_culture=os.getenv("AGENT_LISTEN_CULTURE", "en-US").strip() or "en-US",
        listen_timeout_seconds=max(2, min(30, listen_timeout)),
        fast_reply=_bool_env("AGENT_FAST_REPLY", default=False),
        reasoning_enabled=_bool_env("AGENT_REASONING_ENABLED", default=True),
        reasoning_effort=_choice_env("AGENT_REASONING_EFFORT", "medium", {"low", "medium", "high"}),
        response_max_tokens=max(32, min(2048, response_max_tokens)),
        history_max_messages=max(0, min(40, history_max_messages)),
    )


def _bool_env(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)).strip())
    except ValueError:
        return default


def _choice_env(name: str, default: str, allowed: set[str]) -> str:
    value = os.getenv(name, default).strip().lower()
    return value if value in allowed else default
