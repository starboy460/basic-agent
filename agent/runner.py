from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from openai import OpenAI

from agent.config import Settings
from agent.skills import Skill, SkillCatalog
from agent.tools import TOOL_SPECS, handle_local_command, run_tool


REASONING_INSTRUCTION = {
    "role": "system",
    "content": (
        "For this turn, think through the user's request carefully, but do not reveal hidden "
        "chain-of-thought. Format the final response exactly as:\n"
        "Reasoning summary: <one or two concise sentences about the approach>\n"
        "Answer: <the answer to the user>"
    ),
}


class AgentRunner:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.skills = SkillCatalog.load(settings.skills_dir)
        self.active_skill: Skill | None = None
        self.fast_reply = settings.fast_reply
        client_kwargs: dict[str, Any] = {}
        if settings.api_key:
            client_kwargs["api_key"] = settings.api_key
        if settings.base_url:
            client_kwargs["base_url"] = settings.base_url
        self.client = OpenAI(**client_kwargs)
        self.history = self._load_history()
        if settings.skill_name:
            self.set_skill(settings.skill_name)

    def _supports_tools(self) -> bool:
        return self.settings.tools_enabled and not self.fast_reply

    def _default_history(self) -> list[dict[str, Any]]:
        return [{"role": "system", "content": self.settings.system_prompt}]

    def _skill_message(self) -> dict[str, Any] | None:
        if self.active_skill is None:
            return None
        return self.active_skill.system_message()

    def _load_history(self) -> list[dict[str, Any]]:
        path = self.settings.memory_path
        if not path.exists():
            return self._default_history()

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return self._default_history()

        if not isinstance(data, list) or not data:
            return self._default_history()

        if data[0].get("role") != "system":
            data.insert(0, self._default_history()[0])
        else:
            data[0]["content"] = self.settings.system_prompt
        return data

    def save_history(self) -> None:
        self.settings.memory_path.write_text(
            json.dumps(self.history, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

    def reset_history(self) -> None:
        self.history = self._default_history()
        self.save_history()

    def history_path(self) -> Path:
        return self.settings.memory_path

    def list_skills(self) -> list[str]:
        return self.skills.summary_lines(self.active_skill.name if self.active_skill else None)

    def skill_options(self) -> list[dict[str, Any]]:
        return [
            {
                "name": skill.name,
                "description": skill.description,
                "active": self.active_skill is not None and self.active_skill.name == skill.name,
            }
            for skill in (self.skills.get(name) for name in self.skills.names())
            if skill is not None
        ]

    def set_skill(self, skill_name: str | None) -> str:
        if skill_name is None:
            self.active_skill = None
            return "Skill disabled."

        normalized = skill_name.strip().lower()
        if not normalized:
            self.active_skill = None
            return "Skill disabled."

        skill = self.skills.get(normalized)
        if skill is None:
            raise ValueError(
                f"Unknown skill: {skill_name}. Use /skills to list the available skills."
            )

        self.active_skill = skill
        return f"Skill enabled: {skill.name}"

    def active_skill_summary(self) -> str:
        if self.active_skill is None:
            return "No active skill."
        return (
            f"{self.active_skill.name}: {self.active_skill.description}\n"
            f"File: {self.active_skill.path}"
        )

    def set_fast_reply(self, enabled: bool) -> str:
        self.fast_reply = enabled
        return "Fast reply enabled." if enabled else "Fast reply disabled."

    def fast_reply_summary(self) -> str:
        return "Fast reply is enabled." if self.fast_reply else "Fast reply is disabled."

    def _effective_response_max_tokens(self, requested: int | None = None) -> int:
        base = requested if requested is not None else self.settings.response_max_tokens
        if self.fast_reply:
            base = min(base, 96)
        return max(32, base)

    def _effective_history_max_messages(self) -> int:
        if self.fast_reply:
            if self.settings.history_max_messages == 0:
                return 4
            return min(self.settings.history_max_messages, 4)
        return self.settings.history_max_messages

    def ask(self, user_input: str) -> str:
        return self.ask_with_reasoning(user_input)["reply"]

    def ask_once(self, user_input: str, *, max_tokens: int | None = None) -> str:
        response_max_tokens = self._effective_response_max_tokens(max_tokens)
        request_kwargs: dict[str, Any] = {
            "model": self.settings.model,
            "messages": [
                self._default_history()[0],
                {"role": "user", "content": user_input},
            ],
            "max_tokens": response_max_tokens,
        }
        self._add_provider_options(request_kwargs, request_kwargs["max_tokens"])
        try:
            response = self.client.chat.completions.create(**request_kwargs)
        except Exception as exc:
            return f"I could not get a model response. Reason: {exc.__class__.__name__}: {exc}"
        return (response.choices[0].message.content or "").strip()

    def ask_with_reasoning(self, user_input: str) -> dict[str, str]:
        local_result = handle_local_command(user_input)
        if local_result is not None:
            self.history.append({"role": "user", "content": user_input})
            self.history.append({"role": "assistant", "content": local_result})
            self.save_history()
            return {"reply": local_result, "reasoning": "Handled by a built-in local command."}

        self.history.append({"role": "user", "content": user_input})
        include_reasoning = self.settings.reasoning_enabled and not self.fast_reply

        while True:
            messages = self._request_history()
            skill_message = self._skill_message()
            if skill_message is not None:
                messages = [messages[0], skill_message, *messages[1:]]
            if include_reasoning:
                messages = [*messages, REASONING_INSTRUCTION]
            request_kwargs: dict[str, Any] = {
                "model": self.settings.model,
                "messages": messages,
                "max_tokens": self._effective_response_max_tokens(),
            }
            if self._supports_tools():
                request_kwargs["tools"] = TOOL_SPECS
            if self._supports_reasoning_effort():
                request_kwargs["reasoning_effort"] = self.settings.reasoning_effort
            self._add_provider_options(request_kwargs, request_kwargs["max_tokens"])

            try:
                response = self.client.chat.completions.create(**request_kwargs)
            except Exception as exc:
                self.save_history()
                return {
                    "reply": (
                    "I could not get a model response. "
                    f"Reason: {exc.__class__.__name__}: {exc}"
                    ),
                    "reasoning": "The model request failed before a reasoning summary was available.",
                }

            message = response.choices[0].message
            assistant_message: dict[str, Any] = {"role": "assistant"}

            if message.content:
                content = message.content
                if include_reasoning and not message.tool_calls:
                    parsed = _split_reasoned_response(content)
                    assistant_message["content"] = parsed["reply"]
                else:
                    parsed = {"reply": content, "reasoning": ""}
                    assistant_message["content"] = content
            if message.tool_calls:
                assistant_message["tool_calls"] = [
                    {
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                    for tool_call in message.tool_calls
                ]

            self.history.append(assistant_message)

            if not message.tool_calls:
                self.save_history()
                if not message.content:
                    return {"reply": "", "reasoning": ""}
                return parsed

            for tool_call in message.tool_calls:
                try:
                    arguments = json.loads(tool_call.function.arguments or "{}")
                    result = run_tool(tool_call.function.name, arguments)
                except Exception as exc:
                    result = f"Tool error: {exc}"

                self.history.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                )
            self.save_history()

    def _supports_reasoning_effort(self) -> bool:
        if self.fast_reply or self.settings.provider != "openai":
            return False
        model = self.settings.model.lower()
        return model.startswith(("o", "gpt-5"))

    def _request_history(self) -> list[dict[str, Any]]:
        history_max_messages = self._effective_history_max_messages()
        if history_max_messages == 0:
            return self.history

        system_messages = [message for message in self.history if message.get("role") == "system"]
        non_system_messages = [
            message for message in self.history
            if message.get("role") != "system" and "tool_calls" not in message
        ]
        return [*system_messages[:1], *non_system_messages[-history_max_messages:]]

    def _add_provider_options(self, request_kwargs: dict[str, Any], max_tokens: int) -> None:
        if self.settings.provider == "ollama":
            request_kwargs["extra_body"] = {
                "keep_alive": "10m" if self.fast_reply else "30m",
                "options": {
                    "num_ctx": 1024 if self.fast_reply else 2048,
                    "num_predict": max_tokens,
                },
            }


def _split_reasoned_response(content: str) -> dict[str, str]:
    reasoning_label = "Reasoning summary:"
    answer_label = "Answer:"
    if reasoning_label not in content or answer_label not in content:
        return {"reply": content.strip(), "reasoning": ""}

    _, after_reasoning = content.split(reasoning_label, 1)
    reasoning, answer = after_reasoning.split(answer_label, 1)
    return {
        "reply": answer.strip(),
        "reasoning": reasoning.strip(),
    }
