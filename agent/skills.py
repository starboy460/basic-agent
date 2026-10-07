from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    instructions: str
    path: Path

    def system_message(self) -> dict[str, str]:
        return {
            "role": "system",
            "content": (
                f"Active skill: {self.name}\n"
                f"Skill description: {self.description}\n"
                "Skill instructions:\n"
                f"{self.instructions.strip()}"
            ),
        }


class SkillCatalog:
    def __init__(self, directory: Path, skills: dict[str, Skill]):
        self.directory = directory
        self._skills = skills

    @classmethod
    def load(cls, directory: Path) -> "SkillCatalog":
        skills: dict[str, Skill] = {}
        if directory.exists():
            for path in sorted(directory.glob("*.md")):
                skill = _parse_skill_file(path)
                skills[skill.name] = skill
        return cls(directory, skills)

    def names(self) -> list[str]:
        return sorted(self._skills)

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name.strip().lower())

    def summary_lines(self, active_name: str | None = None) -> list[str]:
        lines: list[str] = []
        active_name = active_name.strip().lower() if active_name else None
        for name in self.names():
            skill = self._skills[name]
            marker = "*" if name == active_name else " "
            lines.append(f"{marker} {skill.name}: {skill.description}")
        if not lines:
            lines.append(f"No skills found in {self.directory}")
        return lines


def _parse_skill_file(path: Path) -> Skill:
    raw = path.read_text(encoding="utf-8")
    metadata, body = _split_front_matter(raw)
    name = (metadata.get("name") or path.stem).strip().lower()
    description = metadata.get("description", "").strip()
    if not description:
        description = _first_nonempty_line(body) or "No description provided."
    instructions = body.strip()
    if not instructions:
        instructions = description
    return Skill(
        name=name,
        description=description,
        instructions=instructions,
        path=path,
    )


def _split_front_matter(text: str) -> tuple[dict[str, str], str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text

    metadata: dict[str, str] = {}
    body_start = 0
    for index in range(1, len(lines)):
        line = lines[index].strip()
        if line == "---":
            body_start = index + 1
            break
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip().lower()] = value.strip().strip('"')
    if body_start == 0:
        return {}, text
    return metadata, "\n".join(lines[body_start:])


def _first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip().lstrip("#").strip()
    return ""


def skill_names(skills: Iterable[Skill]) -> list[str]:
    return sorted(skill.name for skill in skills)
