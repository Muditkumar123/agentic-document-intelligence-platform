"""Prompt registry and rendering helpers."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from jinja2 import StrictUndefined, Template

DEFAULT_PROMPT_DIR = Path("prompts")

PROMPT_REGISTRY = {
    "qa": "qa_v1.txt",
    "brief": "brief_v1.txt",
    "plan": "plan_v1.txt",
    "verify": "verify_v1.txt",
}


@dataclass(frozen=True)
class PromptTemplate:
    task_type: str
    version: str
    path: str
    template: str
    template_hash: str

    def render(self, **values: Any) -> str:
        return Template(self.template, undefined=StrictUndefined).render(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_prompt_template(
    task_type: str,
    prompt_dir: Path = DEFAULT_PROMPT_DIR,
    version: str | None = None,
) -> PromptTemplate:
    normalized_task = task_type if task_type in PROMPT_REGISTRY else "qa"
    filename = version if version is not None else PROMPT_REGISTRY[normalized_task]
    path = prompt_dir.expanduser() / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")

    template = path.read_text(encoding="utf-8")
    return PromptTemplate(
        task_type=normalized_task,
        version=path.stem,
        path=str(path),
        template=template,
        template_hash=hashlib.sha256(template.encode("utf-8")).hexdigest(),
    )
