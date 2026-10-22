"""Prompt registry and rendering helpers."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from jinja2 import Environment, StrictUndefined

DEFAULT_PROMPT_DIR = Path("prompts")

PROMPT_REGISTRY = {
    "qa": "qa_v2.txt",
    "brief": "brief_v2.txt",
    "plan": "plan_v1.txt",
    "verify": "verify_v3.txt",
    "judge": "judge_v1.txt",
    "rewrite": "rewrite_v1.txt",
    "decompose": "decompose_v2.txt",
}


# A "<" that starts an evidence tag, in any case or spacing: documents must not close their
# <evidence> block or open a new one.
EVIDENCE_TAG = re.compile(r"<(?=\s*/?\s*evidence\b)", re.IGNORECASE)


def fence_safe(text: Any) -> str:
    """Text that cannot break out of an ``<evidence>`` block (the ``fence`` template filter)."""
    return EVIDENCE_TAG.sub("&lt;", str(text))


PROMPT_ENV = Environment(undefined=StrictUndefined)
PROMPT_ENV.filters["fence"] = fence_safe


@dataclass(frozen=True)
class PromptTemplate:
    task_type: str
    version: str
    path: str
    template: str
    template_hash: str

    def render(self, **values: Any) -> str:
        return PROMPT_ENV.from_string(self.template).render(**values)

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


# Phrases that address the model rather than inform the reader. Matching text is still used as
# evidence (a document may be *about* prompt injection); it is only flagged for review.
PROMPT_INJECTION_PATTERN = re.compile(
    r"ignore (?:all |any |the )?(?:previous|prior|above|earlier) (?:instructions|prompts|rules)"
    r"|disregard (?:all |any |the )?(?:previous|prior|above|earlier)"
    r"|(?:reveal|print|show) (?:your |the )?system prompt"
    r"|you are now (?:a|an|in)\b"
    r"|new instructions\s*:"
    r"|</?\s*(?:system|evidence)\b",
    re.IGNORECASE,
)


def looks_like_prompt_injection(text: str) -> bool:
    return PROMPT_INJECTION_PATTERN.search(text) is not None

