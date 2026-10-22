"""Question decomposition: split a multi-part question into sub-questions.

A question like "What does OAuth 2.0 require, and how does TLS 1.3 handle
early data?" needs evidence from two documents, but one query for the whole
sentence tends to retrieve chunks of whichever document matches more of its
words. Retrieving per sub-question gives each part its own evidence stream.

Two modes behind one interface:

- ``rules`` — deterministic, syntax-only splitting (CI-safe). It splits at
  sentence boundaries and at coordinated questions (", and which ...",
  "; how ..."), and carries a leading "context:" prefix onto every part. It
  never splits noun phrases ("X and Y"), so ordinary questions stay whole;
  the verifier's coverage check handles what syntax can't see.
- ``llm`` — the planner model writes the sub-questions as JSON (prompt
  ``decompose_v2``, ``SUB_QUESTIONS_SCHEMA``); see ``adip.agents.nodes.plan_task``.
"""

from __future__ import annotations

import re

MAX_SUB_QUESTIONS = 4
MIN_SUB_QUESTION_WORDS = 3

SUB_QUESTIONS_SCHEMA = {
    "title": "sub_questions",
    "type": "object",
    "properties": {"sub_questions": {"type": "array", "items": {"type": "string"}}},
    "required": ["sub_questions"],
    "additionalProperties": False,
}

# Words that open a new question when they follow a conjunction.
QUESTION_OPENERS = (
    r"what|which|who|whom|whose|when|where|why|how|is|are|was|were|does|do|did|can|could|"
    r"should|would|will|must"
)
PREPOSITIONS = r"in|on|for|by|to|from|with|at|under|of"
SENTENCE_SPLIT = re.compile(
    # "...?" always ends a question; "." / "!" end a sentence only after a lowercase
    # word, so "U.S. Securities" and "e.g. TLS" are not split.
    r"(?<=\?)\s+(?=\S)|(?<=[a-z]{2}[.!])\s+(?=[A-Z])"
)
CLAUSE_SPLIT = re.compile(
    rf"(?:,\s*|;\s*|\s+)(?:and|also|plus)\s*,?\s+(?=(?:(?:{PREPOSITIONS})\s+)?(?:{QUESTION_OPENERS})\b)"
    rf"|;\s+(?=(?:{QUESTION_OPENERS})\b)",
    re.IGNORECASE,
)
LEADING_CONNECTIVE = re.compile(r"^(?:and|also|plus|then)\b[,:]?\s*", re.IGNORECASE)
LIST_PREFIX = re.compile(r"^\s*(?:[-*•]|\d+[.)]|q\d+[.:)])\s*", re.IGNORECASE)


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+", text))


def _clean(part: str) -> str:
    part = LEADING_CONNECTIVE.sub("", part.strip()).strip(" ,;")
    return part[:1].upper() + part[1:] if part else part


def split_question(question: str, max_parts: int = MAX_SUB_QUESTIONS) -> list[str]:
    """Deterministic decomposition. Returns ``[question]`` when there is nothing to split."""
    text = " ".join(question.split())
    parts: list[str] = []
    for sentence in SENTENCE_SPLIT.split(text):
        context = ""
        body = sentence
        # "Compare A with B: what is X, and which is Y?" -> carry "Compare A with B" onto each part.
        if ":" in sentence:
            head, tail = sentence.split(":", 1)
            if _word_count(head) >= MIN_SUB_QUESTION_WORDS and CLAUSE_SPLIT.search(tail):
                context, body = head.strip(), tail
        clauses = [_clean(clause) for clause in CLAUSE_SPLIT.split(body)]
        for clause in clauses:
            if not clause:
                continue
            parts.append(f"{context}: {clause}" if context and len(clauses) > 1 else clause)

    # Fragments too short to search on rejoin the part before them.
    merged: list[str] = []
    for part in parts:
        if merged and _word_count(part) < MIN_SUB_QUESTION_WORDS:
            merged[-1] = f"{merged[-1]} {part}"
        else:
            merged.append(part)
    if len(merged) <= 1:
        return [question]
    return merged[:max_parts]


def parse_sub_questions(text: str, question: str, max_parts: int = MAX_SUB_QUESTIONS) -> list[str]:
    """Sub-questions from planner model output: one per line, numbering and bullets tolerated.

    Falls back to ``[question]`` when the output has fewer than two usable lines, so a
    model that declines to split leaves the question whole.
    """
    seen: set[str] = set()
    parts: list[str] = []
    for line in (text or "").splitlines():
        cleaned = LIST_PREFIX.sub("", line).strip().strip('"').strip()
        if not cleaned or cleaned.endswith(":") or _word_count(cleaned) < MIN_SUB_QUESTION_WORDS:
            continue
        key = cleaned.lower()
        if key not in seen:
            seen.add(key)
            parts.append(cleaned)
    if len(parts) < 2:
        return [question]
    return parts[:max_parts]
