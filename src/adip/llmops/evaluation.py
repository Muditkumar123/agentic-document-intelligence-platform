"""Lightweight LLMOps quality checks."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class LLMQualityReport:
    citation_count: int
    visible_citation_count: int
    citation_coverage: float
    evidence_count: int
    unsupported_sentence_count: int
    answer_sentence_count: int
    refusal: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_generation(answer: str, evidence: list[dict[str, Any]]) -> LLMQualityReport:
    citations = [item["citation"] for item in evidence]
    visible = sum(1 for citation in citations if citation in answer)
    citation_coverage = visible / len(citations) if citations else 0.0
    sentences = split_claim_units(answer, citations)
    factual_sentences = [sentence for sentence in sentences if not is_heading_or_note(sentence)]
    unsupported = [
        sentence
        for sentence in factual_sentences
        if sentence.strip() and citations and not any(citation in sentence for citation in citations)
    ]
    refusal = "do not contain enough evidence" in answer.lower() or "insufficient evidence" in answer.lower()
    return LLMQualityReport(
        citation_count=len(citations),
        visible_citation_count=visible,
        citation_coverage=citation_coverage,
        evidence_count=len(evidence),
        unsupported_sentence_count=len(unsupported),
        answer_sentence_count=len(factual_sentences),
        refusal=refusal,
    )


def split_sentences(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1:
        return lines
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [part.strip() for part in parts if part.strip()]


def split_claim_units(text: str, citations: list[str]) -> list[str]:
    units: list[str] = []
    for line in [line.strip() for line in text.splitlines() if line.strip()]:
        if citations and any(citation in line for citation in citations):
            units.append(line)
        else:
            units.extend(split_sentences(line))
    return units


def is_heading_or_note(sentence: str) -> bool:
    lowered = sentence.lower().strip()
    return (
        lowered.endswith(":")
        or lowered.startswith("llmops note")
        or lowered.startswith("question:")
        or lowered.startswith("research brief:")
        or lowered.startswith("domain preset:")
        or lowered.startswith("grounded answer:")
        or lowered.startswith("summary:")
        or lowered.startswith("evidence:")
        or lowered.startswith("source coverage:")
        or lowered.startswith("verification notes:")
        or lowered.startswith("limitations:")
        or lowered.startswith("- ")
    )
