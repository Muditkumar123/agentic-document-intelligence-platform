"""Structured verifier output helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

SECTION_MARKERS = ("Supported claims:", "Missing evidence:", "Verifier decision:")


@dataclass(frozen=True)
class VerifierOutput:
    raw_text: str
    final_text: str
    structured: bool
    normalization_reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_verifier_output(raw_text: str, evidence: list[dict[str, Any]]) -> VerifierOutput:
    """Extract final verifier notes from models that may emit reasoning first."""
    cleaned = raw_text.strip()
    structured_text = extract_structured_sections(cleaned)
    if structured_text:
        return VerifierOutput(
            raw_text=cleaned,
            final_text=structured_text,
            structured=True,
            normalization_reason="structured_sections_found",
        )

    final_text = build_unstructured_fallback(evidence)
    return VerifierOutput(
        raw_text=cleaned,
        final_text=final_text,
        structured=False,
        normalization_reason="structured_sections_missing",
    )


def extract_structured_sections(text: str) -> str | None:
    lowered = text.lower()
    marker_positions = [
        lowered.find(marker.lower())
        for marker in SECTION_MARKERS
        if lowered.find(marker.lower()) >= 0
    ]
    if not marker_positions:
        return None
    return text[min(marker_positions) :].strip()


def build_unstructured_fallback(evidence: list[dict[str, Any]]) -> str:
    if not evidence:
        return (
            "Supported claims:\n"
            "- None. No retrieved evidence was available for verification.\n\n"
            "Missing evidence:\n"
            "- The verifier did not return structured evidence notes.\n\n"
            "Verifier decision:\n"
            "- Needs manual review before relying on the answer."
        )

    top = evidence[0]
    citation = top["citation"]
    return (
        "Supported claims:\n"
        "- The verifier did not return structured supported claims.\n\n"
        "Missing evidence:\n"
        "- Structured verifier sections were missing from the model output.\n\n"
        "Verifier decision:\n"
        f"- Needs manual review; the top retrieved chunk is available for checking ({citation})."
    )
