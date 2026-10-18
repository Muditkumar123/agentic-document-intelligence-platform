"""Multi-hop evaluation: questions whose answer needs evidence from two documents.

Each probe in ``data/eval/multihop_probes.jsonl`` lists its *hops*: the
document(s) that can supply one piece of the answer and the exact phrase that
piece must contain. A run is scored on the evidence the agent finally hands the
writer (capped at ``top_k``, so a variant cannot win by retrieving more) and on
the answer the writer produced:

- ``hop_recall`` — fraction of hops whose phrase appears in a retrieved chunk of
  one of that hop's documents (same substring rule as retrieval hit rate).
- ``full_support_rate`` — fraction of probes with every hop supported.
- ``answer_hop_coverage`` / ``answer_full_coverage`` — the same, measured on the
  answer text with the token-overlap rule used by ``expected_coverage``.

The corpus is the short CI corpus (``data/eval/raw``) plus full-length documents
(``data/eval/long``), so a single query competes with ~170 chunks instead of 19.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from adip.agents.models import AgentState
from adip.ingestion.pipeline import ingest_path
from adip.llmops.generation_eval import score_answer
from adip.llmops.models import meaningful_terms
from adip.llmops.pipeline import build_evidence
from adip.rag.chunks import read_chunks_jsonl
from adip.rag.retriever import RagIndex, build_index

DEFAULT_PROBES_PATH = Path("data/eval/multihop_probes.jsonl")
DEFAULT_CORPUS_DIRS = (Path("data/eval/raw"), Path("data/eval/long"))
ANSWER_OVERLAP_THRESHOLD = 0.6


def read_probes(path: Path = DEFAULT_PROBES_PATH) -> list[dict[str, Any]]:
    probes: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        hops = row.get("hops") or []
        if not row.get("question") or len(hops) < 2:
            raise ValueError(f"{path}:{line_number} needs a question and at least two hops")
        for hop in hops:
            if not hop.get("documents") or not hop.get("expected_substrings"):
                raise ValueError(f"{path}:{line_number} has a hop without documents or expected_substrings")
        probes.append(row)
    if not probes:
        raise ValueError(f"No multi-hop probes found in {path}")
    return probes


def _normalized(text: str) -> str:
    return " ".join(str(text).split()).lower()


def hop_supported(hop: dict[str, Any], retrieved: Iterable[dict[str, Any]]) -> bool:
    documents = set(hop["documents"])
    needles = [_normalized(text) for text in hop["expected_substrings"]]
    for item in retrieved:
        chunk = item["chunk"]
        if chunk.get("filename") in documents and any(needle in _normalized(chunk["text"]) for needle in needles):
            return True
    return False


def answer_covers_hop(hop: dict[str, Any], answer: str, threshold: float = ANSWER_OVERLAP_THRESHOLD) -> bool:
    answer_terms = meaningful_terms(answer)
    for text in hop["expected_substrings"]:
        terms = meaningful_terms(text)
        if terms and len(terms & answer_terms) / len(terms) >= threshold:
            return True
    return False


def loop_activity(state: AgentState) -> dict[str, Any]:
    """How much retrieval work a run did (defaults describe the single-pass pipeline)."""
    sub_questions = list(getattr(state, "sub_questions", []) or [state.question])
    return {
        "retrieval_calls": int(state.metrics.get("retrieval_call_count", 1)),
        "retrieval_rounds": int(state.metrics.get("retrieval_round_count", 1)),
        "sub_questions": sub_questions,
        "decomposed": len(sub_questions) > 1,
        "latency_ms": float(state.metrics.get("workflow_duration_ms", 0.0)),
    }


@dataclass
class MultihopCase:
    id: str
    type: str
    question: str
    hops_supported: list[bool]
    answer_hops_covered: list[bool]
    evidence_files: list[str]
    retrieval_calls: int
    retrieval_rounds: int
    decomposed: bool
    sub_questions: list[str] = field(default_factory=list)
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def score_multihop_case(probe: dict[str, Any], state: AgentState) -> MultihopCase:
    activity = loop_activity(state)
    return MultihopCase(
        id=str(probe.get("id", "")),
        type=str(probe.get("type", "")),
        question=probe["question"],
        hops_supported=[hop_supported(hop, state.retrieved) for hop in probe["hops"]],
        answer_hops_covered=[answer_covers_hop(hop, state.final_answer) for hop in probe["hops"]],
        evidence_files=[item["chunk"].get("filename", "") for item in state.retrieved],
        retrieval_calls=activity["retrieval_calls"],
        retrieval_rounds=activity["retrieval_rounds"],
        decomposed=activity["decomposed"],
        sub_questions=activity["sub_questions"],
        latency_ms=activity["latency_ms"],
    )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate_multihop(cases: Sequence[MultihopCase]) -> dict[str, Any]:
    def summarize(subset: Sequence[MultihopCase]) -> dict[str, float]:
        hops = [supported for case in subset for supported in case.hops_supported]
        answer_hops = [covered for case in subset for covered in case.answer_hops_covered]
        return {
            "probe_count": float(len(subset)),
            "hop_recall": _mean([float(value) for value in hops]),
            "full_support_rate": _mean([float(all(case.hops_supported)) for case in subset]),
            "answer_hop_coverage": _mean([float(value) for value in answer_hops]),
            "answer_full_coverage": _mean([float(all(case.answer_hops_covered)) for case in subset]),
            "avg_retrieval_calls": _mean([float(case.retrieval_calls) for case in subset]),
            "retry_rate": _mean([float(case.retrieval_rounds > 1) for case in subset]),
            "decomposition_rate": _mean([float(case.decomposed) for case in subset]),
            "avg_latency_ms": _mean([case.latency_ms for case in subset]),
        }

    by_type = {kind: summarize([case for case in cases if case.type == kind]) for kind in sorted({c.type for c in cases})}
    return {**summarize(cases), "by_type": by_type}


def score_golden_case(row: dict[str, Any], state: AgentState) -> dict[str, Any]:
    """Single-hop regression check: did the loop keep (or lose) the answer, and what did it cost?"""
    expected = [_normalized(text) for text in row.get("expected_substrings", [])]
    hit = any(any(text in _normalized(item["chunk"]["text"]) for text in expected) for item in state.retrieved)
    case = score_answer(
        row["question"],
        state.final_answer,
        build_evidence(state.retrieved),
        expected_substrings=row.get("expected_substrings"),
    )
    activity = loop_activity(state)
    return {
        "question": row["question"],
        "category": row.get("category"),
        "hit": hit,
        "expected_coverage": case.expected_coverage,
        "retrieval_calls": activity["retrieval_calls"],
        "retrieval_rounds": activity["retrieval_rounds"],
        "decomposed": activity["decomposed"],
        "latency_ms": activity["latency_ms"],
    }


def aggregate_golden(cases: Sequence[dict[str, Any]]) -> dict[str, float]:
    coverage = [case["expected_coverage"] for case in cases if case["expected_coverage"] is not None]
    return {
        "question_count": float(len(cases)),
        "hit_rate": _mean([float(case["hit"]) for case in cases]),
        "expected_coverage": _mean(coverage),
        "avg_retrieval_calls": _mean([float(case["retrieval_calls"]) for case in cases]),
        "retry_rate": _mean([float(case["retrieval_rounds"] > 1) for case in cases]),
        "decomposition_rate": _mean([float(case["decomposed"]) for case in cases]),
        "avg_latency_ms": _mean([case["latency_ms"] for case in cases]),
    }


def build_corpus_index(
    corpus_dirs: Sequence[Path],
    work_dir: Path,
    backend: str = "tfidf",
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> RagIndex:
    """Ingest every corpus directory, index the combined chunks, and save the index under ``work_dir``."""
    work_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[dict[str, Any]] = []
    for position, corpus_dir in enumerate(corpus_dirs):
        chunks_path = work_dir / f"chunks_{position}_{corpus_dir.name}.jsonl"
        ingest_path(
            input_path=corpus_dir,
            output_path=chunks_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks.extend(read_chunks_jsonl(chunks_path))
    index = build_index(chunks, backend=backend, use_faiss=False)
    index.save(work_dir / f"index_{backend}")
    return index
