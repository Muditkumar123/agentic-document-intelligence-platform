"""RAGAS standardized evaluation adapter.

The deterministic generation eval and the LLM judge are project-specific scorers.
RAGAS adds the industry-standard versions of the same questions — faithfulness,
answer relevancy, context precision, context recall — so results can be compared
against other RAG systems using shared definitions. The two context metrics also
give *graded* retrieval evaluation, which matters here because hit@k / MRR are
saturated at 1.0 on the current golden set and can no longer discriminate.

Like the judge and the NLI scorer, RAGAS is opt-in and offline: it needs an LLM
(any OpenAI-compatible endpoint, hosted or local ``adip.serving``) plus a local
sentence-transformers embedding model, so it never runs in the deterministic CI
gate. Heavy dependencies live behind the ``[ragas]`` extra and are imported
lazily; the eval pipeline depends only on the ``RagasScorer`` protocol, so tests
inject a fake. API keys are passed per call and never written to disk or logged.
"""

from __future__ import annotations

import importlib.util
import math
from dataclasses import asdict, dataclass
from typing import Any, Protocol, runtime_checkable

DEFAULT_RAGAS_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RAGAS_METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)


@dataclass(frozen=True)
class RagasScores:
    """Per-case RAGAS metric values (None when a metric failed or was skipped)."""

    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@runtime_checkable
class RagasScorer(Protocol):
    """Scores a batch of RAGAS rows; one RagasScores (or None on failure) per row."""

    def __call__(self, rows: list[dict[str, Any]]) -> list[RagasScores | None]: ...


def build_ragas_row(
    question: str,
    answer: str,
    evidence: list[dict[str, Any]],
    expected_substrings: list[str] | None = None,
) -> dict[str, Any]:
    """Convert one answered eval case into RAGAS's sample format.

    ``reference`` (the ground truth, needed by context_recall) comes from the
    golden row's expected substrings; when a row has none the reference is empty
    and RAGAS reports NaN for reference-based metrics, which we map to None.
    """
    return {
        "user_input": question,
        "response": answer,
        "retrieved_contexts": [str(item.get("text", "")) for item in evidence],
        "reference": " ".join(expected_substrings or []),
    }


def _nan_to_none(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    if math.isnan(number):
        return None
    return max(0.0, min(1.0, number))


class RagasEvaluator:
    """Runs the four standard RAGAS metrics against an OpenAI-compatible LLM.

    The LLM endpoint may be hosted (e.g. Gemini's OpenAI-compatible API) or the
    project's local serving layer. Embeddings (needed by answer_relevancy) are a
    local sentence-transformers model, so no second API key is required.
    """

    def __init__(
        self,
        model_name: str,
        endpoint_url: str | None = None,
        api_key: str | None = None,
        embedding_model: str = DEFAULT_RAGAS_EMBEDDING_MODEL,
        timeout: float = 600.0,
        max_workers: int = 4,
    ) -> None:
        for module in ("ragas", "langchain_openai", "langchain_huggingface"):
            if importlib.util.find_spec(module) is None:
                raise ImportError(
                    f"{module} is required for RAGAS evaluation. "
                    'Install the extra: pip install -e ".[ragas]"'
                )
        self.model_name = model_name
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.embedding_model = embedding_model
        # Applied to both the HTTP client and RAGAS's per-job runner: multi-call
        # metric chains (faithfulness, context_precision) queue behind one another
        # on a single local server, so RAGAS's 180s default times out well before
        # the underlying calls fail.
        self.timeout = timeout
        self.max_workers = max_workers

    def __call__(self, rows: list[dict[str, Any]]) -> list[RagasScores | None]:
        if not rows:
            return []

        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_openai import ChatOpenAI
        from ragas import EvaluationDataset, RunConfig, evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        llm = LangchainLLMWrapper(
            ChatOpenAI(
                model=self.model_name,
                base_url=self.endpoint_url,
                api_key=self.api_key or "not-needed",
                timeout=self.timeout,
                temperature=0.0,
            )
        )
        embeddings = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(model_name=self.embedding_model)
        )
        dataset = EvaluationDataset.from_list(rows)
        try:
            result = evaluate(
                dataset=dataset,
                metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
                llm=llm,
                embeddings=embeddings,
                run_config=RunConfig(timeout=int(self.timeout), max_workers=self.max_workers),
                show_progress=False,
            )
        except Exception as exc:  # endpoint/parsing failure: skip batch, don't crash the eval
            print(f"ragas evaluation failed: {exc}")
            return [None] * len(rows)

        frame = result.to_pandas()
        scores: list[RagasScores | None] = []
        for _, record in frame.iterrows():
            values = {
                name: _nan_to_none(record.get(name)) for name in RAGAS_METRIC_NAMES
            }
            if all(value is None for value in values.values()):
                scores.append(None)
            else:
                scores.append(RagasScores(**values))
        # RAGAS preserves row order; pad defensively if the frame came back short.
        while len(scores) < len(rows):
            scores.append(None)
        return scores
