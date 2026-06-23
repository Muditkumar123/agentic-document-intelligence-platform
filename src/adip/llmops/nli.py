"""Answer-entailment (QNLI) scoring for evidence-gated abstention.

The lexical score threshold in the pipeline catches off-domain questions but not
in-domain-but-unanswerable ones (their evidence is topically related, so it scores
high). This module adds a semantic check: given a question and the retrieved
evidence, estimate whether the evidence actually answers the question, using a
QNLI cross-encoder ("does this sentence answer this question?").

The model is heavy (transformers + a downloaded checkpoint), so it is loaded
lazily and used only as an opt-in offline eval; CI keeps the deterministic
score-threshold gate. The pipeline depends only on the lightweight
``EntailmentScorer`` callable protocol, so tests inject a fake scorer.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

# QNLI cross-encoder: input is (question, sentence), output is P(sentence answers question).
DEFAULT_NLI_MODEL = "cross-encoder/qnli-distilroberta-base"


@runtime_checkable
class EntailmentScorer(Protocol):
    """Maps (question, evidence) to a confidence in [0, 1] that the evidence
    answers the question (1 = strongly answered)."""

    def __call__(self, question: str, evidence: list[dict[str, Any]]) -> float: ...


class NLIEntailmentScorer:
    """Scores answer-entailment with a QNLI cross-encoder.

    For each evidence chunk it scores the (question, chunk) pair and returns the
    maximum answer probability across chunks. Transformers and the model are
    imported/loaded lazily on first call, so importing this module stays light.

    Supports single-logit QNLI heads (sigmoid) and multi-label NLI heads (softmax
    over the ``entailment`` class), resolved from the model's own label map.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_NLI_MODEL,
        device: str | None = None,
        local_files_only: bool = True,
        batch_size: int = 16,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.local_files_only = local_files_only
        self.batch_size = batch_size
        self._model = None
        self._tokenizer = None
        self._num_labels: int | None = None
        self._positive_index: int = 0

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, local_files_only=self.local_files_only
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, local_files_only=self.local_files_only
        )
        model.eval()
        if self.device:
            model.to(self.device)
        self._model = model
        self._num_labels = int(model.config.num_labels)
        if self._num_labels > 1:
            # Multi-label NLI head: find the "entailment" class from the label map.
            id2label = getattr(model.config, "id2label", {}) or {}
            self._positive_index = next(
                (int(idx) for idx, label in id2label.items() if str(label).lower() == "entailment"),
                0,
            )

    def __call__(self, question: str, evidence: list[dict[str, Any]]) -> float:
        if not evidence:
            return 0.0
        self._ensure_loaded()
        import torch

        texts = [str(item.get("text", "")) for item in evidence]
        scores: list[float] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            inputs = self._tokenizer(
                [question] * len(batch),
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            )
            if self.device:
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                logits = self._model(**inputs).logits
                if self._num_labels == 1:
                    probabilities = torch.sigmoid(logits).reshape(-1)
                else:
                    probabilities = torch.softmax(logits, dim=-1)[:, self._positive_index]
            scores.extend(float(value) for value in probabilities.detach().cpu().tolist())
        return max(scores) if scores else 0.0
