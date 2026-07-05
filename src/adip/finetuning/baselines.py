"""Deterministic baselines the LoRA model has to beat (or honestly fail to).

Both run on core dependencies only (scikit-learn), so they are CI-safe: the
majority-class floor and a TF-IDF + logistic-regression classifier — the
classic strong baseline for topic-style classification.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from adip.finetuning.dataset import LabeledChunk


def classification_metrics(true_labels: list[str], predicted: list[str]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(true_labels, predicted)),
        "macro_f1": float(f1_score(true_labels, predicted, average="macro")),
    }


def majority_baseline(
    train: list[LabeledChunk], evaluation: list[LabeledChunk]
) -> dict[str, Any]:
    majority_label = Counter(chunk.label for chunk in train).most_common(1)[0][0]
    predicted = [majority_label] * len(evaluation)
    return {
        "approach": "majority_class",
        "majority_label": majority_label,
        **classification_metrics([chunk.label for chunk in evaluation], predicted),
    }


def tfidf_logreg_baseline(
    train: list[LabeledChunk],
    evaluation: list[LabeledChunk],
    seed: int = 13,
) -> dict[str, Any]:
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))),
            ("logreg", LogisticRegression(max_iter=2000, random_state=seed)),
        ]
    )
    pipeline.fit([chunk.text for chunk in train], [chunk.label for chunk in train])
    predicted = list(pipeline.predict([chunk.text for chunk in evaluation]))
    return {
        "approach": "tfidf_logistic_regression",
        **classification_metrics([chunk.label for chunk in evaluation], predicted),
    }
