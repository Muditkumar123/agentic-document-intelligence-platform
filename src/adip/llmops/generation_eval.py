"""Deterministic answer-quality (generation) evaluation.

Scores generated answers along four axes, all without calling another model so it
runs in CI and is fully reproducible:

- faithfulness: are the answer's claim sentences grounded in the retrieved evidence?
- answer_relevance: how much of the question does the answer actually address?
- expected_coverage: did the answer surface the golden "expected" facts?
- citation_coverage: are the retrieved citations visible in the answer?

The same harness can drive a hosted or local writer (see
``adip.mlops.run_generation_eval``) to compare real models, and an LLM judge can
be layered on later behind the same report shape.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

from adip.llmops.evaluation import evaluate_generation
from adip.llmops.judge import JudgeVerdict
from adip.llmops.models import meaningful_terms


@dataclass(frozen=True)
class GenerationEvalCase:
    question: str
    refusal: bool
    grounded: bool
    faithfulness: float | None
    answer_relevance: float
    expected_coverage: float | None
    citation_coverage: float
    supported_token_count: int
    answer_token_count: int
    answerable: bool = True
    judge_faithfulness: float | None = None
    judge_relevance: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def with_judge(case: GenerationEvalCase, verdict: JudgeVerdict | None) -> GenerationEvalCase:
    """Attach a judge verdict to a scored case (no-op when judging failed)."""
    if verdict is None:
        return case
    return replace(case, judge_faithfulness=verdict.faithfulness, judge_relevance=verdict.relevance)


@dataclass(frozen=True)
class GenerationEvalReport:
    case_count: int
    answered_count: int
    mean_faithfulness: float
    mean_answer_relevance: float
    mean_expected_coverage: float
    mean_citation_coverage: float
    grounded_rate: float
    refusal_rate: float
    unanswerable_count: int
    refusal_precision: float
    refusal_recall: float
    cases: list[dict[str, Any]]
    judged_count: int = 0
    judge_mean_faithfulness: float | None = None
    judge_mean_relevance: float | None = None
    judge_lexical_faithfulness_gap: float | None = None
    judge_lexical_correlation: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def metrics(self) -> dict[str, float]:
        values = {
            "gen_eval_case_count": float(self.case_count),
            "gen_eval_answered_count": float(self.answered_count),
            "gen_eval_mean_faithfulness": self.mean_faithfulness,
            "gen_eval_mean_answer_relevance": self.mean_answer_relevance,
            "gen_eval_mean_expected_coverage": self.mean_expected_coverage,
            "gen_eval_mean_citation_coverage": self.mean_citation_coverage,
            "gen_eval_grounded_rate": self.grounded_rate,
            "gen_eval_refusal_rate": self.refusal_rate,
            "gen_eval_unanswerable_count": float(self.unanswerable_count),
            "gen_eval_refusal_precision": self.refusal_precision,
            "gen_eval_refusal_recall": self.refusal_recall,
        }
        # Judge metrics appear only when a judge actually ran, so the deterministic
        # CI metrics file keeps its exact shape when the judge is off.
        if self.judged_count:
            values["gen_eval_judged_count"] = float(self.judged_count)
            if self.judge_mean_faithfulness is not None:
                values["gen_eval_judge_mean_faithfulness"] = self.judge_mean_faithfulness
            if self.judge_mean_relevance is not None:
                values["gen_eval_judge_mean_relevance"] = self.judge_mean_relevance
            if self.judge_lexical_faithfulness_gap is not None:
                values["gen_eval_judge_lexical_faithfulness_gap"] = self.judge_lexical_faithfulness_gap
            if self.judge_lexical_correlation is not None:
                values["gen_eval_judge_lexical_correlation"] = self.judge_lexical_correlation
        return values


def _coverage_ratio(target: set[str], reference: set[str]) -> float:
    if not target:
        return 0.0
    return len(target & reference) / len(target)


def score_answer(
    question: str,
    answer: str,
    evidence: list[dict[str, Any]],
    expected_substrings: list[str] | None = None,
    *,
    answerable: bool = True,
    grounded_threshold: float = 0.5,
    substring_overlap_threshold: float = 0.6,
) -> GenerationEvalCase:
    """Score a single answer against its evidence and golden expectations.

    Faithfulness is the fraction of the answer's meaningful tokens that also appear
    in the retrieved evidence (a lexical grounding proxy): high when the answer
    stays within the evidence, low when it introduces unsupported content. It is
    ``None`` for refusals, which are scored only by ``refusal_rate`` and
    ``expected_coverage`` so honest "insufficient evidence" answers are not
    penalised as hallucinations.
    """
    quality = evaluate_generation(answer, evidence)
    answer_tokens = meaningful_terms(answer)

    evidence_tokens: set[str] = set()
    for item in evidence:
        evidence_tokens |= meaningful_terms(str(item.get("text", "")))

    supported_tokens = answer_tokens & evidence_tokens
    if quality.refusal:
        faithfulness: float | None = None
    elif answer_tokens:
        faithfulness = len(supported_tokens) / len(answer_tokens)
    else:
        faithfulness = 0.0

    answer_relevance = _coverage_ratio(meaningful_terms(question), answer_tokens)

    expected = [text for text in (expected_substrings or []) if meaningful_terms(text)]
    if expected:
        covered = sum(
            1
            for text in expected
            if _coverage_ratio(meaningful_terms(text), answer_tokens) >= substring_overlap_threshold
        )
        expected_coverage: float | None = covered / len(expected)
    else:
        expected_coverage = None

    grounded = faithfulness is not None and faithfulness >= grounded_threshold
    return GenerationEvalCase(
        question=question,
        refusal=quality.refusal,
        grounded=grounded,
        faithfulness=faithfulness,
        answer_relevance=answer_relevance,
        expected_coverage=expected_coverage,
        citation_coverage=quality.citation_coverage,
        supported_token_count=len(supported_tokens),
        answer_token_count=len(answer_tokens),
        answerable=answerable,
    )


def aggregate_eval(cases: list[GenerationEvalCase]) -> GenerationEvalReport:
    if not cases:
        raise ValueError("No generation evaluation cases to aggregate")

    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    answered = [case for case in cases if not case.refusal]
    faithfulness_values = [case.faithfulness for case in cases if case.faithfulness is not None]
    expected_values = [case.expected_coverage for case in cases if case.expected_coverage is not None]

    grounded_rate = _mean([1.0 if case.grounded else 0.0 for case in answered])
    refusal_rate = _mean([1.0 if case.refusal else 0.0 for case in cases])

    # Abstention quality: treat "should refuse" (unanswerable) as the positive class.
    refused = [case for case in cases if case.refusal]
    unanswerable = [case for case in cases if not case.answerable]
    correct_refusals = sum(1 for case in cases if case.refusal and not case.answerable)
    # Precision: of all refusals, how many were on genuinely unanswerable questions
    # (1.0 when the writer never refused -> no false refusals).
    refusal_precision = correct_refusals / len(refused) if refused else 1.0
    # Recall: of all unanswerable questions, how many the writer correctly refused
    # (1.0 when there are no unanswerable questions to catch).
    refusal_recall = correct_refusals / len(unanswerable) if unanswerable else 1.0

    judged = [case for case in cases if case.judge_faithfulness is not None]
    # Agreement between the lexical faithfulness proxy and the judge, over cases
    # where both scored: mean absolute gap plus Pearson correlation (None when
    # there are too few points or a metric has zero variance).
    paired = [
        (case.faithfulness, case.judge_faithfulness)
        for case in judged
        if case.faithfulness is not None
    ]
    gap = _mean([abs(lexical - judge) for lexical, judge in paired]) if paired else None
    correlation = _pearson(paired)

    return GenerationEvalReport(
        case_count=len(cases),
        answered_count=len(answered),
        mean_faithfulness=_mean(faithfulness_values),
        mean_answer_relevance=_mean([case.answer_relevance for case in cases]),
        mean_expected_coverage=_mean(expected_values),
        mean_citation_coverage=_mean([case.citation_coverage for case in cases]),
        grounded_rate=grounded_rate,
        refusal_rate=refusal_rate,
        unanswerable_count=len(unanswerable),
        refusal_precision=refusal_precision,
        refusal_recall=refusal_recall,
        cases=[case.to_dict() for case in cases],
        judged_count=len(judged),
        judge_mean_faithfulness=_mean([case.judge_faithfulness for case in judged]) if judged else None,
        judge_mean_relevance=_mean(
            [case.judge_relevance for case in judged if case.judge_relevance is not None]
        )
        if judged
        else None,
        judge_lexical_faithfulness_gap=gap,
        judge_lexical_correlation=correlation,
    )


def _pearson(pairs: list[tuple[float, float]]) -> float | None:
    if len(pairs) < 2:
        return None
    xs = [x for x, _ in pairs]
    ys = [y for _, y in pairs]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return None
    return cov / (var_x**0.5 * var_y**0.5)
