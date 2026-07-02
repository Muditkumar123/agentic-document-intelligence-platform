# Continuous Integration & Quality Gates

CI turns the project's deterministic evaluations into automated regression gates. The pipeline lives in [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) and runs on every push to `main`, every pull request, and on manual dispatch.

## Pipeline Stages

### 1. `test` — unit tests

- Installs the package with only the declared dependencies: `pip install -e ".[dev,api]"` (no torch, faiss, or transformers — those are lazy-imported and stubbed in tests).
- Runs the full `pytest` suite on a matrix of Python **3.10, 3.11, 3.12, and 3.14**.

### 2. `eval-gate` — answer-quality regression gate

Runs only after `test` passes, and reproduces the evaluation pipeline end to end on a clean machine, against the **real public-document evaluation corpus** (`data/eval/`, markdown-only — no system PDF tooling needed):

1. **Ingestion** — `run_ingestion` parses `data/eval/raw/` (18 real public docs across 5 categories) into `data/processed/chunks.jsonl`.
2. **Retrieval eval** — `run_rag_eval` builds a TF-IDF index (`--no-faiss`, no reranker, hermetic) over the corpus, scored against `data/eval/golden_qa.jsonl` (45 questions), and writes `data/monitoring/rag_eval_metrics.json` with `hit_rate_at_k`, `mrr`, and per-category slices.
3. **Generation eval** — `run_generation_eval --abstention-threshold 0.10` answers the golden questions with the deterministic extractive writer, refusing when evidence is too weak, and writes `data/monitoring/generation_eval_metrics.json` with the `gen_eval_*` metrics (including refusal precision/recall over the 10 unanswerable questions).
4. **Gate** — `eval_gate` compares those metrics against [`ci/eval_thresholds.json`](../ci/eval_thresholds.json) and exits non-zero if any check fails.

See [EVALUATION_DATASET.md](EVALUATION_DATASET.md) for the corpus, sources, and licensing. Retrieval is saturated (1.0) on this lexically-distinct corpus, so the discriminating gate is generation faithfulness.

The metric JSON files are uploaded as build artifacts (even on failure) for inspection.

## Why The Gate Is Stable, Not Flaky

Both evals use deterministic components — TF-IDF retrieval and the extractive baseline writer call no external model and have no sampling. The same corpus produces the same metrics every run, so a threshold breach means a *real* regression in retrieval or grounding, not noise.

## The Thresholds

Floors (`min`) and a ceiling (`max`) are set with margin around the measured baseline so ordinary corpus tweaks pass but genuine regressions fail:

Measured on the real public-document corpus (`data/eval/`):

| Group | Metric | Bound | Threshold | Baseline |
| --- | --- | --- | --- | --- |
| retrieval | `hit_rate_at_k` | min | 0.85 | 1.00 |
| retrieval | `mrr` | min | 0.80 | 1.00 |
| generation | `gen_eval_mean_faithfulness` | min | 0.45 | 0.594 |
| generation | `gen_eval_grounded_rate` | min | 0.80 | 0.94 |
| generation | `gen_eval_mean_expected_coverage` | min | 0.65 | 0.795 |
| generation | `gen_eval_mean_answer_relevance` | min | 0.80 | 0.909 |
| generation | `gen_eval_mean_citation_coverage` | min | 0.55 | 0.705 |
| generation | `gen_eval_refusal_precision` | min | 0.80 | 1.00 |
| generation | `gen_eval_refusal_recall` | min | 0.40 | 0.50 |
| generation | `gen_eval_refusal_rate` | max | 0.20 | 0.091 |

Generation eval runs with `--abstention-threshold 0.10`: the writer refuses when the best retrieved evidence score is below 0.10. `refusal_precision` / `refusal_recall` treat "should refuse" (the 10 unanswerable golden questions) as the positive class — so the gate enforces that abstention catches off-domain questions (recall) without falsely refusing real ones (precision). If abstention were accidentally disabled, recall would drop to 0 and fail the gate.

A metric that is missing or non-numeric in the report is treated as a **failure** (`MISSING`), so a renamed or dropped metric cannot silently disable a gate.

## Run The Gate Locally

```bash
conda run -n crypto_env env PYTHONPATH=src python -m adip.mlops.run_ingestion \
  --input data/eval/raw --output data/processed/chunks.jsonl
conda run -n crypto_env env PYTHONPATH=src python -m adip.mlops.run_rag_eval \
  --chunks data/processed/chunks.jsonl --index data/processed/vector_index \
  --golden data/eval/golden_qa.jsonl --backend tfidf --no-faiss --reranker none --top-k 5
conda run -n crypto_env env PYTHONPATH=src python -m adip.mlops.run_generation_eval \
  --index data/processed/vector_index --golden data/eval/golden_qa.jsonl
conda run -n crypto_env env PYTHONPATH=src python -m adip.mlops.eval_gate \
  --thresholds ci/eval_thresholds.json
```

The gate prints a table of every check with its bound, threshold, value, and `PASS` / `FAIL` / `MISSING` status, then a summary line, and returns exit code `1` on any failure.

## Updating Thresholds

When an intentional change moves the baseline (new documents, a retriever change, a better writer):

1. Regenerate the metric files with the two eval commands above.
2. Edit `ci/eval_thresholds.json` to re-establish margin around the new baseline.
3. Confirm `eval_gate` passes locally, then commit the new thresholds alongside the change so the gate and the baseline move together.

## Reproducing CI's Environment

CI installs only the declared dependencies, which is stricter than a typical development environment. To catch undeclared imports before pushing (see Story 4 in [INTERVIEW_GUIDE.md](INTERVIEW_GUIDE.md)), verify in a clean venv:

```bash
python -m venv /tmp/ci_check
/tmp/ci_check/bin/pip install -e ".[dev,api]"
/tmp/ci_check/bin/pytest -q
```

## Possible Extensions

- Add a lint/format job (ruff, black) as an additional gate.
- Run the cross-encoder retrieval variant on a schedule (it needs a model download, so keep it out of the per-commit hermetic path).
- Add an LLM-as-judge generation eval behind the same report shape for a semantic faithfulness gate on a nightly cadence.
