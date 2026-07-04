# LLMOps Foundation

This phase adds prompt, model, generation, and quality tracking around the answer writer.

## What Is Implemented

- Versioned prompt templates in `prompts/`.
- Prompt rendering with template hashes.
- Deterministic grounded generation baseline.
- Optional Hugging Face text-generation adapter for local model experiments.
- Token count tracking.
- Generation latency tracking.
- Citation coverage checks.
- Unsupported sentence checks.
- Truncation/answer-budget warnings for hosted thinking models (`answer_warning`).
- Reasoning-block (`<think>`) stripping for both hosted and local reasoning models.
- Deterministic answer-quality evaluation (faithfulness, relevance, expected-fact coverage, citations) over the golden set.
- Evidence-gated abstention in two modes: a deterministic lexical score threshold (the CI gate) and an opt-in QNLI answer-entailment check (`--abstention-mode nli`); measured with refusal precision/recall over unanswerable questions (lexical 1.0/0.5, NLI 1.0/1.0 on the eval corpus).
- Optional LLM-as-judge scoring (`--judge-model-name`) with lexical-vs-judge agreement metrics, behind the same report shape.
- Structured verifier-output normalization for reasoning models.
- LLMOps JSON reports.
- MLOps-tracked LLMOps smoke command.
- Agent writer integration, so agent traces now include LLMOps metadata.

## Prompt Templates

Current prompts:

- `prompts/qa_v1.txt`
- `prompts/brief_v1.txt`
- `prompts/plan_v1.txt`
- `prompts/verify_v1.txt`

Every LLMOps run records:

- Prompt path
- Prompt version
- Prompt hash
- Model profile
- Model provider
- Model name
- Input token count
- Output token count
- Latency
- Citation quality metrics

## Run The LLMOps Smoke Command

```bash
conda run -n crypto_env env PYTHONPATH=src python -m adip.llmops \
  --index data/processed/vector_index \
  --question "What does the platform do with documents?" \
  --task qa \
  --top-k 3
```

Tracked version:

```bash
conda run -n crypto_env env PYTHONPATH=src python -m adip.mlops.run_llmops_smoke \
  --index data/processed/vector_index \
  --question "What does the platform do with documents?" \
  --task qa \
  --top-k 3
```

## Agent Integration

Agent runs now use the LLMOps writer node by default:

```bash
conda run -n crypto_env env PYTHONPATH=src python -m adip.agents \
  --index data/processed/vector_index \
  --question "Create a research brief about the document intelligence platform." \
  --task brief \
  --domain academic \
  --top-k 3
```

The AgentOps trace includes:

- Prompt metadata
- Model metadata
- Token counts
- Latency
- Citation coverage
- Unsupported sentence count

Agents can also run a separate reasoning verifier before the writer:

```bash
conda run -n crypto_env env PYTHONPATH=src python -m adip.agents \
  --index data/processed/vector_index \
  --question "Does the evidence support the platform claims?" \
  --task qa \
  --top-k 3 \
  --model-profile qwen3_8b_default \
  --reasoning-model-profile deepseek_r1_distill_qwen_14b_reasoning \
  --use-reasoning-planner \
  --reasoning-max-new-tokens 256
```

Optional reasoning planner runs are tracked with `planning_llm_*` metrics and `planning_llmops` metadata. Reasoning verifier runs are tracked with `reasoning_llm_*` metrics and `reasoning_llmops` metadata in the AgentOps trace.

Verifier runs keep the raw model output in `generation.text`, then normalize the final verifier notes into `structured_output.final_text` before scoring citation coverage. This prevents reasoning-style models from being judged only on hidden-analysis prose. Verifier metrics include:

- `reasoning_llm_structured_output`
- `reasoning_llm_normalized_answer_char_count`
- `reasoning_llm_citation_coverage`

## Optional Hugging Face Adapter

The default provider is `extractive`, which is deterministic and safe for tests.

When a local Hugging Face text-generation model is available, use:

```bash
conda run -n crypto_env env PYTHONPATH=src python -m adip.llmops \
  --index data/processed/vector_index \
  --question "What does the platform do with documents?" \
  --provider huggingface \
  --model-name /path/to/local/model
```

By default, the Hugging Face adapter uses local files only. Pass `--allow-download` only when you intentionally want the model loader to fetch files.

## Model Profiles

Model profiles live in `config/model_profiles.yaml` and can be selected with `--model-profile`.

Recommended profiles:

- `qwen3_8b_default`
- `deepseek_r1_distill_qwen_14b_reasoning`
- `deepseek_r1_distill_qwen_32b_stretch`

See [MODEL_PROFILES.md](MODEL_PROFILES.md).

## Answer-Quality Evaluation

`python -m adip.mlops.run_generation_eval` scores generated answers over the golden Q&A set and logs a tracked MLOps run:

- **faithfulness**: fraction of the answer's meaningful tokens grounded in the retrieved evidence (a lexical proxy; the optional LLM judge below scores the same answers semantically for comparison). `None` for refusals.
- **answer_relevance**: how much of the question the answer addresses.
- **expected_coverage**: whether the answer surfaced the golden "expected" facts.
- **citation_coverage**: are the retrieved citations visible in the answer.
- **grounded_rate / refusal_rate**: share of answers above the grounding threshold, and how often the writer declined.

It is deterministic with the extractive baseline (CI-safe) and can drive any hosted or local writer for model comparisons. The latest report is exposed at `GET /monitoring/generation-eval` and on the dashboard's **Answer Quality** tiles. See the tracked command in [../README.md](../README.md#answer-quality-evaluation).

### LLM-as-Judge (optional second opinion)

The lexical faithfulness score is a cheap proxy: reproducible and CI-safe, but blind to paraphrase and to subtle unsupported claims. An optional **model judge** scores the same answers semantically, behind the same report shape:

```bash
conda run -n crypto_env env PYTHONPATH=src python -m adip.mlops.run_generation_eval \
  --index data/processed/vector_index \
  --golden data/eval/golden_qa.jsonl \
  --judge-model-name gemini-2.5-flash \
  --judge-endpoint-url https://generativelanguage.googleapis.com/v1beta/openai/chat/completions \
  --judge-api-key "$GEMINI_API_KEY" \
  --judge-limit 10
```

- The judge uses the versioned prompt `prompts/judge_v1.txt` (hashed like every other prompt) and any OpenAI-compatible endpoint; the API key is session-only and never logged or written to disk.
- Each answered case gains `judge_faithfulness` / `judge_relevance`; refusals are skipped (judging a refusal's faithfulness is meaningless); a malformed judge response is counted as a failure, not a crash (reasoning blocks and markdown fences are tolerated, scores are clamped to [0, 1]).
- The report adds `gen_eval_judge_mean_faithfulness`, `gen_eval_judge_mean_relevance`, and two **agreement** metrics against the lexical proxy: `gen_eval_judge_lexical_faithfulness_gap` (mean absolute difference) and `gen_eval_judge_lexical_correlation` (Pearson). A small gap/high correlation means the cheap CI proxy is trustworthy; divergence tells you exactly where it isn't.
- Judge metrics appear **only when a judge ran**, so the deterministic CI gate and its metrics file are completely unaffected.

**First live run** (gemini-2.5-flash judging the extractive baseline over the real-document corpus; 20 of 45 answers judged before the free-tier daily quota — every 429 was skipped gracefully, not fatally):

| Metric | Lexical proxy | LLM judge |
| --- | --- | --- |
| mean faithfulness | 0.594 | **0.965** |
| mean relevance | 0.909 | **0.338** |
| faithfulness gap / correlation | — | 0.407 / ≈0 |

Two findings the proxy could not see:

1. The lexical proxy **systematically underestimates extractive faithfulness**: the writer copies evidence verbatim (nothing fabricated, judge ≈ 1.0), but headers and formatting tokens drag token-overlap down to ~0.6. The near-zero correlation confirms the proxy doesn't even rank cases the way the judge does for this writer.
2. The judge exposed the extractive writer's **real weakness — relevance**: half of the judged answers scored ≤ 0.3 because the writer returns a *relevant chunk* without directly *answering the question*, while lexical answer-relevance said 1.0 (the answer echoes the question's words). This is the concrete argument for a generative writer over the extractive baseline.

### Local judge and inter-judge agreement

The judge works against any OpenAI-compatible endpoint, including the project's own local serving layer — so full-coverage judging needs no API key at all:

```bash
# terminal 1: serve a local model
conda run -n crypto_env env PYTHONPATH=src python -m adip.serving server \
  --model-profile qwen3_8b_default --port 8091

# terminal 2: judge with it
conda run -n crypto_env env PYTHONPATH=src python -m adip.mlops.run_generation_eval \
  --index data/processed/vector_index --golden data/eval/golden_qa.jsonl \
  --abstention-threshold 0.10 \
  --judge-model-name "Qwen/Qwen3-8B" --judge-endpoint-url http://127.0.0.1:8091/v1 \
  --judge-max-new-tokens 2048
```

A full 50-case run with a local Qwen3-8B judge (no quota, no cost) enabled the standard reliability check a single judge can't give you — **inter-judge agreement**, computed per-case against gemini-2.5-flash on the 20 answers both judged:

| Dimension | Qwen↔Gemini mean gap | Pearson | Reading |
| --- | --- | --- | --- |
| relevance | 0.222 | **0.615** | judges broadly agree (Qwen 0.40 vs Gemini 0.34 mean) — the "extractive writer has weak relevance" finding is robust across judges |
| faithfulness | 0.585 | **≈ 0** | judges fundamentally disagree (Qwen 0.40 vs Gemini 0.97 mean) — the 8B judge's faithfulness scores are not trustworthy for this writer |

Interpretation: for verbatim-extractive answers, Gemini's ~1.0 faithfulness is almost certainly correct (nothing is fabricated), so the local 8B judge's low scores look like dimension bleed — punishing "doesn't fully answer the question" inside the faithfulness score. Practical policy that follows: a small local judge is usable for **relevance-style** dimensions and for cheap full-coverage sweeps, but **faithfulness judging needs a stronger model** — and inter-judge agreement is precisely the measurement that tells you which is which.

### Standardized RAGAS metrics (optional)

The deterministic eval and the LLM judge are project-specific scorers. [RAGAS](https://docs.ragas.io/) adds the **industry-standard** versions of the same questions, so results can be compared against other RAG systems using shared metric definitions:

- **faithfulness** — is every claim in the answer supported by the retrieved contexts?
- **answer_relevancy** — does the answer actually address the question? (uses embeddings)
- **context_precision** — are the retrieved chunks relevant to the question?
- **context_recall** — do the retrieved chunks contain what the reference answer needs?

The two context metrics matter here specifically: hit@k and MRR are **saturated at 1.0** on the current golden set, while RAGAS's context metrics are graded (0–1 per case), so they can still discriminate retrieval quality.

**Install and run.** RAGAS is an optional extra (heavy: langchain + datasets; the pins in `pyproject.toml` matter — ragas 0.4 still imports from the langchain-community 0.3 line) and never part of the CI gate:

```bash
pip install -e ".[ragas]"

# terminal 1: serve a local model (or use any hosted OpenAI-compatible endpoint)
conda run -n crypto_env env PYTHONPATH=src python -m adip.serving server \
  --model-profile qwen3_8b_default --port 8091

# terminal 2: generation eval with RAGAS attached
conda run -n crypto_env env PYTHONPATH=src python -m adip.mlops.run_generation_eval \
  --index data/processed/vector_index --golden data/eval/golden_qa.jsonl \
  --abstention-threshold 0.10 \
  --ragas-model-name "Qwen/Qwen3-8B" --ragas-endpoint-url http://127.0.0.1:8091/v1 \
  --ragas-limit 10
```

- The LLM is fully configurable (`--ragas-model-name` / `--ragas-endpoint-url` / `--ragas-api-key`); the key is session-only and never logged or written to disk. Embeddings for `answer_relevancy` use a **local** sentence-transformers model (`--ragas-embedding-model`, default `all-MiniLM-L6-v2`), so no second API key is needed.
- Model policy (from the inter-judge agreement study above): a local 8B-class model is acceptable for **relevance-style** metrics and cheap full-coverage sweeps; **faithfulness-style** judging benefits from a stronger model — treat local-8B RAGAS faithfulness with the same caution as local-8B judge faithfulness.
- Refusals are skipped (nothing to score); per-case failures come back as `None` and reduce coverage instead of crashing; a whole-batch endpoint failure is logged and skipped.
- The report gains `gen_eval_ragas_*` metrics — means for the four RAGAS metrics plus **three-way agreement** on faithfulness: RAGAS-vs-lexical and RAGAS-vs-judge gap and Pearson correlation. Like the judge, RAGAS keys appear **only when RAGAS ran**, so the deterministic CI gate and its metrics file are unaffected. Everything is logged through the same MLOps `start_run` tracking (params: model, embedding model, limit — never the key) and lands in MLflow when `--enable-mlflow` is set.

**First live run** (extractive baseline over the real-document corpus; 10 answers scored end-to-end by a local Qwen/Qwen3-8B via `adip.serving`, zero failed jobs):

| RAGAS metric | Mean | Reading |
| --- | --- | --- |
| faithfulness | **0.925** | agrees with the Gemini judge (~0.97): verbatim-extractive answers fabricate nothing |
| answer_relevancy | 1.000 | saturated — see caveat below |
| context_precision | 1.000 | retrieved chunks are relevant (consistent with saturated hit@k) |
| context_recall | 1.000 | golden evidence present in contexts (consistent with saturated MRR) |
| lexical-vs-RAGAS faithfulness gap / correlation | 0.313 / ≈0 | independently confirms the judge-run finding: the lexical proxy underestimates extractive faithfulness |

Two observations worth keeping:

1. **Metric design beats judge size for faithfulness.** The same local 8B that scored faithfulness 0.40 as a free-form judge scores 0.925 through RAGAS — because RAGAS decomposes the answer into individual claims and verifies each against the contexts, leaving no room for the "doesn't fully answer the question" dimension bleed. Structured decomposition rescued the small model.
2. **Treat `answer_relevancy` = 1.0 with suspicion here.** RAGAS computes it by asking the LLM to regenerate questions from the answer and embedding-comparing them to the real question; extractive answers echo the question's own vocabulary, and the local server returns 1 generation where RAGAS requests 3 (logged as a warning, handled gracefully), reducing this to a single-sample estimate. The Gemini judge's 0.34 relevance remains the more discerning verdict on the extractive writer.

RAGAS timing note: faithfulness and context_precision run multi-call LLM chains per case, which queue on a single local server — the adapter therefore raises RAGAS's per-job timeout (180s default → 600s, `RagasEvaluator(timeout=...)`) and caps concurrency (`max_workers=4`). With the defaults, 10 cases take a few minutes on one A100.

## Current Limitation

This phase establishes LLMOps plumbing and grounded generation quality checks. See [SERVING.md](SERVING.md) for the local serving layer.
