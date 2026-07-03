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

## Current Limitation

This phase establishes LLMOps plumbing and grounded generation quality checks. See [SERVING.md](SERVING.md) for the local serving layer.
