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

- **faithfulness**: fraction of the answer's meaningful tokens grounded in the retrieved evidence (a lexical proxy; an LLM judge can replace it behind the same report shape). `None` for refusals.
- **answer_relevance**: how much of the question the answer addresses.
- **expected_coverage**: whether the answer surfaced the golden "expected" facts.
- **citation_coverage**: are the retrieved citations visible in the answer.
- **grounded_rate / refusal_rate**: share of answers above the grounding threshold, and how often the writer declined.

It is deterministic with the extractive baseline (CI-safe) and can drive any hosted or local writer for model comparisons. The latest report is exposed at `GET /monitoring/generation-eval` and on the dashboard's **Answer Quality** tiles. See the tracked command in [../README.md](../README.md#answer-quality-evaluation).

## Current Limitation

This phase establishes LLMOps plumbing and grounded generation quality checks. See [SERVING.md](SERVING.md) for the local serving layer.
