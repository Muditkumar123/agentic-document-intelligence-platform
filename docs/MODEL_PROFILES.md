# Model Profiles

Model profiles keep local serving choices out of the agent and LLMOps code.

Profile file:

```text
config/model_profiles.yaml
```

## Current Profiles

- `extractive_baseline`: deterministic fallback for tests, CI, and offline demos.
- `qwen3_8b_default`: default local model profile for document QA, summaries, and normal RAG answers.
- `deepseek_r1_distill_qwen_14b_reasoning`: reasoning-specialist profile for verifier, planner, and harder synthesis tasks.
- `deepseek_r1_distill_qwen_32b_stretch`: stretch local profile for high-quality reasoning benchmarks, usually through quantized serving.
- `deepseek_v4_flash_cloud`: hosted DeepSeek profile for fast OpenAI-compatible QA and summaries.
- `deepseek_v4_pro_cloud`: hosted DeepSeek profile for stronger OpenAI-compatible reasoning.
- `deepseek_v3_2_cloud_benchmark`: legacy hosted benchmark alias retained for old run configs.

## List Profiles

```bash
conda run -n crypto_env env PYTHONPATH=src python -m adip.config.list_model_profiles
```

JSON output:

```bash
conda run -n crypto_env env PYTHONPATH=src python -m adip.config.list_model_profiles --json
```

## Use A Profile In LLMOps

Safe local baseline:

```bash
conda run -n crypto_env env PYTHONPATH=src python -m adip.llmops \
  --index data/processed/vector_index \
  --question "What does the platform do with documents?" \
  --task qa \
  --model-profile extractive_baseline
```

Qwen profile:

```bash
conda run -n crypto_env env PYTHONPATH=src python -m adip.llmops \
  --index data/processed/vector_index \
  --question "What does the platform do with documents?" \
  --task qa \
  --model-profile qwen3_8b_default
```

DeepSeek reasoning profile:

```bash
conda run -n crypto_env env PYTHONPATH=src python -m adip.llmops \
  --index data/processed/vector_index \
  --question "Check whether the retrieved evidence supports this answer." \
  --task qa \
  --model-profile deepseek_r1_distill_qwen_14b_reasoning
```

The Hugging Face profiles use local files by default. The model must already be present in the Hugging Face cache or at a local path unless `--allow-download` is passed.

## Use A Profile In Agents

```bash
conda run -n crypto_env env PYTHONPATH=src python -m adip.agents \
  --index data/processed/vector_index \
  --question "Create a research brief about the document intelligence platform." \
  --task brief \
  --domain academic \
  --model-profile extractive_baseline
```

Use Qwen as the answer writer and DeepSeek 14B as the reasoning verifier:

```bash
conda run -n crypto_env env CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python -m adip.mlops.run_agent \
  --index data/processed/vector_index \
  --question "Does the evidence support the platform claims?" \
  --task qa \
  --top-k 3 \
  --model-profile qwen3_8b_default \
  --reasoning-model-profile deepseek_r1_distill_qwen_14b_reasoning \
  --use-reasoning-planner \
  --device cuda:0 \
  --reasoning-device cuda:0 \
  --max-new-tokens 128 \
  --reasoning-max-new-tokens 256
```

This records writer metrics as `llm_*`, planner metrics as `planning_llm_*`, and verifier metrics as `reasoning_llm_*`.

## OpenAI-Compatible Serving

Profiles with provider `openai_compatible` can point to vLLM, SGLang, or a hosted compatible endpoint.

Example local vLLM/SGLang endpoint:

```bash
export ADIP_OPENAI_BASE_URL=http://localhost:8000/v1
export ADIP_OPENAI_API_KEY=local-dev-key
```

Example hosted DeepSeek setup:

```bash
cp .env.example .env
```

Then set:

```bash
DEEPSEEK_BASE_URL=https://api.deepseek.com/chat/completions
DEEPSEEK_API_KEY=your-real-key
```

The API key is read from the environment at runtime and is never written into traces. See [API_KEYS.md](API_KEYS.md) for the full setup.

Then:

```bash
conda run -n crypto_env env PYTHONPATH=src python -m adip.llmops \
  --index data/processed/vector_index \
  --question "Compare the strongest claims in these documents." \
  --task brief \
  --model-profile deepseek_v4_flash_cloud
```

You can also pass an endpoint directly:

```bash
conda run -n crypto_env env PYTHONPATH=src python -m adip.llmops \
  --index data/processed/vector_index \
  --question "Compare the strongest claims in these documents." \
  --task brief \
  --model-profile deepseek_r1_distill_qwen_32b_stretch \
  --endpoint-url http://localhost:8000/v1
```

## Recommended Use

- Use `qwen3_8b_default` as the regular local answer/report model.
- Use `deepseek_r1_distill_qwen_14b_reasoning` for reasoning-heavy nodes.
- Use `deepseek_r1_distill_qwen_32b_stretch` as a benchmark mode through quantized serving.
- Use `deepseek_v4_flash_cloud` for faster hosted DeepSeek runs.
- Use `deepseek_v4_pro_cloud` when benchmarking hosted reasoning quality.
