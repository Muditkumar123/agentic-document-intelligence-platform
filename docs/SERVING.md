# Local LLM Serving

This phase adds serving utilities for Qwen and DeepSeek model profiles.

## Current Environment

The machine is GPU-ready:

- 2x NVIDIA A100 40GB GPUs

The current `crypto_env` now includes:

- `transformers`
- `accelerate`

The current `crypto_env` does not include:

- `vllm`
- `bitsandbytes`

`Qwen/Qwen3-8B` has been downloaded and cached successfully. See [QWEN3_LOCAL_SMOKE.md](QWEN3_LOCAL_SMOKE.md).

`deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` has also been downloaded and run locally. See [DEEPSEEK14B_LOCAL_SMOKE.md](DEEPSEEK14B_LOCAL_SMOKE.md).

## Inspect Serving Readiness

```bash
conda run -n crypto_env env PYTHONPATH=src python -m adip.serving inspect
```

JSON:

```bash
conda run -n crypto_env env PYTHONPATH=src python -m adip.serving inspect --json
```

## Generate A Launch Plan

Qwen default profile:

```bash
conda run -n crypto_env env PYTHONPATH=src python -m adip.serving launch-plan \
  --model-profile qwen3_8b_default
```

DeepSeek reasoning profile:

```bash
conda run -n crypto_env env PYTHONPATH=src python -m adip.serving launch-plan \
  --model-profile deepseek_r1_distill_qwen_14b_reasoning
```

DeepSeek 32B stretch profile:

```bash
conda run -n crypto_env env PYTHONPATH=src python -m adip.serving launch-plan \
  --model-profile deepseek_r1_distill_qwen_32b_stretch
```

## Local Transformers Generation

Safe baseline:

```bash
conda run -n crypto_env env PYTHONPATH=src python -m adip.serving generate \
  --model-profile extractive_baseline \
  --prompt "Summarize the indexed evidence."
```

Qwen when weights are already cached:

```bash
conda run -n crypto_env env PYTHONPATH=src python -m adip.serving generate \
  --model-profile qwen3_8b_default \
  --prompt "Summarize the indexed evidence."
```

Allow an intentional model download:

```bash
conda run -n crypto_env env PYTHONPATH=src python -m adip.serving generate \
  --model-profile qwen3_8b_default \
  --prompt "Summarize the indexed evidence." \
  --allow-download
```

## OpenAI-Compatible Local Server

Start the project server with the safe baseline:

```bash
conda run -n crypto_env env PYTHONPATH=src python -m adip.serving server \
  --model-profile extractive_baseline \
  --host 127.0.0.1 \
  --port 8000
```

Then use:

```bash
export ADIP_OPENAI_BASE_URL=http://127.0.0.1:8000/v1
```

LLMOps and agents can use any OpenAI-compatible endpoint through profiles such as:

- `deepseek_r1_distill_qwen_32b_stretch`
- `deepseek_v4_flash_cloud`
- `deepseek_v4_pro_cloud`

Hosted model API key setup is documented in [API_KEYS.md](API_KEYS.md).

## vLLM Recommendation

Once `vllm` is installed, the project can serve Qwen or DeepSeek with commands from `launch-plan`.

Example:

```bash
vllm serve Qwen/Qwen3-8B \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 32768 \
  --dtype bfloat16 \
  --tensor-parallel-size 1
```

For the DeepSeek 32B stretch profile, use tensor parallelism across the two A100s.
