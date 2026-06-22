# Qwen3-8B Local Smoke Test

This project has successfully run `Qwen/Qwen3-8B` locally through the serving, LLMOps, MLOps, and AgentOps paths.

## Environment

- GPU target: physical GPU 1 through `CUDA_VISIBLE_DEVICES=1`
- GPU: NVIDIA A100-PCIE-40GB
- Python environment: `crypto_env`
- Torch: 2.5.1
- Transformers: 5.12.1
- Accelerate: 1.14.0
- Model cache: `Qwen/Qwen3-8B` cached under Hugging Face Hub cache
- Cache size after download: about 16 GB

## Serving Smoke

Command:

```bash
conda run -n crypto_env env CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python -m adip.serving generate \
  --model-profile qwen3_8b_default \
  --prompt "In one sentence, explain what this document intelligence platform does." \
  --max-new-tokens 64 \
  --json
```

Observed:

- Model provider: `huggingface`
- Model name: `Qwen/Qwen3-8B`
- Output tokens: 64
- Generation latency: about 4.95 seconds

## LLMOps Smoke

Command:

```bash
conda run -n crypto_env env CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python -m adip.mlops.run_llmops_smoke \
  --index data/processed/vector_index \
  --question "What does the platform do with documents?" \
  --task qa \
  --top-k 1 \
  --model-profile qwen3_8b_default \
  --device cuda:0 \
  --max-new-tokens 128 \
  --metrics-output data/monitoring/qwen3_8b_llmops_metrics.json \
  --report-output data/monitoring/qwen3_8b_llmops_report.json
```

Observed:

- Input tokens: 208
- Output tokens: 74
- Generation latency: about 4.02 seconds
- Citation coverage: 1.0
- Unsupported sentence count: 0.0
- GPU allocated: about 15.6 GB
- GPU reserved: about 15.9 GB

## AgentOps Smoke

Command:

```bash
conda run -n crypto_env env CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python -m adip.mlops.run_agent \
  --index data/processed/vector_index \
  --question "What does the platform do with documents?" \
  --task qa \
  --top-k 1 \
  --model-profile qwen3_8b_default \
  --device cuda:0 \
  --max-new-tokens 128 \
  --metrics-output data/monitoring/qwen3_8b_agent_metrics.json
```

Observed:

- Agent trace events: 6
- Retrieved chunks: 1
- Citation coverage: 1.0
- Input tokens: 208
- Output tokens: 74
- Generation latency: about 3.88 seconds
- Workflow duration: about 12.36 seconds
- GPU allocated: about 15.6 GB
- GPU reserved: about 15.9 GB

## Notes

The first non-chat-template Qwen run copied part of the prompt and missed the full citation. The Hugging Face adapter was updated to use the tokenizer chat template, which fixed citation behavior for this smoke test.

Pip warned that `vbench` pins `transformers==4.33.2`. Qwen3 required upgrading Transformers in `crypto_env`; this is acceptable for this project, but it may affect unrelated `vbench` workflows that reuse the same Conda environment.
