# DeepSeek-R1-Distill-Qwen-14B Local Smoke Test

This project has successfully downloaded and run `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` locally through the serving and AgentOps reasoning paths.

## Environment

- GPU target: physical GPU 1 through `CUDA_VISIBLE_DEVICES=1`
- GPU: NVIDIA A100-PCIE-40GB
- Python environment: `crypto_env`
- Model cache: `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` cached under Hugging Face Hub cache
- Cache size after download: about 28 GB
- Qwen cache size: about 16 GB
- Disk free after both cached models: about 125 GB

## Serving Smoke

Command:

```bash
conda run -n crypto_env env CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python -m adip.serving generate \
  --model-profile deepseek_r1_distill_qwen_14b_reasoning \
  --prompt "In one sentence, explain why evidence verification matters in RAG." \
  --max-new-tokens 64 \
  --json
```

Observed:

- Model provider: `huggingface`
- Model name: `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`
- Input tokens: 18
- Output tokens: 64
- Generation latency: about 5.23 seconds after load
- GPU allocated: about 28.3 GB
- GPU reserved: about 28.4 GB

## AgentOps Reasoning Smoke

Command:

```bash
conda run -n crypto_env env CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python -m adip.mlops.run_agent \
  --index data/processed/vector_index \
  --question "Does the evidence support the claim that the platform preserves source metadata?" \
  --task qa \
  --top-k 1 \
  --model-profile extractive_baseline \
  --reasoning-model-profile deepseek_r1_distill_qwen_14b_reasoning \
  --use-reasoning-planner \
  --device cuda:0 \
  --reasoning-device cuda:0 \
  --reasoning-max-new-tokens 192 \
  --metrics-output data/monitoring/deepseek14b_agent_reasoning_metrics.json \
  --json
```

Observed:

- Agent status: `completed`
- Planner prompt: `plan_v1`
- Verifier prompt: `verify_v1`
- Writer prompt: `qa_v1`
- Trace events: 6
- Planning latency: about 12.04 seconds
- Verifier latency: about 12.75 seconds
- Planning GPU allocated: about 28.3 GB
- Verifier GPU allocated: about 28.3 GB
- Total workflow duration: about 41.76 seconds

## Notes

DeepSeek-R1-Distill-Qwen-14B fits on one 40 GB A100 in bf16 for this project. During AgentOps reasoning smoke, GPU 1 used about 31.8 GB total because another small process was already using about 2.7 GB.

The reasoning model tends to spend early tokens on analysis prose. The project now keeps that raw output for auditability, then normalizes verifier notes through `structured_output.final_text` before scoring citation coverage. If the model does not return the requested sections, the normalizer records `structured=false` and creates a conservative review-needed verifier note with the top retrieved citation.
