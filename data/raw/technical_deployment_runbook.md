# Technical Deployment Runbook

The serving layer can inspect CUDA availability, package readiness, model cache status, and local GPU names.

Qwen3-8B is the default local writer model for normal document answers and research briefs.

DeepSeek-R1-Distill-Qwen-14B is a reasoning specialist for planning and evidence verification.

For larger models, the deployment plan recommends quantization, vLLM, SGLang, or tensor parallel serving across multiple GPUs.
