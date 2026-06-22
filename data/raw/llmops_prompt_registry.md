# LLMOps Prompt Registry

The LLMOps layer stores prompt templates for question answering, research briefs, planning, and evidence verification.

Every generation run records the prompt path, prompt version, prompt hash, model profile, token counts, latency, and citation quality metrics.

Structured verifier output keeps raw reasoning text for auditability while scoring the normalized final verifier notes.

Model profiles allow the platform to switch between an extractive baseline, Qwen, DeepSeek, and OpenAI-compatible endpoints.
