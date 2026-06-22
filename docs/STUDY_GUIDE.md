# Study Guide

Use this as the topic map for understanding and explaining the Agentic Document Intelligence Platform.

## Core Python And Software Engineering

- Python packaging with `pyproject.toml`
- CLI design with `argparse`
- Dataclasses and typed function signatures
- JSON and JSONL data formats
- File paths and reproducible project structure
- Unit testing with `pytest`
- Basic Git workflow and clean experiment artifacts

## Document Ingestion And NLP

- PDF/text parsing
- Document metadata: filename, page number, checksum, source type
- Text cleaning and normalization
- Chunking strategies
- Chunk size and overlap tradeoffs
- Token count vs character count
- Why source metadata matters for citations

## Retrieval And RAG

- What Retrieval-Augmented Generation is
- Sparse retrieval with TF-IDF
- Dense retrieval and embeddings
- Latent Semantic Analysis with Truncated SVD
- Sentence Transformers
- Cosine similarity and inner product search
- FAISS vector indexes
- Top-k retrieval
- Candidate retrieval vs final top-k retrieval
- Reranking
- Lexical rerankers
- Cross-encoder rerankers
- Bi-encoder vs cross-encoder tradeoffs
- Recall@k, hit rate@k, MRR
- Retrieval latency and index size tradeoffs
- Why a baseline retriever is important before adding LLMs

## LLM Concepts

- Prompt templates and prompt versions
- Local open-source LLMs
- Hugging Face Transformers
- Chat templates
- Context windows and max new tokens
- Token counts and generation latency
- Grounded generation
- Citation coverage
- Unsupported claim detection
- Reasoning models vs instruction-following models
- Why raw model output and normalized final output are both useful

## Agentic AI

- Agent workflow graphs
- Intent routing
- Planning
- Retrieval as a tool
- Evidence verification
- Writer/synthesis node
- Citation checking
- Agent state
- Agent traces
- Failure replay and debugging
- Guardrails for grounded answers

## MLOps

- Experiment tracking
- Parameters, metrics, and artifacts
- Dataset and index reproducibility
- Local run records
- Optional MLflow tracking
- DVC stages and metrics
- CI smoke tests
- Dockerized reproducibility
- Comparing model or retrieval backends with the same evaluation set

## LLMOps

- Prompt registry
- Model profile registry
- Provider adapters
- Local vs hosted model backends
- Prompt hashes
- Model/version metadata
- Token, latency, and GPU memory metrics
- Evaluation reports
- Regression tests for prompts and outputs

## AgentOps

- Run IDs
- Node-level trace events
- State transition logging
- Planner/verifier observability
- Tool input and output summaries
- Citation and verifier metrics
- Debugging failed or weak agent answers

## Local Serving And GPU Topics

- CUDA visibility with `CUDA_VISIBLE_DEVICES`
- GPU memory allocation vs reservation
- bf16 inference
- Quantization basics
- vLLM and SGLang purpose
- OpenAI-compatible local APIs
- Why a 40 GB GPU can serve 8B and 14B models comfortably

## Evaluation Topics

- Golden QA datasets
- Expected chunk IDs
- Expected substrings
- Retrieval hit rate
- Mean reciprocal rank
- Citation coverage
- Unsupported sentence count
- Latency metrics
- Benchmark reports
- Comparing TF-IDF vs dense retrieval
- Comparing plain retrieval vs retrieval plus reranking

## Suggested Study Order

1. Understand ingestion and chunking.
2. Understand TF-IDF retrieval and why it is the baseline.
3. Learn dense embeddings and FAISS.
4. Learn RAG evaluation metrics.
5. Study prompt templates and LLMOps metrics.
6. Study the agent workflow and AgentOps traces.
7. Study MLOps tracking and DVC pipeline design.
8. Review local model serving with Qwen and DeepSeek.
9. Practice explaining tradeoffs and failure cases.

## Hands-On Commands To Practice

```bash
conda run -n crypto_env env PYTHONPATH=src python -m adip.ingestion
```

```bash
conda run -n crypto_env env PYTHONPATH=src python -m adip.rag.index --backend tfidf
```

```bash
conda run -n crypto_env env PYTHONPATH=src python -m adip.rag.index --backend dense --embedding-model lsa
```

```bash
conda run -n crypto_env env PYTHONPATH=src python -m adip.mlops.run_retrieval_benchmark --backends tfidf dense
```

```bash
conda run -n crypto_env env PYTHONPATH=src python -m adip.mlops.run_agent --model-profile extractive_baseline --reasoning-model-profile extractive_baseline --use-reasoning-planner
```

## Interview Prep Questions

- Why did you start with TF-IDF before dense embeddings?
- What problem does FAISS solve?
- How do you know the retriever is working?
- What does MRR tell you that hit rate does not?
- How do you prevent hallucinations in RAG?
- Why do you store source metadata in every chunk?
- What is the difference between MLOps, LLMOps, and AgentOps?
- How do you compare Qwen and DeepSeek in this project?
- What would you monitor in production?
- What is the next improvement after dense retrieval?
- Why might a reranker not improve metrics on an easy dataset?
- Why is a cross-encoder slower but often more accurate than embedding similarity?
