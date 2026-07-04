# Roadmap

## Phase 0: Project Foundation

Status: started

- Create project folder and design documents.
- Confirm final stack choices.
- Add Python project configuration.
- Add environment setup instructions.
- Add initial test layout.

## Phase 1: Ingestion Pipeline

Goal: convert documents into clean, traceable chunks.

- Add PDF and text parsing.
- Extract metadata such as filename, page number, checksum, and source type.
- Implement chunking with configurable chunk size and overlap.
- Save processed chunks as JSONL.
- Add unit tests with tiny fixture documents.

Deliverable:

- `python -m adip.ingestion ...` creates processed chunks from sample files.

## Phase 2: Baseline RAG

Goal: answer questions with cited context.

- Add embedding model wrapper. Status: done for TF-IDF and dependency-light dense LSA; optional sentence-transformers adapter added.
- Create local vector index. Status: done for TF-IDF, dense NumPy search, and optional FAISS persistence.
- Implement top-k search. Status: done.
- Add hybrid retrieval. Status: done — `hybrid` backend fuses BM25 (Okapi, scikit-learn based, no extra dependencies) with dense rankings via weighted reciprocal-rank fusion, wired through the CLIs, API, and retrieval benchmark.
- Add cited answer generation. Status: done as an extractive non-LLM baseline.
- Add a small golden Q&A evaluation set. Status: done; later replaced by the real public-document corpus in `data/eval/` (45 answerable + 10 unanswerable questions).

Deliverable:

- User can query a document collection and receive an answer with source chunks. Status: done for baseline retrieval.

Next upgrade:

- Install `sentence-transformers` and `faiss-cpu` for production semantic retrieval.
- Add reranking after top-k retrieval. Status: done — lexical and cross-encoder rerankers, with the first-stage candidate pool automatically widened to 3x top_k (min 10) whenever a reranker is enabled.
- Add LLM-generated synthesis after retrieval.

## Phase 3: Agentic Workflow

Goal: convert baseline RAG into a multi-step agent.

- Add intent router. Status: done.
- Add planner node. Status: done.
- Add retriever node. Status: done.
- Add evidence verifier node. Status: done.
- Add report writer node. Status: done.
- Add citation checker node. Status: done.
- Add AgentOps trace persistence. Status: done.

Deliverable:

- Agent can generate a research brief with evidence-backed sections. Status: done for the extractive baseline.

Next upgrade:

- Replace the dependency-light runner with LangGraph once dependencies are added.
- Add LLM-based synthesis behind the writer node.
- Add human review gates for high-risk outputs.

## Phase 4: MLOps Foundation

Goal: make the system reproducible and trackable.

- Initialize DVC. Status: DVC-compatible files added; native `dvc init` waits until DVC is installed.
- Track raw and processed sample datasets. Status: pipeline stages defined in `dvc.yaml`.
- Add MLflow experiment tracking. Status: optional MLflow hooks added; local tracker works now.
- Log chunking parameters, embedding model, prompts, retrieval metrics, and artifacts. Status: done for ingestion, RAG eval, and agent smoke tests.
- Add Dockerfile and compose file. Status: done.
- Add CI smoke test. Status: done as a GitHub Actions template.

Deliverable:

- A reviewer can reproduce ingestion, inspect local MLOps run records, and enable MLflow/DVC once those tools are installed.

## Phase 5: LLMOps Foundation

Goal: demonstrate practical LLM deployment and evaluation on a 40 GB GPU.

- Add local model serving path. Status: adapter interface and model profiles added; real model serving remains a next upgrade.
- Add quantization support. Status: pending local model serving phase.
- Add model configuration profiles. Status: Qwen and DeepSeek profiles added.
- Track latency, token counts, and GPU memory usage. Status: latency and token counts done; GPU memory pending local model serving.
- Keep API model-agnostic so local and hosted models can be swapped. Status: provider adapter added.
- Add prompt template versioning. Status: done.
- Add golden Q&A regression tests. Status: RAG golden set exists; LLMOps smoke tests added.
- Add citation and faithfulness checks. Status: citation coverage and unsupported sentence checks added.
- Add standardized evaluation. Status: done — optional RAGAS integration (`[ragas]` extra) scores faithfulness, answer relevancy, context precision, and context recall behind the same report shape, with three-way faithfulness agreement (RAGAS vs lexical proxy vs LLM judge); see LLMOPS.md.

Deliverable:

- Backend can use a grounded writer for RAG and report generation, with prompt/model/retrieval metrics tracked. Local instruct model serving is the next upgrade.

## Phase 5.5: Local LLM Serving

Goal: prepare local Qwen and DeepSeek serving without forcing model downloads.

- Add serving environment inspector. Status: done.
- Add Qwen and DeepSeek launch plans. Status: done.
- Add local Transformers generation backend. Status: done.
- Add OpenAI-compatible server wrapper. Status: done.
- Add vLLM/SGLang command generation. Status: done.

Deliverable:

- A reviewer can inspect serving readiness and see exactly how to launch Qwen3-8B or DeepSeek profiles locally.

## Phase 6: AgentOps Foundation

Goal: make the agent workflow observable and debuggable.

- Add agent run IDs.
- Log node-by-node state transitions.
- Log tool calls, inputs, and outputs.
- Track verifier and citation checker results.
- Add failure replay records.
- Add run history for agent traces.

Deliverable:

- A reviewer can inspect how the agent planned, retrieved, verified, and wrote each output.

## Phase 6.5: Application API

Goal: expose the core workflow through a demo-ready backend.

- Add health endpoint. Status: done.
- Add RAG query endpoint. Status: done.
- Add agent run endpoint with AgentOps trace output. Status: done.
- Add ingestion plus index rebuild endpoint. Status: done.
- Add endpoint tests. Status: done.

Deliverable:

- A reviewer can run `python -m adip.api` and use OpenAPI docs to rebuild an index, query RAG, and run the agent workflow.

## Phase 7: Fine-Tuning Experiment

Goal: demonstrate LoRA or QLoRA without overbuilding.

- Create a small supervised dataset for one task.
- Suggested task: classify document chunks into claim, evidence, risk, method, result, limitation, or background.
- Fine-tune using PEFT.
- Compare base model vs adapted model.
- Track all runs in MLflow.

Deliverable:

- A concise model comparison report with metrics and artifacts.

## Phase 8: Monitoring and Evaluation Dashboard

Goal: make quality visible.

- Add retrieval evaluation dashboard. Status: started with FastAPI-served dashboard.
- Add answer faithfulness checks.
- Add citation accuracy checks.
- Add text drift report for incoming documents and queries.
- Add run history. Status: done for AgentOps traces and MLOps local run records.

Deliverable:

- UI shows quality metrics, latency, and drift/failure signals.

## Suggested First Sprint

Build only this:

1. Python project setup.
2. Document parser.
3. Chunking pipeline.
4. JSONL output.
5. Basic tests.

This gives us the base layer everything else will depend on.
