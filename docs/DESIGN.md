# Project Design

## 1. Vision

Agentic Document Intelligence Platform is a production-style AI system for analyzing long-form documents across domains. It combines traditional NLP, retrieval-augmented generation, local LLM serving, multi-step agents, and MLOps/LLMOps/AgentOps practices.

The platform should feel like an AI research analyst that can read a document collection, retrieve evidence, reason through a task, verify its own claims, and produce a cited output.

## 2. Goals

- Build a domain-agnostic document intelligence platform.
- Demonstrate LLM concepts beyond simple prompting.
- Demonstrate MLOps with reproducibility, tracking, evaluation, monitoring, and deployment.
- Support local open-source LLM workflows on a 40 GB GPU.
- Keep the first version realistic enough to finish and strong enough for a resume.

## 3. Non-Goals

- Do not train a foundation model from scratch.
- Do not make a 70B model the default dependency.
- Do not build a generic chatbot without citations or evaluation.
- Do not overbuild the UI before the ingestion, RAG, and evaluation layers work.

## 4. Primary Use Cases

### Research Q&A

User uploads a set of documents and asks a question. The system answers using retrieved evidence and cites source chunks.

### Research Brief Generation

User selects documents and a domain preset. The agent produces a structured report with claims, evidence, risks, gaps, and recommendations.

### Document Comparison

User selects two or more documents. The system compares methods, assumptions, results, risks, contradictions, and open questions.

### Risk and Claim Extraction

The system extracts key claims, entities, risks, obligations, metrics, and uncertainties from a document collection.

### Monitoring and Evaluation

The system tracks retrieval quality, answer relevance, citation precision, latency, prompt versions, model versions, and input drift over time.

## 5. Domain Presets

The base pipeline is generic. Domain presets customize extraction fields, prompts, report templates, and evaluation expectations.

### Academic Mode

- Problem statement
- Methodology
- Dataset
- Experiments
- Results
- Limitations
- Reproducibility notes

### Finance Mode

- Business model
- Market signals
- Revenue or adoption metrics
- Risk factors
- Competitive position
- Forward-looking claims

### Crypto/Web3 Mode

- Protocol design
- Consensus mechanism
- Tokenomics
- Security assumptions
- Attack surface
- Governance risks

### Legal/Policy Mode

- Parties
- Obligations
- Deadlines
- Definitions
- Risky clauses
- Compliance implications

## 6. High-Level Architecture

```text
                    User Interface
            Streamlit first, optional web UI later
                          |
                          v
                      FastAPI API
                          |
        +-----------------+-----------------+
        |                 |                 |
        v                 v                 v
  Ingestion Layer     RAG Layer       Agentic Layer
  parse, clean,       chunk, embed,   plan, retrieve,
  normalize           search, rerank  verify, write
        |                 |                 |
        +-----------------+-----------------+
                          |
                          v
                 LLM Serving Layer
           local model, quantization, vLLM
                          |
                          v
                 MLOps / LLMOps Layer
    MLflow, DVC, AgentOps traces, evaluation, Docker
```

## 7. Main Data Flow

1. User uploads or registers documents.
2. Ingestion extracts text, metadata, and page references.
3. Text is cleaned and split into word-window chunks that keep paragraph, heading and list line breaks.
4. Chunks are embedded and stored in a vector index.
5. User asks a question or requests a report.
6. Retriever selects relevant chunks. Backends: TF-IDF (deterministic baseline), dense (LSA, or neural embeddings such as bge-m3 and Qwen3-Embedding-0.6B), and hybrid — BM25 and dense rankings fused with weighted reciprocal-rank fusion (rank-based, so no score-scale calibration is needed; fused scores are normalized so rank-1-in-both equals 1.0). Dense vectors live in the index archive or in a Qdrant vector store (embedded in the index folder, or a server via `ADIP_QDRANT_URL`), which runs filtered nearest-neighbour search.
7. Optional reranker improves the evidence set. When a reranker is on, the first-stage candidate pool automatically widens to 3x top_k (min 10) so reranking can promote chunks the final cut would have missed, not just reorder it.
8. Agent graph plans the task, calls tools, drafts an answer, and verifies citations.
9. Evaluation layer records metrics and failure cases.
10. Output is returned with citations, confidence notes, and trace metadata.

## 8. Agentic Workflow

The agent graph stays small and inspectable, but it is a loop, not a fixed pipeline: the verifier decides whether the evidence is good enough or whether to search again.

```text
User Request
    |
    v
Intent Router
    |
    v
Planner -------- splits a multi-part question into sub-questions
    |            (rule-based, or the planner LLM with --decomposer llm)
    v
Retriever <------------------------------------+
    |   round 1: the whole question + one       |
    |   search per sub-question                 |
    v                                           |
Evidence Verifier -- a (sub-)question's best    |
    |                passage misses a specific  |
    |                term? --> follow-up search +   (at most 2 extra rounds / 8 searches)
    |
    |  covered, or the budget is used up
    v
Answer or Report Writer
    |
    v
Citation Checker
    |
    v
Final Response
```

**Decomposition.** The planner splits coordinated questions ("..., and which ...", "? Also, ...") into sub-questions. Rules are syntax-only and never split noun phrases ("X and Y"), so ordinary questions stay whole (2 of 47 golden questions split, and both really are two questions). With `--decomposer llm` the planner model writes the sub-questions from the versioned `decompose_v2` prompt, as JSON when the provider supports a response schema,, falling back to the rules if it fails. When a question is split, the whole question is still searched first, so decomposition can only add evidence.

**Evidence streams.** Every search is a stream. The evidence list interleaves the accepted streams round-robin (each stream's best result first), dedupes, and caps at `top_k`, so the writer's context stays the same size however many searches ran. Each evidence item records the query that found it, and the extractive writer picks that item's sentence by that query. A question that is never split or retried gets exactly the evidence a plain search returns.

**Coverage-driven follow-ups.** For each (sub-)question the verifier finds the retrieved passage covering the most of its content terms, weighted by IDF (`adip/rag/coverage.py`). Below the coverage threshold, and only if a *specific* term is missing (rare in the corpus, or a name/acronym in the question), the missing terms become a follow-up search. A follow-up is kept only if its best passage actually contains those terms, so a fruitless search adds nothing. The threshold (0.3) was fixed by a rule set before measuring: the 10th percentile of best-passage coverage over 67 single-hop questions.

**Retry records.** Each follow-up round adds an entry to `retrieval_retries` with its round number and, for every search it starts, the query, the (sub-)question it serves, and why it runs (for example `sub-question 2 coverage 0.21 below 0.30, missing dns, label`). The verifier rebuilds its notes each round, so this list is the durable record of why the run went back to retrieval. Every AgentOps trace event summarizes `retrieval_round_count`, `retrieval_call_count`, `retrieval_retry_count` and `pending_query_count`, so a trace shows which verifier step sent the run back and what the next retriever ran.

**Budgets.** Three per-run caps stop the loop: `max_retrieval_retries` (default 2 follow-up rounds), `max_retrieval_calls` (8 searches), and `retrieval_time_budget_ms` (10 s of retrieval wall-clock, checked before each follow-up round). The caps and what the run used (`retrieval_round_count`, `retrieval_call_count`, `retrieval_retry_count`, `retrieval_elapsed_ms`) are in the run's metrics, and `retrieval_stop_reason` names the cap when one ended the loop. There is no token budget because the loop itself calls no model: the optional LLM decomposer, the optional LLM verifier (on the final evidence) and the writer each run once per question, however many rounds there are. As a backstop, the sequential engine stops after a step count derived from the retry cap, and the LangGraph engine has a recursion limit.

**Engines.** The workflow runs on two interchangeable engines behind one definition: a real **LangGraph `StateGraph`** (default when the `[agents]` extra is installed) and a dependency-light sequential runner. Both read the same node list and the same router table: failure routes to `END`, and the verifier's router chooses `retriever` or `writer`. They share the same node functions, the same `AgentState`, and the same tracing wrapper, so AgentOps traces are identical (a retried run shows `retriever` and `evidence_verifier` twice), and every trace records which engine produced it (`workflow_engine`).

Measured on two-document questions in [EVALUATION_DATASET.md](EVALUATION_DATASET.md#multi-hop-probes-and-the-retrieval-loop).

### Agent State

```text
request_id
user_query
domain_preset
document_ids
sub_questions
retrieval_streams      every search: query, reason, round, accepted, chunk ids
sub_question_support   coverage, missing terms, follow-up terms per (sub-)question
retrieval_retries      every follow-up round: round, queries, and why each one ran
retrieved_chunks
evidence_summary
draft_answer
verification_notes
final_answer
metrics
```

### Agent Tools

- Search document chunks.
- Fetch source text by document and page.
- Extract entities and claims.
- Generate report sections.
- Check whether answer sentences are supported by evidence.
- Log run metadata to MLflow.

Current baseline tool support:

- Search document chunks through the local RAG index, once per question and sub-question.
- Check whether the evidence covers each (sub-)question, and search again for what is missing.
- Write extractive cited answers and research briefs.
- Persist node-by-node AgentOps traces, including every search stream.

## 9. NLP and LLM Concepts To Demonstrate

### NLP

- PDF parsing and text normalization.
- Chunking strategies.
- Embeddings.
- Semantic search.
- Named entity extraction.
- Topic or intent classification.
- Claim and risk extraction.

### LLM Engineering

- Prompt templates.
- RAG.
- Context window management.
- Retrieval reranking.
- Tool calling.
- Agent graph state.
- Citation grounding.
- Quantized local inference.
- LoRA or QLoRA fine-tuning.
- LLM evaluation.

### LLMOps

- Prompt versioning.
- Model selection records.
- Evaluation datasets.
- Golden question-answer pairs.
- Latency and token tracking.
- Failure analysis.
- Drift reports for incoming documents and queries.

Current implementation adds versioned prompt files, prompt hashes, token counts, latency, citation coverage checks, unsupported sentence checks, a deterministic grounded writer, and an optional Hugging Face text-generation adapter.

Model profiles are defined in `config/model_profiles.yaml`, including Qwen3-8B, DeepSeek-R1-Distill-Qwen-14B, DeepSeek-R1-Distill-Qwen-32B, and a DeepSeek cloud benchmark profile.

## 10. Model Plan for 40 GB GPU

Use the 40 GB GPU for practical local LLM engineering.

Recommended default:

- Primary generation model: 7B or 8B instruct model.
- Optional stronger model: 14B model, possibly quantized.
- Experimental larger model: 30B or 32B quantized inference.
- Embedding model: small or medium sentence embedding model.
- Fine-tuning: LoRA or QLoRA on a narrow task such as domain classification, risk extraction, or answer-style adaptation.

Avoid making a 70B model part of the required MVP. It can be listed as an optional future experiment.

## 11. MLOps Design

The project has three explicit operational layers:

- MLOps handles datasets, models, experiments, deployments, and monitoring.
- LLMOps handles prompts, retrieval, model serving, generation quality, and hallucination checks.
- AgentOps handles agent traces, tool calls, state transitions, failures, guardrails, and replay.

See `OPS_PIPELINES.md` for the detailed pipeline design.

### DVC

Use DVC to version:

- Raw sample documents.
- Processed text.
- Chunked datasets.
- Evaluation datasets.
- Fine-tuning datasets.

### MLflow

Use MLflow to track:

- Ingestion runs.
- Embedding model versions.
- Chunking parameters.
- Retrieval metrics.
- LLM model name and quantization mode.
- Prompt template versions.
- Fine-tuning runs.
- Evaluation artifacts.

### Docker

Use Docker to package:

- FastAPI backend.
- Optional Streamlit UI.
- MLflow service.
- Vector database service if needed.

### CI/CD

Use GitHub Actions for:

- Python linting.
- Unit tests.
- Small ingestion test.
- RAG smoke test.
- Evaluation script on a tiny fixture dataset.

## 12. Evaluation Design

### Retrieval Metrics

- Recall at k.
- MRR.
- Source coverage.
- Citation chunk precision.

### Generation Metrics

- Answer relevance.
- Faithfulness to retrieved context.
- Citation accuracy.
- Refusal quality when evidence is missing.
- Report section completeness.

### Operational Metrics

- Ingestion time.
- Embedding time.
- Retrieval latency.
- Generation latency.
- Tokens per request.
- GPU memory usage.

### Monitoring Metrics

- Input document length distribution.
- Language distribution.
- Query topic distribution.
- Embedding drift.
- Text quality drift.
- Failure rate by domain preset.

## 12.1 AgentOps Design

AgentOps makes the agent workflow observable and debuggable.

Track these for every agent run:

- Agent run ID.
- User task type.
- Domain preset.
- Node-by-node state transitions.
- Tool calls.
- Tool inputs and outputs.
- Retrieved chunk IDs.
- Verification notes.
- Citation checker results.
- Retry count.
- Failure reason.
- Human approval or correction events.
- Final output quality score.

## 13. Storage Design

### Document Store

Stores original document metadata:

```text
document_id
filename
source_type
domain_preset
uploaded_at
page_count
checksum
```

### Chunk Store

Stores searchable chunks:

```text
chunk_id
document_id
page_number
section_title
text
embedding_id
token_count
metadata
```

### Run Store

Stores execution traces:

```text
run_id
request_id
pipeline_version
model_version
prompt_version
retrieved_chunk_ids
metrics
created_at
```

## 14. API Design

Initial endpoints:

```text
POST /documents/upload
POST /documents/ingest
GET  /documents
POST /query
POST /reports/research-brief
POST /compare
GET  /runs/{run_id}
GET  /health
```

## 15. UI Design

Build the first UI in Streamlit for speed.

Core screens:

- Document upload and collection view.
- Query screen with cited answers.
- Report generation screen.
- Comparison screen.
- Evaluation dashboard.
- Run history view.

The UI should behave like a work tool, not a landing page. The first screen should let the user upload documents or choose an existing collection.

## 16. Implementation Milestones

### Milestone 1: Document Ingestion

- PDF/text parsing.
- Metadata extraction.
- Chunking.
- Local processed dataset output.

### Milestone 2: Baseline RAG

- Embedding model integration.
- Vector index.
- Question answering with citations.
- Basic retrieval evaluation.

Current implementation starts with a local TF-IDF vector baseline using `scikit-learn`. This gives us a working retrieval pipeline now, while leaving a clean path to add dense sentence-transformer embeddings and FAISS/Chroma later.

### Milestone 3: Agent Graph

- Router.
- Planner.
- Retriever.
- Verifier.
- Writer.
- Citation checker.
- AgentOps trace persistence.

Current implementation completes these as an extractive baseline running on LangGraph orchestration (with a sequential fallback), with a verifier-driven retrieval loop and question decomposition. LLM synthesis as the default writer is the planned upgrade.

### Milestone 4: MLOps Foundation

- DVC setup.
- MLflow tracking.
- Dockerized backend.
- CI smoke tests.

### Milestone 5: LLMOps Foundation

- Prompt registry.
- Golden evaluation set.
- Model serving profiles.
- Citation and faithfulness checks.

### Milestone 6: AgentOps Foundation

- Agent trace logging.
- Tool-call history.
- State transition records.
- Failure replay records.

### Milestone 7: Fine-Tuning

- LoRA or QLoRA experiment.
- Model comparison report.

### Milestone 8: Monitoring

- Drift report.
- Latency dashboard.
- Failure case logging.

## 17. Risks and Mitigations

### Risk: Scope Creep

Mitigation: Build two domain presets first: Academic and Finance/Crypto.

### Risk: Weak Resume Signal

Mitigation: Emphasize evaluation, model tracking, deployment, and monitoring in addition to the agent.

### Risk: Hallucinated Answers

Mitigation: Require citations, add evidence verification, and make the system say when evidence is insufficient.

### Risk: GPU Memory Limits

Mitigation: Default to 7B/8B models, use quantization, and keep 14B/32B models optional.

### Risk: Poor Data Quality

Mitigation: Add document checksums, parsing diagnostics, chunk quality reports, and fixture tests.

## 18. Success Criteria

The project is successful when a reviewer can:

- Upload a document collection.
- Ask questions and receive grounded answers with citations.
- Generate a useful structured report.
- Inspect tracked MLflow runs.
- Reproduce data processing with DVC.
- Run the backend locally with Docker.
- See evaluation results and failure cases.
- Understand the agent workflow from the code and docs.
