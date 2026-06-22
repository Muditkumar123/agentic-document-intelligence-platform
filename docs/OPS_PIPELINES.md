# MLOps, LLMOps, and AgentOps Pipelines

This project will explicitly demonstrate three related but distinct operational pipelines.

## 1. MLOps Pipeline

The MLOps pipeline manages the lifecycle of datasets, traditional NLP models, fine-tuned models, experiments, and deployments.

Current implementation note: `crypto_env` does not currently include MLflow or DVC, so the project includes a working local JSON run tracker plus DVC-compatible pipeline files. MLflow logging activates automatically when MLflow is installed and `--enable-mlflow` is passed.

### Purpose

Make the machine learning parts reproducible, measurable, versioned, and deployable.

### Pipeline Flow

```text
Raw Documents
    |
    v
Data Validation
    |
    v
Document Parsing and Chunking
    |
    v
Dataset Versioning with DVC
    |
    v
Training or Fine-Tuning
    |
    v
Experiment Tracking with MLflow
    |
    v
Model Registry
    |
    v
Dockerized Deployment
    |
    v
Monitoring and Drift Reports
```

### What To Track

- Dataset version
- Parser version
- Chunk size and overlap
- Model name
- Training parameters
- Evaluation metrics
- Model artifacts
- Drift reports
- Deployment version

### Resume Signal

Shows that the project is not just an AI demo. It has reproducibility, model lifecycle management, evaluation, and deployment discipline.

## 2. LLMOps Pipeline

The LLMOps pipeline manages prompts, retrieval, generation, LLM serving, model comparisons, and answer quality.

Current implementation note: prompt versioning, prompt hashes, generation metrics, citation coverage checks, unsupported sentence checks, and a deterministic grounded baseline are implemented. A Hugging Face adapter exists for local model experiments when a model is available.

### Purpose

Make LLM behavior observable, testable, and improvable.

### Pipeline Flow

```text
User Query
    |
    v
Prompt Template Selection
    |
    v
Retrieval and Reranking
    |
    v
Context Assembly
    |
    v
LLM Generation
    |
    v
Citation and Faithfulness Evaluation
    |
    v
Prompt / Model / Retrieval Metrics Logged
    |
    v
Regression Test Against Golden Q&A Set
```

### What To Track

- Prompt template version
- System prompt version
- Model name and quantization mode
- Retrieval top-k
- Reranker configuration
- Context token count
- Output token count
- Latency
- Cost estimate or local GPU usage
- Answer relevance
- Faithfulness
- Citation accuracy
- Refusal quality when evidence is missing

### Resume Signal

Shows practical LLM engineering: RAG, prompt management, local model serving, evaluation, and hallucination control.

## 3. AgentOps Pipeline

The AgentOps pipeline manages the operational behavior of the agent graph: planning, tool use, state transitions, verification, failures, and replay.

### Purpose

Make multi-step agent behavior inspectable instead of treating the agent as a black box.

### Pipeline Flow

```text
User Task
    |
    v
Intent Router
    |
    v
Planner Node
    |
    v
Tool Calls
    |
    v
Retriever Node
    |
    v
Evidence Verifier Node
    |
    v
Writer Node
    |
    v
Citation Checker Node
    |
    v
Final Answer
    |
    v
Agent Trace, Tool Logs, Metrics, and Failure Cases
```

### What To Track

- Agent run ID
- User task type
- Domain preset
- Node-by-node state transitions
- Tool calls
- Tool inputs and outputs
- Retrieved chunk IDs
- Verification notes
- Citation checker results
- Agent retries
- Failure reason
- Human approval or correction events
- Final output quality score

### Resume Signal

Shows agentic AI beyond simple tool calling. It proves the system can trace, debug, evaluate, and improve agent workflows.

## How The Three Pipelines Work Together

```text
MLOps
  owns data, model artifacts, training, deployment, monitoring

LLMOps
  owns prompts, RAG quality, LLM serving, generation evaluation

AgentOps
  owns agent traces, tool calls, workflow reliability, failure replay
```

In the final project, one user request should produce:

- An MLflow run.
- A prompt and model version record.
- A retrieval evaluation record.
- An agent trace.
- A final cited answer or report.
- Quality metrics and failure notes.

## MVP Implementation Order

1. MLOps foundation: DVC plus MLflow tracking for ingestion and RAG.
2. LLMOps foundation: prompt versions, RAG evaluation, citation checks.
3. AgentOps foundation: graph traces, node logs, tool-call history.

This order keeps the project buildable while still making all three pipelines visible.
