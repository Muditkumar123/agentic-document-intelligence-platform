# Source Layout

Internal package name: `adip`

This avoids spaces in Python imports while keeping the user-facing project folder name intact.

Planned modules:

- `ingestion`: document parsing, cleaning, metadata extraction, chunking.
- `rag`: embeddings, vector search, reranking, cited retrieval.
- `agents`: graph nodes, agent state transitions, trace persistence, and workflow runners.
- `serving`: local LLM loading and inference adapters.
- `evaluation`: retrieval, generation, citation, and regression tests.
- `llmops`: prompt templates, model adapters, generation metrics, and citation quality checks.
- `mlops`: local run tracking, optional MLflow hooks, reproducible pipeline commands.
- `api`: FastAPI routes and app wiring.
- `schemas`: typed request, response, document, chunk, and run models.
- `config`: project settings and model profiles.
