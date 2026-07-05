"""FastAPI app for RAG, agent, and indexing workflows."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from adip.api.cache import cache_stats
from adip.api.schemas import (
    AgentRunRequest,
    ModelCheckRequest,
    RagQueryRequest,
    RebuildIndexRequest,
)
from adip.api.services import (
    check_model_connection,
    delete_raw_document,
    generation_eval_summary,
    get_agent_trace,
    get_mlops_run,
    health_payload,
    indexed_documents,
    list_agent_trace_history,
    list_mlops_run_history,
    list_raw_documents,
    model_profiles_summary,
    offline_eval_snapshot,
    rebuild_index,
    retrieval_benchmark_summary,
    run_agent_workflow,
    run_rag_query,
    save_uploaded_document,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app():
    try:
        from fastapi import FastAPI, File, Form, HTTPException, UploadFile
        from fastapi.responses import FileResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError as exc:  # pragma: no cover - import guard for optional extra
        raise ImportError("Install the API extra with `pip install -e .[api]` to use FastAPI.") from exc

    app = FastAPI(
        title="Agentic Document Intelligence Platform API",
        version=health_payload()["version"],
        description="HTTP API for document ingestion, RAG search, and agentic cited answers.",
    )
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def dashboard():
        # Never cache the HTML shell: it pins the versioned (?v=) JS/CSS, so a stale
        # cached page would keep loading old assets after an update.
        return FileResponse(
            STATIC_DIR / "dashboard.html",
            headers={"Cache-Control": "no-cache"},
        )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return health_payload()

    @app.get("/monitoring/retrieval-benchmark")
    def retrieval_benchmark() -> dict[str, Any]:
        return retrieval_benchmark_summary()

    @app.get("/monitoring/generation-eval")
    def generation_eval() -> dict[str, Any]:
        return generation_eval_summary()

    @app.get("/monitoring/offline-eval")
    def offline_eval() -> dict[str, Any]:
        return offline_eval_snapshot()

    @app.get("/monitoring/cache")
    def cache_monitoring() -> dict[str, Any]:
        return cache_stats()

    @app.get("/model-profiles")
    def model_profiles() -> dict[str, Any]:
        return call_service(model_profiles_summary, HTTPException)

    @app.get("/index/documents")
    def index_documents(index_path: str = "data/processed/vector_index") -> dict[str, Any]:
        return call_service(lambda: indexed_documents(Path(index_path)), HTTPException)

    @app.get("/documents")
    def raw_documents(
        raw_dir: str = "data/raw",
        index_path: str = "data/processed/vector_index",
    ) -> dict[str, Any]:
        return call_service(lambda: list_raw_documents(Path(raw_dir), Path(index_path)), HTTPException)

    @app.delete("/documents/{filename}")
    def remove_document(filename: str, raw_dir: str = "data/raw") -> dict[str, Any]:
        return call_service(lambda: delete_raw_document(filename, Path(raw_dir)), HTTPException)

    @app.post("/documents/upload")
    async def upload_document(
        file: UploadFile = File(...),
        raw_dir: str = Form("data/raw"),
    ) -> dict[str, Any]:
        content = await file.read()
        return call_service(
            lambda: save_uploaded_document(
                filename=file.filename or "",
                content=content,
                raw_dir=Path(raw_dir),
            ),
            HTTPException,
        )

    @app.get("/history/agent-traces")
    def agent_trace_history(limit: int = 25) -> dict[str, Any]:
        return call_service(lambda: list_agent_trace_history(limit=limit), HTTPException)

    @app.get("/history/agent-traces/{run_id}")
    def agent_trace_detail(run_id: str) -> dict[str, Any]:
        return call_service(lambda: get_agent_trace(run_id), HTTPException)

    @app.get("/history/mlops-runs")
    def mlops_run_history(limit: int = 25) -> dict[str, Any]:
        return call_service(lambda: list_mlops_run_history(limit=limit), HTTPException)

    @app.get("/history/mlops-runs/{run_id}")
    def mlops_run_detail(run_id: str) -> dict[str, Any]:
        return call_service(lambda: get_mlops_run(run_id), HTTPException)

    @app.post("/rag/query")
    def rag_query(request: RagQueryRequest) -> dict[str, Any]:
        return call_service(lambda: run_rag_query(request), HTTPException)

    @app.post("/agent/run")
    def agent_run(request: AgentRunRequest) -> dict[str, Any]:
        return call_service(lambda: run_agent_workflow(request), HTTPException)

    @app.post("/models/check")
    def model_check(request: ModelCheckRequest) -> dict[str, Any]:
        return call_service(lambda: check_model_connection(request), HTTPException)

    @app.post("/pipeline/rebuild-index")
    def pipeline_rebuild_index(request: RebuildIndexRequest) -> dict[str, Any]:
        return call_service(lambda: rebuild_index(request), HTTPException)

    return app


def call_service(service_call: Callable[[], dict[str, Any]], http_exception_type):
    try:
        return service_call()
    except FileNotFoundError as exc:
        raise http_exception_type(status_code=404, detail=str(exc)) from exc
    except (ImportError, TypeError, ValueError) as exc:
        raise http_exception_type(status_code=400, detail=str(exc)) from exc


app = create_app()
