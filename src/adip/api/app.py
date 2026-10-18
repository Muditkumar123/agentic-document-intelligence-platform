"""FastAPI app for RAG, agent, and indexing workflows."""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from adip.api.cache import cache_stats
from adip.api.schemas import (
    AgentRunRequest,
    IndexBackend,
    ModelCheckRequest,
    RagQueryRequest,
    RebuildIndexRequest,
)
from adip.api.security import RequestSizeLimitMiddleware, write_access_error
from adip.api.services import (
    check_model_connection,
    delete_raw_document,
    drift_summary,
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
from adip.api.settings import ADMIN_TOKEN_ENV, ApiSettings

STATIC_DIR = Path(__file__).resolve().parent / "static"
logger = logging.getLogger(__name__)


def create_app(settings: ApiSettings | None = None):
    try:
        from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
        from fastapi.responses import FileResponse
        from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
        from fastapi.staticfiles import StaticFiles
    except ImportError as exc:  # pragma: no cover - import guard for optional extra
        raise ImportError("Install the API extra with `pip install -e .[api]` to use FastAPI.") from exc

    settings = settings or ApiSettings.from_env()
    if not settings.admin_token:
        logger.warning(
            "%s is not set: upload, delete, and rebuild endpoints only accept requests from loopback clients.",
            ADMIN_TOKEN_ENV,
        )

    app = FastAPI(
        title="Agentic Document Intelligence Platform API",
        version=health_payload()["version"],
        description="HTTP API for document ingestion, RAG search, and agentic cited answers.",
    )
    app.state.settings = settings
    max_upload_mb = settings.max_upload_bytes / (1024 * 1024)
    app.add_middleware(
        RequestSizeLimitMiddleware,
        max_bytes=settings.max_request_bytes,
        detail=f"Request is too large: uploads are limited to {max_upload_mb:g} MB.",
    )
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    bearer_scheme = HTTPBearer(auto_error=False, description="Admin token (ADIP_ADMIN_TOKEN) for write endpoints.")

    def require_write_access(
        request: Request,
        credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    ) -> None:
        error = write_access_error(
            settings,
            client_host=request.client.host if request.client else None,
            bearer_token=credentials.credentials if credentials else None,
        )
        if error is not None:
            status_code, detail = error
            headers = {"WWW-Authenticate": "Bearer"} if status_code == 401 else None
            raise HTTPException(status_code=status_code, detail=detail, headers=headers)

    write_guard = [Depends(require_write_access)]

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
        return {
            **health_payload(),
            "write_access": settings.write_access,
            "max_upload_mb": max_upload_mb,
        }

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

    @app.get("/monitoring/drift")
    def drift_monitoring() -> dict[str, Any]:
        return drift_summary()

    @app.get("/model-profiles")
    def model_profiles() -> dict[str, Any]:
        return call_service(model_profiles_summary, HTTPException)

    @app.get("/index/documents")
    def index_documents(backend: IndexBackend = "tfidf") -> dict[str, Any]:
        return call_service(lambda: indexed_documents(settings.index_path(backend)), HTTPException)

    @app.get("/documents")
    def raw_documents() -> dict[str, Any]:
        return call_service(
            lambda: list_raw_documents(settings.raw_dir, settings.index_path("tfidf")),
            HTTPException,
        )

    @app.delete("/documents/{filename}", dependencies=write_guard)
    def remove_document(filename: str) -> dict[str, Any]:
        return call_service(lambda: delete_raw_document(filename, settings.raw_dir), HTTPException)

    @app.post("/documents/upload", dependencies=write_guard)
    async def upload_document(file: UploadFile = File(...)) -> dict[str, Any]:
        # Read at most one byte past the limit, so an oversized file never sits in memory whole.
        content = await file.read(settings.max_upload_bytes + 1)
        if len(content) > settings.max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"{file.filename or 'Upload'} is larger than the {max_upload_mb:g} MB upload limit.",
            )
        return call_service(
            lambda: save_uploaded_document(
                filename=file.filename or "",
                content=content,
                raw_dir=settings.raw_dir,
            ),
            HTTPException,
        )

    @app.get("/history/agent-traces")
    def agent_trace_history(limit: int = 25) -> dict[str, Any]:
        return call_service(
            lambda: list_agent_trace_history(trace_dir=settings.trace_dir, limit=limit), HTTPException
        )

    @app.get("/history/agent-traces/{run_id}")
    def agent_trace_detail(run_id: str) -> dict[str, Any]:
        return call_service(lambda: get_agent_trace(run_id, trace_dir=settings.trace_dir), HTTPException)

    @app.get("/history/mlops-runs")
    def mlops_run_history(limit: int = 25) -> dict[str, Any]:
        return call_service(
            lambda: list_mlops_run_history(run_dir=settings.mlops_run_dir, limit=limit), HTTPException
        )

    @app.get("/history/mlops-runs/{run_id}")
    def mlops_run_detail(run_id: str) -> dict[str, Any]:
        return call_service(lambda: get_mlops_run(run_id, run_dir=settings.mlops_run_dir), HTTPException)

    @app.post("/rag/query")
    def rag_query(request: RagQueryRequest) -> dict[str, Any]:
        return call_service(lambda: run_rag_query(request, settings.index_path(request.backend)), HTTPException)

    @app.post("/agent/run")
    def agent_run(request: AgentRunRequest) -> dict[str, Any]:
        return call_service(
            lambda: run_agent_workflow(
                request,
                index_path=settings.index_path(request.backend),
                trace_dir=settings.trace_dir,
            ),
            HTTPException,
        )

    @app.post("/models/check")
    def model_check(request: ModelCheckRequest) -> dict[str, Any]:
        return call_service(lambda: check_model_connection(request), HTTPException)

    @app.post("/pipeline/rebuild-index", dependencies=write_guard)
    def pipeline_rebuild_index(request: RebuildIndexRequest) -> dict[str, Any]:
        return call_service(lambda: rebuild_index(request, settings), HTTPException)

    return app


def call_service(service_call: Callable[[], dict[str, Any]], http_exception_type):
    try:
        return service_call()
    except FileNotFoundError as exc:
        raise http_exception_type(status_code=404, detail=str(exc)) from exc
    except (ImportError, TypeError, ValueError) as exc:
        raise http_exception_type(status_code=400, detail=str(exc)) from exc


app = create_app()
