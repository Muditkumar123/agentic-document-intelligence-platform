"""Local MLOps run tracking with optional MLflow integration."""

from __future__ import annotations

import importlib.util
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from adip.mlops.fingerprint import git_commit, git_dirty_status


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class MLOpsRun:
    run_name: str
    run_dir: Path
    enable_mlflow: bool = False
    mlflow_tracking_uri: str | None = None
    tags: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    run_id: str = field(default_factory=lambda: f"run_{uuid.uuid4().hex[:12]}")
    status: str = "created"
    started_at: str = field(default_factory=utc_now_iso)
    ended_at: str | None = None
    duration_ms: float | None = None
    error: str | None = None
    mlflow_run_id: str | None = None

    def __enter__(self) -> "MLOpsRun":
        self.status = "running"
        self._started_perf = time.perf_counter()
        self._start_mlflow_if_available()
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if exc is not None:
            self.status = "failed"
            self.error = str(exc)
        elif self.status == "running":
            self.status = "completed"

        self.ended_at = utc_now_iso()
        self.duration_ms = (time.perf_counter() - self._started_perf) * 1000
        self.write()
        self._end_mlflow()
        return False

    def log_param(self, key: str, value: Any) -> None:
        self.params[key] = value
        if self._mlflow_active:
            self._mlflow.log_param(key, value)

    def log_params(self, values: dict[str, Any]) -> None:
        for key, value in values.items():
            self.log_param(key, value)

    def log_metric(self, key: str, value: float | int) -> None:
        numeric_value = float(value)
        self.metrics[key] = numeric_value
        if self._mlflow_active:
            self._mlflow.log_metric(key, numeric_value)

    def log_metrics(self, values: dict[str, float | int]) -> None:
        for key, value in values.items():
            self.log_metric(key, value)

    def log_artifact(self, key: str, path: Path | str) -> None:
        artifact_path = str(Path(path))
        self.artifacts[key] = artifact_path
        if self._mlflow_active and Path(path).exists():
            self._mlflow.log_artifact(artifact_path)

    def write(self) -> Path:
        path = self.run_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return path

    @property
    def run_path(self) -> Path:
        return self.run_dir.expanduser() / self.run_id / "run.json"

    @property
    def _mlflow_active(self) -> bool:
        return hasattr(self, "_mlflow") and self._mlflow is not None and self.mlflow_run_id is not None

    def _start_mlflow_if_available(self) -> None:
        self._mlflow = None
        if not self.enable_mlflow or importlib.util.find_spec("mlflow") is None:
            return

        import mlflow

        self._mlflow = mlflow
        if self.mlflow_tracking_uri:
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        active_run = mlflow.start_run(run_name=self.run_name)
        self.mlflow_run_id = active_run.info.run_id
        for key, value in self.tags.items():
            mlflow.set_tag(key, value)

    def _end_mlflow(self) -> None:
        if self._mlflow_active:
            self._mlflow.set_tag("status", self.status)
            if self.error:
                self._mlflow.set_tag("error", self.error)
            self._mlflow.end_run()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["run_dir"] = str(self.run_dir)
        payload["run_path"] = str(self.run_path)
        payload["git_commit"] = git_commit(Path.cwd())
        payload["git_dirty_status"] = git_dirty_status(Path.cwd())
        payload["mlflow_available"] = importlib.util.find_spec("mlflow") is not None
        payload["mlflow_enabled"] = self.enable_mlflow
        return payload


def start_run(
    run_name: str,
    run_dir: Path = Path("data/monitoring/mlops_runs"),
    enable_mlflow: bool = False,
    mlflow_tracking_uri: str | None = None,
    tags: dict[str, Any] | None = None,
) -> MLOpsRun:
    return MLOpsRun(
        run_name=run_name,
        run_dir=run_dir,
        enable_mlflow=enable_mlflow,
        mlflow_tracking_uri=mlflow_tracking_uri,
        tags=tags or {},
    )
