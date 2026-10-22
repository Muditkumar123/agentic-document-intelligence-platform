"""OpenTelemetry spans for agent runs, agent nodes, retrieval searches and model calls.

Tracing is optional. Install the ``tracing`` extra and point the standard OTLP variables
at a collector (Langfuse, Arize Phoenix, Jaeger, an OpenTelemetry Collector)::

    OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:6006    # or OTEL_EXPORTER_OTLP_TRACES_ENDPOINT
    OTEL_EXPORTER_OTLP_HEADERS=Authorization=Basic%20...  # when the collector needs auth
    OTEL_SERVICE_NAME=adip

Without the packages, or without an endpoint, every span is a no-op. If the host process
already installed a tracer provider (``opentelemetry-instrument``, a test), it is used as is.

Spans carry counts, ids, model names and token usage, never question text, prompts,
document text or API keys.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

try:
    from opentelemetry import trace as otel_trace
    from opentelemetry.trace import Status, StatusCode
except ImportError:  # the tracing extra is not installed
    otel_trace = None

TRACER_NAME = "adip"
ENDPOINT_ENVS = ("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "OTEL_EXPORTER_OTLP_ENDPOINT")

_configure_lock = threading.Lock()
_configured = False


def _configure_exporter() -> None:
    """Install an OTLP/HTTP exporter once, when an endpoint is set and no provider exists yet."""
    global _configured
    with _configure_lock:
        if _configured:
            return
        _configured = True
        if not any(os.environ.get(name) for name in ENDPOINT_ENVS):
            return
        if not isinstance(otel_trace.get_tracer_provider(), otel_trace.ProxyTracerProvider):
            return  # someone else already configured tracing
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
        except ImportError:
            return
        # The exporter and the resource read the remaining OTEL_* variables themselves.
        provider = TracerProvider(resource=Resource.create({"service.name": os.environ.get("OTEL_SERVICE_NAME", "adip")}))
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        otel_trace.set_tracer_provider(provider)


def _clean(attributes: dict[str, Any] | None) -> dict[str, Any]:
    """Keep only values OpenTelemetry accepts; drop ``None``."""
    return {
        key: value
        for key, value in (attributes or {}).items()
        if isinstance(value, (str, bool, int, float))
    }


@contextmanager
def span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[Any]:
    """A span around the block (``None`` when tracing is off). Exceptions are recorded on it."""
    if otel_trace is None:
        yield None
        return
    _configure_exporter()
    with otel_trace.get_tracer(TRACER_NAME).start_as_current_span(name, attributes=_clean(attributes)) as current:
        yield current


def set_attributes(current: Any, attributes: dict[str, Any]) -> None:
    if current is not None:
        current.set_attributes(_clean(attributes))


def mark_error(current: Any, message: str) -> None:
    """Flag a span whose block handled its own exception (so the context manager never saw it)."""
    if current is not None:
        current.set_status(Status(StatusCode.ERROR, message))
