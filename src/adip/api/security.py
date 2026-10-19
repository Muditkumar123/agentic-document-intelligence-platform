"""Access control and request-size limits for the HTTP API.

Write endpoints (upload, delete, index rebuild) change what every user of the
instance sees, so they are gated:

- With ``ADIP_ADMIN_TOKEN`` set, a request must send ``Authorization: Bearer <token>``.
- Without it, writes are accepted only from loopback clients, so a laptop dev
  server keeps working while any network-exposed deployment fails closed.

Read endpoints (query, agent run, listings, monitoring) stay open, but the ones
that do work share a per-client rate limit, and a model endpoint URL sent by a
remote client must be a public https host (no server-side request forgery) and
come with the client's own API key.
"""

from __future__ import annotations

import hmac
import ipaddress
import socket
import threading
import time
from collections import OrderedDict, deque
from typing import Any
from urllib.parse import urlparse

from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse

from adip.api.settings import ADMIN_TOKEN_ENV, ApiSettings

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


def write_access_error(
    settings: ApiSettings,
    client_host: str | None,
    bearer_token: str | None,
) -> tuple[int, str] | None:
    """``(status, detail)`` when a write must be refused, ``None`` when it is allowed."""
    if settings.admin_token:
        if not bearer_token:
            return (
                401,
                "This endpoint changes server data and needs the admin token. Send "
                "`Authorization: Bearer <token>`, or enter it under Admin Token in the "
                "dashboard's Documents tab.",
            )
        if not hmac.compare_digest(bearer_token.encode("utf-8"), settings.admin_token.encode("utf-8")):
            return 403, "The admin token is not valid."
        return None
    if client_host in LOOPBACK_HOSTS:
        return None
    return (
        403,
        f"Write endpoints only accept requests from this machine because no admin token is "
        f"configured. Set {ADMIN_TOKEN_ENV} on the server to allow authenticated remote writes.",
    )


class RequestSizeLimitMiddleware:
    """Reject request bodies over ``max_bytes`` before they are buffered or spooled.

    A declared ``Content-Length`` over the limit is refused immediately; bodies
    without one (chunked uploads) are counted as they stream in.
    """

    def __init__(self, app: Any, max_bytes: int, detail: str = "Request body is too large.") -> None:
        self.app = app
        self.max_bytes = max_bytes
        self.detail = detail

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        declared = dict(scope.get("headers") or []).get(b"content-length")
        if declared is not None:
            try:
                declared_size = int(declared)
            except ValueError:
                await JSONResponse({"detail": "Invalid Content-Length header"}, status_code=400)(scope, receive, send)
                return
            if declared_size > self.max_bytes:
                await JSONResponse({"detail": self.detail}, status_code=413)(scope, receive, send)
                return

        received = 0

        async def limited_receive() -> dict[str, Any]:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    # FastAPI re-raises HTTPException from body parsing, so this becomes a 413.
                    raise HTTPException(status_code=413, detail=self.detail)
            return message

        await self.app(scope, limited_receive, send)


def model_endpoint_error(url: str | None, client_host: str | None, api_key: str | None = None) -> str | None:
    """Why a client-supplied model endpoint is refused, or ``None`` when it may be called.

    The server sends requests to this URL, so a remote client could otherwise make it reach
    internal services or the cloud metadata address, or collect the server's own provider key
    (which a profile falls back to when the request carries none). Loopback clients are
    trusted, so local development can still use a local model server and the server's keys.
    """
    if not url or client_host in LOOPBACK_HOSTS:
        return None
    if not api_key:
        return "A custom model endpoint needs its own api_key; the server's keys are never sent to client-supplied endpoints."
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        return "Model endpoint must be an https URL."
    try:
        addresses = {info[4][0].split("%")[0] for info in socket.getaddrinfo(parsed.hostname, parsed.port or 443)}
    except (socket.gaierror, UnicodeError):
        return f"Model endpoint host {parsed.hostname!r} does not resolve."
    # shortcut: checks the addresses at request time, not the ones the HTTP client later connects to
    # (DNS rebinding); pin the resolved IP or route model calls through an egress proxy to close that.
    if any(not ipaddress.ip_address(address).is_global for address in addresses):
        return "Model endpoint must be a public internet host; private, loopback and link-local addresses are refused."
    return None


class RateLimiter:
    """At most ``per_minute`` requests per client in any 60-second window.

    shortcut: in memory and per process, keyed by the connecting address. Behind a proxy that
    hides client addresses every caller shares one budget; put the limit in the gateway for
    multi-instance or proxied deployments.
    """

    MAX_TRACKED_CLIENTS = 10_000

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._hits: OrderedDict[str, deque[float]] = OrderedDict()  # least recently seen client first
        self._lock = threading.Lock()

    def retry_after(self, client: str, now: float | None = None) -> int | None:
        """Seconds until ``client`` may try again, or ``None`` (and the hit is counted) if allowed."""
        if self.per_minute <= 0:
            return None
        now = time.monotonic() if now is None else now
        with self._lock:
            hits = self._hits.get(client)
            if hits is None:
                hits = self._hits[client] = deque()
                if len(self._hits) > self.MAX_TRACKED_CLIENTS:
                    # a hard cap, so a flood of new addresses cannot grow memory; the evicted client
                    # starts a fresh window, which only matters when more clients are active than the cap
                    self._hits.popitem(last=False)
            else:
                self._hits.move_to_end(client)
            while hits and now - hits[0] >= 60:
                hits.popleft()
            if len(hits) >= self.per_minute:
                return max(1, int(60 - (now - hits[0])) + 1)
            hits.append(now)
            return None

