"""Access control and request-size limits for the HTTP API.

Write endpoints (upload, delete, index rebuild) change what every user of the
instance sees, so they are gated:

- With ``ADIP_ADMIN_TOKEN`` set, a request must send ``Authorization: Bearer <token>``.
- Without it, writes are accepted only from loopback clients, so a laptop dev
  server keeps working while any network-exposed deployment fails closed.

Read endpoints (query, agent run, listings, monitoring) stay open.
"""

from __future__ import annotations

import hmac
from typing import Any

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
