"""Request guards for the local command center.

When ``API_TOKEN`` is empty the app stays in dev mode and ``auth_enabled`` is
false. When it is set, mutating routes and evidence files require the token.
The token is compared in constant time and is never written to a log line.
"""

from __future__ import annotations

import secrets
import time
from collections import defaultdict, deque
from typing import Any
from urllib.parse import parse_qs

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response


def mask_credentials(value: object) -> object:
    """Hide a password embedded in a URL. Other values pass through."""
    if not isinstance(value, str) or "://" not in value or "@" not in value:
        return value
    scheme, rest = value.split("://", 1)
    userinfo, separator, host = rest.rpartition("@")
    if not separator or ":" not in userinfo:
        return value
    user, _password = userinfo.split(":", 1)
    return f"{scheme}://{user}:***@{host}"


def tokens_match(expected: str | None, presented: str | None) -> bool:
    if not expected or not presented:
        return False
    try:
        return secrets.compare_digest(expected, presented)
    except (TypeError, ValueError):
        return False


def _bearer(header: str) -> str:
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    return ""


def _query_token(query: str) -> str:
    values = parse_qs(query, keep_blank_values=False).get("token") or []
    return values[0] if values else ""


class SecurityMiddleware(BaseHTTPMiddleware):
    """Auth, a per-IP mutation limit, a body cap, and safe response headers."""

    def __init__(self, app: Any, runtime: Any) -> None:
        super().__init__(app)
        self.runtime = runtime
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        config = self.runtime.config
        if _body_too_large(request, int(config.max_body_bytes)):
            return _json(413, "request body too large")
        if _is_mutation(request.method) and not self._allow(request, int(config.rate_limit_per_minute)):
            return _json(429, "too many requests")
        if _needs_auth(request.method, request.url.path) and config.api_token:
            presented = _bearer(request.headers.get("authorization", ""))
            if request.method == "GET":
                presented = presented or _query_token(request.url.query)
            if not tokens_match(config.api_token, presented):
                response = _json(401, "unauthorized")
                response.headers["WWW-Authenticate"] = "Bearer"
                _stamp(response)
                return response
        response = await call_next(request)
        _stamp(response)
        return response

    def _allow(self, request: Request, limit: int) -> bool:
        if limit <= 0:
            return True
        now = time.monotonic()
        host = request.client.host if request.client else "local"
        bucket = self._hits[host]
        while bucket and now - bucket[0] > 60.0:
            bucket.popleft()
        if len(bucket) >= limit:
            return False
        bucket.append(now)
        return True


def _needs_auth(method: str, path: str) -> bool:
    if method in {"POST", "PUT", "DELETE", "PATCH"}:
        return True
    return path.startswith("/api/v1/evidence")


def _is_mutation(method: str) -> bool:
    return method in {"POST", "PUT", "DELETE", "PATCH"}


def _body_too_large(request: Request, limit: int) -> bool:
    if limit <= 0 or not _is_mutation(request.method):
        return False
    raw = request.headers.get("content-length")
    if raw is None:
        return False
    try:
        return int(raw) > limit
    except ValueError:
        return True


def _json(status: int, detail: str) -> JSONResponse:
    return JSONResponse({"detail": detail}, status_code=status)


def _stamp(response: Response) -> None:
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
