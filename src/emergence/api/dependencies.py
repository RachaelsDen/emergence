"""Typed service access, credential roles, and bounded HTTP request bodies."""

from dataclasses import dataclass
from hmac import compare_digest
from typing import Annotated, Final, Literal

from anyio import fail_after
from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from emergence.config import ConfigurationError, SafeModel, Settings
from emergence.inference.gateway import Gateway
from emergence.resources.quota_polling import QuotaMonitor
from emergence.storage.delivery import ArchiveDelivery


@dataclass(frozen=True, slots=True)
class Services:
    gateway: Gateway
    monitor: QuotaMonitor
    archive: ArchiveDelivery


class Identity(SafeModel):
    actor: Literal["operator", "reader"]


class ErrorResponse(SafeModel):
    error: str
    request_id: str | None = None


ERROR_DESCRIPTIONS: Final = {
    400: "Bad Request",
    401: "Unauthorized",
    402: "Payment Required",
    403: "Forbidden",
    404: "Not Found",
    408: "Request Timeout",
    409: "Conflict",
    413: "Content Too Large",
    422: "Unprocessable Content",
    429: "Too Many Requests",
    503: "Service Unavailable",
    507: "Insufficient Storage",
}


class Authorization:
    """Snapshot credential sources at startup; only role names leave this boundary."""

    def __init__(self, settings: Settings) -> None:
        operator = settings.secret(settings.operator_token_env)
        reader = settings.secret(settings.reader_token_env)
        self._operator = operator.get_secret_value().encode() if operator else None
        self._reader = reader.get_secret_value().encode() if reader else None
        if any(token is not None and len(token) < 32 for token in (self._operator, self._reader)):
            raise ConfigurationError("api_token_too_short")
        if (
            self._operator is not None
            and self._reader is not None
            and compare_digest(self._operator, self._reader)
        ):
            raise ConfigurationError("api_roles_require_distinct_tokens")

    async def reader(
        self,
        credentials: Annotated[
            HTTPAuthorizationCredentials | None, Security(HTTPBearer(auto_error=False))
        ] = None,
    ) -> Identity:
        if credentials is None:
            raise HTTPException(401, "unauthorized", headers={"WWW-Authenticate": "Bearer"})
        if len(credentials.credentials) > 4096:
            raise HTTPException(401, "unauthorized", headers={"WWW-Authenticate": "Bearer"})
        received = credentials.credentials.encode()
        operator = self._operator is not None and compare_digest(received, self._operator)
        reader = self._reader is not None and compare_digest(received, self._reader)
        if operator:
            return Identity(actor="operator")
        if reader:
            return Identity(actor="reader")
        raise HTTPException(401, "unauthorized", headers={"WWW-Authenticate": "Bearer"})

    async def operator(
        self,
        credentials: Annotated[
            HTTPAuthorizationCredentials | None, Security(HTTPBearer(auto_error=False))
        ] = None,
    ) -> Identity:
        identity = await self.reader(credentials)
        if identity.actor != "operator":
            raise HTTPException(403, "operator_required")
        return identity


class RequestBodyLimit:
    """Buffer at most the configured body bytes before FastAPI parses JSON."""

    def __init__(self, app: ASGIApp, limit: int, timeout: float) -> None:
        self.app = app
        self.limit = limit
        self.timeout = timeout

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] not in {"POST", "PUT", "PATCH"}:
            await self.app(scope, receive, send)
            return
        body = bytearray()
        try:
            with fail_after(self.timeout):
                while True:
                    message = await receive()
                    if message["type"] == "http.disconnect":
                        return
                    body.extend(message.get("body", b""))
                    if len(body) > self.limit:
                        await JSONResponse({"error": "request_body_too_large"}, status_code=413)(
                            scope, receive, send
                        )
                        return
                    if not message.get("more_body", False):
                        break
        except TimeoutError:
            await JSONResponse({"error": "request_body_timeout"}, status_code=408)(
                scope, receive, send
            )
            return
        delivered = False

        async def replay() -> Message:
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, replay, send)
