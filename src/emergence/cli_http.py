"""Authenticated REST client and sanitized operator-command output."""

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, NoReturn
from urllib.parse import urlsplit

import anyio
import httpx
import typer
from pydantic import JsonValue, TypeAdapter, ValidationError

from emergence.config import ConfigurationError, Settings, load_settings
from emergence.telemetry.logging import Redactor

API: Final = "/api/v1"
DEADLINE: Final = 60.0
JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


def _emit(payload: JsonValue, *, err: bool = False) -> None:
    typer.echo(json.dumps(payload), err=err)


def fail(code: str, *, status: int = 0, exit_code: int = 2) -> NoReturn:
    payload: dict[str, JsonValue] = {"error": code}
    if status:
        payload["status"] = status
    _emit(payload, err=True)
    raise typer.Exit(exit_code)


def load_client_settings(config: Path | None) -> Settings:
    try:
        return load_settings(config)
    except ConfigurationError as error:
        fail(error.code)


@dataclass(frozen=True, slots=True)
class ClientOptions:
    config: Path | None
    url: str | None
    token_env: str | None


def client_options(ctx: typer.Context) -> ClientOptions:
    options = ctx.find_object(ClientOptions)
    assert options is not None  # Set by the root callback for every subcommand.
    return options


@dataclass(frozen=True, slots=True)
class Request:
    path: str
    method: Literal["GET", "POST"] = "GET"
    params: tuple[tuple[str, str | int | None], ...] = ()
    body: JsonValue = None
    public: bool = False
    idempotency_key: str | None = None


async def _send(url: str, token: str | None, request: Request) -> httpx.Response:
    headers: dict[str, str] = {}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    if request.idempotency_key is not None:
        headers["Idempotency-Key"] = request.idempotency_key
    with anyio.fail_after(DEADLINE):
        async with httpx.AsyncClient(
            base_url=url,
            timeout=httpx.Timeout(DEADLINE, connect=5.0),
            follow_redirects=False,
            trust_env=False,
        ) as client:
            return await client.request(
                request.method,
                request.path,
                headers=headers,
                json=request.body,
                params={key: value for key, value in request.params if value is not None},
            )


def _safe_value(value: JsonValue, redactor: Redactor) -> JsonValue:
    # The shared redactor handles values and sensitive fields, but not JSON keys.
    match value:
        case dict():
            return {redactor.text(key): _safe_value(item, redactor) for key, item in value.items()}
        case list():
            return [_safe_value(item, redactor) for item in value]
        case _:
            return value


def _emit_response(payload: JsonValue, redactor: Redactor, *, err: bool = False) -> None:
    _emit(_safe_value(redactor.value(payload), redactor), err=err)


def _uncertain(request: Request, redactor: Redactor, error: str) -> NoReturn:
    payload: dict[str, JsonValue] = {"error": error}
    if request.idempotency_key is not None:
        payload["outcome_uncertain"] = True
        payload["idempotency_key"] = request.idempotency_key
    _emit_response(payload, redactor, err=True)
    raise typer.Exit(3)


def perform_request(ctx: typer.Context, request: Request) -> None:
    options = client_options(ctx)
    settings = load_client_settings(options.config)
    host = settings.host
    if host in {"0.0.0.0", "::"}:
        host = "127.0.0.1" if host == "0.0.0.0" else "::1"
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    url = options.url if options.url is not None else f"http://{host}:{settings.port}"
    try:
        parsed = urlsplit(url)
        valid = (
            parsed.scheme in {"http", "https"}
            and bool(parsed.hostname)
            and parsed.username is None
            and parsed.password is None
            and "?" not in url
            and "#" not in url
            and (parsed.port is None or 0 < parsed.port <= 65535)
            and not any(char.isspace() or ord(char) < 32 for char in url)
        )
    except ValueError:
        valid = False
    if not valid:
        fail("invalid_url")

    token = None
    if options.token_env is not None and not re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_]*", options.token_env
    ):
        fail("invalid_token_env")
    if not request.public:
        if options.token_env is not None:
            token = os.environ.get(options.token_env)
            if not token:
                fail("token_env_unset")
        else:
            token = os.environ.get(settings.operator_token_env) or os.environ.get(
                settings.reader_token_env
            )
        if token and not re.fullmatch(r"[\x21-\x7e]+", token):
            fail("invalid_token")
    explicit_secret = os.environ.get(options.token_env) if options.token_env else None
    redactor = Redactor((*settings.secrets(), *((explicit_secret,) if explicit_secret else ())))
    try:
        response = anyio.run(_send, url, token, request)
    except httpx.InvalidURL:
        fail("invalid_url")
    except (httpx.RequestError, TimeoutError, KeyboardInterrupt):
        _uncertain(request, redactor, "transport_error")

    try:
        body = JSON_OBJECT.validate_json(response.content)
    except ValidationError:
        if response.is_success or request.idempotency_key is not None:
            _uncertain(request, redactor, "invalid_response")
        fail("http_error", status=response.status_code, exit_code=1)
    if not response.is_success:
        error = body.get("error")
        payload: dict[str, JsonValue] = {
            "error": error
            if isinstance(error, str) and re.fullmatch(r"[a-z0-9_]{1,96}", error)
            else "http_error",
            "status": response.status_code,
        }
        for name in ("request_id", "status"):
            if isinstance(body.get(name), str):
                payload["request_id" if name == "request_id" else "health_status"] = body[name]
        if request.idempotency_key is not None and (
            response.status_code >= 500 or response.is_redirect
        ):
            payload.update(outcome_uncertain=True, idempotency_key=request.idempotency_key)
        _emit_response(payload, redactor, err=True)
        raise typer.Exit(1)

    if request.path == f"{API}/inference":
        if not (
            isinstance(body.get("request_id"), str)
            and isinstance(body.get("replayed"), bool)
            and body.get("state")
            in (
                "admitted",
                "dispatched",
                "completed",
                "failed",
                "ambiguous",
                "resolved",
            )
        ):
            _uncertain(request, redactor, "invalid_response")
        _emit_response(body, redactor)
        if body["state"] != "completed":
            raise typer.Exit(1)
    elif request.idempotency_key is not None:
        if not (
            isinstance(body.get("adjustment_id"), str) and isinstance(body.get("replayed"), bool)
        ):
            _uncertain(request, redactor, "invalid_response")
        body["idempotency_key"] = request.idempotency_key
        _emit_response(body, redactor)
    else:
        _emit_response(body, redactor)
