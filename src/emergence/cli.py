"""Emergence process entry point; operator commands use HTTP."""

import json
import os
import re
import sys
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Final, Literal, NoReturn
from urllib.parse import urlsplit
from uuid import uuid4

import anyio
import httpx
import typer
import uvicorn
from pydantic import JsonValue, TypeAdapter, ValidationError

from emergence.config import ConfigurationError, Settings, load_settings
from emergence.telemetry.logging import Redactor

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)
budgets = typer.Typer(no_args_is_help=True)
inference = typer.Typer(no_args_is_help=True)
app.add_typer(budgets, name="budgets")
app.add_typer(inference, name="inference")

API: Final = "/api/v1"
DEADLINE: Final = 60.0
PROMPT_LIMIT: Final = 1024 * 1024
JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])

ProviderOption = Annotated[str | None, typer.Option("--provider")]
RequestID = Annotated[str, typer.Argument(help="Request ID returned by the service.")]


def _emit(payload: JsonValue, *, err: bool = False) -> None:
    typer.echo(json.dumps(payload), err=err)


def _fail(code: str, *, status: int = 0, exit_code: int = 2) -> NoReturn:
    payload: dict[str, JsonValue] = {"error": code}
    if status:
        payload["status"] = status
    _emit(payload, err=True)
    raise typer.Exit(exit_code)


def _settings(config: Path | None) -> Settings:
    try:
        return load_settings(config)
    except ConfigurationError as error:
        _fail(error.code)


def _serve(config: Path | None) -> None:
    from emergence.api.app import create_app

    settings = _settings(config)
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        access_log=False,
        log_config=None,
        limit_concurrency=64,
    )


@dataclass(frozen=True, slots=True)
class _ClientOptions:
    config: Path | None
    url: str | None
    token_env: str | None


@app.callback(invoke_without_command=True)
def root(
    ctx: typer.Context,
    config: Annotated[Path | None, typer.Option(help="Non-secret YAML configuration.")] = None,
    url: Annotated[str | None, typer.Option(envvar="EMERGENCE_URL", help="Emergence URL.")] = None,
    token_env: Annotated[
        str | None, typer.Option(help="Name of the token environment variable.")
    ] = (None),
) -> None:
    """Run Emergence or operate a running service over HTTP."""
    if ctx.invoked_subcommand is None:
        if config is None:
            typer.echo(ctx.get_help())
            return
        _serve(config)
        return
    ctx.obj = _ClientOptions(config, url, token_env)


@app.command()
def serve(ctx: typer.Context, config: Annotated[Path | None, typer.Option()] = None) -> None:
    """Run one core process with a loopback listener by default."""
    _serve(config or _client_options(ctx).config)


def _client_options(ctx: typer.Context) -> _ClientOptions:
    options = ctx.find_object(_ClientOptions)
    assert options is not None  # Set by the root callback for every subcommand.
    return options


@dataclass(frozen=True, slots=True)
class _Request:
    path: str
    method: Literal["GET", "POST"] = "GET"
    params: tuple[tuple[str, str | int | None], ...] = ()
    body: JsonValue = None
    public: bool = False
    idempotency_key: str | None = None


async def _send(url: str, token: str | None, request: _Request) -> httpx.Response:
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


def _uncertain(request: _Request, redactor: Redactor, error: str) -> NoReturn:
    payload: dict[str, JsonValue] = {"error": error}
    if request.idempotency_key is not None:
        payload["outcome_uncertain"] = True
        payload["idempotency_key"] = request.idempotency_key
    _emit_response(payload, redactor, err=True)
    raise typer.Exit(3)


def _request(ctx: typer.Context, request: _Request) -> None:
    options = _client_options(ctx)
    settings = _settings(options.config)
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
        _fail("invalid_url")

    token = None
    if options.token_env is not None and not re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_]*", options.token_env
    ):
        _fail("invalid_token_env")
    if not request.public:
        if options.token_env is not None:
            token = os.environ.get(options.token_env)
            if not token:
                _fail("token_env_unset")
        else:
            token = os.environ.get(settings.operator_token_env) or os.environ.get(
                settings.reader_token_env
            )
        if token and not re.fullmatch(r"[\x21-\x7e]+", token):
            _fail("invalid_token")
    explicit_secret = os.environ.get(options.token_env) if options.token_env else None
    redactor = Redactor((*settings.secrets(), *((explicit_secret,) if explicit_secret else ())))
    try:
        response = anyio.run(_send, url, token, request)
    except httpx.InvalidURL:
        _fail("invalid_url")
    except (httpx.RequestError, TimeoutError, KeyboardInterrupt):
        _uncertain(request, redactor, "transport_error")

    try:
        body = JSON_OBJECT.validate_json(response.content)
    except ValidationError:
        if response.is_success or request.idempotency_key is not None:
            _uncertain(request, redactor, "invalid_response")
        _fail("http_error", status=response.status_code, exit_code=1)
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

    if request.idempotency_key is not None:
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
    else:
        _emit_response(body, redactor)


@app.command()
def health(ctx: typer.Context, live: Annotated[bool, typer.Option("--live")] = False) -> None:
    """Check readiness by default; use --live for liveness."""
    _request(ctx, _Request("/health/live" if live else "/health/ready", public=True))


@app.command()
def providers(ctx: typer.Context) -> None:
    """List configured providers."""
    _request(ctx, _Request(f"{API}/providers"))


@app.command()
def models(ctx: typer.Context) -> None:
    """List configured model aliases."""
    _request(ctx, _Request(f"{API}/models"))


@app.command()
def resources(ctx: typer.Context) -> None:
    """Show provider resource observations."""
    _request(ctx, _Request(f"{API}/resources"))


@app.command()
def usage(
    ctx: typer.Context,
    provider: ProviderOption = None,
    model: Annotated[str | None, typer.Option()] = None,
    start: Annotated[str | None, typer.Option(help="Start of range (ISO 8601).")] = None,
    end: Annotated[str | None, typer.Option(help="End of range (ISO 8601).")] = None,
    limit: Annotated[int | None, typer.Option(min=1)] = None,
    offset: Annotated[int | None, typer.Option(min=0)] = None,
) -> None:
    """Show a page of usage history."""
    _request(
        ctx,
        _Request(
            f"{API}/usage",
            params=(
                ("provider", provider),
                ("model", model),
                ("start", start),
                ("end", end),
                ("limit", limit),
                ("offset", offset),
            ),
        ),
    )


@budgets.command()
def show(ctx: typer.Context, provider: ProviderOption = None) -> None:
    """Show virtual budget balances."""
    _request(ctx, _Request(f"{API}/budgets", params=(("provider", provider),)))


class _AdjustmentKind(StrEnum):
    GRANT = "grant"
    REDUCTION = "reduction"
    CORRECTION = "correction"


def _decimal_text(amount: str) -> str:
    try:
        value = Decimal(amount)
    except InvalidOperation:
        _fail("invalid_amount")
    if not value.is_finite():
        _fail("invalid_amount")
    return str(value)


@budgets.command()
def adjust(
    ctx: typer.Context,
    provider: Annotated[str, typer.Option()],
    amount: Annotated[str, typer.Option(help="Exact decimal USD amount.")],
    kind: Annotated[_AdjustmentKind, typer.Option()],
    reason: Annotated[str, typer.Option()],
) -> None:
    """Append an audited virtual budget adjustment."""
    _request(
        ctx,
        _Request(
            f"{API}/budgets/adjustments",
            "POST",
            body={
                "provider": provider,
                "amount": _decimal_text(amount),
                "kind": kind.value,
                "reason": reason,
            },
        ),
    )


def _read_prompt(prompt_file: Path | None) -> str:
    try:
        if prompt_file is None:
            data = sys.stdin.buffer.read(PROMPT_LIMIT + 1)
        else:
            with prompt_file.open("rb") as stream:
                data = stream.read(PROMPT_LIMIT + 1)
        prompt = data.decode("utf-8")
    except (OSError, UnicodeError):
        _fail("prompt_unreadable")
    if len(data) > PROMPT_LIMIT:
        _fail("prompt_too_large")
    if not prompt:
        _fail("prompt_empty")
    return prompt


@inference.command("test")
def inference_test(
    ctx: typer.Context,
    provider: Annotated[str, typer.Option(help="Configured provider name.")],
    model: Annotated[str, typer.Option(help="Configured model alias.")],
    max_output_tokens: Annotated[int, typer.Option(min=1, max=131072)] = 128,
    prompt_file: Annotated[Path | None, typer.Option(help="UTF-8 file; defaults to stdin.")] = None,
    request_id: Annotated[
        str | None, typer.Option(help="Idempotency key; generated if omitted.")
    ] = None,
) -> None:
    """Request bounded inference without retrying automatically."""
    key = str(uuid4()) if request_id is None else request_id
    if not re.fullmatch(r"[\x21-\x7e]{1,128}", key):
        _fail("invalid_idempotency_key")
    _request(
        ctx,
        _Request(
            f"{API}/inference",
            "POST",
            idempotency_key=key,
            body={
                "provider": provider,
                "model": model,
                "max_output_tokens": max_output_tokens,
                "messages": [{"role": "user", "content": _read_prompt(prompt_file)}],
            },
        ),
    )


def _request_path(request_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", request_id):
        _fail("invalid_request_id")
    return f"{API}/inference/{request_id}"


@inference.command()
def status(ctx: typer.Context, request_id: RequestID) -> None:
    """Fetch an inference request by its service-issued ID."""
    _request(ctx, _Request(_request_path(request_id)))


@inference.command()
def resolve(
    ctx: typer.Context,
    request_id: RequestID,
    reason: Annotated[str, typer.Option()],
    amount: Annotated[
        str | None, typer.Option(help="Operator-estimated charge (decimal USD).")
    ] = None,
    no_charge: Annotated[
        bool, typer.Option("--no-charge", help="Confirm no charge occurred.")
    ] = False,
) -> None:
    """Append an audited resolution to an unresolved inference request."""
    if (amount is not None) == no_charge:
        _fail("choose_amount_or_no_charge")
    _request(
        ctx,
        _Request(
            f"{_request_path(request_id)}/resolution",
            "POST",
            body={
                "amount": _decimal_text(amount) if amount is not None else None,
                "reason": reason,
            },
        ),
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
