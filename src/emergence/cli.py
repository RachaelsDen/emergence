"""Emergence process entry point; operator commands use HTTP."""

import re
import sys
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Final
from uuid import uuid4

import typer
import uvicorn

from emergence.cli_http import (
    API,
    ClientOptions,
    Request,
    client_options,
    fail,
    load_client_settings,
    perform_request,
)

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)
budgets = typer.Typer(no_args_is_help=True)
inference = typer.Typer(no_args_is_help=True)
app.add_typer(budgets, name="budgets")
app.add_typer(inference, name="inference")

PROMPT_LIMIT: Final = 1024 * 1024

ProviderOption = Annotated[str | None, typer.Option("--provider")]
RequestID = Annotated[str, typer.Argument(help="Request ID returned by the service.")]


def _serve(config: Path | None) -> None:
    from emergence.api.app import create_app

    settings = load_client_settings(config)
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        access_log=False,
        log_config=None,
        limit_concurrency=64,
    )


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
    ctx.obj = ClientOptions(config, url, token_env)


@app.command()
def serve(ctx: typer.Context, config: Annotated[Path | None, typer.Option()] = None) -> None:
    """Run one core process with a loopback listener by default."""
    _serve(config or client_options(ctx).config)


@app.command()
def health(ctx: typer.Context, live: Annotated[bool, typer.Option("--live")] = False) -> None:
    """Check readiness by default; use --live for liveness."""
    perform_request(ctx, Request("/health/live" if live else "/health/ready", public=True))


@app.command()
def providers(ctx: typer.Context) -> None:
    """List configured providers."""
    perform_request(ctx, Request(f"{API}/providers"))


@app.command()
def models(ctx: typer.Context) -> None:
    """List configured model aliases."""
    perform_request(ctx, Request(f"{API}/models"))


@app.command()
def resources(ctx: typer.Context) -> None:
    """Show provider resource observations."""
    perform_request(ctx, Request(f"{API}/resources"))


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
    perform_request(
        ctx,
        Request(
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
    perform_request(ctx, Request(f"{API}/budgets", params=(("provider", provider),)))


class _AdjustmentKind(StrEnum):
    GRANT = "grant"
    REDUCTION = "reduction"
    CORRECTION = "correction"


def _decimal_text(amount: str) -> str:
    try:
        value = Decimal(amount)
    except InvalidOperation:
        fail("invalid_amount")
    if not value.is_finite():
        fail("invalid_amount")
    return str(value)


@budgets.command()
def adjust(
    ctx: typer.Context,
    provider: Annotated[str, typer.Option()],
    amount: Annotated[str, typer.Option(help="Exact decimal USD amount.")],
    kind: Annotated[_AdjustmentKind, typer.Option()],
    reason: Annotated[str, typer.Option()],
    adjustment_id: Annotated[
        str | None, typer.Option(help="Stable idempotency key; generated if omitted.")
    ] = None,
) -> None:
    """Append an audited virtual budget adjustment."""
    key = str(uuid4()) if adjustment_id is None else adjustment_id
    if not re.fullmatch(r"[\x21-\x7e]{1,128}", key):
        fail("invalid_idempotency_key")
    perform_request(
        ctx,
        Request(
            f"{API}/budgets/adjustments",
            "POST",
            idempotency_key=key,
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
        fail("prompt_unreadable")
    if len(data) > PROMPT_LIMIT:
        fail("prompt_too_large")
    if not prompt:
        fail("prompt_empty")
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
        fail("invalid_idempotency_key")
    perform_request(
        ctx,
        Request(
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
        fail("invalid_request_id")
    return f"{API}/inference/{request_id}"


@inference.command()
def status(ctx: typer.Context, request_id: RequestID) -> None:
    """Fetch an inference request by its service-issued ID."""
    perform_request(ctx, Request(_request_path(request_id)))


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
        fail("choose_amount_or_no_charge")
    perform_request(
        ctx,
        Request(
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
