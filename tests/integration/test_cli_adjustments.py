"""Durable financial operation identities through actual CLI child processes."""

import socket
from decimal import Decimal
from uuid import UUID

import pytest
from test_cli_service import JSON_OBJECT, OPERATOR, Service, success
from test_cli_service import launch_config as launch_config
from test_cli_service import service as service
from test_cli_service import upstream as upstream


def test_operator_adjusts_exact_decimal_budget(service: Service) -> None:
    args = (
        "budgets",
        "adjust",
        "--provider",
        "offline",
        "--amount",
        "0.123456789012",
        "--kind",
        "grant",
        "--reason",
        "offline CLI grant",
        "--adjustment-id",
        "one-cli-grant",
    )
    adjusted = success(service.run(*args))
    replay = success(service.run(*args))
    assert adjusted["provider"] == "offline"
    assert isinstance(amount := adjusted["adjustments"], str)
    assert Decimal(amount) == Decimal("0.123456789012")
    assert replay["adjustments"] == amount
    assert adjusted["replayed"] is False and replay["replayed"] is True
    assert replay["adjustment_id"] == adjusted["adjustment_id"]
    assert replay["idempotency_key"] == "one-cli-grant"


@pytest.mark.parametrize(
    ("command", "key_option"),
    [
        (("inference", "test", "--provider", "offline", "--model", "alias"), "--request-id"),
        (
            (
                "budgets",
                "adjust",
                "--provider",
                "offline",
                "--amount",
                "1",
                "--kind",
                "grant",
                "--reason",
                "confirmed",
            ),
            "--adjustment-id",
        ),
    ],
)
def test_missing_reply_exposes_retry_key_without_echoing_credentials(
    launch_config: Service,
    command: tuple[str, ...],
    key_option: str,
) -> None:
    with socket.socket() as closed:
        # Hold an unused local port without listening, so no request can leave this machine.
        closed.bind(("127.0.0.1", 0))
        url = f"http://127.0.0.1:{closed.getsockname()[1]}"
        args = (
            "--url",
            url,
            *command,
        )
        missing = launch_config.run(*args, stdin="offline")
        redacted = launch_config.run(*args, key_option, OPERATOR, stdin="offline")

    for result in (missing, redacted):
        assert result.returncode == 3, (result.stdout, result.stderr)
        assert result.stdout == ""
        assert OPERATOR not in result.stderr
        assert JSON_OBJECT.validate_json(result.stderr)["outcome_uncertain"] is True
    key = JSON_OBJECT.validate_json(missing.stderr)["idempotency_key"]
    assert isinstance(key, str) and UUID(key).version == 4
    assert JSON_OBJECT.validate_json(redacted.stderr)["idempotency_key"] == "[REDACTED]"
