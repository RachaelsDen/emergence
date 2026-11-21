"""Secrets are scrubbed before logs or archive serialization."""

import logging

from emergence.telemetry.logging import JsonFormatter, Redactor


def test_nested_credentials_scrubbed_without_losing_conversation() -> None:
    redactor = Redactor(("SENTINEL",))
    result = redactor.value(
        {"messages": ["hello SENTINEL"], "Authorization": "Bearer unknown", "output": "answer"}
    )
    assert result == {
        "messages": ["hello [REDACTED]"],
        "Authorization": "[REDACTED]",
        "output": "answer",
    }


def test_log_formatter_does_not_emit_secret_or_exception() -> None:
    record = logging.LogRecord("test", logging.ERROR, "", 0, "key SENTINEL", (), None)
    formatted = JsonFormatter(Redactor(("SENTINEL",))).format(record)
    assert "SENTINEL" not in formatted
    assert "[REDACTED]" in formatted


def test_redaction_does_not_expand_its_own_marker_on_archive_reentry() -> None:
    redactor = Redactor(("A", "E"))
    once = redactor.text("A E")
    assert once == "[REDACTED] [REDACTED]"
    assert redactor.text(once) == once


def test_longer_secrets_starting_with_the_marker_are_not_shadowed() -> None:
    secret = "[REDACTED]synthetic"
    redactor = Redactor(("[RE", secret, f"{secret}-longer"))
    once = redactor.text(f"prefix {secret}-longer {secret} [REDACTED]")
    assert once == "prefix [REDACTED] [REDACTED] [REDACTED]"
    assert redactor.text(once) == once
