"""Redact credential material before serialization or logging."""

import json
import logging
import re
from collections.abc import Mapping
from typing import Final

from pydantic import JsonValue

SECRET_FIELD: Final = re.compile(
    r"(?i)(authorization|api[_-]?key|password|secret|access[_-]?token|credential)"
)
TOKEN_TEXT: Final = re.compile(
    r"(?i)(bearer\s+)[^\s\"',;]+|"
    r"((?:api[_-]?key|password|secret|access[_-]?token)\s*[=:]\s*)[^\s\"',;]+|"
    r"\bsk-[A-Za-z0-9_-]{8,}"
)


class Redactor:
    """Scrub known secrets and common credential fields recursively."""

    def __init__(self, secrets: tuple[str, ...] = ()) -> None:
        values = tuple(sorted((secret for secret in secrets if secret), key=len, reverse=True))
        self._known = (
            re.compile(r"\[REDACTED\]|" + "|".join(re.escape(secret) for secret in values))
            if values
            else None
        )
        self.expansion_factor = max(
            (len("[REDACTED]") + len(secret.encode("utf-8")) - 1) // len(secret.encode("utf-8"))
            for secret in (*values, "[REDACTED]")
        )

    def text(self, value: str) -> str:
        if self._known is not None:
            value = self._known.sub("[REDACTED]", value)
        return TOKEN_TEXT.sub("[REDACTED]", value)

    def value(self, value: JsonValue) -> JsonValue:
        match value:
            case str():
                return self.text(value)
            case list():
                return [self.value(item) for item in value]
            case dict():
                return {
                    key: "[REDACTED]" if SECRET_FIELD.search(key) else self.value(item)
                    for key, item in value.items()
                }
            case _:
                return value


class JsonFormatter(logging.Formatter):
    def __init__(self, redactor: Redactor) -> None:
        super().__init__()
        self.redactor = redactor

    def format(self, record: logging.LogRecord) -> str:
        # Do not serialize exception objects, locals, or request/response bodies.
        entry: Mapping[str, str] = {
            "level": record.levelname,
            "logger": record.name,
            "message": self.redactor.text(record.getMessage()),
        }
        return json.dumps(entry)


def configure_logging(redactor: Redactor) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter(redactor))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
