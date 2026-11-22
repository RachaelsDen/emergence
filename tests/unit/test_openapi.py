"""Keep the shipped machine-consumed API schema synchronized with the service."""

from pathlib import Path
from typing import Final

from pydantic import JsonValue, TypeAdapter

from emergence.api.app import create_app
from emergence.config import Settings

PROJECT: Final = Path(__file__).resolve().parents[2]


def test_shipped_openapi_matches_the_running_contract_with_optional_metrics() -> None:
    settings = Settings(
        metrics_enabled=True,
        providers={},
        archive_path=None,
        archive_mount_path=None,
        operator_token_env="SCHEMA_UNUSED_OPERATOR",
        reader_token_env="SCHEMA_UNUSED_READER",
    )
    shipped = TypeAdapter(dict[str, JsonValue]).validate_json(
        (PROJECT / "docs/openapi.json").read_bytes()
    )
    assert shipped == create_app(settings).openapi()
