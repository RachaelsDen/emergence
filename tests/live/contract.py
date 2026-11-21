"""Parse an operator's live selection before opening a network client."""

from collections.abc import Mapping
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Annotated

from pydantic import Field, TypeAdapter, ValidationError

from emergence.config import ProviderKind, SafeModel, Settings


class LiveConfigurationError(Exception):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class LiveOptions(SafeModel):
    config: Path
    providers: Annotated[tuple[str, ...], Field(min_length=1, max_length=4)]
    models: dict[str, str]
    max_output_tokens: Annotated[int, Field(gt=0, le=32)]
    request_cap: Annotated[
        Decimal, Field(gt=0, le=1, allow_inf_nan=False, max_digits=15, decimal_places=12)
    ]
    total_cap: Annotated[
        Decimal, Field(gt=0, le=4, allow_inf_nan=False, max_digits=15, decimal_places=12)
    ]

    @classmethod
    def from_environment(cls, environment: Mapping[str, str]) -> "LiveOptions":
        if environment.get("EMERGENCE_LIVE") != "1":
            raise LiveConfigurationError("live_not_enabled")
        try:
            result = cls(
                config=Path(environment["EMERGENCE_CONFIG"]),
                providers=tuple(
                    part.strip() for part in environment["EMERGENCE_LIVE_PROVIDERS"].split(",")
                ),
                models=TypeAdapter(dict[str, str]).validate_json(
                    environment.get("EMERGENCE_LIVE_MODELS", "{}")
                ),
                max_output_tokens=TypeAdapter(int).validate_python(
                    environment["EMERGENCE_LIVE_MAX_OUTPUT_TOKENS"]
                ),
                request_cap=TypeAdapter(Decimal).validate_python(
                    environment["EMERGENCE_LIVE_MAX_REQUEST_USD"]
                ),
                total_cap=TypeAdapter(Decimal).validate_python(
                    environment["EMERGENCE_LIVE_TOTAL_CAP_USD"]
                ),
            )
        except (KeyError, ValidationError):
            raise LiveConfigurationError("live_configuration_incomplete") from None
        if not result.config.is_absolute():
            raise LiveConfigurationError("live_config_requires_absolute_path")
        if len(set(result.providers)) != len(result.providers):
            raise LiveConfigurationError("duplicate_live_provider")
        return result

    def bounded_settings(self, settings: Settings, path: Path) -> Settings:
        """Isolate state and validate combined ceilings before any call."""
        if any(name not in settings.providers for name in self.providers):
            raise LiveConfigurationError("live_provider_not_configured")
        selected = {name: settings.providers[name] for name in self.providers}
        if any(not config.enabled for config in selected.values()):
            raise LiveConfigurationError("live_provider_disabled")
        cloud_count = sum(config.kind is not ProviderKind.LLAMACPP for config in selected.values())
        with localcontext(prec=64):
            if self.request_cap * cloud_count > self.total_cap:
                raise LiveConfigurationError("live_total_cap_insufficient")
        return settings.model_copy(
            update={
                "providers": selected,
                "prices": tuple(price for price in settings.prices if price.provider in selected),
                "database_path": path / "smoke.sqlite3",
                "spool_path": path / "spool",
                "archive_path": None,
                "archive_mount_path": None,
                "enforce_budgets": True,
            }
        )

    def model(self, settings: Settings, provider: str) -> str:
        configured = settings.providers[provider].models
        alias = self.models.get(provider)
        if alias is None:
            if len(configured) != 1:
                raise LiveConfigurationError("live_model_selection_required")
            alias = next(iter(configured))
        if alias not in configured:
            raise LiveConfigurationError("live_model_not_configured")
        return alias
