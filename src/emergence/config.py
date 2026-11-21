"""Non-secret configuration; credential values live exclusively in the environment."""

import os
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Final, Self
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml
from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    model_validator,
)
from pydantic_core import PydanticCustomError
from pydantic_settings import BaseSettings, SettingsConfigDict

type PositiveInt = Annotated[int, Field(gt=0)]
type EnvName = Annotated[str, Field(pattern=r"^[A-Z][A-Z0-9_]*$")]
MONTHLY_BUDGET_USD: Final = Decimal("100")


class ConfigurationError(Exception):
    """A configuration failure whose input must not be serialized."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class ProviderKind(StrEnum):
    ZAI = "zai"
    MINIMAX = "minimax"
    OPENAI = "openai"
    LLAMACPP = "llamacpp"


class QuotaPolicy(StrEnum):
    REQUIRE_FRESH = "require_fresh"
    LOCAL_LIMITS_ONLY = "local_limits_only"


class SafeModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)


class ModelConfig(SafeModel):
    id: str
    max_input_tokens: PositiveInt = 4096
    max_output_tokens: PositiveInt = 256
    output_bound_verified: bool = False
    bound_evidence: str | None = None
    input_bound_verified: bool = False
    input_bound_evidence: str | None = None
    input_overhead_tokens: Annotated[int, Field(ge=0)] = 512
    reasoning_in_output_bound_verified: bool = False
    reasoning_bound_evidence: str | None = None


class ProviderConfig(SafeModel):
    kind: ProviderKind
    enabled: bool = True
    quota_policy: QuotaPolicy
    base_url: str
    models: dict[str, ModelConfig] = Field(default_factory=dict)
    credential_env: EnvName | None = None
    reporting_credential_env: EnvName | None = None
    quota_url: str | None = None
    reporting_base_url: str = "https://api.openai.com/v1"
    reporting_key_expires_at: AwareDatetime | None = None
    reporting_lookback_days: Annotated[int, Field(gt=0, le=366)] = 31
    organization: str | None = None
    project: str | None = None
    concurrency: PositiveInt = 1
    timeout_seconds: Annotated[float, Field(gt=0, le=300)] = 30
    response_limit_bytes: PositiveInt = 4 * 1024 * 1024
    poll_interval_seconds: Annotated[float, Field(gt=0)] = 300
    stale_after_seconds: PositiveInt = 900
    monthly_budget: Annotated[
        Decimal, Field(ge=MONTHLY_BUDGET_USD, le=MONTHLY_BUDGET_USD, allow_inf_nan=False)
    ] = MONTHLY_BUDGET_USD
    safety_limit: Annotated[Decimal, Field(gt=0, allow_inf_nan=False)] = Decimal("95")
    safety_limit_confirmed: bool = False

    @model_validator(mode="after")
    def validate_endpoints(self) -> Self:
        """Credentials belong in secret sources, never in persisted endpoint URLs."""
        for endpoint in (self.base_url, self.quota_url, self.reporting_base_url):
            if endpoint is None:
                continue
            try:
                parsed = urlsplit(endpoint)
                valid = (
                    parsed.scheme in {"http", "https"}
                    and bool(parsed.hostname)
                    and parsed.username is None
                    and parsed.password is None
                    and not parsed.query
                    and not parsed.fragment
                    and (parsed.port is None or 0 < parsed.port <= 65535)
                )
            except ValueError:
                valid = False
            if not valid:
                raise PydanticCustomError("endpoint", "endpoint must be a credential-free HTTP URL")
        return self


class PriceConfig(SafeModel):
    """Operator-sourced price versions; amounts are USD per million tokens."""

    version: Annotated[str, Field(min_length=1, max_length=128)]
    provider: str
    model: str
    effective_at: AwareDatetime
    input_per_million: Annotated[
        Decimal, Field(ge=0, le=1000000, max_digits=24, decimal_places=12, allow_inf_nan=False)
    ]
    output_per_million: Annotated[
        Decimal, Field(ge=0, le=1000000, max_digits=24, decimal_places=12, allow_inf_nan=False)
    ]
    cached_input_per_million: Annotated[
        Decimal | None,
        Field(ge=0, le=1000000, max_digits=24, decimal_places=12, allow_inf_nan=False),
    ] = None
    source: Annotated[str, Field(min_length=1)]

    @model_validator(mode="after")
    def validate_cache_discount(self) -> Self:
        if (
            self.cached_input_per_million is not None
            and self.cached_input_per_million > self.input_per_million
        ):
            raise PydanticCustomError("cache_price", "cache surcharges are not a bounded discount")
        return self


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="EMERGENCE_",
        env_nested_delimiter="__",
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
    )

    host: str = "127.0.0.1"
    port: Annotated[int, Field(ge=1, le=65535)] = 8000
    database_path: Path = Path("var/core.sqlite3")
    spool_path: Path = Path("var/spool")
    archive_path: Path | None = None
    archive_mount_path: Path | None = None
    accounting_timezone: str = "UTC"
    operator_token_env: EnvName = "EMERGENCE_OPERATOR_TOKEN"
    reader_token_env: EnvName = "EMERGENCE_READER_TOKEN"
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    prices: tuple[PriceConfig, ...] = ()
    metrics_enabled: bool = False
    spool_limit_bytes: PositiveInt = 256 * 1024 * 1024
    disk_headroom_bytes: PositiveInt = 64 * 1024 * 1024
    request_limit_bytes: PositiveInt = 1024 * 1024
    request_body_timeout_seconds: Annotated[float, Field(gt=0, le=300)] = 10
    requests_per_minute: PositiveInt = 30
    poll_concurrency: PositiveInt = 4
    enforce_budgets: bool = True

    @model_validator(mode="after")
    def validate_timezone(self) -> Self:
        try:
            ZoneInfo(self.accounting_timezone)
        except ZoneInfoNotFoundError:
            raise PydanticCustomError("timezone", "unknown accounting timezone") from None
        return self

    def secret(self, name: str | None) -> SecretStr | None:
        value = os.environ.get(name) if name is not None else None
        return SecretStr(value) if value else None

    def secrets(self) -> tuple[str, ...]:
        names = {self.operator_token_env, self.reader_token_env}
        for provider in self.providers.values():
            if provider.credential_env:
                names.add(provider.credential_env)
            if provider.reporting_credential_env:
                names.add(provider.reporting_credential_env)
        return tuple(os.environ[name] for name in names if os.environ.get(name))


def load_settings(path: Path | None = None) -> Settings:
    """Parse a non-secret YAML file and environment overrides without leaking inputs."""
    selected = path or (
        Path(os.environ["EMERGENCE_CONFIG"]) if os.environ.get("EMERGENCE_CONFIG") else None
    )
    try:
        if selected is None:
            return Settings()
        parsed = yaml.safe_load(selected.read_text(encoding="utf-8"))
        return Settings.model_validate(parsed)
    except (OSError, yaml.YAMLError, ValidationError):
        raise ConfigurationError("configuration_invalid") from None
