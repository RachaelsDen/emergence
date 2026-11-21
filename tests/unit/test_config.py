"""Configuration protects policy choices and secret boundaries."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from emergence.config import (
    ConfigurationError,
    ProviderConfig,
    QuotaPolicy,
    Settings,
    load_settings,
)


@pytest.mark.parametrize("provider", ["zai", "minimax", "openai", "llamacpp"])
def test_local_policy_survives_startup(provider: str) -> None:
    settings = Settings.model_validate(
        {
            "providers": {
                provider: {
                    "kind": provider,
                    "base_url": "http://127.0.0.1:9999/v1",
                    "quota_policy": "local_limits_only",
                }
            }
        }
    )
    assert settings.providers[provider].quota_policy is QuotaPolicy.LOCAL_LIMITS_ONLY


@pytest.mark.parametrize("policy", [None, "unlimited"])
def test_missing_or_unknown_policy_rejected(policy: str | None) -> None:
    with pytest.raises(ValidationError):
        ProviderConfig.model_validate(
            {"kind": "zai", "base_url": "https://invalid.test", "quota_policy": policy}
        )


def test_invalid_config_never_exposes_input(tmp_path: Path) -> None:
    config = tmp_path / "invalid.yaml"
    config.write_text("password: SENTINEL_SECRET\n", encoding="utf-8")
    with pytest.raises(ConfigurationError) as captured:
        load_settings(config)
    assert str(captured.value) == "configuration_invalid"


def test_invalid_timezone_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(accounting_timezone="Not/A_Real_Zone")


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://user:password@fixture.invalid",
        "https://fixture.invalid?key=secret",
        "file:///tmp/key",
    ],
)
def test_credentials_and_non_http_endpoints_cannot_enter_configuration(endpoint: str) -> None:
    with pytest.raises(ValidationError):
        ProviderConfig.model_validate(
            {"kind": "zai", "base_url": endpoint, "quota_policy": "local_limits_only"}
        )


@pytest.mark.parametrize("opening", ["0", "99.99", "150"])
def test_monthly_opening_changes_require_audited_adjustments(opening: str) -> None:
    with pytest.raises(ValidationError):
        ProviderConfig.model_validate(
            {
                "kind": "openai",
                "base_url": "https://fixture.invalid",
                "quota_policy": "local_limits_only",
                "monthly_budget": opening,
            }
        )
