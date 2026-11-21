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


@pytest.mark.parametrize("contents", ["password: SENTINEL_SECRET\n", "", "[]", "[invalid"])
def test_invalid_config_never_exposes_input(tmp_path: Path, contents: str) -> None:
    config = tmp_path / "invalid.yaml"
    config.write_text(contents, encoding="utf-8")
    with pytest.raises(ConfigurationError) as captured:
        load_settings(config)
    assert str(captured.value) == "configuration_invalid"


@pytest.mark.parametrize("explicit", [True, False])
def test_yaml_environment_overrides_retain_siblings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, explicit: bool
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "host: localhost\nport: 8001\n"
        "providers:\n"
        "  local:\n"
        "    kind: llamacpp\n"
        "    quota_policy: local_limits_only\n"
        "    base_url: http://127.0.0.1:9999/v1\n"
        "    concurrency: 2\n"
        "    models:\n"
        "      small:\n"
        "        id: fixture-model\n"
        "        max_output_tokens: 128\n",
        encoding="utf-8",
    )
    alternate = tmp_path / "alternate.yaml"
    alternate.write_text("password: SENTINEL_SECRET\n", encoding="utf-8")
    monkeypatch.setenv("EMERGENCE_CONFIG", str(alternate if explicit else config))
    monkeypatch.setenv("EMERGENCE_HOST", "127.0.0.2")
    monkeypatch.setenv("EMERGENCE_PORT", "8123")
    monkeypatch.setenv("EMERGENCE_PROVIDERS__LOCAL__CONCURRENCY", "3")
    monkeypatch.setenv("EMERGENCE_PROVIDERS__LOCAL__MODELS__SMALL__MAX_OUTPUT_TOKENS", "64")

    settings = load_settings(config if explicit else None)

    assert (settings.host, settings.port) == ("127.0.0.2", 8123)
    provider = settings.providers["local"]
    assert provider.concurrency == 3
    assert provider.kind.value == "llamacpp"
    assert provider.quota_policy is QuotaPolicy.LOCAL_LIMITS_ONLY
    assert provider.base_url == "http://127.0.0.1:9999/v1"
    assert provider.models["small"].id == "fixture-model"
    assert provider.models["small"].max_output_tokens == 64


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("PORT", "SENTINEL_SECRET"),
        ("PROVIDERS__LOCAL__BASE_URL", "https://user:SENTINEL_SECRET@fixture.invalid"),
        ("PROVIDERS__LOCAL__MONTHLY_BUDGET", "99"),
        ("PROVIDERS", '{"credential": "SENTINEL_SECRET"'),
    ],
)
def test_invalid_environment_override_never_exposes_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "port: 8001\n"
        "providers:\n"
        "  local:\n"
        "    kind: llamacpp\n"
        "    quota_policy: local_limits_only\n"
        "    base_url: http://127.0.0.1:9999/v1\n"
        "    monthly_budget: 100\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(f"EMERGENCE_{name}", value)

    with pytest.raises(ConfigurationError) as captured:
        load_settings(config)

    assert str(captured.value) == "configuration_invalid"
    assert captured.value.__cause__ is None
    assert captured.value.__suppress_context__


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
