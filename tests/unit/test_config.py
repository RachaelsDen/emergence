"""Configuration protects policy choices and secret boundaries."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from emergence.config import (
    ConfigurationError,
    ModelConfig,
    PriceConfig,
    ProviderConfig,
    ProviderKind,
    QuotaPolicy,
    Settings,
    load_settings,
)
from emergence.inference.schemas import InferenceRequest, Message


@pytest.mark.parametrize("field", ["organization", "project"])
@pytest.mark.parametrize("unsafe", ["org-é", "org\ninvalid", " org", "org\x7f"])
def test_openai_scope_must_be_a_header_safe_identifier(field: str, unsafe: str) -> None:
    with pytest.raises(ValidationError, match="OpenAI scope must be visible ASCII"):
        ProviderConfig.model_validate(
            {
                "kind": ProviderKind.OPENAI,
                "base_url": "https://fixture.invalid/v1",
                "quota_policy": QuotaPolicy.LOCAL_LIMITS_ONLY,
                field: unsafe,
            }
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


@pytest.mark.parametrize(
    "contents",
    [
        b"port: 8001\n'SENTINEL_SECRET': first\nSENTINEL_SECRET: second\n",
        b"port: 8001\nport: 8002\n",
        (
            b"providers:\n  local:\n    kind: llamacpp\n    quota_policy: local_limits_only\n"
            b"    base_url: https://first.invalid\n    'base_url': https://second.invalid\n"
        ),
        (
            b"providers:\n  local:\n"
            b"    <<: {kind: llamacpp, quota_policy: local_limits_only, "
            b"base_url: https://first.invalid, base_url: https://second.invalid}\n"
        ),
        b"host: SENTINEL_SECRET\xff\n",
    ],
)
def test_duplicate_yaml_keys_and_invalid_utf8_are_sanitized(
    tmp_path: Path, contents: bytes
) -> None:
    config = tmp_path / "invalid.yaml"
    config.write_bytes(contents)
    with pytest.raises(ConfigurationError, match="^configuration_invalid$") as captured:
        load_settings(config)
    assert captured.value.__cause__ is None
    assert captured.value.__suppress_context__


def test_yaml_merge_still_allows_an_explicit_override(tmp_path: Path) -> None:
    config = tmp_path / "merge.yaml"
    config.write_text(
        "providers:\n"
        "  first: &defaults\n"
        "    kind: llamacpp\n"
        "    quota_policy: local_limits_only\n"
        "    base_url: https://first.invalid\n"
        "  second:\n"
        "    <<: *defaults\n"
        "    base_url: https://second.invalid\n",
        encoding="utf-8",
    )
    providers = load_settings(config).providers
    assert providers["first"].base_url == "https://first.invalid"
    assert providers["second"].base_url == "https://second.invalid"


@pytest.mark.parametrize("length", [0, 1, 128, 129])
def test_configured_model_alias_has_the_same_bounds_as_inference(length: int) -> None:
    alias = "m" * length
    provider = {
        "kind": "llamacpp",
        "quota_policy": "local_limits_only",
        "base_url": "https://fixture.invalid",
        "models": {alias: {"id": "upstream"}},
    }
    request = {
        "provider": "local",
        "model": alias,
        "messages": (Message(role="user", content="x"),),
    }
    if length in (1, 128):
        assert alias in ProviderConfig.model_validate(provider).models
        assert InferenceRequest.model_validate(request).model == alias
    else:
        with pytest.raises(ValidationError):
            ProviderConfig.model_validate(provider)
        with pytest.raises(ValidationError):
            InferenceRequest.model_validate(request)


@pytest.mark.parametrize("length", [0, 1, 128, 129])
def test_configured_provider_name_has_the_same_bounds_as_inference(length: int) -> None:
    name = "p" * length
    config = {
        "providers": {
            name: {
                "kind": "llamacpp",
                "quota_policy": "local_limits_only",
                "base_url": "https://fixture.invalid",
            }
        }
    }
    request = {"provider": name, "model": "m", "messages": (Message(role="user", content="x"),)}
    if length in (1, 128):
        assert name in Settings.model_validate(config).providers
        assert InferenceRequest.model_validate(request).provider == name
    else:
        with pytest.raises(ValidationError):
            Settings.model_validate(config)
        with pytest.raises(ValidationError):
            InferenceRequest.model_validate(request)


@pytest.mark.parametrize("limit", [2**63 - 1, 2**63])
def test_configured_input_bound_fits_sqlite_integer(tmp_path: Path, limit: int) -> None:
    config = tmp_path / "input-limit.yaml"
    config.write_text(
        "providers:\n  local:\n    kind: llamacpp\n    quota_policy: local_limits_only\n"
        "    base_url: https://fixture.invalid\n    models:\n      m:\n        id: upstream\n"
        f"        max_input_tokens: {limit}\n",
        encoding="utf-8",
    )
    if limit == 2**63 - 1:
        assert load_settings(config).providers["local"].models["m"].max_input_tokens == limit
    else:
        with pytest.raises(ConfigurationError, match="^configuration_invalid$"):
            load_settings(config)


@pytest.mark.parametrize("name", ["team/vendor", "p%2Fq", ".", "..", "p q", "p?", "p#", "p\\q"])
def test_provider_name_must_fit_in_recovery_url_segment(name: str) -> None:
    with pytest.raises(ValidationError):
        Settings.model_validate(
            {
                "providers": {
                    name: {
                        "kind": "zai",
                        "quota_policy": "local_limits_only",
                        "base_url": "https://fixture.invalid",
                    }
                }
            }
        )
    with pytest.raises(ValidationError):
        InferenceRequest(provider=name, model="m", messages=(Message(role="user", content="x"),))
    request = InferenceRequest(
        provider="team_vendor.eu-1~a", model="m", messages=(Message(role="user", content="x"),)
    )
    assert request.provider == "team_vendor.eu-1~a"


@pytest.mark.parametrize(
    ("flag", "field"),
    [
        ("input_bound_verified", "input_bound_evidence"),
        ("output_bound_verified", "bound_evidence"),
        ("reasoning_in_output_bound_verified", "reasoning_bound_evidence"),
    ],
)
def test_verified_bounds_require_nonblank_evidence(flag: str, field: str) -> None:
    for invalid in (None, "", " \t\n\u2003"):
        with pytest.raises(ValidationError, match="verified bounds require nonblank evidence"):
            ModelConfig.model_validate({"id": "fixture", flag: True, field: invalid})
    model = ModelConfig.model_validate({"id": "fixture", flag: True, field: "  verified limit \n"})
    assert model.model_dump()[field] == "verified limit"


@pytest.mark.parametrize(("input_rate", "output_rate"), [(0, 0), (0, 1), (1, 0)])
def test_cloud_price_requires_positive_input_and_output_rates(
    input_rate: int, output_rate: int
) -> None:
    values = {
        "version": "v1",
        "provider": "p",
        "model": "m",
        "source": "synthetic tariff",
        "effective_at": "2026-01-01T00:00:00Z",
        "input_per_million": input_rate,
        "output_per_million": output_rate,
        "cached_input_per_million": 0,
    }
    with pytest.raises(ValidationError, match="greater than 0"):
        PriceConfig.model_validate(values)
    values.update(input_per_million=1, output_per_million=1)
    assert PriceConfig.model_validate(values).cached_input_per_million == 0


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
        "https://fixture.invalid/path#secret",
        "https://fixture.invalid/v1?",
        "https://fixture.invalid/v1#",
        "file:///tmp/key",
        "https://fixture.invalid:secret/path",
        "https://fixture.invalid:65536/path",
        "https://fixture.invalid:0/path",
        "https://fixture.invalid:-1/path",
        "https://[::1/path",
        "https://\u2603.invalid/path",
        "https://fixture.invalid/\ud800",
        *[f"https://fixture.invalid/v1{char}secret" for char in "\x00\t\n\r\x01\x1f\x7f"],
    ],
)
@pytest.mark.parametrize("field", ["base_url", "quota_url", "reporting_base_url"])
def test_invalid_endpoints_cannot_enter_loaded_configuration(
    tmp_path: Path, field: str, endpoint: str
) -> None:
    config = tmp_path / "endpoints.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "providers": {
                    "p": {
                        "kind": "openai",
                        "quota_policy": "local_limits_only",
                        "base_url": "https://fixture.invalid/v1",
                        field: endpoint,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError) as captured:
        load_settings(config)

    assert str(captured.value) == "configuration_invalid"
    assert captured.value.__cause__ is None
    assert captured.value.__suppress_context__


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://fixture.invalid/v1",
        "http://127.0.0.1:8080/v1",
        "http://[::1]:8080/v1",
        "https://fixture.invalid:65535/path%20segment",
        "https://fixture.invalid/v1/path%3F%23",
        "https://b\u00fccher.example/\u00e9tude",
    ],
)
def test_valid_endpoints_survive_configuration_loading(tmp_path: Path, endpoint: str) -> None:
    config = tmp_path / "endpoints.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "providers": {
                    "p": {
                        "kind": "openai",
                        "quota_policy": "local_limits_only",
                        "base_url": endpoint,
                        "quota_url": endpoint,
                        "reporting_base_url": endpoint,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    provider = load_settings(config).providers["p"]

    assert (provider.base_url, provider.quota_url, provider.reporting_base_url) == (endpoint,) * 3


@pytest.mark.parametrize(("seconds", "accepted"), [(1, True), (31622400, True), (31622401, False)])
def test_reporting_freshness_is_bounded(seconds: int, accepted: bool) -> None:
    values = {
        "kind": "openai",
        "quota_policy": "local_limits_only",
        "base_url": "https://fixture.invalid/v1",
        "stale_after_seconds": seconds,
    }
    if accepted:
        assert ProviderConfig.model_validate(values).stale_after_seconds == seconds
    else:
        with pytest.raises(ValidationError):
            ProviderConfig.model_validate(values)


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
