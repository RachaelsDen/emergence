"""Live selection must fail closed before any provider operation is available."""

from decimal import Decimal, localcontext
from pathlib import Path

import pytest

from tests.integration._gateway_fixture import settings_for
from tests.live.contract import LiveConfigurationError, LiveOptions


def environment() -> dict[str, str]:
    return {
        "EMERGENCE_LIVE": "1",
        "EMERGENCE_CONFIG": "/operator/selected.yaml",
        "EMERGENCE_LIVE_PROVIDERS": "p",
        "EMERGENCE_LIVE_MAX_OUTPUT_TOKENS": "8",
        "EMERGENCE_LIVE_MAX_REQUEST_USD": "0.01",
        "EMERGENCE_LIVE_TOTAL_CAP_USD": "0.04",
    }


@pytest.mark.parametrize("flag", ["0", "", "true"])
def test_live_requires_exact_optin(flag: str) -> None:
    with pytest.raises(LiveConfigurationError, match="live_not_enabled"):
        LiveOptions.from_environment(environment() | {"EMERGENCE_LIVE": flag})


@pytest.mark.parametrize(
    "missing",
    [
        "EMERGENCE_CONFIG",
        "EMERGENCE_LIVE_PROVIDERS",
        "EMERGENCE_LIVE_MAX_OUTPUT_TOKENS",
        "EMERGENCE_LIVE_MAX_REQUEST_USD",
        "EMERGENCE_LIVE_TOTAL_CAP_USD",
    ],
)
def test_live_missing_selection_or_cap_is_a_failure(missing: str) -> None:
    values = environment()
    del values[missing]
    with pytest.raises(LiveConfigurationError, match="live_configuration_incomplete"):
        LiveOptions.from_environment(values)


def test_smoke_uses_isolated_enforced_state_and_preserves_operator_settings(tmp_path: Path) -> None:
    source = settings_for().model_copy(update={"enforce_budgets": False})
    bounded = LiveOptions.from_environment(environment()).bounded_settings(source, tmp_path)
    assert bounded.database_path == tmp_path / "smoke.sqlite3"
    assert bounded.spool_path == tmp_path / "spool"
    assert bounded.archive_path is None and bounded.archive_mount_path is None
    assert bounded.enforce_budgets
    assert bounded.providers["p"].monthly_budget == Decimal("100")
    assert not source.enforce_budgets
    assert source.providers["p"].monthly_budget == Decimal("100")


def test_total_cap_counts_all_cloud_selections_under_low_decimal_precision(tmp_path: Path) -> None:
    source = settings_for()
    source = source.model_copy(
        update={
            "providers": {
                "p": source.providers["p"],
                "q": source.providers["p"],
            }
        }
    )
    options = LiveOptions.from_environment(
        environment()
        | {
            "EMERGENCE_LIVE_PROVIDERS": "p,q",
            "EMERGENCE_LIVE_MAX_REQUEST_USD": "0.0100001",
            "EMERGENCE_LIVE_TOTAL_CAP_USD": "0.02",
        }
    )
    with (
        localcontext(prec=2),
        pytest.raises(LiveConfigurationError, match="live_total_cap_insufficient"),
    ):
        options.bounded_settings(source, tmp_path)


@pytest.mark.parametrize("alias", [None, "alternative"])
def test_live_multiple_aliases_require_explicit_selection(alias: str | None) -> None:
    source = settings_for()
    provider = source.providers["p"]
    provider = provider.model_copy(
        update={
            "models": {
                "m": provider.models["m"],
                "alternative": provider.models["m"].model_copy(update={"id": "different-model"}),
            }
        }
    )
    source = source.model_copy(update={"providers": {"p": provider}})
    values = environment()
    if alias is not None:
        values["EMERGENCE_LIVE_MODELS"] = '{"p":"alternative"}'
    options = LiveOptions.from_environment(values)
    if alias is None:
        with pytest.raises(LiveConfigurationError, match="live_model_selection_required"):
            options.model(source, "p")
    else:
        assert options.model(source, "p") == "alternative"
