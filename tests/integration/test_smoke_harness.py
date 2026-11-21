"""Exercise the actual opt-in runner with a synthetic transport and no live enablement."""

from decimal import Decimal
from functools import partial
from pathlib import Path

import httpx
import pytest
from _gateway_fixture import NOW, Clock, answer, settings_for
from anyio import Path as AsyncPath
from httpx import AsyncClient

from emergence.resources.ledger import BudgetError
from tests.live.contract import LiveOptions
from tests.live.test_providers import prepared_run
from tests.live.test_providers import test_selected_provider_inference as invoke_smoke


@pytest.mark.anyio
@pytest.mark.parametrize("transport_stalls", [False, True])
async def test_smoke_harness_uses_temporary_ledger_and_never_retries_a_provider_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, transport_stalls: bool
) -> None:
    monkeypatch.delenv("EMERGENCE_LIVE", raising=False)
    monkeypatch.setenv("TEST_E005_INFERENCE", "synthetic-smoke-inference-key")
    calls: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if transport_stalls:
            raise httpx.ReadTimeout("synthetic provider stall", request=request)
        return answer()

    monkeypatch.setattr(
        httpx, "AsyncClient", partial(AsyncClient, transport=httpx.MockTransport(respond))
    )
    source = settings_for()
    untouched = tmp_path / "original-state" / "do-not-write.sqlite3"
    source = source.model_copy(
        update={
            "database_path": untouched,
            "prices": tuple(
                price.model_copy(
                    update={"input_per_million": Decimal("1"), "output_per_million": Decimal("1")}
                )
                for price in source.prices
            ),
        }
    )
    config = tmp_path / "operator-selected.yaml"
    await AsyncPath(config).write_text(source.model_dump_json())
    options = LiveOptions(
        config=config,
        providers=("p",),
        models={},
        max_output_tokens=8,
        request_cap=Decimal("0.01"),
        total_cap=Decimal("0.01"),
    )
    async with prepared_run(options, tmp_path, monkeypatch) as run:
        if transport_stalls:
            with pytest.raises(AssertionError, match="provider_transport_error"):
                await invoke_smoke(run)
        else:
            await invoke_smoke(run)
        balance = await run[0].ledger.balance("p")
        assert (balance.opening_budget, balance.adjustments) == (Decimal("100"), Decimal("0"))
        assert (balance.spent, balance.reserved) == (
            (Decimal("0"), Decimal("0.000028"))
            if transport_stalls
            else (Decimal("0.000008"), Decimal("0"))
        )
    assert calls == ["/v1/chat/completions"]
    assert not await AsyncPath(untouched).exists()


@pytest.mark.anyio
async def test_smoke_ceiling_is_checked_inside_admission_after_calendar_renewal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_E005_INFERENCE", "synthetic-smoke-key")
    calls: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return answer()

    monkeypatch.setattr(
        httpx, "AsyncClient", partial(AsyncClient, transport=httpx.MockTransport(respond))
    )
    config = tmp_path / "ceiling.yaml"
    await AsyncPath(config).write_text(settings_for().model_dump_json())
    options = LiveOptions(
        config=config,
        providers=("p",),
        models={},
        max_output_tokens=8,
        request_cap=Decimal("0.01"),
        total_cap=Decimal("0.01"),
    )
    clock = Clock(NOW)
    periods: list[str | None] = []
    async with prepared_run(options, tmp_path, monkeypatch) as run:
        run[0].clock = clock
        for instant in (NOW, NOW.replace(month=12, day=1)):
            clock.at = instant
            with pytest.raises(BudgetError, match="live_request_cap_exceeded"):
                await invoke_smoke(run)
            balance = await run[0].ledger.balance("p", instant)
            periods.append(balance.period_id)
            assert (balance.opening_budget, balance.adjustments, balance.reserved) == (
                Decimal(100),
                Decimal(0),
                Decimal(0),
            )
    assert calls == []
    assert len(set(periods)) == 2
