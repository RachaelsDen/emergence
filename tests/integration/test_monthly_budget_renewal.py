"""Calendar renewal restores virtual funds without renewing real provider capacity."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Literal

import httpx
import pytest
from _gateway_fixture import NOW, Clock, answer, gateway_for, invocation, settings_for
from test_migrations import migrated_path as migrated_path

from emergence.config import ProviderKind
from emergence.inference.gateway import GatewayError, RequestState
from emergence.resources.ledger import BudgetError
from emergence.resources.reporting_store import ReportingStore
from emergence.resources.schemas import QuotaObservation


@pytest.fixture(autouse=True)
def gateway_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_E005_INFERENCE", "synthetic-e005-inference-key")


@pytest.mark.anyio
async def test_period_opening_is_fixed_even_when_internal_settings_bypass_validation(
    migrated_path: Path,
) -> None:
    settings = settings_for()
    settings = settings.model_copy(
        update={
            "providers": {
                "p": settings.providers["p"].model_copy(update={"monthly_budget": Decimal(150)})
            }
        }
    )
    async with gateway_for(migrated_path, settings, lambda _: answer()) as (_, ledger, _, _):
        balance = await ledger.balance("p", NOW)
        assert (balance.opening_budget, balance.adjustments, balance.available) == (
            Decimal(100),
            Decimal(0),
            Decimal(100),
        )


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("zone", "boundary"),
    [
        ("UTC", datetime(2026, 12, 1, tzinfo=UTC)),
        ("America/New_York", datetime(2026, 12, 1, 5, tzinfo=UTC)),
    ],
)
async def test_exhausted_virtual_budget_renews_after_restart_at_calendar_boundary(
    migrated_path: Path, zone: str, boundary: datetime
) -> None:
    settings = settings_for()
    provider = settings.providers["p"]
    settings = settings.model_copy(
        update={
            "accounting_timezone": zone,
            "providers": {
                "p": provider.model_copy(
                    update={
                        "models": {
                            "m": provider.models["m"].model_copy(
                                update={
                                    "max_input_tokens": 50,
                                    "max_output_tokens": 50,
                                }
                            )
                        }
                    }
                )
            },
            "prices": (
                settings.prices[0].model_copy(
                    update={
                        "input_per_million": Decimal("1000000"),
                        "output_per_million": Decimal("1000000"),
                    }
                ),
            ),
        }
    )
    clock = Clock(boundary - timedelta(seconds=1))
    calls: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return answer(50, 50)

    async with gateway_for(migrated_path, settings, respond, clock) as (gateway, ledger, _, _):
        assert (await gateway.infer(invocation(50), "old-month")).state is RequestState.COMPLETED
        prior = await ledger.balance("p", clock())
        assert prior.spent == Decimal("100")
        assert prior.available == 0
        with pytest.raises(BudgetError, match="budget_insufficient"):
            await gateway.infer(invocation(50), "old-month-exhausted")

    clock.at = boundary
    async with gateway_for(migrated_path, settings, respond, clock) as (gateway, ledger, _, _):
        renewed = await ledger.balance("p", clock())
        assert renewed.opening_budget == Decimal("100")
        assert renewed.available == Decimal("100")
        assert renewed.period_id != prior.period_id
        assert (await gateway.infer(invocation(50), "new-month")).state is RequestState.COMPLETED
        assert (await ledger.balance("p", boundary - timedelta(seconds=1))).spent == Decimal("100")
        assert (await ledger.balance("p", clock())).spent == Decimal("100")
    assert calls == ["/v1/chat/completions", "/v1/chat/completions"]


@pytest.mark.anyio
@pytest.mark.parametrize("block", ["quota", "credit", "safety"])
async def test_real_capacity_still_blocks_after_virtual_month_renewal(
    migrated_path: Path, block: Literal["quota", "credit", "safety"]
) -> None:
    boundary = datetime(2026, 12, 1, tzinfo=UTC)
    clock = Clock(boundary - timedelta(seconds=1))
    settings = settings_for(ProviderKind.OPENAI)
    provider = settings.providers["p"]
    settings = settings.model_copy(
        update={
            "providers": {
                "p": provider.model_copy(
                    update={
                        "safety_limit": Decimal("0.008") if block == "safety" else Decimal("95"),
                        "models": {
                            "m": provider.models["m"].model_copy(
                                update={
                                    "max_input_tokens": 5,
                                    "max_output_tokens": 3,
                                }
                            )
                        },
                    }
                )
            }
        }
    )
    calls: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return answer()

    async with gateway_for(migrated_path, settings, respond, clock) as (
        gateway,
        ledger,
        database,
        spool,
    ):
        assert (await gateway.infer(invocation(3), "old-month")).state is RequestState.COMPLETED
        await ledger.adjust(
            "p",
            Decimal("-99.992"),
            "reduction",
            "exhaust test virtual period",
            "operator",
            at=clock(),
        )
        assert (await ledger.balance("p", clock())).available == 0
        if block != "safety":
            await ReportingStore(database, spool).append(
                (
                    QuotaObservation(
                        provider="p",
                        resource="inference" if block == "quota" else "credit",
                        unit="requests" if block == "quota" else "USD",
                        remaining=Decimal(0),
                        observed_at=clock(),
                        source="inference",
                        status="observed",
                        confidence="synthetic known exhaustion",
                    ),
                ),
                86400,
            )

    clock.at = boundary
    async with gateway_for(migrated_path, settings, respond, clock) as (gateway, ledger, _, _):
        renewed = await ledger.balance("p", clock())
        assert renewed.available == Decimal("100")
        expected = BudgetError if block == "safety" else GatewayError
        code = "safety_limit_exceeded" if block == "safety" else "provider_capacity_exhausted"
        with pytest.raises(expected, match=code):
            await gateway.infer(invocation(3), "new-month-blocked")
        assert (await ledger.balance("p", clock())).reserved == 0
    assert calls == ["/v1/chat/completions"]
