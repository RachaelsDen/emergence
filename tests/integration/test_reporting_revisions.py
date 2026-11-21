from decimal import Decimal
from pathlib import Path

import httpx
import pytest
from sqlalchemy import select
from test_budget_concurrency import NOW, admit, configuration, prepared
from test_migrations import migrated_path as migrated_path
from test_quota_polling import cost, monitor_for, page

from emergence.config import ProviderKind
from emergence.resources.schemas import ReportingIssue
from emergence.storage.models import usage_transactions


@pytest.mark.anyio
async def test_complete_daily_dimension_replacement_is_append_only(migrated_path: Path) -> None:
    settings = configuration(ProviderKind.OPENAI, project="proj_fixture")
    cfg = settings.providers["p"].model_copy(update={"reporting_credential_env": "TEST_REV_ADMIN"})
    settings = settings.model_copy(update={"providers": {"p": cfg}})
    label = "A"

    def handler(request: httpx.Request) -> httpx.Response:
        rows = cost("7" if label == "A" else "8").replace('"tokens"', f'"{label}"')
        return page(request, rows=rows if request.url.path.endswith("/costs") else "")

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("TEST_REV_ADMIN", "fixture-revision-admin")
        async with monitor_for(migrated_path, settings, handler, lambda: NOW) as (monitor, db):
            ledger = await prepared(db, settings)
            refs: list[str | None] = []
            for label, expected in (("A", 7), ("B", 8), ("B", 8)):
                _, observation = await monitor.poll_once("p")
                assert observation.status == "verified"
                assert [cost.line_item for cost in observation.costs] == [label]
                refs.append(observation.raw_payload_ref)
                balance = await ledger.balance("p", NOW)
                assert (balance.measured_total, balance.safety_exposure) == (expected, expected)
            async with db.session() as session:
                entries = list(
                    await session.execute(
                        select(
                            usage_transactions.c.measured_cost,
                            usage_transactions.c.detail_archive_record_id,
                        )
                    )
                )
            assert len(entries) == 3
            assert set(entries) == {(7, refs[0]), (-7, refs[1]), (8, refs[1])}


@pytest.mark.anyio
async def test_only_explicit_complete_days_can_remove_previously_measured_dimensions(
    migrated_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_REV_ADMIN", "fixture-revision-admin")
    settings = configuration(ProviderKind.OPENAI, project="proj_fixture")
    cfg = settings.providers["p"].model_copy(update={"reporting_credential_env": "TEST_REV_ADMIN"})
    settings = settings.model_copy(update={"providers": {"p": cfg}})
    stage = "seed"
    exact = "7.00000000000000000000000000000000000001"

    def handler(request: httpx.Request) -> httpx.Response:
        if not request.url.path.endswith("/costs"):
            return page(request, rows="")
        if stage == "seed":
            return page(request, rows=cost(exact))
        if stage == "foreign":
            return page(request, rows=cost("0", project="other"))
        if stage == "partial":
            if request.url.params.get("page"):
                return httpx.Response(503)
            return page(request, rows="", cursor="second")
        if stage == "omitted" or not request.url.params.get("page"):
            return page(
                request, rows=cost("3"), offset=1, cursor=None if stage == "omitted" else "second"
            )
        return page(request, rows="")

    async with monitor_for(migrated_path, settings, handler, lambda: NOW) as (monitor, db):
        ledger = await prepared(db, settings)
        async with db.write() as session:
            assert await admit(session, ledger, "unrelated", 1) == 2
        for stage, issue, expected in (
            ("seed", None, exact),
            ("partial", ReportingIssue.HTTP_ERROR, exact),
            ("foreign", ReportingIssue.SCOPE_MISMATCH, exact),
            ("omitted", None, "10.00000000000000000000000000000000000001"),
            ("empty", None, "3"),
            ("empty", None, "3"),
        ):
            _, observation = await monitor.poll_once("p")
            assert observation.issue == issue, stage
            balance = await ledger.balance("p", NOW)
            assert (balance.measured_total, balance.reserved) == (Decimal(expected), 2)
