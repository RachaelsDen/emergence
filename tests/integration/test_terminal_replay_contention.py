"""Reading archived replay detail does not hold the financial writer."""

from decimal import Decimal
from pathlib import Path

import httpx
import pytest
from _gateway_fixture import NOW, answer, gateway_for, invocation, settings_for
from anyio import Event, create_task_group, fail_after
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from test_migrations import migrated_path as migrated_path
from test_provider_quota_policy import gateway_key as gateway_key

from emergence.inference.gateway import GatewayResult, RequestState
from emergence.resources.budgets import AdjustmentReceipt
from emergence.storage.archive import RecordKey
from emergence.storage.repositories import ArchiveReference


@pytest.mark.anyio
@pytest.mark.parametrize("terminal", [True, False])
async def test_replay_archive_io_does_not_block_financial_writes(
    migrated_path: Path, monkeypatch: pytest.MonkeyPatch, terminal: bool
) -> None:
    entered, release = Event(), Event()
    receipts: list[GatewayResult] = []
    calls: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return answer()

    async with gateway_for(migrated_path, settings_for(), respond) as (gateway, ledger, db, spool):
        if not terminal:
            async with db.write() as session:
                await session.execute(
                    text(
                        "CREATE TRIGGER finish_fault BEFORE INSERT ON request_events "
                        "WHEN NEW.kind = 'completed' "
                        "BEGIN SELECT RAISE(ABORT, 'simulated storage failure'); END"
                    )
                )
            with pytest.raises(IntegrityError, match="simulated storage failure"):
                await gateway.infer(invocation(), "replay")
            async with db.write() as session:
                await session.execute(text("DROP TRIGGER finish_fault"))
        first = await gateway.infer(invocation(), "replay") if terminal else None
        if first is None:
            async with db.session() as session:
                output_id = (
                    await session.execute(text("SELECT output_record_id FROM request_capacity"))
                ).scalar_one()
        else:
            assert first.state is RequestState.COMPLETED
            output_id = RecordKey(first.request_id, "output").record_id
        read = spool.read

        async def delayed(reference: ArchiveReference) -> bytes:
            if reference.record_id == output_id:
                entered.set()
                await release.wait()
            return await read(reference)

        monkeypatch.setattr(spool, "read", delayed)
        adjusted: AdjustmentReceipt | None = None

        async def replay() -> None:
            receipts.append(await gateway.infer(invocation(), "replay"))

        with fail_after(3):
            async with create_task_group() as tasks:
                tasks.start_soon(replay)
                await entered.wait()
                try:
                    adjusted = await ledger.adjust(
                        "p",
                        Decimal(1),
                        "grant",
                        "independent grant",
                        "operator",
                        NOW,
                        idempotency_key="while-reading-archive",
                    )
                finally:
                    release.set()
        assert adjusted is not None
        assert adjusted.available == Decimal("100.992" if terminal else "100.97")
        assert len(receipts) == 1 and receipts[0].replayed
        assert receipts[0].state is RequestState.COMPLETED
        assert calls == ["/v1/chat/completions"]
