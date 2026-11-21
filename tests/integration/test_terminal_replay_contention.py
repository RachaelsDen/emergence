"""Reading completed archive detail does not hold the financial writer."""

from decimal import Decimal
from pathlib import Path

import httpx
import pytest
from _gateway_fixture import NOW, answer, gateway_for, invocation, settings_for
from anyio import Event, create_task_group, fail_after
from test_migrations import migrated_path as migrated_path
from test_provider_quota_policy import gateway_key as gateway_key

from emergence.inference.gateway import GatewayResult, RequestState
from emergence.resources.budgets import AdjustmentReceipt
from emergence.storage.archive import RecordKey
from emergence.storage.repositories import ArchiveReference


@pytest.mark.anyio
async def test_terminal_replay_archive_io_does_not_block_financial_writes(
    migrated_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered, release = Event(), Event()
    receipts: list[GatewayResult] = []
    calls: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return answer()

    async with gateway_for(migrated_path, settings_for(), respond) as (gateway, ledger, _, spool):
        first = await gateway.infer(invocation(), "terminal")
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
            receipts.append(await gateway.infer(invocation(), "terminal"))

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
        assert adjusted is not None and adjusted.available == Decimal("100.992")
        assert len(receipts) == 1 and receipts[0].replayed
        assert receipts[0].state is RequestState.COMPLETED
        assert calls == ["/v1/chat/completions"]
