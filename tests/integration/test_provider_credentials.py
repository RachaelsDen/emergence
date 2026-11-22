"""Invalid inference header credentials fail before admission or transport."""

from pathlib import Path

import httpx
import pytest
from _gateway_fixture import answer, gateway_for, invocation, settings_for
from sqlalchemy import func, select
from test_migrations import migrated_path as migrated_path

from emergence.config import ProviderKind
from emergence.providers.base import ChatAdapter, ProviderFailure
from emergence.storage.models import request_reservations, requests


@pytest.mark.anyio
@pytest.mark.parametrize("kind", list(ProviderKind))
@pytest.mark.parametrize("enforce_budgets", [True, False], ids=["enforced", "shadow"])
@pytest.mark.parametrize(
    "credential",
    ["synthetic-\u00e9", "synthetic key", " key", "key ", "key\t", "key\r\n", "key\x01", "key\x7f"],
)
async def test_invalid_header_credential_rejects_before_admission(
    migrated_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: ProviderKind,
    enforce_budgets: bool,
    credential: str,
) -> None:
    monkeypatch.setenv("TEST_E005_INFERENCE", credential)
    settings = settings_for(kind).model_copy(update={"enforce_budgets": enforce_budgets})
    calls: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return answer()

    async with gateway_for(migrated_path, settings, respond) as (gateway, _, database, _):
        with pytest.raises(ProviderFailure, match="invalid_inference_credential") as failure:
            await gateway.infer(invocation(), "invalid-credential")
        assert not failure.value.ambiguous
        assert failure.value.raw_response is None
        assert credential not in str(failure.value)
        assert calls == []
        async with database.session() as session:
            assert await session.scalar(select(func.count()).select_from(requests)) == 0
            assert await session.scalar(select(func.count()).select_from(request_reservations)) == 0


@pytest.mark.anyio
@pytest.mark.parametrize("credential", ["synthetic-\u00e9", "key\t", "key\r\n", "key\x7f"])
async def test_actual_header_resolution_rejects_invalid_rotated_credential(
    monkeypatch: pytest.MonkeyPatch, credential: str
) -> None:
    monkeypatch.setenv("TEST_E005_INFERENCE", "synthetic-valid-key")
    settings = settings_for()
    async with httpx.AsyncClient() as client:
        adapter = ChatAdapter("p", settings.providers["p"], settings, client)
        adapter.preflight(invocation(), cost_enforced=True)
        monkeypatch.setenv("TEST_E005_INFERENCE", credential)
        with pytest.raises(ProviderFailure, match="invalid_inference_credential") as failure:
            adapter.headers()
    assert not failure.value.ambiguous
    assert credential not in str(failure.value)
