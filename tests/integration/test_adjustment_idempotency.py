"""Financial operation identities survive retry, conflict and concurrent delivery."""

import pytest
from anyio import create_task_group
from httpx import AsyncClient
from pydantic import JsonValue
from test_api import JSON_OBJECT
from test_api import api as api
from test_api import metrics_enabled as metrics_enabled
from test_api import provider_enabled as provider_enabled
from test_migrations import migrated_path as migrated_path


@pytest.mark.anyio
@pytest.mark.parametrize("concurrent", [False, True])
async def test_same_adjustment_key_applies_once(
    api: tuple[AsyncClient, list[str]], concurrent: bool
) -> None:
    client, calls = api
    receipts: list[dict[str, JsonValue]] = []

    async def deliver() -> None:
        response = await client.post(
            "/api/v1/budgets/adjustments",
            json={"provider": "p", "amount": "10", "kind": "grant", "reason": "Confirmed grant"},
            headers={"Idempotency-Key": "one-grant"},
        )
        assert response.status_code == 200
        receipts.append(JSON_OBJECT.validate_json(response.content))

    if concurrent:
        async with create_task_group() as tasks:
            tasks.start_soon(deliver)
            tasks.start_soon(deliver)
    else:
        await deliver()
        await deliver()
    current = JSON_OBJECT.validate_json((await client.get("/api/v1/budgets")).content)
    budgets = current["budgets"]
    assert isinstance(budgets, list) and isinstance(budgets[0], dict)
    assert budgets[0]["adjustments"] == "10"
    assert sum(item["replayed"] is True for item in receipts) == 1
    assert sum(item["replayed"] is False for item in receipts) == 1
    assert receipts[0]["adjustment_id"] == receipts[1]["adjustment_id"]
    assert calls == []


@pytest.mark.anyio
async def test_adjustment_key_cannot_be_reused_for_different_money_or_reason(
    api: tuple[AsyncClient, list[str]],
) -> None:
    client, _ = api
    body = {"provider": "p", "amount": "1", "kind": "grant", "reason": "password=first"}
    headers = {"Idempotency-Key": "conflict"}
    assert (
        await client.post("/api/v1/budgets/adjustments", json=body, headers=headers)
    ).status_code == 200
    for altered in ({**body, "amount": "2"}, {**body, "reason": "password=second"}):
        response = await client.post("/api/v1/budgets/adjustments", json=altered, headers=headers)
        assert response.status_code == 409
        assert (
            JSON_OBJECT.validate_json(response.content)["error"]
            == "adjustment_idempotency_conflict"
        )


@pytest.mark.anyio
@pytest.mark.parametrize("key", [None, "", "two words", "x" * 129])
async def test_adjustment_requires_a_bounded_visible_operation_key(
    api: tuple[AsyncClient, list[str]], key: str | None
) -> None:
    client, _ = api
    response = await client.post(
        "/api/v1/budgets/adjustments",
        json={"provider": "p", "amount": "1", "kind": "grant", "reason": "confirmed"},
        headers={} if key is None else {"Idempotency-Key": key},
    )
    assert response.status_code == 422
    assert JSON_OBJECT.validate_json(response.content)["error"] == "invalid_request"
