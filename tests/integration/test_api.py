"""Authenticated HTTP boundaries over real migrated storage and provider transport."""

from collections.abc import AsyncGenerator
from functools import partial
from pathlib import Path
from typing import Final

import httpx
import pytest
from _gateway_fixture import answer, invocation, settings_for
from fastapi.openapi.models import OpenAPI, Reference, Response, Schema
from httpx import AsyncClient
from pydantic import JsonValue, TypeAdapter
from test_migrations import migrated_path as migrated_path

from emergence.api.app import create_app
from emergence.config import ConfigurationError

OPERATOR: Final = "api-operator-synthetic-aaaaaaaaaaaaaaaa"
READER: Final = "api-reader-synthetic-bbbbbbbbbbbbbbbb"
JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


@pytest.fixture
def metrics_enabled() -> bool:
    return False


@pytest.fixture
async def api(
    migrated_path: Path, monkeypatch: pytest.MonkeyPatch, metrics_enabled: bool
) -> AsyncGenerator[tuple[AsyncClient, list[str]]]:
    monkeypatch.setenv("TEST_API_OPERATOR", OPERATOR)
    monkeypatch.setenv("TEST_API_READER", READER)
    monkeypatch.setenv("TEST_E005_INFERENCE", "synthetic-api-upstream")
    calls: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return answer()

    monkeypatch.setattr(
        httpx, "AsyncClient", partial(AsyncClient, transport=httpx.MockTransport(respond))
    )
    settings = settings_for().model_copy(
        update={
            "database_path": migrated_path,
            "spool_path": migrated_path.parent / "api-spool",
            "operator_token_env": "TEST_API_OPERATOR",
            "reader_token_env": "TEST_API_READER",
            "request_limit_bytes": 1024,
            "requests_per_minute": 2,
            "metrics_enabled": metrics_enabled,
        }
    )
    app = create_app(settings)
    async with (
        app.router.lifespan_context(app),
        AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://core",
            headers={"Authorization": f"Bearer {OPERATOR}"},
        ) as client,
    ):
        yield client, calls


@pytest.mark.anyio
async def test_http_inference_preserves_exact_usage_and_protected_history(
    api: tuple[AsyncClient, list[str]],
) -> None:
    client, calls = api
    response = await client.post(
        "/api/v1/inference",
        json=invocation().model_dump(mode="json"),
        headers={"Idempotency-Key": "api-receipt"},
    )
    assert response.status_code == 200
    receipt = JSON_OBJECT.validate_json(response.content)
    request_id = receipt["request_id"]
    assert isinstance(request_id, str)
    assert receipt["state"] == "completed"
    history = await client.get(f"/api/v1/history/{request_id}")
    assert history.status_code == 200
    records = JSON_OBJECT.validate_json(history.content)["records"]
    assert isinstance(records, list) and len(records) == 2
    usage = await client.get("/api/v1/usage", params={"provider": "p", "model": "m"})
    rows = JSON_OBJECT.validate_json(usage.content)["items"]
    assert isinstance(rows, list) and len(rows) == 1
    row = rows[0]
    assert isinstance(row, dict)
    assert (row["estimated_cost"], row["measured_cost"], row["request_id"]) == (
        "0.008",
        None,
        request_id,
    )
    assert calls == ["/v1/chat/completions"]


@pytest.mark.anyio
@pytest.mark.parametrize("resource", ["history/private", "inference/private"])
async def test_reader_cannot_retrieve_full_conversations(
    api: tuple[AsyncClient, list[str]], resource: str
) -> None:
    client, _ = api
    response = await client.get(
        f"/api/v1/{resource}", headers={"Authorization": f"Bearer {READER}"}
    )
    assert response.status_code == 403
    assert JSON_OBJECT.validate_json(response.content)["error"] == "operator_required"


@pytest.mark.anyio
async def test_chunked_oversize_body_is_rejected_before_dispatch(
    api: tuple[AsyncClient, list[str]],
) -> None:
    client, calls = api

    async def body() -> AsyncGenerator[bytes]:
        yield b'{"messages":["'
        yield b"x" * 2048
        yield b'"]}'

    response = await client.post(
        "/api/v1/inference", content=body(), headers={"Idempotency-Key": "oversize"}
    )
    assert response.status_code == 413
    assert calls == []


@pytest.mark.anyio
async def test_http_rate_limit_prevents_another_provider_dispatch(
    api: tuple[AsyncClient, list[str]],
) -> None:
    client, calls = api
    statuses: list[int] = []
    for key in ("first", "second", "third"):
        response = await client.post(
            "/api/v1/inference",
            json=invocation().model_dump(mode="json"),
            headers={"Idempotency-Key": key},
        )
        statuses.append(response.status_code)
    assert statuses == [200, 200, 429]
    assert calls == ["/v1/chat/completions", "/v1/chat/completions"]
    assert (await client.get("/api/v1/budgets")).status_code == 200


@pytest.mark.anyio
async def test_numeric_money_and_secret_validation_inputs_are_not_accepted_or_echoed(
    api: tuple[AsyncClient, list[str]],
) -> None:
    client, _ = api
    response = await client.post(
        "/api/v1/budgets/adjustments",
        json={"provider": "p", "amount": 0.1, "kind": "grant", "reason": OPERATOR},
    )
    assert response.status_code == 422
    assert response.content == b'{"error":"invalid_request"}'
    assert OPERATOR not in response.text
    document = OpenAPI.model_validate_json((await client.get("/openapi.json")).content)
    assert document.paths is not None and document.components is not None
    operation = document.paths["/api/v1/budgets/adjustments"].post
    assert operation is not None and operation.responses is not None
    documented = operation.responses["422"]
    assert isinstance(documented, Response) and documented.content is not None
    reference = documented.content["application/json"].schema_
    assert isinstance(reference, Reference | Schema) and reference.ref is not None
    assert document.components.schemas is not None
    error_schema = document.components.schemas[reference.ref.rsplit("/", 1)[1]]
    assert isinstance(error_schema, Schema) and error_schema.properties is not None
    body = JSON_OBJECT.validate_json(response.content)
    assert set(body) <= set(error_schema.properties)
    assert set(error_schema.required or ()) <= set(body)


@pytest.mark.parametrize("reader", ["too-short", OPERATOR])
def test_unsafe_role_credentials_reject_configuration(
    monkeypatch: pytest.MonkeyPatch, reader: str
) -> None:
    monkeypatch.setenv("TEST_API_OPERATOR", OPERATOR)
    monkeypatch.setenv("TEST_API_READER", reader)
    settings = settings_for().model_copy(
        update={"operator_token_env": "TEST_API_OPERATOR", "reader_token_env": "TEST_API_READER"}
    )
    with pytest.raises(ConfigurationError):
        create_app(settings)
