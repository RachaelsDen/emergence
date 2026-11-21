"""Prometheus is optional; authenticated storage status does not depend on it."""

import pytest
from _gateway_fixture import invocation
from httpx import AsyncClient
from prometheus_client.parser import text_string_to_metric_families
from pydantic import JsonValue, TypeAdapter
from test_api import OPERATOR, READER
from test_api import api as api
from test_api import metrics_enabled as metrics_enabled
from test_migrations import migrated_path as migrated_path


@pytest.mark.anyio
@pytest.mark.parametrize("metrics_enabled", [False, True])
async def test_metrics_are_optional_and_protected_while_storage_status_remains_available(
    api: tuple[AsyncClient, list[str]], metrics_enabled: bool
) -> None:
    client, _ = api
    response = await client.get("/metrics", headers={"Authorization": ""})
    assert response.status_code == (401 if metrics_enabled else 404)
    metadata = await client.get("/api/v1/resources")
    assert metadata.status_code == 200
    storage = TypeAdapter(dict[str, JsonValue]).validate_json(metadata.content)["storage"]
    assert isinstance(storage, dict)
    assert storage["limit_bytes"] == 1024 * 1024
    assert (await client.get("/health/ready")).status_code == 200


@pytest.mark.anyio
@pytest.mark.parametrize("metrics_enabled", [True])
async def test_metrics_report_actual_requests_usage_and_process_latency(
    api: tuple[AsyncClient, list[str]], metrics_enabled: bool
) -> None:
    client, _ = api
    assert metrics_enabled
    inference = await client.post(
        "/api/v1/inference",
        json=invocation().model_dump(mode="json"),
        headers={"Idempotency-Key": "metrics-request"},
    )
    assert inference.status_code == 200
    metrics = await client.get("/metrics", headers={"Authorization": f"Bearer {READER}"})
    assert metrics.status_code == 200
    samples = tuple(
        sample
        for family in text_string_to_metric_families(metrics.text)
        for sample in family.samples
    )
    scalars = {sample.name: sample.value for sample in samples if "dimension" not in sample.labels}
    tokens = {
        sample.labels["dimension"]: sample.value
        for sample in samples
        if sample.name == "emergence_observed_tokens_total"
    }
    assert scalars["emergence_requests_total"] == 1
    assert scalars["emergence_inference_duration_seconds_count"] == 1
    assert scalars["emergence_budget_available_usd"] == 99.992
    assert tokens == {"input": 5, "output": 3}
    assert OPERATOR not in metrics.text and READER not in metrics.text
    assert "bounded answer" not in metrics.text
