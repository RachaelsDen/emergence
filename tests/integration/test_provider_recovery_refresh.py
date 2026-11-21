"""Fresh operator confirmation requires the current recovery observation token."""

from dataclasses import replace
from datetime import timedelta

import pytest
from test_api import JSON_OBJECT
from test_migrations import migrated_path as migrated_path
from test_provider_recovery import PATH, Harness, Receipt, denial, infer
from test_provider_recovery import harness as harness

from emergence.config import QuotaPolicy


@pytest.mark.anyio
async def test_expired_recovery_can_be_confirmed_again_without_replaying_old_authority(
    harness: Harness,
) -> None:
    harness.replies.append(denial())
    async with harness.open() as client:
        rejected = Receipt.model_validate_json((await infer(client, "denial")).content)
        body = {"request_id": rejected.request_id, "reason": "Independently confirmed top-up"}
        first = await client.post(PATH, json=body)
        assert first.status_code == 200
        observation_id = JSON_OBJECT.validate_json(first.content)["observation_id"]
    provider = harness.settings.providers["p"].model_copy(
        update={"quota_policy": QuotaPolicy.REQUIRE_FRESH}
    )
    settings = harness.settings.model_copy(update={"providers": {"p": provider}})
    harness.clock.at += timedelta(seconds=provider.stale_after_seconds)
    async with replace(harness, settings=settings).open() as client:
        blocked = await infer(client, "expired")
        assert JSON_OBJECT.validate_json(blocked.content)["error"] == "provider_capacity_unknown"
        assert (await client.post(PATH, json=body)).status_code == 409
        confirmation = {**body, "observation_id": observation_id}
        refreshed = await client.post(PATH, json=confirmation)
        assert refreshed.status_code == 200
        assert (await client.post(PATH, json=confirmation)).status_code == 409
        assert (
            Receipt.model_validate_json((await infer(client, "fresh-confirmation")).content).state
            == "completed"
        )
    assert len(harness.calls) == 2
