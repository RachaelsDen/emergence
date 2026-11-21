from datetime import UTC, datetime, timedelta

import pytest

from emergence.resources.budgets import month


@pytest.mark.parametrize(
    ("zone", "start", "end"),
    [
        ("UTC", "2024-02-01T00:00:00Z", "2024-03-01T00:00:00Z"),
        ("America/New_York", "2026-03-01T05:00:00Z", "2026-04-01T04:00:00Z"),
        ("America/New_York", "2026-11-01T04:00:00Z", "2026-12-01T05:00:00Z"),
        ("Asia/Kathmandu", "2026-12-31T18:15:00Z", "2027-01-31T18:15:00Z"),
    ],
)
def test_calendar_month_is_half_open_in_operator_timezone(zone: str, start: str, end: str) -> None:
    first, following = datetime.fromisoformat(start), datetime.fromisoformat(end)
    assert (
        month(first, zone)
        == month(following - timedelta(microseconds=1), zone)
        == (
            first,
            following,
        )
    )
    assert month(following.astimezone(UTC), zone)[0] == following
