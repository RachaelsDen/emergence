"""Live provider tests must be deliberately enabled."""

import os

import pytest


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    if os.environ.get("EMERGENCE_LIVE") != "1":
        for item in items:
            if item.get_closest_marker("live"):
                item.add_marker(
                    pytest.mark.skip(reason="set EMERGENCE_LIVE=1 for capped live tests")
                )
