"""Configuration validates the actual endpoint including its completion suffix."""

import pytest
from pydantic import ValidationError

from emergence.config import ProviderConfig


@pytest.mark.parametrize("fits", [False, True])
def test_base_url_must_leave_room_for_the_actual_inference_path(fits: bool) -> None:
    prefix = "https://fixture.invalid/"
    maximum = 65536 - (len("/chat/completions") if fits else 0)
    endpoint = prefix + "x" * (maximum - len(prefix))
    raw = {"kind": "zai", "quota_policy": "local_limits_only", "base_url": endpoint}
    if fits:
        assert ProviderConfig.model_validate(raw).base_url == endpoint
    else:
        with pytest.raises(ValidationError):
            ProviderConfig.model_validate(raw)
