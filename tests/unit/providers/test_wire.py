"""Storage-safe counter boundaries without dispatching billable requests."""

import pytest
from pydantic import ValidationError

from emergence.inference.schemas import TokenUsage
from emergence.providers.wire import Completion


@pytest.mark.parametrize("count", [0, 2**63 - 1])
def test_wire_and_normalized_usage_accept_storage_boundaries(count: int) -> None:
    completion = Completion.model_validate(
        {
            "model": "configured-model",
            "choices": [{"message": {"content": "schema only"}}],
            "usage": {
                "prompt_tokens": count,
                "completion_tokens": count,
                "total_tokens": count,
                "prompt_tokens_details": {"cached_tokens": count},
                "completion_tokens_details": {"reasoning_tokens": count},
            },
        }
    )
    assert completion.usage is not None
    assert completion.usage.model_dump() == {
        "prompt_tokens": count,
        "completion_tokens": count,
        "total_tokens": count,
        "prompt_tokens_details": {"cached_tokens": count},
        "completion_tokens_details": {"reasoning_tokens": count},
    }
    normalized = TokenUsage(
        input_tokens=count,
        output_tokens=count,
        cached_input_tokens=count,
        reasoning_tokens=count,
    )
    assert normalized.complete
    assert normalized.model_dump() == {
        "input_tokens": count,
        "output_tokens": count,
        "cached_input_tokens": count,
        "reasoning_tokens": count,
    }


@pytest.mark.parametrize(
    "field", ["input_tokens", "output_tokens", "cached_input_tokens", "reasoning_tokens"]
)
@pytest.mark.parametrize("count", [2**63, -1, False, 0.5])
def test_normalized_usage_rejects_unstorable_counters(field: str, count: int | float) -> None:
    with pytest.raises(ValidationError) as failure:
        TokenUsage.model_validate({field: count})
    assert failure.value.errors()[0]["loc"] == (field,)
