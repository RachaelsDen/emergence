from decimal import Decimal
from pathlib import Path

import pytest

from emergence.resources.reporting_parsers import decode_body, parse_minimax, parse_zai
from emergence.resources.schemas import ReportingError, ReportingIssue

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "reporting"


def test_zai_retains_only_defensible_percent_and_categories() -> None:
    body = decode_body((FIXTURES / "zai-synthetic.json").read_bytes())
    assert dict(parse_zai(body)) == {
        "TOKENS_LIMIT": Decimal("21.75"),
        "TIME_LIMIT": None,
        "FUTURE_LIMIT": None,
    }


@pytest.mark.parametrize("percent", [b"null", b"-1", b"101", b'"8"', b"true"])
def test_zai_invalid_percent_does_not_turn_into_capacity(percent: bytes) -> None:
    body = b'{"data":{"limits":[{"type":"TOKENS_LIMIT","percentage":' + percent + b"}]}}"
    assert parse_zai(decode_body(body)) == (("TOKENS_LIMIT", None),)


@pytest.mark.parametrize("bad", [b"HTML", b'{"p":NaN}', b'{"p":1,"p":2}', b'{"p":'])
def test_decoder_rejects_non_json_and_ambiguous_objects(bad: bytes) -> None:
    with pytest.raises(ReportingError) as error:
        decode_body(bad)
    assert error.value.issue is ReportingIssue.INVALID_JSON


def test_minimax_unverified_counters_produce_no_measurements() -> None:
    body = decode_body((FIXTURES / "minimax-synthetic-unverified.json").read_bytes())
    assert parse_minimax(body) is None
    with pytest.raises(ReportingError, match="invalid_schema"):
        parse_minimax({})
