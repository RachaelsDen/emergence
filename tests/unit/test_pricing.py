from decimal import Decimal, localcontext

import pytest

from emergence.resources.pricing import Rates


@pytest.mark.parametrize(
    ("cached", "expected"),
    [
        (None, "0.000008000000000003"),
        (0, "0.000008000000000003"),
        (2, "0.000006500000000001"),
    ],
)
def test_cache_replaces_input_only_and_prices_remain_exact(
    cached: int | None, expected: str
) -> None:
    price = Rates(Decimal("1.000000000001"), Decimal("2.5"), Decimal("0.25"))
    with localcontext(prec=4):
        assert price.cost(3, 2, cached) == Decimal(expected)


def test_absent_cache_rate_charges_cache_at_full_input_price() -> None:
    assert Rates(Decimal("4"), Decimal("1"), None).cost(10, 2, 10) == Decimal("0.000042")
