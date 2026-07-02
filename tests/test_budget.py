import pytest

from dah.budget import Budget, BudgetExceeded


def test_charge_and_remaining():
    b = Budget(max_tokens=100)
    b.charge(tokens=40)
    assert b.remaining("tokens") == 60
    assert b.remaining_fraction("tokens") == pytest.approx(0.6)
    assert b.pressure() == pytest.approx(0.4)


def test_unbounded_dimension_has_full_headroom():
    b = Budget(max_tokens=100)  # steps unbounded
    assert b.remaining("steps") == float("inf")
    assert b.remaining_fraction("steps") == 1.0
    # pressure only considers bounded dims
    b.charge(steps=10_000)
    assert b.pressure() == 0.0


def test_can_afford_is_nonmutating():
    b = Budget(max_tokens=100)
    assert b.can_afford(tokens=100) is True
    assert b.can_afford(tokens=101) is False
    assert b.tokens == 0  # unchanged


def test_charge_raises_when_over_limit():
    b = Budget(max_tokens=10)
    with pytest.raises(BudgetExceeded) as ei:
        b.charge(tokens=11)
    assert ei.value.dimension == "tokens"
    assert ei.value.spent == 11


def test_pressure_is_worst_case_across_dims():
    b = Budget(max_tokens=100, max_tool_calls=10)
    b.charge(tokens=10, tool_calls=9)  # 10% tokens, 90% tools
    assert b.pressure() == pytest.approx(0.9)


def test_snapshot_shape():
    b = Budget(max_tokens=100)
    b.charge(tokens=5, steps=2)
    snap = b.snapshot()
    assert snap["tokens"] == 5 and snap["steps"] == 2
