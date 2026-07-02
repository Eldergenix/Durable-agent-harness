import numpy as np

from dah.providers.simulated import AccessibilityParams, SimulatedProvider
from dah.types import AssembledContext, ContextItem, Turn


def test_accessibility_decreases_with_length():
    p = AccessibilityParams()
    a_short = p.accessibility(total_tokens=100, rel=0.0)
    a_long = p.accessibility(total_tokens=3000, rel=0.0)
    assert a_short > a_long


def test_accessibility_is_u_shaped_in_position():
    p = AccessibilityParams()
    edge = p.accessibility(total_tokens=800, rel=0.0)
    middle = p.accessibility(total_tokens=800, rel=0.5)
    end = p.accessibility(total_tokens=800, rel=1.0)
    assert edge > middle and end > middle


def test_absent_fact_is_never_recalled():
    prov = SimulatedProvider(AccessibilityParams())
    ctx = AssembledContext()
    ctx.add(ContextItem(source="working", text="x", tokens=50, fact_ids=("present",)))
    q = Turn(index=0, session=0, kind="query", text="q", tokens=10, required_fact_ids=["absent"], query_id="q0")
    rng = np.random.default_rng(0)
    # over many draws, an absent fact never appears in the recalled set
    for _ in range(200):
        resp = prov.respond(q, ctx, rng)
        assert "absent" not in resp.recalled


def test_recall_probability_matches_empirical_frequency():
    prov = SimulatedProvider(AccessibilityParams())
    ctx = AssembledContext()
    ctx.add(ContextItem(source="working", text="x", tokens=200, fact_ids=("f",)))
    q = Turn(index=0, session=0, kind="query", text="q", tokens=10, required_fact_ids=["f"], query_id="q0")
    rng = np.random.default_rng(42)
    n = 4000
    hits = sum("f" in prov.respond(q, ctx, rng).recalled for _ in range(n))
    expected = prov.p.accessibility(200 + 10 + prov.p.token_overhead, 0.5)
    assert abs(hits / n - expected) < 0.03


def test_determinism_given_seed():
    prov = SimulatedProvider(AccessibilityParams())
    ctx = AssembledContext()
    ctx.add(ContextItem(source="working", text="x", tokens=100, fact_ids=("a", "b", "c")))
    q = Turn(index=0, session=0, kind="query", text="q", tokens=10, required_fact_ids=["a"], query_id="q0")
    r1 = prov.respond(q, ctx, np.random.default_rng(7)).recalled
    r2 = prov.respond(q, ctx, np.random.default_rng(7)).recalled
    assert r1 == r2
