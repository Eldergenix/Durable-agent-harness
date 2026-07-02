import numpy as np

from dah.env.tasks import TaskConfig, generate_suite, generate_task
from dah.env.multisession import MultiSessionEnv
from dah.providers.simulated import AccessibilityParams, SimulatedProvider
from dah.providers.base import ModelResponse


def test_required_facts_introduced_before_query():
    task = generate_task(TaskConfig(), seed=3)
    for turn in task.queries:
        for fid in turn.required_fact_ids:
            assert task.facts[fid].turn_introduced < turn.index


def test_cross_session_detection():
    task = generate_task(TaskConfig(p_cross_session=1.0), seed=5)
    prov = SimulatedProvider(AccessibilityParams(), topic_of={f: v.topic for f, v in task.facts.items()})
    env = MultiSessionEnv(task, prov, seed=5)
    # with p_cross_session=1.0, every query in sessions > 0 that *can* be cross
    # should be flagged when it depends on an earlier session
    cross = [env.is_cross_session(t) for t in task.queries if t.session > 0]
    assert any(cross)


def test_query_embedding_correlates_with_required_topics():
    task = generate_task(TaskConfig(embed_noise=0.05), seed=1)
    prov = SimulatedProvider(AccessibilityParams())
    env = MultiSessionEnv(task, prov, query_noise=0.05, seed=1)
    # the query embedding should be most similar to its own required facts
    for turn in task.queries[:20]:
        qe = env.query_embedding(turn)
        req_sim = np.mean([qe @ task.facts[f].embedding for f in turn.required_fact_ids])
        rand_ids = list(task.facts)[:10]
        rand_sim = np.mean([qe @ task.facts[f].embedding for f in rand_ids])
        assert req_sim > rand_sim


def test_reformulation_reduces_noise_toward_base():
    task = generate_task(TaskConfig(), seed=2)
    prov = SimulatedProvider(AccessibilityParams())
    env = MultiSessionEnv(task, prov, query_noise=0.8, reformulation_gain=0.5, seed=2)
    turn = task.queries[0]
    base = np.mean(np.stack([task.facts[f].embedding for f in turn.required_fact_ids]), axis=0)
    base = base / np.linalg.norm(base)
    sim0 = env.query_embedding(turn, reformulation=0) @ base
    sim2 = env.query_embedding(turn, reformulation=2) @ base
    assert sim2 >= sim0  # sharper query is closer to the true topic signal


def test_grade_requires_all_facts():
    task = generate_task(TaskConfig(), seed=4)
    prov = SimulatedProvider(AccessibilityParams())
    env = MultiSessionEnv(task, prov, seed=4)
    turn = next(t for t in task.queries if len(t.required_fact_ids) == 2)
    partial = ModelResponse(recalled={turn.required_fact_ids[0]})
    full = ModelResponse(recalled=set(turn.required_fact_ids))
    assert env.grade(turn, partial)[0] is False
    assert env.grade(turn, full)[0] is True


def test_suite_is_deterministic():
    a = generate_suite(TaskConfig(), n_tasks=5, base_seed=0)
    b = generate_suite(TaskConfig(), n_tasks=5, base_seed=0)
    assert [t.id for t in a] == [t.id for t in b]
    assert len(a[0].facts) == len(b[0].facts)


# ---------------------------------------------------------------------------
# Frozen GOAL.md metrics on EpisodeResult
# ---------------------------------------------------------------------------
def _qr(session, correct, answered=True, confident_wrong=False, tokens=100):
    from dah.env.multisession import QueryResult

    return QueryResult(
        query_id=f"q{session}",
        session=session,
        n_required=1,
        n_recalled_required=int(correct),
        correct=correct,
        recall_fraction=float(correct),
        cross_session=False,
        confident_wrong=confident_wrong,
        context_tokens=tokens,
        retrieval_performed=False,
        n_retrieved=0,
        tokens_spent=float(tokens),
        retries=0,
        rolled_back=False,
        answered=answered,
    )


def test_episode_metrics_definitions():
    from dah.env.multisession import EpisodeResult

    res = EpisodeResult(
        task_id="t",
        memory="none",
        results=[_qr(0, True), _qr(0, True), _qr(1, False, confident_wrong=True), _qr(2, True)],
        budget={"tokens": 900.0},
    )
    assert res.query_accuracy() == 0.75
    assert res.episode_success(threshold=0.8) is False
    assert res.episode_success(threshold=0.7) is True
    assert res.tokens_per_correct() == 300.0  # 900 / 3 correct
    assert res.hallucination_rate() == 0.25
    # per-session accuracy: s0=1.0, s1=0.0, s2=1.0 -> slope of OLS fit exists
    slope = res.degradation_slope()
    assert isinstance(slope, float)


def test_tokens_per_correct_undefined_when_nothing_correct():
    import math

    from dah.env.multisession import EpisodeResult

    res = EpisodeResult(task_id="t", memory="none", results=[_qr(0, False)], budget={"tokens": 100.0})
    assert math.isnan(res.tokens_per_correct())


def test_degradation_slope_detects_decay():
    from dah.env.multisession import EpisodeResult

    decaying = EpisodeResult(
        task_id="t",
        memory="none",
        results=[_qr(0, True), _qr(1, True), _qr(2, False), _qr(3, False)],
        budget={},
    )
    assert decaying.degradation_slope() < 0
