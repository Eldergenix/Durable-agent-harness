"""Behavioral tests for dah.agents: determinism, budget stops, recovery value."""

import numpy as np

from dah.agents import Agent, BaselineAgent, HarnessAgent
from dah.budget import Budget
from dah.env.multisession import MultiSessionEnv
from dah.env.tasks import TaskConfig, generate_task
from dah.memory.manager import make_memory
from dah.planner import StaticPlanner
from dah.providers.simulated import AccessibilityParams, SimulatedProvider
from dah.verifier import ConsistencyVerifier


def _make(memory="full", seed=0):
    task = generate_task(TaskConfig(), seed=seed)
    topic = {f: v.topic for f, v in task.facts.items()}
    prov = SimulatedProvider(AccessibilityParams(), topic_of=topic)
    env = MultiSessionEnv(task, prov, seed=seed)
    mem = make_memory(memory, task.facts, seed=seed)
    return task, env, prov, mem


def test_agent_run_is_deterministic():
    def once():
        task, env, prov, mem = _make(seed=1)
        return Agent(mem, prov, StaticPlanner(), seed=1).run(env)

    r1, r2 = once(), once()
    assert [x.correct for x in r1.results] == [x.correct for x in r2.results]
    assert r1.query_accuracy() == r2.query_accuracy()


def test_none_memory_fails_all_cross_session_queries():
    task, env, prov, mem = _make(memory="none", seed=2)
    res = Agent(mem, prov, StaticPlanner(), seed=2).run(env)
    cross = [r for r in res.results if r.cross_session]
    assert cross  # there are cross-session queries
    assert all(not r.correct for r in cross)


def test_budget_exhaustion_marks_tail_unanswered():
    task, env, prov, mem = _make(memory="episodic", seed=3)
    tiny = Budget(max_tokens=2500)
    agent = Agent(mem, prov, StaticPlanner(k=8), budget=tiny, seed=3)
    res = agent.run(env)
    assert len(res.results) == len(task.queries)  # every query is accounted for
    assert any(not r.answered for r in res.results)  # some abandoned


def test_recovery_reduces_lost_ingestions():
    # same task/seed/budget; only the harness features differ
    task = generate_task(TaskConfig(), seed=4)
    topic = {f: v.topic for f, v in task.facts.items()}

    def run(harness: bool):
        prov = SimulatedProvider(AccessibilityParams(), topic_of=topic)
        env = MultiSessionEnv(task, prov, seed=4)
        mem = make_memory("episodic", task.facts, seed=4, overrides={"retrieval_k": 8})
        kw = dict(budget=Budget(max_tokens=40000), base_k=8, fault_rate=0.3, seed=4)
        cls = HarnessAgent if harness else BaselineAgent
        return cls(mem, prov, **kw).run(env)

    base = run(False)
    harn = run(True)
    # the harness should give up on far fewer ingestions thanks to retry+rollback
    assert harn.trace.get("ingest.giveup", 0) < base.trace.get("ingest.giveup", 0)
    # and its recoveries are recorded for the RecoveryRate metric
    assert harn.trace.get("ingest.recovered", 0) > 0


def test_strict_verifier_abstains_instead_of_guessing():
    # tau=1.0 makes the relevance check unsatisfiable -> every answer is
    # rejected -> an abstaining agent answers nothing and hallucinates nothing.
    task, env, prov, mem = _make(memory="episodic", seed=5)
    agent = Agent(
        mem,
        prov,
        StaticPlanner(k=5, max_retries=1, escalate_factor=2.0),
        verifier=ConsistencyVerifier(relevance_threshold=1.0),
        abstain_on_reject=True,
        seed=5,
    )
    res = agent.run(env)
    assert res.answered_rate() == 0.0
    assert res.hallucination_rate() == 0.0
    assert res.trace.get("query.abstain", 0) == len(task.queries)


def test_harness_beats_baseline_success_under_budget_and_failures():
    # This is an *aggregate* claim (single episodes are noisy): averaged over many
    # tasks, the full harness should improve QueryAccuracy and cut hallucination
    # versus the naive baseline, at the same budget and failure rate.
    from dataclasses import replace

    from dah.experiments import config as C
    from dah.experiments.runner import harness_ladder, paired_delta_ci

    cfg = replace(C.BASELINE, n_tasks=40)
    df = harness_ladder(C.TASKS, C.ACCESS, cfg, modes=["baseline", "full"])
    d, lo, hi, n = paired_delta_ci(df, "query_accuracy", "baseline", "full")
    assert n == 40
    assert d > 0 and lo > 0  # paired 95% CI excludes zero
    by = df.groupby("condition")["hallucination_rate"].mean()
    assert by["full"] < by["baseline"]
