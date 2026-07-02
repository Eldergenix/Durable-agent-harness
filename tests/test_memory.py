import numpy as np

from dah.budget import Budget
from dah.env.tasks import TaskConfig, generate_task
from dah.memory.manager import MemoryConfig, MemoryManager, make_memory
from dah.memory.stores import Compactor, WItem, WorkingMemory
from dah.types import Turn


def _task():
    return generate_task(TaskConfig(), seed=11)


def test_working_memory_evicts_oldest_over_window():
    wm = WorkingMemory(window_tokens=100)
    wm.add(WItem("working", "a", 60, ("f0",)))
    evicted = wm.add(WItem("working", "b", 60, ("f1",)))
    assert [it.fact_ids for it in evicted] == [("f0",)]
    assert wm.fact_ids() == {"f1"}


def test_compaction_is_lossy_but_smaller():
    comp = Compactor(recall=0.5, tokens_per_fact=5, base_tokens=10, seed=0)
    items = [WItem("working", "t", 80, (f"f{i}",)) for i in range(200)]
    comp.ingest(items)
    # ~50% retained
    assert 70 <= len(comp.retained) <= 130
    # summary is far smaller than the 200*80 tokens of raw turns
    assert comp.tokens() < 200 * 80


def test_compaction_recall_one_keeps_all():
    comp = Compactor(recall=1.0, tokens_per_fact=5, base_tokens=10, seed=1)
    comp.ingest([WItem("working", "t", 10, (f"f{i}",)) for i in range(50)])
    assert len(comp.retained) == 50


def test_semantic_salience_gating():
    task = _task()
    mem = make_memory("semantic", task.facts, seed=0)
    # feed all info turns
    for session in task.sessions:
        mem.start_session(session[0].session)
        for turn in session:
            if turn.kind == "info":
                mem.add_info(turn)
    thr = mem.config.sem_salience_threshold
    for fid in mem.semantic.ids:
        assert task.facts[fid].salience >= thr


def test_retrieval_is_appended_last_for_recency():
    task = _task()
    mem = make_memory("episodic", task.facts, seed=0, overrides={"retrieval_k": 3})
    mem.start_session(0)
    for turn in task.sessions[0]:
        if turn.kind == "info":
            mem.add_info(turn)
    # a query embedding aimed at an early fact
    fid = task.sessions[0][0].fact_ids[0]
    qe = task.facts[fid].embedding
    ctx, stats = mem.assemble(qe, query_tokens=10, budget=Budget())
    if stats.n_retrieved:
        assert ctx.items[-1].source == "retrieved"


def test_context_cap_trims_retrieved_first():
    task = _task()
    mem = make_memory("full", task.facts, seed=0, overrides={"retrieval_k": 10})
    mem.start_session(0)
    for turn in task.sessions[0]:
        if turn.kind == "info":
            mem.add_info(turn)
    qe = task.facts[task.sessions[0][0].fact_ids[0]].embedding
    ctx, stats = mem.assemble(qe, query_tokens=10, budget=Budget(), context_cap=400)
    assert ctx.total_tokens + 10 <= 400


def test_snapshot_restore_roundtrip():
    task = _task()
    mem = make_memory("full", task.facts, seed=0)
    mem.start_session(0)
    for turn in task.sessions[0]:
        if turn.kind == "info":
            mem.add_info(turn)
    snap = mem.snapshot()
    n_working = len(mem.working.items)
    n_epi = len(mem.episodic.ids)
    # mutate
    mem.working.items.clear()
    mem.episodic.ids.clear()
    mem.restore(snap)
    assert len(mem.working.items) == n_working
    assert len(mem.episodic.ids) == n_epi


def test_start_session_resets_working_memory():
    task = _task()
    mem = make_memory("episodic", task.facts, seed=0)
    mem.start_session(0)
    for turn in task.sessions[0]:
        if turn.kind == "info":
            mem.add_info(turn)
    assert len(mem.working.items) > 0
    persisted = len(mem.episodic.ids)
    mem.start_session(1)
    assert len(mem.working.items) == 0  # working cleared
    assert len(mem.episodic.ids) == persisted  # episodic persists


def test_none_preset_has_no_cross_session_memory():
    task = _task()
    mem = make_memory("none", task.facts, seed=0)
    mem.start_session(0)
    for turn in task.sessions[0]:
        if turn.kind == "info":
            mem.add_info(turn)
    mem.start_session(1)
    qe = task.facts[task.sessions[0][0].fact_ids[0]].embedding
    ctx, stats = mem.assemble(qe, query_tokens=10, budget=Budget())
    assert stats.n_retrieved == 0
    assert ctx.total_tokens == 0  # nothing carried across the session boundary
