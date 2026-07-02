"""Governance tests (T0.4): the rules that must hold *behaviorally*.

The static linter (scripts/rules_lint.py) greps for forbidden patterns; these
tests enforce the same contracts dynamically:

* R5  - no answer-key leakage into decision-making code (token scan + the
        import surface of every module stays importable without grading access)
* R6  - snapshot totality: restore reproduces bit-identical behavior incl. RNG
* R7  - budget conservation: spend is monotone, no refunds, no negative charges
* R1  - determinism: seeded end-to-end; zero-rate fault injector is a no-op
* R11 - every library module is importable (the dotted names below are also the
        references the linter looks for)
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src" / "dah"

# R11: the canonical module inventory. Adding a module without listing it here
# (or in another test) fails scripts/rules_lint.py.
ALL_MODULES = [
    "dah.types",
    "dah.telemetry",
    "dah.budget",
    "dah.checkpoint",
    "dah.verifier",
    "dah.recovery",
    "dah.planner",
    "dah.faults",
    "dah.agents",
    "dah.memory.stores",
    "dah.memory.manager",
    "dah.env.tasks",
    "dah.env.multisession",
    "dah.providers.base",
    "dah.providers.simulated",
    "dah.providers.openai_compat",
    "dah.experiments.config",
    "dah.experiments.runner",
    "dah.experiments.run_ablation",
    "dah.experiments.run_baseline",
    "dah.experiments.analyze",
    "dah.experiments.live_check",
    "dah.experiments.gallery",
]


def test_every_module_imports():
    for mod in ALL_MODULES:
        importlib.import_module(mod)


def test_module_inventory_is_complete():
    # if a new module appears on disk it must be added to ALL_MODULES
    on_disk = {
        ".".join(p.relative_to(SRC.parent).with_suffix("").parts)
        for p in SRC.rglob("*.py")
        if p.name != "__init__.py"
    }
    assert on_disk == set(ALL_MODULES)


# ---------------------------------------------------------------------------
# R5: answer-key leakage
# ---------------------------------------------------------------------------
LEAKAGE_TOKENS = ("required_fact_ids", "confident_wrong")
LEAKAGE_PATHS = ("memory", "planner.py", "verifier.py", "recovery.py", "agents.py")


def test_no_answer_key_tokens_in_decision_code():
    offenders = []
    for path in SRC.rglob("*.py"):
        rel = path.relative_to(SRC)
        if not any(str(rel).startswith(p) for p in LEAKAGE_PATHS):
            continue
        text = path.read_text(encoding="utf-8")
        for tok in LEAKAGE_TOKENS:
            if tok in text:
                offenders.append(f"{rel}: contains '{tok}'")
    assert not offenders, offenders


# ---------------------------------------------------------------------------
# R7: budget conservation
# ---------------------------------------------------------------------------
def test_budget_spend_is_monotone_and_rejects_negative():
    from dah.budget import Budget

    b = Budget(max_tokens=100)
    b.charge(tokens=10)
    with pytest.raises(ValueError):
        b.charge(tokens=-1)
    assert b.tokens == 10  # unchanged after the rejected charge
    # there is no refund API at all
    assert not any("refund" in name for name in dir(b))


# ---------------------------------------------------------------------------
# R6: snapshot totality (including RNG state)
# ---------------------------------------------------------------------------
def test_compactor_snapshot_restores_rng_draws():
    from dah.memory.stores import Compactor, WItem

    comp = Compactor(recall=0.5, tokens_per_fact=5, base_tokens=10, seed=3)
    comp.ingest([WItem("working", "t", 10, (f"a{i}",)) for i in range(20)])
    snap = comp.snapshot()
    comp.ingest([WItem("working", "t", 10, (f"b{i}",)) for i in range(50)])
    after_first = list(comp.retained)
    comp.restore(snap)
    comp.ingest([WItem("working", "t", 10, (f"b{i}",)) for i in range(50)])
    assert comp.retained == after_first  # identical draws after restore


def test_fault_injector_snapshot_restores_rng_draws():
    from dah.faults import FaultConfig, FaultInjector

    inj = FaultInjector(FaultConfig(ingest_fail_rate=0.5), seed=9)
    _ = [inj.ingest_fails() for _ in range(10)]
    snap = inj.snapshot()
    seq1 = [inj.ingest_fails() for _ in range(30)]
    inj.restore(snap)
    seq2 = [inj.ingest_fails() for _ in range(30)]
    assert seq1 == seq2


def test_memory_manager_rollback_is_total():
    from dah.checkpoint import Checkpointer
    from dah.env.tasks import TaskConfig, generate_task
    from dah.memory.manager import make_memory

    task = generate_task(TaskConfig(), seed=6)
    mem = make_memory("full", task.facts, seed=6)
    cp = Checkpointer()
    cp.register("memory", mem)
    mem.start_session(0)
    for turn in task.sessions[0]:
        if turn.kind == "info":
            mem.add_info(turn)
    cp.commit("mid")
    before = (list(mem.working.items), list(mem.episodic.ids), list(mem.compactor.retained))
    mem.ingest_poison(500)
    mem.start_session(1)
    cp.rollback()
    after = (list(mem.working.items), list(mem.episodic.ids), list(mem.compactor.retained))
    assert repr(before) == repr(after)


# ---------------------------------------------------------------------------
# R1: determinism / fault-injection identity
# ---------------------------------------------------------------------------
def _run_episode(fault_injector):
    from dah.agents import Agent
    from dah.env.multisession import MultiSessionEnv
    from dah.env.tasks import TaskConfig, generate_task
    from dah.memory.manager import make_memory
    from dah.planner import StaticPlanner
    from dah.providers.simulated import AccessibilityParams, SimulatedProvider

    task = generate_task(TaskConfig(n_sessions=3, turns_per_session=8), seed=11)
    topic = {f: v.topic for f, v in task.facts.items()}
    prov = SimulatedProvider(AccessibilityParams(), topic_of=topic)
    env = MultiSessionEnv(task, prov, seed=11)
    mem = make_memory("episodic", task.facts, seed=11)
    agent = Agent(mem, prov, StaticPlanner(k=4), fault_injector=fault_injector, seed=11)
    return agent.run(env)


def test_zero_rate_injector_is_bit_identical_to_none():
    import dataclasses

    from dah.faults import FaultConfig, FaultInjector

    res_none = _run_episode(None)
    res_zero = _run_episode(FaultInjector(FaultConfig(ingest_fail_rate=0.0), seed=99))
    assert [dataclasses.asdict(r) for r in res_none.results] == [
        dataclasses.asdict(r) for r in res_zero.results
    ]
    assert res_none.budget == res_zero.budget


def test_verifier_is_deterministic():
    from dah.verifier import ConsistencyVerifier

    rng = np.random.default_rng(0)
    q = rng.standard_normal(16)
    q /= np.linalg.norm(q)
    embs = rng.standard_normal((5, 16))
    v = ConsistencyVerifier(relevance_threshold=0.4)
    verdicts = {(v.verify(q, embs).ok, v.verify(q, embs).reason) for _ in range(20)}
    assert len(verdicts) == 1  # never flips on identical inputs
    assert not v.verify(q, None).ok  # empty recall is always rejected


# ---------------------------------------------------------------------------
# Telemetry (dah.telemetry) basic contract
# ---------------------------------------------------------------------------
def test_trace_counts_and_sums():
    from dah.telemetry import Trace

    tr = Trace()
    tr.emit("x", cost=2)
    tr.emit("x", cost=3)
    tr.emit("y")
    assert tr.count("x") == 2
    assert tr.sum("x", "cost") == 5.0
    assert tr.last("x").data["cost"] == 3
    assert tr.summary() == {"x": 2, "y": 1}


# ---------------------------------------------------------------------------
# Recovery policy (dah.recovery) contract
# ---------------------------------------------------------------------------
def test_recovery_policy_presets():
    from dah.recovery import RecoveryPolicy

    none = RecoveryPolicy.none()
    robust = RecoveryPolicy.robust()
    assert none.ingestion_retries == 0 and not none.rollback_on_failure
    assert robust.ingestion_retries > 0 and robust.rollback_on_failure


# ---------------------------------------------------------------------------
# Results schema (R3) - checked when artifacts exist
# ---------------------------------------------------------------------------
def test_results_csvs_carry_seed_and_config_id():
    results = REPO / "results"
    csvs = sorted(results.glob("*.csv")) if results.is_dir() else []
    if not csvs:
        pytest.skip("no results generated yet")
    for csv in csvs:
        header = csv.read_text(encoding="utf-8").splitlines()[0].split(",")
        assert "seed" in header and "config_id" in header, f"{csv.name} missing R3 columns"


# ---------------------------------------------------------------------------
# Cursor hooks (T0.3): each guard script, fed a sample stdin payload, must
# return the documented JSON decision. These run the real scripts the way the
# hook runner does, so a regression in the guards fails the suite (R13/R15).
# ---------------------------------------------------------------------------
import json  # noqa: E402
import os  # noqa: E402
import stat  # noqa: E402
import subprocess  # noqa: E402

HOOKS = REPO / ".cursor" / "hooks"


def _run_hook(script: str, payload: dict | str, env: dict | None = None) -> dict:
    path = HOOKS / script
    assert path.exists(), f"hook script missing: {path}"
    assert path.stat().st_mode & stat.S_IXUSR, f"hook script not executable: {path}"
    data = payload if isinstance(payload, str) else json.dumps(payload)
    proc = subprocess.run(
        [str(path)],
        input=data,
        capture_output=True,
        text=True,
        cwd=REPO,
        env={**os.environ, **(env or {})},
        timeout=60,
    )
    assert proc.returncode == 0, f"{script} exited {proc.returncode}: {proc.stderr}"
    return json.loads(proc.stdout)


def test_hook_governance_denies_frozen_file_edit():
    out = _run_hook(
        "protect-governance.sh",
        {"tool_name": "Write", "tool_input": {"path": str(REPO / "RULES.md")}},
    )
    assert out["permission"] == "deny"
    assert "R13" in out["user_message"]


def test_hook_governance_denies_results_edit_and_hook_edit():
    for target in ("results/ablation_main.csv", ".cursor/hooks/shell-guard.sh"):
        out = _run_hook(
            "protect-governance.sh",
            {"tool_name": "StrReplace", "tool_input": {"path": str(REPO / target)}},
        )
        assert out["permission"] == "deny", target


def test_hook_governance_allows_ordinary_edits_and_reads():
    # ordinary source edit -> allow
    out = _run_hook(
        "protect-governance.sh",
        {"tool_name": "Write", "tool_input": {"path": str(REPO / "src/dah/budget.py")}},
    )
    assert out["permission"] == "allow"
    # non-mutating tool -> allow even on a frozen path
    out = _run_hook(
        "protect-governance.sh",
        {"tool_name": "Read", "tool_input": {"path": str(REPO / "RULES.md")}},
    )
    assert out["permission"] == "allow"


def test_hook_governance_bootstrap_escape_hatch():
    out = _run_hook(
        "protect-governance.sh",
        {"tool_name": "Write", "tool_input": {"path": str(REPO / "GOAL.md")}},
        env={"DAH_BOOTSTRAP": "1"},
    )
    assert out["permission"] == "allow"


def test_hook_shell_guard_denies_no_verify_and_force_push():
    out = _run_hook("shell-guard.sh", {"command": "git commit --no-verify -m x"})
    assert out["permission"] == "deny"
    out = _run_hook("shell-guard.sh", {"command": "git push --force origin main"})
    assert out["permission"] == "deny"
    # --force-with-lease is the sanctioned variant: not hard-denied
    out = _run_hook("shell-guard.sh", {"command": "git push --force-with-lease origin dev"})
    assert out["permission"] == "allow"


def test_hook_shell_guard_asks_on_results_redirect_and_git_config():
    out = _run_hook("shell-guard.sh", {"command": "echo hacked > results/ablation_main.csv"})
    assert out["permission"] == "ask"
    out = _run_hook("shell-guard.sh", {"command": "git config user.email evil@example.com"})
    assert out["permission"] == "ask"
    # the documented setup command stays frictionless
    out = _run_hook("shell-guard.sh", {"command": "git config core.hooksPath .githooks"})
    assert out["permission"] == "allow"


def test_hook_shell_guard_allows_ordinary_commands():
    for cmd in ("ls -la", "python scripts/quality_gate.py --fast", "git status", "git config --list"):
        out = _run_hook("shell-guard.sh", {"command": cmd})
        assert out["permission"] == "allow", cmd


def test_hook_stop_quality_no_action_when_skipped():
    out = _run_hook("stop-quality.sh", {"status": "completed", "loop_count": 0}, env={"DAH_SKIP_GATE": "1"})
    assert out == {}


def test_hook_lint_context_ignores_non_python_paths():
    out = _run_hook(
        "lint-context.sh",
        {"tool_name": "Write", "tool_input": {"path": str(REPO / "docs/RESULTS.md")}},
    )
    assert out == {}
