"""Pure functions that turn configs into tidy result frames (T3.1).

Studies:

* :func:`memory_ablation` - hold the harness fixed (no failures, generous
  budget, static plan) and vary only the memory stack. Isolates *which memory
  components matter* (H2-H4).
* :func:`harness_ladder` / :func:`budget_sweep` - hold the memory fixed and
  switch harness features on cumulatively under a binding budget and injected
  failures. Isolates *what the control-plane machinery is worth* (H1).
* :func:`tau_sweep` - the verifier's operating point (relevance threshold).
* :func:`recovery_study` - recovery on/off across fault rates (H5).

Every returned DataFrame has one row per (condition, task) and always carries
``seed`` and ``config_id`` columns (R3), so any row can be traced to the exact
code path that produced it. Seed ``i`` generates the same task in every arm, so
all deltas are paired (R4); :func:`paired_delta_ci` implements the paired
bootstrap with >= 10,000 resamples.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from ..agents import Agent
from ..budget import Budget
from ..env.multisession import EpisodeResult, MultiSessionEnv
from ..env.tasks import TaskConfig, generate_suite
from ..faults import FaultConfig, FaultInjector
from ..memory.manager import make_memory
from ..planner import BudgetAwarePlanner, StaticPlanner
from ..providers.simulated import AccessibilityParams, SimulatedProvider
from ..recovery import RecoveryPolicy
from ..verifier import ConsistencyVerifier, NullVerifier
from . import config as C

N_BOOT_LEVEL = 4_000  # percentile bootstrap for per-arm level CIs
N_BOOT_DELTA = 10_000  # paired bootstrap for deltas (R4 floor: 10,000)


def _seed_for(task_index: int) -> int:
    return C.GLOBAL_SEED + task_index


def _episode_row(res: EpisodeResult, *, seed: int, config_id: str, **extra) -> dict:
    """Flatten one episode into a CSV row (frozen GOAL.md metrics + counters)."""
    tr = res.trace
    row = {
        "seed": seed,
        "config_id": config_id,
        "task_id": res.task_id,
        "memory": res.memory,
        # frozen metrics (GOAL.md definitions)
        "query_accuracy": res.query_accuracy(),
        "query_accuracy_cross": res.query_accuracy(cross=True),
        "query_accuracy_within": res.query_accuracy(cross=False),
        "episode_success": float(res.episode_success()),
        "tokens_per_correct": res.tokens_per_correct(),
        "hallucination_rate": res.hallucination_rate(),
        "degradation_slope": res.degradation_slope(),
        # secondary metrics
        "mean_recall": res.mean_recall(),
        "answered_rate": res.answered_rate(),
        "total_tokens": res.total_tokens(),
        "mean_ctx_tokens": res.mean_context_tokens(),
        "n_queries": len(res.results),
        # trace counters
        "rollbacks": tr.get("checkpoint.rollback", 0),
        "ingest_fail": tr.get("ingest.fail", 0),
        "ingest_giveup": tr.get("ingest.giveup", 0),
        "ingest_recovered": tr.get("ingest.recovered", 0),
        "verify_reject": tr.get("verify.reject", 0),
        "abstained": tr.get("query.abstain", 0),
        "budget_exhausted": tr.get("budget.exhausted", 0),
    }
    row.update(extra)
    return row


# ---------------------------------------------------------------------------
# Study 1: memory ablation (+ noise / context-fill sweeps)
# ---------------------------------------------------------------------------
def _run_memory_episode(
    task,
    task_index: int,
    memory_name: str,
    access: AccessibilityParams,
    query_noise: float,
    overrides: dict | None = None,
) -> tuple[EpisodeResult, int]:
    seed = _seed_for(task_index)
    topic = {f: v.topic for f, v in task.facts.items()}
    provider = SimulatedProvider(access, topic_of=topic)
    env = MultiSessionEnv(task, provider, query_noise=query_noise, seed=seed)
    memory = make_memory(memory_name, task.facts, seed=seed, overrides=overrides)
    agent = Agent(
        memory=memory,
        provider=provider,
        planner=StaticPlanner(),  # k=None -> use each preset's own retrieval policy
        verifier=NullVerifier(),
        budget=Budget(),  # unbounded: this study is about memory, not budget
        recovery=RecoveryPolicy.none(),
        use_checkpointing=False,
        fault_injector=None,  # no failures: this study is about memory, not recovery
        seed=seed,
    )
    return agent.run(env), seed


def memory_ablation(
    presets: list[str],
    task_cfg: TaskConfig,
    access: AccessibilityParams,
    n_tasks: int,
    query_noise: float | None = None,
    window_tokens: int | None = None,
    sweep: dict | None = None,
) -> pd.DataFrame:
    """Run each memory preset over the same task suite; one row per episode.

    ``window_tokens`` overrides the working-memory window (the context-pressure
    axis of the ablation matrix). ``sweep`` adds constant annotation columns
    (e.g. ``{"embed_noise": 0.5}``) for sweep studies.
    """
    suite = generate_suite(task_cfg, n_tasks=n_tasks, base_seed=0)
    qn = task_cfg.embed_noise if query_noise is None else query_noise
    overrides = {"working_window_tokens": window_tokens} if window_tokens else None
    win = window_tokens or 700
    rows = []
    for name in presets:
        for i, task in enumerate(suite):
            res, seed = _run_memory_episode(task, i, name, access, qn, overrides)
            cid = f"ablation:mem={name};win={win};noise={qn};fill={task_cfg.filler_tokens_per_turn}"
            rows.append(
                _episode_row(res, seed=seed, config_id=cid, condition=name, window=win, **(sweep or {}))
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Study 2: harness feature ladder
# ---------------------------------------------------------------------------
LADDER = ["baseline", "verify", "verify+recovery", "full"]


def _injector(rate: float, seed: int) -> FaultInjector | None:
    if rate <= 0.0:
        return None
    return FaultInjector(FaultConfig(ingest_fail_rate=rate), seed=seed + 7_919)


def _build_ladder_agent(
    mode: str,
    memory,
    provider,
    *,
    budget: Budget,
    cfg: C.BaselineExpConfig,
    seed: int,
) -> Agent:
    base_k = cfg.retrieval_k
    verifier = ConsistencyVerifier(relevance_threshold=cfg.verifier_tau)
    common = dict(
        memory=memory,
        provider=provider,
        budget=budget,
        fault_injector=_injector(cfg.fault_rate, seed),
        seed=seed,
    )

    if mode == "baseline":
        return Agent(
            planner=StaticPlanner(k=base_k, max_retries=0, escalate_factor=1.0),
            verifier=NullVerifier(),
            recovery=RecoveryPolicy.none(),
            use_checkpointing=False,
            abstain_on_reject=False,
            **common,
        )
    if mode == "verify":
        # verification loop: reject -> retry deeper + reformulated -> abstain if
        # still failing. No state recovery, no budget logic.
        return Agent(
            planner=StaticPlanner(k=base_k, max_retries=2, escalate_factor=2.0),
            verifier=verifier,
            recovery=RecoveryPolicy.none(),
            use_checkpointing=False,
            abstain_on_reject=True,
            **common,
        )
    if mode == "verify+recovery":
        # add checkpoint + rollback + ingestion retries on top of verification
        return Agent(
            planner=StaticPlanner(k=base_k, max_retries=2, escalate_factor=2.0),
            verifier=verifier,
            recovery=RecoveryPolicy.robust(),
            use_checkpointing=True,
            abstain_on_reject=True,
            **common,
        )
    if mode == "full":
        # add budget-aware planning (depth + context cap scale with budget)
        return Agent(
            planner=BudgetAwarePlanner(base_k=base_k, min_k=1, max_retries=2, escalate_factor=2.0),
            verifier=verifier,
            recovery=RecoveryPolicy.robust(),
            use_checkpointing=True,
            abstain_on_reject=True,
            **common,
        )
    raise ValueError(f"unknown ladder mode {mode!r}")


def harness_ladder(
    task_cfg: TaskConfig,
    access: AccessibilityParams,
    cfg: C.BaselineExpConfig,
    modes: list[str] | None = None,
) -> pd.DataFrame:
    """Switch harness features on cumulatively; same memory/budget/faults per arm."""
    modes = modes or LADDER
    suite = generate_suite(task_cfg, n_tasks=cfg.n_tasks, base_seed=0)
    rows = []
    for mode in modes:
        for i, task in enumerate(suite):
            seed = _seed_for(i)
            topic = {f: v.topic for f, v in task.facts.items()}
            provider = SimulatedProvider(access, topic_of=topic)
            env = MultiSessionEnv(task, provider, query_noise=task_cfg.embed_noise, seed=seed)
            memory = make_memory(
                cfg.memory_preset, task.facts, seed=seed, overrides={"retrieval_k": cfg.retrieval_k}
            )
            budget = Budget(max_tokens=cfg.max_tokens)
            agent = _build_ladder_agent(mode, memory, provider, budget=budget, cfg=cfg, seed=seed)
            cid = f"ladder:mode={mode};budget={cfg.max_tokens};pfail={cfg.fault_rate};tau={cfg.verifier_tau}"
            rows.append(_episode_row(agent.run(env), seed=seed, config_id=cid, condition=mode))
    return pd.DataFrame(rows)


def budget_sweep(
    task_cfg: TaskConfig,
    access: AccessibilityParams,
    cfg: C.BaselineExpConfig,
    budgets: list[int],
    modes: list[str] | None = None,
) -> pd.DataFrame:
    """Quality-vs-budget curve for each ladder step (the headline figure)."""
    modes = modes or LADDER
    frames = []
    for b in budgets:
        sub = replace(cfg, max_tokens=b)
        df = harness_ladder(task_cfg, access, sub, modes=modes)
        df["max_tokens"] = b
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def tau_sweep(
    task_cfg: TaskConfig,
    access: AccessibilityParams,
    cfg: C.BaselineExpConfig,
    taus: list[float],
) -> pd.DataFrame:
    """Sweep the ConsistencyVerifier operating point (relevance threshold)."""
    frames = []
    for tau in taus:
        sub = replace(cfg, verifier_tau=tau)
        df = harness_ladder(task_cfg, access, sub, modes=["verify+recovery"])
        df["tau"] = tau
        df["condition"] = f"tau={tau:.2f}"
        df["config_id"] = df["config_id"].str.replace("mode=verify+recovery", f"tau-sweep={tau:.2f}", regex=False)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Study 3: failure recovery (H5)
# ---------------------------------------------------------------------------
def recovery_study(
    task_cfg: TaskConfig,
    access: AccessibilityParams,
    cfg: C.BaselineExpConfig,
    fault_rates: list[float],
) -> pd.DataFrame:
    """Recovery on/off at several fault rates; everything else identical.

    Both arms use the same static planner and no query verifier, so the *only*
    difference is checkpoint + rollback + ingestion retries. This isolates H5.
    """
    suite = generate_suite(task_cfg, n_tasks=cfg.n_tasks, base_seed=0)
    rows = []
    for rate in fault_rates:
        for arm, recovery, checkpointing in (
            ("recovery_off", RecoveryPolicy.none(), False),
            ("recovery_on", RecoveryPolicy.robust(), True),
        ):
            for i, task in enumerate(suite):
                seed = _seed_for(i)
                topic = {f: v.topic for f, v in task.facts.items()}
                provider = SimulatedProvider(access, topic_of=topic)
                env = MultiSessionEnv(task, provider, query_noise=task_cfg.embed_noise, seed=seed)
                memory = make_memory(
                    cfg.memory_preset, task.facts, seed=seed, overrides={"retrieval_k": cfg.retrieval_k}
                )
                agent = Agent(
                    memory=memory,
                    provider=provider,
                    planner=StaticPlanner(k=cfg.retrieval_k, max_retries=0, escalate_factor=1.0),
                    verifier=NullVerifier(),
                    budget=Budget(max_tokens=cfg.max_tokens),
                    recovery=recovery,
                    use_checkpointing=checkpointing,
                    fault_injector=_injector(rate, seed),
                    abstain_on_reject=False,
                    seed=seed,
                )
                cid = f"recovery:arm={arm};pfail={rate};budget={cfg.max_tokens}"
                rows.append(
                    _episode_row(agent.run(env), seed=seed, config_id=cid, condition=arm, fault_rate=rate)
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------
def bootstrap_ci(values: np.ndarray, n_boot: int = N_BOOT_LEVEL, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI for a mean *level*. Returns (mean, lo, hi)."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(values.mean()), float(lo), float(hi)


def paired_delta_ci(
    df: pd.DataFrame,
    metric: str,
    cond_a: str,
    cond_b: str,
    by: str = "seed",
    cond_col: str = "condition",
    n_boot: int = N_BOOT_DELTA,
    alpha: float = 0.05,
    seed: int = 0,
):
    """Paired bootstrap CI for mean(``cond_b`` - ``cond_a``) matched on ``by`` (R4).

    Rows are matched per task seed, so the delta distribution is over *tasks*,
    not over independent samples - this is the pairing GOAL.md requires. Returns
    ``(mean_delta, lo, hi, n_pairs)``.
    """
    a = df[df[cond_col] == cond_a].set_index(by)[metric]
    b = df[df[cond_col] == cond_b].set_index(by)[metric]
    joined = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if joined.empty:
        return float("nan"), float("nan"), float("nan"), 0
    deltas = (joined["b"] - joined["a"]).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(deltas), size=(n_boot, len(deltas)))
    means = deltas[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(deltas.mean()), float(lo), float(hi), int(len(deltas))


def summarize(df: pd.DataFrame, metrics: list[str], by: str = "condition") -> pd.DataFrame:
    """Mean + bootstrap CI for each metric, grouped by ``by``, preserving order."""
    rows = []
    for cond, g in df.groupby(by, sort=False):
        row = {by: cond, "n": len(g)}
        for m in metrics:
            mean, lo, hi = bootstrap_ci(g[m].to_numpy())
            row[f"{m}"] = mean
            row[f"{m}_lo"] = lo
            row[f"{m}_hi"] = hi
        rows.append(row)
    return pd.DataFrame(rows)
