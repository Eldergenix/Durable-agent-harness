"""Single source of truth for experiment settings.

Everything that affects a number in ``docs/RESULTS.md`` is here, so a reader can
audit or change the study in one place. All randomness downstream is seeded from
:data:`GLOBAL_SEED` plus per-task offsets - seed ``i`` generates the *same* task
for every arm (paired design, R4) - so results are bit-for-bit reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..env.tasks import TaskConfig
from ..providers.simulated import AccessibilityParams

GLOBAL_SEED = 20240601

# The provider's context-degradation model, held fixed across every condition.
ACCESS = AccessibilityParams(l_half=1200.0, gamma=1.5, m_mid=0.60, confusion_rate=0.5)

# The default task distribution for the main ablation.
TASKS = TaskConfig(
    n_sessions=6,
    turns_per_session=14,
    query_fraction=0.35,
    n_topics=12,
    embed_dim=32,
    embed_noise=0.35,
    facts_per_info_turn=2,
    fact_tokens=20,
    filler_tokens_per_turn=55,
    query_tokens=15,
    max_required_facts=2,
    p_cross_session=0.55,
)

N_TASKS = 120  # paired seeds per arm (R4 floor is 20; we run 120 for tight CIs)

# ---------------------------------------------------------------------------
# Study 1: memory ablation
# ---------------------------------------------------------------------------
# The ablation matrix: 8 memory configurations x 2 context-pressure levels
# (working-window sizes). H2 is evaluated from the leave-one-out subset of this
# grid (full minus one component at a time).
ABLATION_PRESETS = [
    "none",
    "compaction",
    "episodic",
    "semantic",
    "episodic+semantic",
    "compaction+episodic",
    "compaction+semantic",
    "full",
]
WINDOW_LEVELS = [700, 450]  # working-memory tokens: default and tight (H2: <= 800)

# Leave-one-out mapping for H2: full minus each single component.
LOO_ARMS = {
    "compaction": "episodic+semantic",
    "episodic": "compaction+semantic",
    "semantic": "compaction+episodic",
}

# Retrieval-quality sweep (H4 + the negative-result study). ``semantic`` and
# ``episodic+semantic`` are in the sweep so H4's crossover claim is testable.
NOISE_SWEEP = [0.2, 0.35, 0.5, 0.7, 0.9, 1.2]
NOISE_SWEEP_PRESETS = ["none", "compaction", "episodic", "semantic", "episodic+semantic", "full"]

# Context-fill sweep (long-horizon pressure axis).
FILL_SWEEP = [25, 55, 90, 140]
FILL_SWEEP_PRESETS = ["none", "compaction", "episodic", "full"]

# Robustness re-run (T4.2): re-check the headline memory ordering on two harder
# task distributions - a longer horizon (10 sessions) and a cross-session-heavy
# mix (85% of queries depend on an earlier session). Conclusions that hold only
# at the default setting are knife-edge and must be called out in RESULTS.md.
ROBUSTNESS_PRESETS = ["none", "compaction", "episodic", "full"]
ROBUSTNESS_SETTINGS: dict[str, dict] = {
    "longer-horizon": {"n_sessions": 10},
    "cross-heavy": {"p_cross_session": 0.85},
}

# ---------------------------------------------------------------------------
# Study 2: harness value (feature ladder, budget sweep, verifier operating
# point, failure recovery)
# ---------------------------------------------------------------------------


@dataclass
class BaselineExpConfig:
    """Settings for the harness-feature ladder (baseline -> full harness)."""

    memory_preset: str = "episodic"
    retrieval_k: int = 8
    max_tokens: int = 18000  # a binding token budget (the baseline over-spends it)
    fault_rate: float = 0.12  # transient ingestion-failure rate
    verifier_tau: float = 0.35  # ConsistencyVerifier relevance threshold
    n_tasks: int = 120


BASELINE = BaselineExpConfig()

# Token-budget sweep: the headline figure (quality vs budget for each ladder step).
BUDGET_SWEEP = [11000, 13000, 15000, 17000, 19000, 21000, 24000]

# Verifier operating-point sweep: relevance threshold tau of the (oracle-free)
# ConsistencyVerifier. tau=0 only rejects empty recall; higher tau rejects more.
TAU_SWEEP = [0.0, 0.2, 0.35, 0.5, 0.65]

# Failure-recovery study (H5): fault rates x {recovery off, recovery on},
# everything else identical (static plan, no verifier interference).
RECOVERY_FAULT_RATES = [0.05, 0.12, 0.25]
