"""Bit-for-bit reproducibility (R1) and paired-statistics sanity (R4)."""

import numpy as np
import pandas as pd

from dah.experiments import config as C
from dah.experiments.runner import (
    N_BOOT_DELTA,
    harness_ladder,
    memory_ablation,
    paired_delta_ci,
    recovery_study,
)


def test_memory_ablation_reproducible():
    df1 = memory_ablation(["none", "episodic"], C.TASKS, C.ACCESS, n_tasks=6)
    df2 = memory_ablation(["none", "episodic"], C.TASKS, C.ACCESS, n_tasks=6)
    pd.testing.assert_frame_equal(df1, df2)


def test_harness_ladder_reproducible():
    from dataclasses import replace

    cfg = replace(C.BASELINE, n_tasks=6)
    df1 = harness_ladder(C.TASKS, C.ACCESS, cfg)
    df2 = harness_ladder(C.TASKS, C.ACCESS, cfg)
    pd.testing.assert_frame_equal(df1, df2)


def test_recovery_study_reproducible():
    from dataclasses import replace

    cfg = replace(C.BASELINE, n_tasks=4)
    df1 = recovery_study(C.TASKS, C.ACCESS, cfg, [0.2])
    df2 = recovery_study(C.TASKS, C.ACCESS, cfg, [0.2])
    pd.testing.assert_frame_equal(df1, df2)


def test_rows_carry_seed_and_config_id():
    df = memory_ablation(["none"], C.TASKS, C.ACCESS, n_tasks=3)
    assert "seed" in df.columns and "config_id" in df.columns
    assert df["seed"].nunique() == 3  # one distinct seed per task
    assert df["config_id"].str.startswith("ablation:").all()


def test_none_is_worst_on_cross_session():
    df = memory_ablation(["none", "compaction", "episodic", "full"], C.TASKS, C.ACCESS, n_tasks=30)
    by = df.groupby("condition")["query_accuracy_cross"].mean()
    assert by["none"] < by["episodic"]
    assert by["none"] < by["compaction"]
    assert by["none"] < by["full"]


def test_paired_delta_ci_matches_hand_computation():
    df = pd.DataFrame(
        {
            "condition": ["a"] * 4 + ["b"] * 4,
            "seed": [1, 2, 3, 4] * 2,
            "m": [0.1, 0.2, 0.3, 0.4, 0.3, 0.4, 0.5, 0.6],
        }
    )
    d, lo, hi, n = paired_delta_ci(df, "m", "a", "b")
    assert n == 4
    assert np.isclose(d, 0.2)
    assert lo <= d <= hi
    # constant per-pair delta => degenerate CI collapses onto the mean
    assert np.isclose(lo, 0.2) and np.isclose(hi, 0.2)


def test_delta_bootstrap_meets_r4_floor():
    assert N_BOOT_DELTA >= 10_000
