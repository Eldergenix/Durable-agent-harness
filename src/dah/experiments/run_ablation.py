"""Study 1 driver (T3.3): the memory ablation matrix + noise and fill sweeps.

Run with::

    python -m dah.experiments.run_ablation          # full study
    python -m dah.experiments.run_ablation --quick  # fast smoke (few tasks)

Writes raw per-episode CSVs into ``results/`` (with ``seed`` and ``config_id``
columns per R3). Aggregation + figures are done by ``dah.experiments.analyze``.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import replace

import pandas as pd

from . import config as C
from .runner import memory_ablation, summarize

RESULTS_DIR = os.path.join(os.getcwd(), "results")

MAIN_METRICS = ["query_accuracy", "query_accuracy_cross", "query_accuracy_within", "mean_ctx_tokens"]


def main(argv: list[str] | None = None) -> None:
    """Run the ablation matrix (presets x window levels) and both sweeps."""
    ap = argparse.ArgumentParser(description="Memory ablation study")
    ap.add_argument("--quick", action="store_true", help="use few tasks for a fast smoke run")
    ap.add_argument("--n-tasks", type=int, default=None)
    ap.add_argument("--out", type=str, default=RESULTS_DIR)
    args = ap.parse_args(argv)

    n_tasks = args.n_tasks or (12 if args.quick else C.N_TASKS)
    os.makedirs(args.out, exist_ok=True)
    t0 = time.perf_counter()

    # -- ablation matrix: presets x context-pressure (window) levels -------
    print(
        f"[ablation] matrix: {len(C.ABLATION_PRESETS)} presets x "
        f"{len(C.WINDOW_LEVELS)} window levels x {n_tasks} tasks"
    )
    frames = []
    for win in C.WINDOW_LEVELS:
        df = memory_ablation(C.ABLATION_PRESETS, C.TASKS, C.ACCESS, n_tasks=n_tasks, window_tokens=win)
        frames.append(df)
    main_df = pd.concat(frames, ignore_index=True)
    main_df.to_csv(os.path.join(args.out, "ablation_main.csv"), index=False)
    for win in C.WINDOW_LEVELS:
        print(f"\n  window={win} tokens:")
        print(
            summarize(main_df[main_df["window"] == win], MAIN_METRICS)
            .to_string(index=False, float_format=lambda x: f"{x:.3f}")
        )

    # -- retrieval-noise sweep (H4 + negative-result study) ----------------
    print(f"\n[ablation] noise sweep: {C.NOISE_SWEEP} x {len(C.NOISE_SWEEP_PRESETS)} presets")
    noise_rows = []
    for noise in C.NOISE_SWEEP:
        cfg = replace(C.TASKS, embed_noise=noise)
        df = memory_ablation(
            C.NOISE_SWEEP_PRESETS, cfg, C.ACCESS, n_tasks=n_tasks,
            query_noise=noise, sweep={"embed_noise": noise},
        )
        noise_rows.append(df)
    noise_df = pd.concat(noise_rows, ignore_index=True)
    noise_df.to_csv(os.path.join(args.out, "ablation_noise.csv"), index=False)

    # -- context-fill sweep -------------------------------------------------
    print(f"[ablation] context-fill sweep: {C.FILL_SWEEP}")
    fill_rows = []
    for fill in C.FILL_SWEEP:
        cfg = replace(C.TASKS, filler_tokens_per_turn=fill)
        df = memory_ablation(
            C.FILL_SWEEP_PRESETS, cfg, C.ACCESS, n_tasks=n_tasks,
            sweep={"filler_tokens": fill},
        )
        fill_rows.append(df)
    fill_df = pd.concat(fill_rows, ignore_index=True)
    fill_df.to_csv(os.path.join(args.out, "ablation_fill.csv"), index=False)

    # -- robustness re-run (T4.2): harder task distributions ----------------
    print(f"[ablation] robustness re-run: {list(C.ROBUSTNESS_SETTINGS)}")
    rob_rows = []
    for setting, overrides in C.ROBUSTNESS_SETTINGS.items():
        cfg = replace(C.TASKS, **overrides)
        df = memory_ablation(
            C.ROBUSTNESS_PRESETS, cfg, C.ACCESS, n_tasks=n_tasks,
            sweep={"setting": setting},
        )
        df["config_id"] = df["config_id"] + f";setting={setting}"
        rob_rows.append(df)
    rob_df = pd.concat(rob_rows, ignore_index=True)
    rob_df.to_csv(os.path.join(args.out, "ablation_robustness.csv"), index=False)
    for setting in C.ROBUSTNESS_SETTINGS:
        print(f"\n  setting={setting}:")
        print(
            summarize(rob_df[rob_df["setting"] == setting], MAIN_METRICS)
            .to_string(index=False, float_format=lambda x: f"{x:.3f}")
        )

    print(f"\n[ablation] done in {time.perf_counter() - t0:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
