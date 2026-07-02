"""Study 2 driver (T3.2): harness ladder, budget sweep, tau sweep, recovery.

Run with::

    python -m dah.experiments.run_baseline
    python -m dah.experiments.run_baseline --quick
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import replace

from . import config as C
from .runner import budget_sweep, harness_ladder, recovery_study, summarize, tau_sweep

RESULTS_DIR = os.path.join(os.getcwd(), "results")

LADDER_METRICS = [
    "query_accuracy",
    "query_accuracy_cross",
    "answered_rate",
    "hallucination_rate",
    "total_tokens",
    "rollbacks",
    "ingest_giveup",
]


def main(argv: list[str] | None = None) -> None:
    """Run the harness ladder plus the budget, tau, and recovery studies."""
    ap = argparse.ArgumentParser(description="Harness feature-ladder study")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n-tasks", type=int, default=None)
    ap.add_argument("--out", type=str, default=RESULTS_DIR)
    args = ap.parse_args(argv)

    cfg = C.BASELINE
    if args.quick:
        cfg = replace(cfg, n_tasks=12)
    if args.n_tasks:
        cfg = replace(cfg, n_tasks=args.n_tasks)
    os.makedirs(args.out, exist_ok=True)
    t0 = time.perf_counter()

    print(
        f"[baseline] harness ladder: {cfg.n_tasks} tasks, budget={cfg.max_tokens} tokens, "
        f"p_fail={cfg.fault_rate}, tau={cfg.verifier_tau}"
    )
    ladder_df = harness_ladder(C.TASKS, C.ACCESS, cfg)
    ladder_df.to_csv(os.path.join(args.out, "harness_ladder.csv"), index=False)
    print(summarize(ladder_df, LADDER_METRICS).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    budgets = (
        [C.BUDGET_SWEEP[0], C.BUDGET_SWEEP[len(C.BUDGET_SWEEP) // 2], C.BUDGET_SWEEP[-1]]
        if args.quick
        else C.BUDGET_SWEEP
    )
    print(f"\n[baseline] token-budget sweep: {budgets}")
    bud_df = budget_sweep(C.TASKS, C.ACCESS, cfg, budgets)
    bud_df.to_csv(os.path.join(args.out, "harness_budget_sweep.csv"), index=False)

    print(f"\n[baseline] verifier operating-point sweep (tau): {C.TAU_SWEEP}")
    tau_df = tau_sweep(C.TASKS, C.ACCESS, cfg, C.TAU_SWEEP)
    tau_df.to_csv(os.path.join(args.out, "verifier_tau_sweep.csv"), index=False)
    print(
        summarize(tau_df, ["query_accuracy", "answered_rate", "hallucination_rate", "verify_reject"])
        .to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )

    print(f"\n[baseline] failure-recovery study (H5): fault rates {C.RECOVERY_FAULT_RATES}")
    rec_df = recovery_study(C.TASKS, C.ACCESS, cfg, C.RECOVERY_FAULT_RATES)
    rec_df.to_csv(os.path.join(args.out, "recovery.csv"), index=False)
    print(
        summarize(
            rec_df,
            ["query_accuracy", "ingest_fail", "ingest_recovered", "ingest_giveup", "total_tokens"],
        ).to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )

    print(f"\n[baseline] done in {time.perf_counter() - t0:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
