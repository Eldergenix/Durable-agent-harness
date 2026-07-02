"""Render figures, tables, and H1-H5 verdicts from result CSVs (T3.4, T4.1).

Reads the CSVs produced by ``run_ablation`` / ``run_baseline`` from ``results/``
and writes:

* ``results/fig_*.png``      - every figure cited by docs/RESULTS.md
* ``results/hypotheses.csv`` - machine-computed verdicts for GOAL.md H1-H5,
                               each with a paired-bootstrap 95% CI (R4)
* ``results/summary.md``     - generated tables

Safe to run after a ``--quick`` study; it renders whatever conditions exist.

    python -m dah.experiments.analyze
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from .runner import LADDER, bootstrap_ci, paired_delta_ci, summarize

RESULTS_DIR = os.path.join(os.getcwd(), "results")

PALETTE = {
    "none": "#9aa0a6",
    "compaction": "#4c78a8",
    "episodic": "#f58518",
    "semantic": "#54a24b",
    "episodic+semantic": "#b279a2",
    "compaction+episodic": "#e45756",
    "compaction+semantic": "#bab0ac",
    "full": "#72b7b2",
    "baseline": "#9aa0a6",
    "verify": "#f58518",
    "verify+recovery": "#4c78a8",
    "recovery_off": "#9aa0a6",
    "recovery_on": "#4c78a8",
}


def _color(name: str) -> str:
    return PALETTE.get(name, "#333333")


def _load(name: str) -> pd.DataFrame | None:
    path = os.path.join(RESULTS_DIR, name)
    return pd.read_csv(path) if os.path.exists(path) else None


def _errbars(summary: pd.DataFrame, metric: str):
    lo = summary[metric] - summary[f"{metric}_lo"]
    hi = summary[f"{metric}_hi"] - summary[metric]
    return np.vstack([lo, hi])


def _mean_ci(series):
    return bootstrap_ci(series.to_numpy())


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _fig_accessibility(out: str) -> None:
    p = C.ACCESS
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    rels = np.linspace(0, 1, 200)
    for L in [200, 600, 1200, 2400]:
        axes[0].plot(rels, [p.accessibility(L, r) for r in rels], label=f"context={L} tok")
    axes[0].set_xlabel("relative position in context (0=start, 1=end)")
    axes[0].set_ylabel("P(use fact)")
    axes[0].set_title("Lost-in-the-middle: recall vs. position")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    Ls = np.linspace(50, 3500, 200)
    for rel, lbl in [(0.0, "edge (rel=0)"), (0.5, "middle (rel=0.5)")]:
        axes[1].plot(Ls, [p.accessibility(L, rel) for L in Ls], label=lbl)
    axes[1].axvline(p.l_half, ls="--", c="grey", alpha=0.7, label=f"l_half={p.l_half:.0f}")
    axes[1].set_xlabel("total context tokens")
    axes[1].set_ylabel("P(use fact)")
    axes[1].set_title("Length decay: recall vs. context size")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.suptitle("Simulated provider: context-degradation model (held fixed across all conditions)")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def _fig_memory_matrix(df: pd.DataFrame, out: str) -> pd.DataFrame:
    windows = sorted(df["window"].unique(), reverse=True)
    fig, axes = plt.subplots(1, len(windows) + 1, figsize=(6.5 * (len(windows) + 1), 5))
    order = [c for c in C.ABLATION_PRESETS if c in set(df["condition"])]
    summaries = []
    for ax, win in zip(axes[:-1], windows):
        s = summarize(df[df["window"] == win], ["query_accuracy", "query_accuracy_cross", "query_accuracy_within", "mean_ctx_tokens"])
        s = s.set_index("condition").loc[order].reset_index()
        s["window"] = win
        summaries.append(s)
        x = np.arange(len(s))
        w = 0.26
        for i, (m, lbl) in enumerate(
            [("query_accuracy_cross", "cross-session"), ("query_accuracy_within", "within-session"), ("query_accuracy", "overall")]
        ):
            ax.bar(x + (i - 1) * w, s[m], w, yerr=_errbars(s, m), capsize=3, label=lbl,
                   color=plt.cm.Blues(0.4 + 0.25 * i))
        ax.set_xticks(x)
        ax.set_xticklabels(s["condition"], rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("QueryAccuracy")
        ax.set_title(f"working window = {win} tokens")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    sw = summaries[0]
    ax = axes[-1]
    for _, r in sw.iterrows():
        ax.scatter(r["mean_ctx_tokens"], r["query_accuracy_cross"], s=90, color=_color(r["condition"]), zorder=3)
        ax.annotate(r["condition"], (r["mean_ctx_tokens"], r["query_accuracy_cross"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("mean context tokens per query (cost)")
    ax.set_ylabel("cross-session QueryAccuracy")
    ax.set_title(f"Quality vs. cost (window={windows[0]})")
    ax.grid(alpha=0.3)
    fig.suptitle("Memory ablation matrix: 8 configurations x 2 context-pressure levels")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return pd.concat(summaries, ignore_index=True)


def _fig_loo(df: pd.DataFrame, out: str) -> pd.DataFrame:
    """H2 figure: per-component accuracy drop when removed from `full`."""
    rows = []
    for win in sorted(df["window"].unique(), reverse=True):
        sub = df[df["window"] == win]
        for comp, loo_arm in C.LOO_ARMS.items():
            d, lo, hi, n = paired_delta_ci(sub, "query_accuracy", loo_arm, "full")
            rows.append({"window": win, "component": comp, "drop": d, "lo": lo, "hi": hi, "n": n})
    loo = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    comps = list(C.LOO_ARMS)
    x = np.arange(len(comps))
    windows = sorted(loo["window"].unique(), reverse=True)
    w = 0.35
    for j, win in enumerate(windows):
        sub = loo[loo["window"] == win].set_index("component").loc[comps]
        err = np.vstack([sub["drop"] - sub["lo"], sub["hi"] - sub["drop"]])
        ax.bar(x + (j - 0.5) * w, sub["drop"], w, yerr=err, capsize=3, label=f"window={win}")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(comps)
    ax.set_ylabel("QueryAccuracy gained by having the component\n(paired: full − full-minus-component)")
    ax.set_title("Leave-one-out value of each memory component (H2)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return loo


def _fig_noise_sweep(df: pd.DataFrame, out: str) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for cond in C.NOISE_SWEEP_PRESETS:
        rows = []
        for noise, g in df[df["condition"] == cond].groupby("embed_noise"):
            m, lo, hi = _mean_ci(g["query_accuracy_cross"])
            rows.append((noise, m, lo, hi))
        if not rows:
            continue
        rows.sort()
        xs, ms, los, his = map(np.array, zip(*rows))
        ax.plot(xs, ms, "-o", color=_color(cond), label=cond, ms=4)
        ax.fill_between(xs, los, his, color=_color(cond), alpha=0.13)
    ax.set_xlabel("retrieval noise (embedding + query noise)")
    ax.set_ylabel("cross-session QueryAccuracy")
    ax.set_title("When retrieval stops helping: cross-session accuracy vs. retrieval noise")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def _fig_fill_sweep(df: pd.DataFrame, out: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for cond in C.FILL_SWEEP_PRESETS:
        rows = []
        for fill, g in df[df["condition"] == cond].groupby("filler_tokens"):
            m, lo, hi = _mean_ci(g["query_accuracy"])
            rows.append((fill, m, lo, hi))
        if not rows:
            continue
        rows.sort()
        fills, ms, los, his = map(np.array, zip(*rows))
        ax.plot(fills, ms, "-o", color=_color(cond), label=cond)
        ax.fill_between(fills, los, his, color=_color(cond), alpha=0.15)
    ax.set_xlabel("filler tokens per turn (context-fill rate)")
    ax.set_ylabel("QueryAccuracy")
    ax.set_title("Long-horizon pressure: accuracy vs. how fast context fills")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def _fig_ladder(df: pd.DataFrame, out: str) -> pd.DataFrame:
    s = summarize(df, ["query_accuracy", "query_accuracy_cross", "answered_rate", "hallucination_rate"])
    order = [c for c in LADDER if c in set(s["condition"])]
    s = s.set_index("condition").loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(s))
    w = 0.2
    for i, (m, lbl) in enumerate(
        [("query_accuracy", "QueryAccuracy"), ("query_accuracy_cross", "cross-session"),
         ("answered_rate", "answered rate"), ("hallucination_rate", "hallucination rate")]
    ):
        ax.bar(x + (i - 1.5) * w, s[m], w, yerr=_errbars(s, m), capsize=3, label=lbl,
               color=plt.cm.viridis(0.15 + 0.22 * i))
    ax.set_xticks(x)
    ax.set_xticklabels(s["condition"], fontsize=9)
    ax.set_ylabel("fraction")
    ax.set_title("Harness feature ladder (same budget, same failures, same memory)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return s


def _fig_budget_sweep(df: pd.DataFrame, out: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for cond in LADDER:
        rows = []
        for b, g in df[df["condition"] == cond].groupby("max_tokens"):
            sm, slo, shi = _mean_ci(g["query_accuracy"])
            am, alo, ahi = _mean_ci(g["answered_rate"])
            rows.append((b, sm, slo, shi, am, alo, ahi))
        if not rows:
            continue
        rows.sort()
        b, sm, slo, shi, am, alo, ahi = map(np.array, zip(*rows))
        axes[0].plot(b, sm, "-o", color=_color(cond), label=cond)
        axes[0].fill_between(b, slo, shi, color=_color(cond), alpha=0.15)
        axes[1].plot(b, am, "-o", color=_color(cond), label=cond)
        axes[1].fill_between(b, alo, ahi, color=_color(cond), alpha=0.15)
    axes[0].set_xlabel("token budget")
    axes[0].set_ylabel("QueryAccuracy")
    axes[0].set_title("Accuracy vs. budget")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("token budget")
    axes[1].set_ylabel("answered rate")
    axes[1].set_title("Task completion vs. budget")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.suptitle("Budget-aware planning: converting budget into correctness (H1)")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def _fig_tau_sweep(df: pd.DataFrame, out: str) -> None:
    rows = []
    for tau, g in df.groupby("tau"):
        qm, qlo, qhi = _mean_ci(g["query_accuracy"])
        am, *_ = _mean_ci(g["answered_rate"])
        hm, *_ = _mean_ci(g["hallucination_rate"])
        rows.append((tau, qm, qlo, qhi, am, hm))
    rows.sort()
    tau, qm, qlo, qhi, am, hm = map(np.array, zip(*rows))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(tau, qm, "-o", color="#4c78a8", label="QueryAccuracy")
    ax.fill_between(tau, qlo, qhi, color="#4c78a8", alpha=0.15)
    ax.plot(tau, am, "-s", color="#f58518", label="answered rate")
    ax.plot(tau, hm, "-^", color="#e45756", label="hallucination rate")
    ax.set_xlabel("verifier relevance threshold tau (oracle-free ConsistencyVerifier)")
    ax.set_ylabel("fraction")
    ax.set_title("Verifier operating point: strictness trades completion for trust")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def _fig_recovery(df: pd.DataFrame, out: str) -> pd.DataFrame:
    """H5 figure + per-rate stats: accuracy by arm; step recovery; token overhead."""
    stats = []
    for rate in sorted(df["fault_rate"].unique()):
        sub = df[df["fault_rate"] == rate]
        d, lo, hi, n = paired_delta_ci(sub, "query_accuracy", "recovery_off", "recovery_on")
        on = sub[sub["condition"] == "recovery_on"]
        off = sub[sub["condition"] == "recovery_off"]
        recovered = on["ingest_recovered"].sum()
        giveup = on["ingest_giveup"].sum()
        step_recovery = recovered / max(1, recovered + giveup)
        overhead = (on["total_tokens"].mean() - off["total_tokens"].mean()) / off["total_tokens"].mean()
        stats.append(
            {
                "fault_rate": rate,
                "acc_delta": d,
                "acc_delta_lo": lo,
                "acc_delta_hi": hi,
                "n": n,
                "step_recovery_rate": step_recovery,
                "token_overhead": overhead,
                "acc_on": float(on["query_accuracy"].mean()),
                "acc_off": float(off["query_accuracy"].mean()),
            }
        )
    st = pd.DataFrame(stats)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for cond in ["recovery_off", "recovery_on"]:
        rows = []
        for rate, g in df[df["condition"] == cond].groupby("fault_rate"):
            m, lo, hi = _mean_ci(g["query_accuracy"])
            rows.append((rate, m, lo, hi))
        rows.sort()
        xs, ms, los, his = map(np.array, zip(*rows))
        axes[0].plot(xs, ms, "-o", color=_color(cond), label=cond)
        axes[0].fill_between(xs, los, his, color=_color(cond), alpha=0.15)
    axes[0].set_xlabel("injected ingestion-failure rate")
    axes[0].set_ylabel("QueryAccuracy")
    axes[0].set_title("Recovery keeps accuracy flat as failures rise")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(st["fault_rate"], st["step_recovery_rate"], "-o", color="#4c78a8", label="step recovery rate")
    axes[1].plot(st["fault_rate"], st["token_overhead"], "-s", color="#f58518", label="token overhead (on vs off)")
    axes[1].axhline(0.6, ls="--", c="grey", alpha=0.6, label="H5 recovery floor (0.60)")
    axes[1].axhline(0.25, ls=":", c="grey", alpha=0.6, label="H5 overhead ceiling (0.25)")
    axes[1].set_xlabel("injected ingestion-failure rate")
    axes[1].set_ylabel("fraction")
    axes[1].set_title("H5 criteria: recovery rate and token overhead")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return st


# ---------------------------------------------------------------------------
# Hypothesis verdicts (GOAL.md H1-H5)
# ---------------------------------------------------------------------------
def _verdicts(
    ladder: pd.DataFrame | None,
    budget: pd.DataFrame | None,
    ablation: pd.DataFrame | None,
    noise: pd.DataFrame | None,
    recovery: pd.DataFrame | None,
) -> pd.DataFrame:
    rows: list[dict] = []

    # -- H1: full harness vs naive baseline at the same budget --------------
    if ladder is not None:
        d, lo, hi, n = paired_delta_ci(ladder, "query_accuracy", "baseline", "full")
        verdict = "supported" if (d >= 0.15 and lo > 0) else ("directional" if lo > 0 else "refuted")
        rows.append(
            dict(hypothesis="H1", setting=f"budget={C.BASELINE.max_tokens}", metric="QueryAccuracy delta (full-baseline)",
                 value=d, lo=lo, hi=hi, n=n, criterion=">= +0.15 pp, CI>0", verdict=verdict)
        )
    if budget is not None:
        for b in sorted(budget["max_tokens"].unique()):
            sub = budget[budget["max_tokens"] == b]
            d, lo, hi, n = paired_delta_ci(sub, "query_accuracy", "baseline", "full")
            rows.append(
                dict(hypothesis="H1", setting=f"budget={b}", metric="QueryAccuracy delta (full-baseline)",
                     value=d, lo=lo, hi=hi, n=n, criterion=">= +0.15, CI>0",
                     verdict="supported" if (d >= 0.15 and lo > 0) else ("directional" if lo > 0 else "refuted"))
            )

    # -- H2: compaction the most valuable component at tight windows --------
    if ablation is not None:
        for win in sorted(ablation["window"].unique()):
            sub = ablation[ablation["window"] == win]
            drops = {}
            for comp, loo_arm in C.LOO_ARMS.items():
                d, lo, hi, n = paired_delta_ci(sub, "query_accuracy", loo_arm, "full")
                drops[comp] = (d, lo, hi, n)
                rows.append(
                    dict(hypothesis="H2", setting=f"window={win}", metric=f"drop when removing {comp}",
                         value=d, lo=lo, hi=hi, n=n, criterion="largest drop = compaction", verdict="")
                )
            largest = max(drops, key=lambda c: drops[c][0])
            rows.append(
                dict(hypothesis="H2", setting=f"window={win}", metric="largest single-component drop",
                     value=drops[largest][0], lo=drops[largest][1], hi=drops[largest][2], n=drops[largest][3],
                     criterion="compaction has the largest drop",
                     verdict="supported" if largest == "compaction" else f"refuted (largest={largest})")
            )

    # -- H3: episodic helps only cross-session, hurts within ----------------
    if ablation is not None:
        sub = ablation[ablation["window"] == 700]
        dc, dclo, dchi, n1 = paired_delta_ci(sub, "query_accuracy_cross", "none", "episodic")
        dw, dwlo, dwhi, n2 = paired_delta_ci(sub, "query_accuracy_within", "none", "episodic")
        if dclo > 0 and dwhi < 0:
            verdict = "supported"
        elif dclo > 0 and dwlo <= 0 <= dwhi:
            verdict = "partially supported (cross-only; within-session neutral, not negative)"
        elif dclo > 0 and dwlo > 0:
            verdict = "refuted (retrieval helped everywhere)"
        else:
            verdict = "refuted"
        rows.append(dict(hypothesis="H3", setting="window=700", metric="cross-session delta (episodic-none)",
                         value=dc, lo=dclo, hi=dchi, n=n1, criterion="CI>0", verdict=verdict))
        rows.append(dict(hypothesis="H3", setting="window=700", metric="within-session delta (episodic-none)",
                         value=dw, lo=dwlo, hi=dwhi, n=n2, criterion="CI<0 for full support", verdict=verdict))

    # -- H4: semantic value flips sign as embedding noise grows -------------
    if noise is not None and {"episodic", "episodic+semantic"} <= set(noise["condition"].unique()):
        signs = []
        for nz in sorted(noise["embed_noise"].unique()):
            sub = noise[noise["embed_noise"] == nz]
            d, lo, hi, n = paired_delta_ci(sub, "query_accuracy_cross", "episodic", "episodic+semantic")
            signs.append((nz, d, lo, hi))
            rows.append(dict(hypothesis="H4", setting=f"noise={nz}", metric="marginal semantic value (epi+sem - epi)",
                             value=d, lo=lo, hi=hi, n=n, criterion="sign flip across noise", verdict=""))
        sig_pos = [s for s in signs if s[2] > 0]
        sig_neg = [s for s in signs if s[3] < 0]
        if sig_pos and sig_neg:
            verdict = "supported (sign flips with noise)"
        elif sig_pos and not sig_neg:
            verdict = "refuted (semantic stays additive; no crossover)"
        elif sig_neg and not sig_pos:
            verdict = "refuted (semantic never helps)"
        else:
            verdict = "inconclusive (no significant effect at any noise)"
        rows.append(dict(hypothesis="H4", setting="all noise levels", metric="crossover exists",
                         value=float("nan"), lo=float("nan"), hi=float("nan"), n=len(signs),
                         criterion="significant + and - regions", verdict=verdict))

    # -- H5: recovery >= 60% at <= 25% token overhead ------------------------
    if recovery is not None:
        for rate in sorted(recovery["fault_rate"].unique()):
            sub = recovery[recovery["fault_rate"] == rate]
            on = sub[sub["condition"] == "recovery_on"]
            off = sub[sub["condition"] == "recovery_off"]
            recovered = on["ingest_recovered"].sum()
            giveup = on["ingest_giveup"].sum()
            step = recovered / max(1, recovered + giveup)
            overhead = (on["total_tokens"].mean() - off["total_tokens"].mean()) / off["total_tokens"].mean()
            d, lo, hi, n = paired_delta_ci(sub, "query_accuracy", "recovery_off", "recovery_on")
            ok = step >= 0.60 and overhead <= 0.25
            rows.append(dict(hypothesis="H5", setting=f"fault_rate={rate}", metric="step recovery rate / token overhead",
                             value=step, lo=overhead, hi=float("nan"), n=n,
                             criterion=">=0.60 recovery at <=0.25 overhead",
                             verdict="supported" if ok else "refuted"))
            rows.append(dict(hypothesis="H5", setting=f"fault_rate={rate}", metric="QueryAccuracy delta (on-off)",
                             value=d, lo=lo, hi=hi, n=n, criterion="CI>0 (secondary)",
                             verdict="supported" if lo > 0 else "not significant"))
            # frozen-definition RecoveryRate (episode-level), reported for completeness
            with_fault = on[on["ingest_fail"] > 0]
            ep_rr = float(with_fault["episode_success"].mean()) if len(with_fault) else float("nan")
            rows.append(dict(hypothesis="H5", setting=f"fault_rate={rate}",
                             metric="episode-level RecoveryRate (frozen definition)",
                             value=ep_rr, lo=float("nan"), hi=float("nan"), n=len(with_fault),
                             criterion="reported as defined (see RESULTS note)", verdict="reported"))
    return pd.DataFrame(rows)


def _fmt(s: pd.DataFrame, metrics: list[str]) -> str:
    cols = ["condition", "n"] + metrics
    disp = s[cols].copy()
    for m in metrics:
        disp[m] = disp[m].map(lambda x: f"{x:.3f}")
    return disp.to_markdown(index=False)


def main(argv: list[str] | None = None) -> None:
    """Render every figure/table and compute H1-H5 verdicts from results CSVs."""
    ap = argparse.ArgumentParser(description="Render figures + verdicts from result CSVs")
    ap.add_argument("--out", type=str, default=RESULTS_DIR)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    sections: list[str] = []
    _fig_accessibility(os.path.join(args.out, "fig_accessibility.png"))
    sections.append("## Provider model\n\n![accessibility](fig_accessibility.png)\n")

    main_df = _load("ablation_main.csv")
    if main_df is not None:
        s = _fig_memory_matrix(main_df, os.path.join(args.out, "fig_memory_ablation.png"))
        loo = _fig_loo(main_df, os.path.join(args.out, "fig_loo.png"))
        # R3 traceability on derived artifacts: the seed base and source arm
        loo["seed"] = C.GLOBAL_SEED
        loo["config_id"] = loo.apply(lambda r: f"loo:win={r['window']};component={r['component']}", axis=1)
        loo.to_csv(os.path.join(args.out, "loo_summary.csv"), index=False)
        for win in sorted(s["window"].unique(), reverse=True):
            sections.append(
                f"## Ablation matrix (window={win})\n\n"
                + _fmt(s[s["window"] == win],
                       ["query_accuracy", "query_accuracy_cross", "query_accuracy_within", "mean_ctx_tokens"])
                + "\n"
            )
        sections.append("![memory ablation](fig_memory_ablation.png)\n\n![leave one out](fig_loo.png)\n")

    noise_df = _load("ablation_noise.csv")
    if noise_df is not None:
        _fig_noise_sweep(noise_df, os.path.join(args.out, "fig_noise_sweep.png"))
        sections.append("## Retrieval-noise sweep\n\n![noise sweep](fig_noise_sweep.png)\n")

    fill_df = _load("ablation_fill.csv")
    if fill_df is not None:
        _fig_fill_sweep(fill_df, os.path.join(args.out, "fig_fill_sweep.png"))
        sections.append("## Context-fill sweep\n\n![fill sweep](fig_fill_sweep.png)\n")

    rob_df = _load("ablation_robustness.csv")
    if rob_df is not None:
        for setting, g in rob_df.groupby("setting", sort=False):
            s = summarize(g, ["query_accuracy", "query_accuracy_cross", "query_accuracy_within"])
            sections.append(
                f"## Robustness re-run (T4.2): {setting}\n\n"
                + _fmt(s, ["query_accuracy", "query_accuracy_cross", "query_accuracy_within"])
                + "\n"
            )

    ladder_df = _load("harness_ladder.csv")
    if ladder_df is not None:
        s = _fig_ladder(ladder_df, os.path.join(args.out, "fig_harness_ladder.png"))
        sections.append(
            "## Harness feature ladder\n\n"
            + _fmt(s, ["query_accuracy", "query_accuracy_cross", "answered_rate", "hallucination_rate"])
            + "\n\n![harness ladder](fig_harness_ladder.png)\n"
        )

    budget_df = _load("harness_budget_sweep.csv")
    if budget_df is not None:
        _fig_budget_sweep(budget_df, os.path.join(args.out, "fig_budget_sweep.png"))
        sections.append("## Budget sweep\n\n![budget sweep](fig_budget_sweep.png)\n")

    tau_df = _load("verifier_tau_sweep.csv")
    if tau_df is not None:
        _fig_tau_sweep(tau_df, os.path.join(args.out, "fig_tau_sweep.png"))
        sections.append("## Verifier operating point\n\n![tau sweep](fig_tau_sweep.png)\n")

    rec_df = _load("recovery.csv")
    if rec_df is not None:
        st = _fig_recovery(rec_df, os.path.join(args.out, "fig_recovery.png"))
        st["seed"] = C.GLOBAL_SEED
        st["config_id"] = st["fault_rate"].map(lambda r: f"recovery-summary:pfail={r}")
        st.to_csv(os.path.join(args.out, "recovery_summary.csv"), index=False)
        sections.append("## Failure recovery\n\n![recovery](fig_recovery.png)\n")

    verdicts = _verdicts(ladder_df, budget_df, main_df, noise_df, rec_df)
    if not verdicts.empty:
        verdicts["seed"] = C.GLOBAL_SEED
        verdicts["config_id"] = verdicts.apply(
            lambda r: f"verdict:{r['hypothesis']}:{r['setting']}", axis=1
        )
        verdicts.to_csv(os.path.join(args.out, "hypotheses.csv"), index=False)
        printable = verdicts[verdicts["verdict"] != ""].copy()
        for c in ("value", "lo", "hi"):
            printable[c] = printable[c].map(lambda x: f"{x:+.3f}" if pd.notna(x) else "-")
        print("\nHypothesis verdicts (full table in results/hypotheses.csv):")
        print(printable.to_string(index=False))
        sections.append("## Hypothesis verdicts\n\nSee `hypotheses.csv` for every row with CIs.\n")

    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write("# Generated results summary\n\n")
        f.write("\n".join(sections))
    print(f"[analyze] wrote figures + hypotheses.csv + summary.md to {args.out}")


if __name__ == "__main__":
    main()
