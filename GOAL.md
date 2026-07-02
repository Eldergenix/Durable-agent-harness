# GOAL.md — What this project must achieve

> Read order for agents: **GOAL.md → RULES.md → FEATURES.md → TASKS.md → AGENTS.md**.
> This file defines *what winning looks like*. It changes only with explicit user approval (Rule R13).

## Mission

Build **durable-agent-harness (`dah`)**: a compact, research-grade harness for long-horizon
agents that demonstrates — with reproducible numbers, not vibes — that a control plane of
**verification loops, checkpoint/rollback, budget-aware planning, and failure recovery**
measurably beats a naive plan-and-execute baseline, and that ships a **memory system with
real ablations** showing which context-management components actually matter on
multi-session tasks.

The audience is **agent engineers at research labs**. The bar is: a skeptical engineer can
clone the repo, run one command, regenerate every number in `docs/RESULTS.md`, and find the
experimental design honest (paired seeds, confidence intervals, pre-declared metrics, and
negative results reported as prominently as positive ones).

## Why this is worth building

Long-horizon agents degrade as context fills. Compaction, episodic memory, and retrieval
policies are ad hoc across the industry — teams ship them without measuring whether each
component pays for its tokens. Meanwhile the reliability primitives (rollback, verification,
budget pressure) are chronically under-used because nobody quantifies their value. This
project quantifies both, in a controlled environment where ground truth is knowable.

## Primary hypotheses (pre-declared)

| ID | Hypothesis | Falsifiable prediction |
|----|-----------|------------------------|
| H1 | The full harness (verify + rollback + budget-aware planning + recovery) beats a naive agent given the **same budget**. | ≥ +15 percentage-point mean QueryAccuracy on the 6-session suite; 95% paired-bootstrap CI excluding 0. |
| H2 | Compaction is the highest-value memory component under tight context budgets. | Removing the compactor causes the largest single-component accuracy drop at `window_tokens ≤ 800`. |
| H3 | Episodic retrieval pays off **only** on cross-session queries and can *hurt* within-session accuracy via distractors. | Ablation split by query type shows opposite-sign effects; if retrieval helps everywhere or nowhere, report that instead. |
| H4 | Semantic (salience-gated) memory trades recall for precision; its value flips sign as embedding noise grows. | Sweep over `embed_noise` shows a crossover; if no crossover exists, that is a publishable negative result. |
| H5 | Rollback-on-verifier-failure recovers a majority of injected failures at bounded extra token cost. | ≥ 60% recovery rate at ≤ 25% token overhead vs. no-recovery; otherwise report the actual trade-off curve. |

A **well-executed negative result is a success**, not a failure: if retrieval-based memory
doesn't help, `docs/RESULTS.md` must say so and explain *why* (with the supporting plot).
Deleting or hiding a negative result violates Rule R3.

## Pre-declared metrics (definitions are frozen; changing them requires user approval)

- **QueryAccuracy** — fraction of query turns where the model recalled *all* required facts.
- **EpisodeSuccess** — fraction of episodes with QueryAccuracy ≥ 0.8.
- **TokensPerCorrect** — total context tokens spent ÷ number of correct queries (efficiency).
- **HallucinationRate** — fraction of queries flagged `confident_wrong`.
- **RecoveryRate** — fraction of injected failures after which the agent still completes the episode successfully.
- **DegradationSlope** — OLS slope of per-session QueryAccuracy vs. session index (closer to 0 is better).

## Statistical standard (Rule R4)

- ≥ 20 paired seeds per experimental arm (default 30). The same seed generates the same
  task for every arm — comparisons are always paired.
- Report mean, and 95% CI via paired bootstrap (≥ 10,000 resamples) for all deltas.
- All randomness flows from explicit seeds; a result that cannot be regenerated bit-for-bit
  from its recorded seed + config is not a result.
- Exploratory (post hoc) analyses are allowed but must be labeled "exploratory" in RESULTS.md.

## Definition of Done

The project is done when **all** of the following hold:

1. `python scripts/quality_gate.py --strict` exits 0 (all rules, all tests).
2. `dah-baseline` and `dah-ablation` regenerate `results/*.csv` from scratch with one command each.
3. `dah-analyze` regenerates every figure and table cited in `docs/RESULTS.md`.
4. `docs/RESULTS.md` contains: H1–H5 verdicts with CIs, an ablation matrix (≥ 6 memory
   configurations × ≥ 2 context-pressure levels), a failure-injection study, and a
   **"What didn't work"** section with at least one honest negative or neutral finding.
5. Every feature row in `FEATURES.md` is `verified` with evidence paths that exist.
6. A stranger can go from `git clone` to reproduced headline number using only `README.md`.

## Non-goals

- Not a production framework or SDK. No plugin systems, no YAML config empires.
- Not a prompt-engineering study. The simulated provider isolates *context mechanics* from
  prompt wording by design; an optional real-LLM spot-check (T5.2) is a stretch goal only.
- No web UI, no dashboard. Plots are static files in `results/`.
- No benchmark-chasing against public leaderboards; the controlled environment is the point
  (it gives us an oracle for "which fact did the model use", which no public benchmark has).

## Current status

See `FEATURES.md` for the live inventory and `TASKS.md` for the ordered plan. As of M4,
the full stack is implemented (control plane, ablatable memory, environment, experiment
drivers, analysis) and the studies run end to end via `make reproduce`; remaining work is
tracked in TASKS.md M4/M5 (results-doc conformance, robustness re-runs, release tag).
