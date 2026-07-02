# TASKS.md — Ordered plan

> Work top to bottom; do not start a milestone while the previous one has open blockers.
> Task IDs are unique and stable (`T<milestone>.<n>`) — cite them in commits and
> `docs/JOURNAL.md`. A task is closed only when its acceptance criteria hold **and**
> `python scripts/quality_gate.py --fast` is green (R15). Status marks: `[ ]` open,
> `[~]` in progress, `[x]` done, `[!]` blocked (say by what).

## M0 — Governance bootstrap — **done**

- [x] **T0.1** Write GOAL.md, RULES.md, FEATURES.md, TASKS.md, AGENTS.md; add README.md and docs/JOURNAL.md.
  - *Evidence:* all files exist; linter enforces the cross-references (R12).
- [x] **T0.2** Implement `scripts/rules_lint.py` (R1, R2, R3, R5, R8–R12) and `scripts/quality_gate.py` (`--fast` / `--strict`).
  - *Evidence:* `python scripts/rules_lint.py` exits 0 on the current tree; violations exit 1.
- [x] **T0.3** Create `.cursor/hooks.json` + hook scripts: frozen-file/results deny-guard (R13/R3), shell guard (R15), post-edit lint feedback, stop-gate + journal reminder (R15/R14).
  - *Evidence:* payload tests in `tests/test_governance.py` exercise every decision path; scripts executable. Note: `preToolUse` cannot show approval cards, so frozen files **deny** (amendments go through the user directly).
- [x] **T0.4** Add `tests/test_governance.py` and `.githooks/pre-commit`; `git config core.hooksPath .githooks` set.
  - *Evidence:* `pytest tests/test_governance.py -q` green.

## M1 — Make the package whole (single-agent memory loop) — **done**

- [x] **T1.1** `src/dah/memory/manager.py`: `MemoryConfig` (all toggles), `MemoryManager` (ingest → assemble → session-reset; density-ordered trimming), 10 `PRESETS`, `make_memory`.
  - *Evidence:* `tests/test_memory.py`; whole-stack snapshot/RNG round-trip in `tests/test_governance.py` (R6); leakage grep clean (R5).
- [x] **T1.2** `src/dah/env/multisession.py`: `MultiSessionEnv` (noisy query embeddings + reformulation), sole grader (R5), `QueryResult`/`EpisodeResult` with the frozen GOAL.md metrics.
  - *Evidence:* `tests/test_env.py` (incl. multi-fact AND grading and metric definitions).
- [x] **T1.3** `BaselineAgent` in `src/dah/agents.py` (single toggleable `Agent` loop; baseline = features off).
  - *Evidence:* `tests/test_agent.py` end-to-end smoke + determinism.
- [x] **T1.4** Unit tests for F1, F9–F12, F15, F18 contracts.
  - *Evidence:* `tests/test_memory.py`, `tests/test_env.py`, `tests/test_provider.py`; R11 lint clear.
- [x] **T1.5** FEATURES.md statuses + journal entry.

## M2 — Control plane (verify / recover / plan / inject) — **done**

- [x] **T2.1** `src/dah/verifier.py`: oracle-free `ConsistencyVerifier` (`Verdict` API; empty-recall + low-relevance checks; deterministic; R5-clean).
  - *Scope note:* an earlier tunable-oracle design (`detect_wrong` knob) was discarded for violating R5's spirit; verification quality now emerges from embedding geometry. See JOURNAL 2026-07-01.
- [x] **T2.2** `src/dah/recovery.py`: `RecoveryPolicy` (ingestion retries + rollback-before-retry, `none`/`robust`).
  - *Scope note:* simpler than the planned 4-verb policy engine — query-side escalation lives in the planner (`max_retries`/`escalate_factor`), state-side recovery here. The split is deliberate and documented in DESIGN.md §5.
- [x] **T2.3** `src/dah/planner.py`: `BudgetAwarePlanner` (remaining-budget/remaining-queries allocation, depth scaling, retry suppression under pressure, focused retries) vs `StaticPlanner`; plans are data (`QueryPlan`).
  - *Evidence:* `tests/test_agent.py`; budget sweep shows the allocation effect (`results/harness_budget_sweep.csv`).
- [x] **T2.4** `src/dah/faults.py`: seeded `FaultInjector` (transient ingest failure + context poison), own RNG stream, rate-0 bit-identity, snapshot-safe, deliberately not checkpointed.
  - *Evidence:* identity + RNG round-trip tests in `tests/test_governance.py`.
- [x] **T2.5** `HarnessAgent`: verify → retry deeper/reformulated → abstain; checkpoint+rollback around ingestion; budget charged with no refunds (R7).
  - *Evidence:* `tests/test_agent.py` (rollback path asserted via trace events; paired harness>baseline check on 40 seeds).
- [x] **T2.6** FEATURES.md (F5–F8, F21) + journal.

## M3 — Experiments — **done**

- [x] **T3.1** `experiments/config.py` + `experiments/runner.py`: run matrix → tidy DataFrames; `seed` + `config_id` on every row (R3); paired bootstrap ≥ 10k resamples (R4).
  - *Evidence:* `tests/test_reproducibility.py` (bit-identical re-runs, schema, hand-checked paired CI).
- [x] **T3.2** `dah-baseline` (`run_baseline.py`): harness ladder + budget sweep + tau sweep + recovery study, 120 paired seeds.
  - *Evidence:* `results/harness_ladder.csv`, `harness_budget_sweep.csv`, `verifier_tau_sweep.csv`, `recovery.csv`.
- [x] **T3.3** `dah-ablation` (`run_ablation.py`): 8 presets × 2 windows × 120 seeds; noise sweep (6 levels); fill sweep (4 levels).
  - *Evidence:* `results/ablation_main.csv`, `ablation_noise.csv`, `ablation_fill.csv`.
- [x] **T3.4** `dah-analyze` (`analyze.py`): all figures, `loo_summary.csv`, `recovery_summary.csv`, machine-computed `hypotheses.csv`, `summary.md`.
  - *Evidence:* `make reproduce` end-to-end (~90 s laptop); every figure cited by RESULTS.md regenerated.
- [x] **T3.5** Journal + FEATURES.md (F22–F24, F31–F32).

## M4 — Science pass — **done**

- [x] **T4.1** `docs/RESULTS.md` with required sections (Headline / Ablation matrix / Failure recovery / What didn't work / Reproduce) and explicit H1–H5 verdicts with CIs.
  - *Evidence:* linter checks the sections; verdicts quoted from `results/hypotheses.csv` (H1 directional, H2 supported w/ caveat, H3 partial, H4 refuted-opposite, H5 supported).
- [x] **T4.2** Robustness re-run at 2 extra settings: longer-horizon (`n_sessions=10`) and cross-heavy (`p_cross_session=0.85`); noise/fill sweeps already cover embedding quality and context pressure.
  - *Evidence:* `results/ablation_robustness.csv`; table + caveats (summary-growth cost at long horizons; no ordering flips) in RESULTS.md.
- [x] **T4.3** Failure-mode gallery (`experiments/gallery.py` → `results/traces.md`): rollback-saves-episode, distractor-hurts, budget-degrade-still-succeeds — all seed-reproducible.
- [x] **T4.4** GOAL.md DoD items checked: strict gate green; one-command regeneration; RESULTS.md complete; FEATURES.md verified with live evidence; README quickstart (see JOURNAL for the verdict log).

## M5 — Polish & stretch

- [x] **T5.1** README quickstart (clone → install → reproduce in ≤ 3 commands) + architecture/limitations.
- [ ] **T5.2** *(stretch; needs `OPENAI_API_KEY`)* Live-LLM spot check via `dah.experiments.live_check`: directional agreement of the top ablation finding on a real endpoint; document divergences in RESULTS.md.
- [ ] **T5.3** *(stretch)* Promote `gap_triggered` retrieval into the ablation grid with a dedicated test, or delete the policy (currently implemented-but-unmeasured; see FEATURES.md Known gaps).
- [ ] **T5.4** Tag `v0.1.0` — requires strict gate green **and explicit user go-ahead** (no commits/tags without it).

## Task hygiene

- New work discovered mid-milestone gets a new task ID here (never silently absorbed).
- If a task is blocked, mark `[!]` with the blocker and move on to the next unblocked task.
- Scope changes to hypotheses/metrics are GOAL.md amendments (R13), not task edits.
