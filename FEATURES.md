# FEATURES.md — Live feature inventory

> Status enum (R12): `planned` → `in-progress` → `implemented` → `verified`.
> `implemented` = code exists and imports. `verified` = tests/evidence exercise the
> contract; the Evidence column must point at a file that exists. `scripts/rules_lint.py`
> checks that every module under `src/dah/` appears here and that `verified` evidence paths
> exist. Update this file in the same change that alters a feature's reality.

## A. Core primitives (control plane)

| ID | Feature | Module | Status | Evidence |
|----|---------|--------|--------|----------|
| F1 | Shared data model: `Fact`, `Turn`, `Task`, `ContextItem`, `AssembledContext`, position map | `src/dah/types.py` | verified | `tests/test_provider.py`, `tests/test_memory.py` |
| F2 | Multi-dimensional budget: hard `charge`, `can_afford`, soft `pressure()`; monotone, no refunds (R7) | `src/dah/budget.py` | verified | `tests/test_budget.py`, `tests/test_governance.py` |
| F3 | Checkpoint/rollback stack over `Snapshotable` components; deep, total snapshots | `src/dah/checkpoint.py` | verified | `tests/test_checkpoint.py` |
| F4 | Structured telemetry: `Trace`, `Event`, lazy counters — sole reporting channel (R9) | `src/dah/telemetry.py` | verified | `tests/test_governance.py` |
| F5 | Oracle-free verification loop: `ConsistencyVerifier` (`Verdict` API; empty-recall + low-relevance checks; deterministic; R5-clean) | `src/dah/verifier.py` | verified | `tests/test_governance.py`, `tests/test_agent.py` |
| F6 | Recovery policy: ingestion retries + rollback-before-retry (`none`/`robust` presets) | `src/dah/recovery.py` | verified | `tests/test_agent.py`, `tests/test_governance.py` |
| F7 | Budget-aware planner: allocates remaining budget over remaining queries, scales retrieval depth, focused retries; `StaticPlanner` contrast | `src/dah/planner.py` | verified | `tests/test_agent.py`, `results/harness_budget_sweep.csv` |
| F8 | Seeded failure injection: own RNG stream; rate 0 bit-identical to no injector; snapshot-safe | `src/dah/faults.py` | verified | `tests/test_governance.py` |

## B. Memory stack (the ablation subject)

| ID | Feature | Module | Status | Evidence |
|----|---------|--------|--------|----------|
| F9  | `WorkingMemory`: bounded in-context window, eviction returns overflow | `src/dah/memory/stores.py` | verified | `tests/test_memory.py` |
| F10 | `Compactor`: lossy summarisation, seeded survival RNG in snapshot (R6) | `src/dah/memory/stores.py` | verified | `tests/test_memory.py`, `tests/test_governance.py` |
| F11 | `EpisodicStore`: append-only, cosine top-k retrieval, exclusion of in-context facts | `src/dah/memory/stores.py` | verified | `tests/test_memory.py` |
| F12 | `SemanticStore`: salience-gated compact store (precision/recall trade) | `src/dah/memory/stores.py` | verified | `tests/test_memory.py` |
| F13 | `MemoryManager` + `MemoryConfig` + 10 `PRESETS`; session-reset semantics; density-ordered context packing; whole-stack snapshot/restore | `src/dah/memory/manager.py` | verified | `tests/test_memory.py`, `tests/test_governance.py` |
| F14 | Retrieval policies as a config axis: `off` / `always` / `budget_aware` / `gap_triggered` | `src/dah/memory/manager.py` | implemented | `always` + `budget_aware` exercised by the studies; `gap_triggered` untested (see Known gaps) |

## C. Environment & providers (the ground truth)

| ID | Feature | Module | Status | Evidence |
|----|---------|--------|--------|----------|
| F15 | Synthetic multi-session task generator: cross-session deps, topic-clustered distractors, multi-fact AND queries, tunable fill | `src/dah/env/tasks.py` | verified | `tests/test_env.py` |
| F16 | `MultiSessionEnv`: noisy query embeddings, reformulation, sole grader (R5), frozen GOAL.md metrics on `EpisodeResult` | `src/dah/env/multisession.py` | verified | `tests/test_env.py` |
| F17 | Provider protocol + `ModelResponse` | `src/dah/providers/base.py` | verified | `tests/test_provider.py` |
| F18 | `SimulatedProvider`: length decay × lost-in-the-middle, seeded recall rolls, confusion metric; blind to answer key | `src/dah/providers/simulated.py` | verified | `tests/test_provider.py` |
| F19 | `OpenAICompatProvider`: optional real-LLM adapter (extra, never default, sole network egress per R2) | `src/dah/providers/openai_compat.py` | implemented | needs live key: `python -m dah.experiments.live_check` |

## D. Agents (the comparison)

| ID | Feature | Module | Status | Evidence |
|----|---------|--------|--------|----------|
| F20 | `BaselineAgent`: static plan, no verify/checkpoint/recovery — same loop, features off | `src/dah/agents.py` | verified | `tests/test_agent.py` |
| F21 | `HarnessAgent`: verify → retry deeper/reformulated → abstain; checkpoint+rollback; budget-aware plan | `src/dah/agents.py` | verified | `tests/test_agent.py`, `tests/test_governance.py` |

## E. Experiments & analysis (the deliverable)

| ID | Feature | Module | Status | Evidence |
|----|---------|--------|--------|----------|
| F22 | `dah-baseline`: harness ladder, budget sweep, tau sweep, recovery study (H1, H5) | `src/dah/experiments/run_baseline.py` | verified | `results/harness_ladder.csv`, `tests/test_reproducibility.py` |
| F23 | `dah-ablation`: 8-preset × 2-window matrix, noise + fill sweeps (H2–H4), paired seeds | `src/dah/experiments/run_ablation.py` | verified | `results/ablation_main.csv`, `tests/test_reproducibility.py` |
| F24 | `dah-analyze`: paired-bootstrap CIs, all figures, machine-computed H1–H5 verdicts | `src/dah/experiments/analyze.py` | verified | `results/hypotheses.csv`, `results/summary.md` |
| F25 | `docs/RESULTS.md` with required sections (Headline / Ablation matrix / Failure recovery / What didn't work / Reproduce) | `docs/RESULTS.md` | verified | `scripts/rules_lint.py` checks the sections |
| F31 | Experiment config: single source of truth for every number in RESULTS.md | `src/dah/experiments/config.py` | verified | `tests/test_reproducibility.py` |
| F32 | Study runners: pure config→DataFrame functions; `seed`+`config_id` on every row (R3); paired bootstrap ≥10k (R4) | `src/dah/experiments/runner.py` | verified | `tests/test_reproducibility.py` |
| F33 | Live-endpoint smoke test for the OpenAI-compat path (stretch T5.2) | `src/dah/experiments/live_check.py` | implemented | needs `OPENAI_API_KEY`; not part of the deterministic study |
| F34 | Failure-mode gallery: three annotated, seed-reproducible traces (T4.3) | `src/dah/experiments/gallery.py` | verified | `results/traces.md` |

## F. Governance & quality controls

| ID | Feature | Module | Status | Evidence |
|----|---------|--------|--------|----------|
| F26 | Rules linter: machine checks for R1/R2/R3/R5/R8–R12 | `scripts/rules_lint.py` | verified | `python scripts/rules_lint.py` exits 0; violations exit 1 |
| F27 | Quality gate: lint + compile + tests (+ imports + full pytest in `--strict`) in one exit code | `scripts/quality_gate.py` | verified | `python scripts/quality_gate.py --strict` |
| F28 | Cursor hooks: frozen-file/results deny-guard, shell guard, post-edit lint feedback, stop-gate + journal reminder | `.cursor/hooks.json` + `.cursor/hooks/` | verified | `tests/test_governance.py` (hook payload tests) |
| F29 | Governance tests: leakage, snapshot-RNG round-trips, budget monotonicity, injector identity, module inventory, hook behavior | `tests/test_governance.py` | verified | `pytest tests/test_governance.py -q` |
| F30 | Git pre-commit hook mirroring the fast gate | `.githooks/pre-commit` | verified | `git config core.hooksPath .githooks` (set in this repo) |

## Known gaps (tracked)

- **F14 `gap_triggered` retrieval policy** is implemented and presettable
  (`full_gap_triggered`) but not in the committed ablation grid and has no dedicated test.
  Promote it into the matrix or delete it before v0.2 (tracked as T5.3 scope).
- **F19/F33 live-LLM path** is deliberately unverified: it needs an API key, is excluded
  from the deterministic study by design, and `respond()` is marked `pragma: no cover`.
- **Frozen episode-level RecoveryRate** is degenerate on this suite (no episode reaches
  the 0.8 EpisodeSuccess bar under this degradation model); reported as defined in
  `results/hypotheses.csv`, with step-level recovery as the informative measure. Changing
  the definition requires a GOAL.md amendment (R13).
