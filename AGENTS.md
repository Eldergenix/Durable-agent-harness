# AGENTS.md — Operating manual for agents in this repo

You are working on **durable-agent-harness (`dah`)**: a research harness for long-horizon
agents (verification loops, checkpoint/rollback, budget-aware planning, failure recovery)
plus an **ablatable memory system** evaluated on multi-session tasks. The deliverable is
reproducible evidence, judged by agent engineers at labs. Honest negative results are wins.

## Read order (once per session)

1. `GOAL.md` — mission, pre-declared hypotheses H1–H5, frozen metric definitions, Definition of Done.
2. `RULES.md` — non-negotiables R1–R15 and which machine enforces each.
3. `FEATURES.md` — what exists vs. what's planned (includes tracked known gaps).
4. `TASKS.md` — the ordered plan; find the first unblocked `[ ]` task in the active milestone.
5. `docs/JOURNAL.md` — what the last session did and what it left for you.

## Ground truth commands

```bash
source .venv/bin/activate                     # Python 3.11 venv at repo root
python scripts/quality_gate.py --fast         # lint + compile + governance tests (~seconds)
python scripts/quality_gate.py --strict       # + full pytest, import checks, results integrity
python scripts/rules_lint.py                  # just the rule checks, verbose per-rule output
pytest -q                                     # full test suite
dah-baseline / dah-ablation / dah-analyze     # experiment CLIs (M3+; console scripts from pyproject)
```

The **fast gate green** is the precondition for saying "done" on anything (R15) — the
`stop` hook runs it and will bounce you back if it's red. The **strict gate** closes
milestones.

## Repo map

```
src/dah/
  types.py            # Fact/Turn/Task/ContextItem/AssembledContext — shared data model
  budget.py           # Budget: hard charge + soft pressure() in [0,1]; no refunds ever
  checkpoint.py       # Checkpointer over Snapshotable components; deep, total snapshots
  telemetry.py        # Trace.emit — the ONLY way library code reports what happened
  verifier.py         # oracle-free ConsistencyVerifier -> Verdict (empty-recall, low-relevance)
  recovery.py         # RecoveryPolicy: ingestion retries + rollback-before-retry
  planner.py          # StaticPlanner vs BudgetAwarePlanner (allocation, focused retries)
  faults.py           # seeded FaultInjector — own RNG stream, rate 0 == no injector
  agents.py           # one toggleable Agent loop; BaselineAgent vs HarnessAgent configs
  memory/
    stores.py         # WorkingMemory / Compactor / EpisodicStore / SemanticStore
    manager.py        # MemoryConfig + MemoryManager + 10 PRESETS — the ablation surface
  env/
    tasks.py          # synthetic multi-session task generator (owns ground truth)
    multisession.py   # MultiSessionEnv — sole grader; frozen metrics on EpisodeResult
  providers/
    base.py           # Provider protocol + ModelResponse
    simulated.py      # deterministic context-degradation model (length decay × middle loss)
    openai_compat.py  # optional live-LLM adapter; sole sanctioned network egress (R2)
  experiments/
    config.py         # single source of truth for every number in RESULTS.md
    runner.py         # config -> tidy DataFrame studies; paired bootstrap (R4)
    run_ablation.py   # dah-ablation: memory matrix + noise/fill sweeps
    run_baseline.py   # dah-baseline: harness ladder + budget/tau/recovery studies
    analyze.py        # dah-analyze: figures + hypotheses.csv verdicts + summary.md
    gallery.py        # results/traces.md — three annotated, seed-reproducible episodes
    live_check.py     # manual live-endpoint smoke test (needs OPENAI_API_KEY)
scripts/              # rules_lint.py, quality_gate.py (governance machinery)
tests/                # pytest; test_governance.py enforces R5/R6/R7/R1 + hook behavior
results/              # GENERATED ONLY — never hand-edit (R3); `make reproduce` rebuilds all
docs/                 # DESIGN.md (methodology), RESULTS.md (findings), JOURNAL.md (log)
.cursor/hooks*        # guardrails — do not weaken without user approval (R13)
```

## Architecture invariants (violating these invalidates the science)

1. **The env owns truth.** Only `env/` and `experiments/` may read `required_fact_ids` or
   grade. Memory/planner/verifier/recovery must never see the answer key (R5 — the linter
   greps for the tokens; don't alias them to sneak past it, the test suite also checks
   behaviorally).
2. **The provider is blind.** `SimulatedProvider` rolls recall for every fact in context;
   it cannot favor required facts. A fact absent from assembled context has recall
   probability 0 — that asymmetry is what makes memory load-bearing.
3. **Same degradation model for every arm.** Memory policies win by *placing the right
   facts at accessible positions under a token budget*, never by changing accessibility
   parameters. If an experiment varies `AccessibilityParams`, it's a sweep axis, applied
   identically to all arms.
4. **Determinism end to end (R1).** Seeds in, identical bytes out. Every RNG is an
   explicitly seeded `np.random.default_rng`; RNG state goes into snapshots (copy the
   `Compactor` pattern: `self.rng.bit_generator.state` in `snapshot()`/`restore()`).
5. **Rollback restores state, not budget (R7).** Retries are paid for. This is the tension
   that makes budget-aware planning a real subject, so never "fix" it.
6. **Paired everything (R4).** Seed `i` = same task for every arm. Deltas get paired
   bootstrap CIs (≥ 10k resamples, seeded). ≥ 20 seeds/arm, default 30.

## Working style

- **One task at a time**, from `TASKS.md`, respecting milestone order. Mark it `[~]` when
  you start, `[x]` when the acceptance criteria hold and the fast gate is green.
- **Tests land with code (R11).** New module ⇒ new tests in the same change. The linter
  only checks a reference exists; you are responsible for the tests being real.
- **Sync the docs you touched (R12).** Feature reality changed ⇒ FEATURES.md row updated
  (status + evidence). New scope discovered ⇒ new task ID in TASKS.md.
- **Journal before you stop (R14).** Append to `docs/JOURNAL.md`: date, task IDs, what
  changed, evidence (gate/test output, result files), next step. Record dead ends — a
  failed approach is data the next session needs.
- **Experiments are code.** Change an experiment ⇒ regenerate its `results/` artifacts via
  the CLI; never edit CSVs (the results hook denies it) and never cherry-pick seeds
  (R3 — that's falsification).
- Python ≥ 3.10, 4-space indent, type hints on public APIs, `from __future__ import
  annotations` everywhere, docstrings that explain *why* (R10). No `print` in library code
  — `Trace.emit` (R9). No `pickle`/`eval`/`exec` (R8). No network in `src/dah/` except
  `providers/openai_compat.py` (R2).
- Dependencies: numpy/matplotlib/pandas (+pytest dev). Adding anything else needs a
  written rationale in the PR/journal and user approval.

## Guardrails you will encounter (don't fight them, don't weaken them)

| Trigger | Hook | What happens |
|---------|------|--------------|
| Editing GOAL.md / RULES.md / hooks / linter / gate / .githooks | `protect-governance.sh` (preToolUse) | **Denied** — propose the change to the user; they apply it or lift the guard (R13) |
| Editing `results/*` with any editor tool | `protect-governance.sh` (preToolUse) | **Denied** — regenerate via `make reproduce` (R3) |
| `git commit --no-verify`, force-push, `rm -rf` at dangerous paths | `shell-guard.sh` (beforeShellExecution) | **Denied** outright (R15) |
| `git config` writes, shell redirection into `results/`, sed -i on governance files | `shell-guard.sh` | **Approval card** shown to the user |
| Any Python edit under `src/`, `scripts/`, `tests/` | `lint-context.sh` (postToolUse) | Runs the rules linter; violations injected back as context |
| Declaring completion with a dirty tree | `stop-quality.sh` (stop, loop_limit 3) | Runs the fast gate; red gate bounces the completion; green-but-unjournaled code bounces once for R14 |

If a guardrail seems wrong, propose an amendment (RULES.md § Amendment protocol) — never
route around it. Weakening enforcement silently is the one unforgivable move in this repo.

## Current state & first moves (as of M4 complete)

- M0–M4 closed (incl. the T4.2 robustness re-run); the strict gate is green; all studies
  are generated and `docs/RESULTS.md` carries machine-computed H1–H5 verdicts (H1
  directional, H4 refuted-opposite — reported honestly in "What didn't work"). See
  `docs/JOURNAL.md` for session history, including discarded designs.
- Open items in order: stretch **T5.2** (live-LLM spot check, needs `OPENAI_API_KEY`),
  **T5.3** (`gap_triggered` policy: measure it or delete it), **T5.4** (`v0.1.0` tag —
  needs explicit user approval; no commits without it).
- If you change *any* experiment or provider/memory behavior, regenerate all of
  `results/` via `make reproduce` before touching RESULTS.md — numbers in docs must
  never drift from artifacts (R3/R12). FEATURES.md "Known gaps" lists the three
  tracked gaps; never re-litigate closed hypotheses without new runs.

## Definition of Done (summary — full version in GOAL.md)

Strict gate green; one-command regeneration of all results; RESULTS.md with H1–H5 verdicts
+ CIs + "What didn't work"; every FEATURES.md row `verified` with live evidence; a stranger
reproduces the headline number from README alone.
