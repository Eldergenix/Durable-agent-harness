# RULES.md — Non-negotiable engineering rules

> These rules are **enforced by machines** wherever possible. Column "Enforced by" tells you
> which mechanism will catch a violation: `lint` = `scripts/rules_lint.py`,
> `gate` = `scripts/quality_gate.py`, `test` = `tests/test_governance.py`,
> `hook` = `.cursor/hooks/*`, `review` = human/agent judgment.
> Changing this file or GOAL.md requires explicit user approval (R13); the edit hook
> denies agent edits to frozen files — the user applies the change directly or
> explicitly authorizes lifting the guard for one change.

## Index

| ID  | Rule (short form) | Enforced by |
|-----|-------------------|-------------|
| R1  | Determinism: all randomness from explicit seeds | lint |
| R2  | Isolation: no network I/O in the library | lint |
| R3  | Results are generated artifacts; negative results are never buried | hook + lint |
| R4  | Statistical floor: ≥ 20 paired seeds, bootstrap CIs on deltas | lint + review |
| R5  | No answer-key leakage into memory/planner/verifier code | lint + test |
| R6  | Snapshot totality: rollback restores bit-identical behavior | test |
| R7  | Budget conservation: no refunds, no negative charges | test |
| R8  | No `pickle`, `eval`, or `exec` in the library | lint |
| R9  | Library code is silent: telemetry, not `print` | lint |
| R10 | API hygiene: docstrings + `from __future__ import annotations` | lint |
| R11 | Tests land with code, in the same milestone | lint + review |
| R12 | Docs stay synchronized with reality | lint |
| R13 | GOAL.md and RULES.md are frozen without user approval | hook |
| R14 | Work sessions leave a journal trail | review (hook reminder) |
| R15 | Nothing is "done" while the quality gate is red | hook |

---

## R1 — Determinism

Every stochastic choice flows from an explicitly seeded `numpy.random.Generator`
(`np.random.default_rng(seed)`), threaded through call signatures — never global state.

- Forbidden in `src/dah/`: `import random`, `np.random.seed(...)`, no-argument
  `default_rng()`, `time.time()`, `datetime.now()` as inputs to any decision.
  (`time.perf_counter()` is allowed for telemetry timestamps only.)
- RNG state is part of component state: anything holding an RNG must include
  `bit_generator.state` in its snapshot (see `Compactor` for the pattern).
- Two runs with the same seed and config must produce byte-identical metrics.

## R2 — Isolation

The library (`src/dah/`) performs no network I/O. Simulation is the point: ablations must
be free, fast, and hermetic.

- Forbidden imports in `src/dah/`: `requests`, `httpx`, `aiohttp`, `urllib`, `socket`.
- Sole exception: `src/dah/providers/openai_compat.py` (the sanctioned egress point,
  optional extra, never imported by default).

## R3 — Results integrity

- Files under `results/` are **generated** by `dah-baseline` / `dah-ablation` /
  `dah-analyze` (equivalently `make reproduce`). Hand-editing them is forbidden; the edit
  hook denies any editor-tool write into `results/`. Regenerate instead.
- Every results CSV carries `seed` and `config_id` columns so any row can be traced to the
  exact code path that produced it.
- `docs/RESULTS.md` must always contain the sections: `Headline`, `Ablation matrix`,
  `Failure recovery`, `What didn't work`, `Reproduce`. A negative or neutral finding goes
  into "What didn't work" with the same rigor as a win. Deleting a negative result, or
  quietly re-running until a favorable seed appears, is falsification.

## R4 — Statistical floor

- ≥ 20 paired seeds per experimental arm (30 is the default). Seed `i` generates the same
  task instance for every arm, so all comparisons are paired.
- Every reported delta carries a 95% CI from a paired bootstrap with ≥ 10,000 resamples.
- No metric definitions may change after data collection without re-running all arms
  (metric definitions live in GOAL.md and are frozen).
- Post-hoc analyses are welcome but must be labeled `exploratory` in RESULTS.md.

## R5 — No answer-key leakage

The environment owns the ground truth (`Turn.required_fact_ids`, grading). Code that
*assembles context or makes decisions for the agent* must never touch it.

- The tokens `required_fact_ids` and `confident_wrong` are forbidden in
  `src/dah/memory/`, `src/dah/planner.py`, `src/dah/verifier.py`, `src/dah/recovery.py`.
- Memory may use env-provided metadata that a real system would have (embeddings, token
  counts, salience) — but never which facts a future query needs.
- Only `src/dah/env/` and `src/dah/experiments/` may grade responses.
- The provider is blind too: `SimulatedProvider` rolls recall for every fact in context
  without knowing which are required (its `required_fact_ids` access is confined to the
  post-hoc confusion metric, which does not affect recall).

## R6 — Snapshot totality

Any component registered with `Checkpointer` implements `snapshot()`/`restore()` such that:

- The snapshot is a deep copy — later mutation of live state must not alter it.
- `restore()` reproduces bit-identical future behavior, **including RNG draws**.
- Rolling back and replaying the same inputs yields the same outputs (verified by
  round-trip tests in `tests/test_governance.py`).

## R7 — Budget conservation

`Budget` spend is monotonically non-decreasing. Rollback restores *state*, never budget —
you do not get tokens back for work you already paid for (that asymmetry is what makes
budget-aware planning a real problem).

- No method may reduce spend; negative charge amounts raise `ValueError`.
- `BudgetExceeded` must report the true post-charge spend (for post-mortem telemetry).

## R8 — No dynamic code, no pickle

`pickle`, `eval(`, `exec(` are forbidden in `src/dah/`. Checkpoints are in-memory deep
copies; anything persisted to disk is JSON or CSV.

## R9 — Library silence

No `print` in `src/dah/` — components speak through `Trace.emit`. Exception:
`src/dah/experiments/` (CLI entry points) may print user-facing output.

## R10 — API hygiene

Every non-`__init__` module in `src/dah/`:

- has a module docstring that explains *why the module exists*, not just what it does;
- includes `from __future__ import annotations`;
- gives every public class and public module-level function a docstring.

## R11 — Tests land with code

Every non-`__init__` module under `src/dah/` is referenced by name in at least one file
under `tests/`. This is a floor, not a target: the linter checks the reference exists; the
milestone review checks the tests are real. A feature moves to `verified` in FEATURES.md
only when its tests actually exercise its contract.

## R12 — Docs synchronized with reality

- Every module under `src/dah/` appears in the FEATURES.md inventory.
- FEATURES.md statuses come from the enum `planned | in-progress | implemented | verified`.
- A `verified` row must cite evidence (test path or results path) that exists on disk.
- TASKS.md task IDs (`T<milestone>.<n>`) are unique.
- GOAL.md, RULES.md, FEATURES.md, TASKS.md, AGENTS.md, README.md, docs/JOURNAL.md all exist.

## R13 — Frozen governance

GOAL.md (mission, hypotheses, metric definitions) and RULES.md change only with explicit
user approval. The pre-edit hook **denies** agent writes to these files; on approval the
user applies the edit directly (hooks do not intercept the human editor) or temporarily
lifts the guard in `.cursor/hooks.json`. Guardrail infrastructure (`.cursor/hooks/`,
`scripts/rules_lint.py`, `scripts/quality_gate.py`, `.githooks/`) gets the same treatment:
an agent must not weaken its own enforcement without the user seeing it happen.

## R14 — Journal trail

Any session that changes behavior (code or experiments) appends an entry to
`docs/JOURNAL.md`: date, what changed, evidence (gate output, test names, result files),
and what's next. Failed approaches are recorded — they are data.

## R15 — Gate before done

- `python scripts/quality_gate.py --fast` must be green before an agent declares any task
  complete. The `stop` hook runs it and will bounce a completion attempt while it's red.
- `python scripts/quality_gate.py --strict` (adds full pytest, package import checks,
  results integrity) must be green to close a milestone or tag a release.
- Committing with `--no-verify`, force-pushing, and hand-editing `git config` are blocked
  by the shell hook.

---

## Amendment protocol

1. Propose the change as a diff to this file with a one-paragraph rationale.
2. Get explicit user approval in-conversation (the R13 hook will surface the edit anyway).
3. Record the amendment in `docs/JOURNAL.md` with date and reason.
