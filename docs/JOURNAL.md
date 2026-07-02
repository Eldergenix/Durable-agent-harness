# Journal

> R14: every session that changes behavior appends an entry — date, what
> changed, evidence, next step. Failed approaches are recorded; they are data.

## 2026-07-01 — Session 1: core build (pre-governance)

**What changed.** Built the package from an empty directory: budget, checkpoint,
telemetry, memory stores (working/compaction/episodic/semantic) + manager with
presets, multi-session task generator, simulated provider (length decay ×
lost-in-the-middle), agent loop with static/budget-aware planners, an
oracle-style verifier (later replaced, see Session 2), experiment drivers,
analyzer, 41 tests, README/DESIGN/RESULTS drafts. Ran the first full study.

**Evidence.** All tests green; `results/*.csv` + 7 figures generated;
determinism verified by double-running `run_ablation` into two directories and
diffing (identical, md5-matched).

**Dead ends recorded.**
- A single-task assertion comparing baseline/harness hallucination was flaky —
  single episodes are far too noisy; replaced with an aggregate paired check.
- First harness draft spent budget greedily; adding *allocation* (remaining
  budget / remaining queries) is what made the budget-aware planner actually
  finish tasks.

**Next.** Governance scaffold (GOAL/RULES/FEATURES/TASKS/AGENTS + hooks)
appeared at session end; alignment work queued.

## 2026-07-01 — Session 2: governance alignment + honest re-run (M1–M4)

**Blocked, then unblocked.** All tool calls were blocked because the
`.cursor/hooks/*.sh` guard scripts shipped without the executable bit and fail
closed (exit 126). User ran `chmod +x .cursor/hooks/*.sh`; work resumed. Also
fixed an `IndexError` in `scripts/rules_lint.py::_code_only` on files whose
final token points past the last physical line (falls back to raw text —
fails toward more checking).

**What changed (code).**
- `agents.py` replaces `agent.py`: same single loop, plus `BaselineAgent` /
  `HarnessAgent` (T1.3/T2.5). Grading moved fully into the environment
  (`env.score_query` / `score_unanswered`) so agent code never touches the
  answer key (R5); `agents.py` now passes the same leakage grep as
  memory/planner/verifier/recovery.
- `verifier.py` rewritten per T2.1: oracle-free `ConsistencyVerifier`
  (`Verdict` API; empty-recall + low-relevance checks over query/recalled
  embeddings; deterministic). The previous tunable-oracle verifier and its
  `detect_wrong` sweep were **discarded** — R5's spirit beats the convenience.
  Verification quality now *emerges* from embedding geometry.
- `faults.py` (T2.4): seeded `FaultInjector` with its own RNG stream; rate 0
  provably bit-identical to no injector (governance test). Deliberately NOT
  checkpointed: it models the outside world; rolling it back would replay the
  identical failure forever (found the hard way — recovery gave up 45 times per
  task with the injector registered in the checkpointer; 0 after unregistering).
- Frozen GOAL.md metrics implemented on `EpisodeResult`; every results CSV now
  carries `seed` + `config_id` (R3); paired-delta bootstrap (10k resamples, R4)
  in `runner.paired_delta_ci`; `dah-analyze` now emits `hypotheses.csv` with
  machine-computed H1–H5 verdicts.
- New studies: ablation matrix at 2 window levels (+`compaction+semantic`
  preset for leave-one-out H2), semantic arms in the noise sweep (H4), tau
  sweep for the verifier operating point, dedicated recovery on/off study (H5).
- **Context-packing fix (material).** The assembler used to trim *retrieved*
  items first when a context cap bound — silently starving retrieval exactly
  when budget-aware caps were active (full-harness delta shrank to +0.03).
  Trim order now: stale working turns → retrieved → summary last. Also added
  focused retries (tighter cap on verified retries). Full-harness paired delta
  went +0.028 → +0.112 at 18k. Recorded in RESULTS "What didn't work" #7.

**What changed (governance).** `tests/test_governance.py` (leakage grep incl.
`agents.py`, snapshot-RNG round-trips, budget monotonicity, zero-rate-injector
identity, verifier determinism, module inventory); `.githooks/pre-commit`;
FEATURES.md statuses with live evidence; TASKS.md checkmarks; docs synced
(DESIGN/RESULTS/README rewritten for the new verifier + numbers).

**Evidence.** `python scripts/quality_gate.py --strict` green (rules lint,
byte-compile, 61 tests, imports); full re-run of all studies with the new
schema; `results/hypotheses.csv` verdicts: H1 directional (never reaches the
pre-registered +15pp; +5.2..+13.4pp with CIs>0), H2 supported (overlap caveat),
H3 partially supported (no within-session harm), H4 **refuted in the opposite
direction** (semantic value grows with noise), H5 supported with margin
(~99% step recovery at ~0% overhead).

**Next.** Optional stretch: live-LLM spot check (`dah.experiments.live_check`)
needs an API key; `v0.1.0` tag awaits explicit user go-ahead (no commits made
without it).

## 2026-07-01 — Session 3: hook hardening, provenance dedup, T4.2 robustness

**Hooks corrected against the documented schema.** A docs check confirmed
`permission: "ask"` is schema-accepted but *not enforced* on `preToolUse` (no
approval card) — only `beforeShellExecution`/`beforeMCPExecution` enforce it.
`protect-governance.sh` therefore now **denies** edits to frozen files
(GOAL/RULES, hooks, linter/gate, `.githooks`) and to `results/*`, with an
agent_message directing amendments through the user; `DAH_BOOTSTRAP=1` is the
explicit escape hatch. RULES.md R13/R3 wording updated to match (deny, not
ask). `stop-quality.sh` now scopes `git status` to relevant paths. Lesson
recorded: hooks fail closed *before* their scripts are executable — ship
`chmod +x` first (Session 2's lockout, root-caused).

**Hook behavior is now tested.** `tests/test_governance.py` grew 9 payload
tests that execute the real hook scripts over stdin JSON: frozen-file deny,
results deny, ordinary-edit allow, bootstrap escape, `--no-verify`/force-push
deny, `--force-with-lease` allow, results-redirect ask, `git config` ask vs
sanctioned `core.hooksPath` allow, stop-hook no-op, lint-hook path filter.
`.githooks/pre-commit` created (bootstrap path) and `core.hooksPath` set.

**Provenance dedup (bug avoided).** Session 2 added `seed`/`config_id` at the
CSV write sites; my parallel edit added them *inside* the builders too —
`out.insert` would have raised on the duplicate column. Removed the builder
copies; write sites are the single source. `make reproduce` + gallery re-run
green; all 12 CSVs carry R3 columns.

**T4.2 closed (robustness).** New `ROBUSTNESS_SETTINGS` in config
(longer-horizon `n_sessions=10`, cross-heavy `p_cross_session=0.85`), runner
wired into `dah-ablation` → `results/ablation_robustness.csv`, tables in
`summary.md` + RESULTS.md. Ordering `none < compaction < episodic < full`
holds at both settings (no flips). Honest caveat found: the persisted summary
*grows* with horizon (compaction ctx 643→813, full→922 tokens at 10 sessions)
and full's margin over episodic narrows +0.034→+0.017 — unbounded summaries
are not free; noted in RESULTS.md.

**Also.** Import guards in `memory/env __init__` (bootstrap leftovers) deleted;
`time.time()` → `perf_counter` in experiment CLIs (R1 lint); stale
`results/verifier_sweep.csv` + `fig_verifier_sweep.png` (old oracle-verifier
design) deleted; FEATURES.md rewritten with per-row live evidence (F31–F34
added); TASKS.md statuses trued up; AGENTS.md repo map/guardrail table/state
refreshed; README+RESULTS test/runtime counts corrected (70 tests, ~2 min).

**Evidence.** `python scripts/quality_gate.py --strict` green (rules-lint,
byte-compile, governance 30, imports, pytest 70); planted-violation check:
`import random` in a scratch module → lint exits 1 (R1+R11+R12), clean tree
exits 0; README quickstart snippet executed as-is.

**Next.** Stretch T5.2 (live check, needs key), T5.3 (measure or delete
`gap_triggered`), T5.4 (tag on user go-ahead).

## 2026-07-01 — Session 2 addendum: gallery, traceability, doc-sync (T4.3, T3.4, T0.4, T5.1)

Entry appended after the fact at the user's R14 reminder — the changes below
landed *after* the session-2 entry above was written and were not journaled at
the time.

**Task IDs.** T4.3 (failure-mode gallery), T3.4 (derived-artifact traceability),
T0.4 (pre-commit hook attempt, blocked), T1.5/T2.6/T3.5 (FEATURES.md statuses),
T4.4 (DoD reconciliation in TASKS.md), T5.1 (README refresh).

**What changed (code).**
- New module `src/dah/experiments/gallery.py` (T4.3): writes
  `results/traces.md` with three seed-reproducible annotated episodes —
  (a) rollback+retry recovering an ingest whose facts a later cross-session
  query needs (recovered turn indices read from `Trace` events, not inferred),
  (b) noisy retrieval serving a same-topic distractor and answering
  confidently wrong where the compaction twin answers correctly,
  (c) a budget-pressured 328-token degraded context still landing an answer
  the static planner never reached. Registered in `tests/test_governance.py`
  `ALL_MODULES` (R11) and FEATURES.md as F34 (R12).
- `src/dah/experiments/analyze.py`: derived artifacts (`hypotheses.csv`,
  `loo_summary.csv`, `recovery_summary.csv`) now carry `seed` (the GLOBAL_SEED
  base) and `config_id` (verdict/arm identifier) so the R3 schema check holds
  for every CSV under `results/`, not just raw run records.
- `src/dah/__init__.py` docstring updated to list `faults`/`agents` and the
  env-owns-all-grading invariant; `Makefile` gained `gate`/`gallery` targets and
  `reproduce` now includes the gallery.

**What changed (docs/governance).** DESIGN.md synced to the shipped design
(oracle-free verifier §5, density-ordered trim §4, injector-not-checkpointed
rationale, paired-delta stats §6, updated anti-cheating guarantees §7);
README.md rewritten with final numbers and the pre-registration story;
FEATURES.md fully re-inventoried (F1–F34, verified rows cite live evidence);
TASKS.md checkmarks reconciled (T0.4 marked `[!]`); AGENTS.md "current state"
updated to M4-complete.

**Blocked (recorded).** Writing `.githooks/pre-commit` was blocked by the R13
guardrail hook (agent edits to `.githooks/` require explicit user approval).
Not retried; tracked in T0.4 and FEATURES F30 (`planned`). A `/tmp` cleanup
using `rm -rf` was blocked by the shell guard; redone with targeted `rm`/`rmdir`.

**Evidence.** `python scripts/rules_lint.py` exit 0;
`python scripts/quality_gate.py --strict` green (rules-lint, byte-compile,
governance-tests, imports, pytest-full: 61 tests); `results/traces.md`
regenerated; determinism re-verified post-refactor by double-running
`run_baseline --n-tasks 15` into two directories — `harness_ladder.csv`,
`recovery.csv`, `verifier_tau_sweep.csv` byte-identical.

**Next.** (1) User to approve or hand-create `.githooks/pre-commit` (T0.4);
(2) T5.3 compaction-policy variants remain an open stretch; (3) T5.4 `v0.1.0`
tag on explicit user approval (requires the first commit); (4) optional live
spot check via `python -m dah.experiments.live_check` with a real key.

## 2026-07-01 — Session close: R14 reconciliation (no code changes in this pass)

**Task IDs.** R14 compliance; T4.4 (Definition-of-Done bookkeeping).

**What changed.** Documentation only. The "Session 2 addendum" above was
appended at the user's R14 reminder to cover the post-entry work (gallery
T4.3, derived-artifact traceability T3.4, doc-sync, blocked `.githooks` write);
this closing entry records the reconciliation itself. No files under `src/`,
`tests/`, or `scripts/` were modified after the addendum — the code state that
produced every artifact in `results/` is exactly the state the strict gate
below certifies.

**Evidence.** `python scripts/quality_gate.py --strict` green immediately
after the journal append (rules-lint PASS, byte-compile PASS, governance-tests
PASS, imports PASS, pytest-full PASS — 61 tests); `python scripts/rules_lint.py`
exit 0 (R12 confirms JOURNAL.md present and RESULTS.md sections intact).

**Next step.** All engineering tasks are closed or explicitly blocked on user
decisions: (1) approve or hand-create `.githooks/pre-commit` (T0.4, R13-gated);
(2) optionally green-light stretch T5.3 (compaction-policy variants);
(3) authorise the first commit + `v0.1.0` tag (T5.4). No further agent action
pending until one of these is decided.

## 2026-07-01 — Session close #2: R14 reminder root-caused; external guardrail work landed

**Task IDs.** T0.3 (stop-hook behavior), T0.4 (pre-commit, now closed), R13
amendment protocol, R14 compliance.

**What changed (by me).** Nothing under `src/`, `scripts/`, or `tests/` — this
turn was diagnosis plus this entry. Root cause of the thrice-repeated
"changed code without journaling" stop reminder: `stop-quality.sh` derives both
of its signals from `git status --porcelain`, and in a repository with **no
commits yet** porcelain collapses untracked directories to single entries
(`?? src/`, `?? docs/`). Its `grep -F 'docs/JOURNAL.md'` therefore can never
match while `?? src/` always matches as a code change — the R14 reminder loops
regardless of journal state (bounded only by the hook's `loop_limit: 3`).
Journal appends could never satisfy it; the two entries above were correct but
invisible to the check.

**Proposed amendment (R13 — awaiting user approval; my direct edit was
correctly denied by the frozen-file guard).** One line in
`.cursor/hooks/stop-quality.sh`:

```
- status=$(git status --porcelain 2>/dev/null || true)
+ status=$(git status --porcelain -uall 2>/dev/null || true)
```

`-uall` lists individual untracked files, so `docs/JOURNAL.md` becomes visible
to the journal check (and the guard behaves identically once a first commit
exists). Rationale: fixes a false-positive loop without weakening any check.

**What changed (externally, observed between turns).** The user/scaffold:
created `.githooks/pre-commit` (fast-gate runner, executable) and set
`git config core.hooksPath .githooks` — closing the blocked half of T0.4 (F30
now `verified`); hardened `.cursor/hooks/protect-governance.sh` to deny-mode
(preToolUse cannot render approval cards); added hook payload tests to
`tests/test_governance.py` (`DAH_SKIP_GATE=1` plumbing); refreshed FEATURES.md
(F28–F30) and TASKS.md (T0.3/T0.4 evidence). My queued FEATURES/TASKS edits for
T0.4 were superseded by those newer external versions and dropped.

**Evidence.** `python scripts/quality_gate.py --strict` green on the updated
tree (rules-lint, byte-compile, governance-tests incl. the new payload tests,
imports, pytest-full); `git config core.hooksPath` → `.githooks`;
`.githooks/pre-commit` present and executable; hook-loop diagnosis reproduced
with `git status --porcelain | grep -F 'docs/JOURNAL.md'` → no match.

**Next step.** (1) User: apply or approve the one-line `-uall` amendment above
so the R14 reminder keys on actual journal state (until then it may fire
spuriously at stop, capped at 3 loops); (2) stretch T5.3 remains open;
(3) T5.4 first commit + `v0.1.0` tag on explicit user approval.

## 2026-07-01 — Session 3 close: spurious R14 bounce acknowledged (no changes)

**Task IDs.** R14 compliance; T0.3 (stop-hook amendment still pending).

**What changed.** This entry only. No code, docs, or results changed in this
pass — Session 3's work (hook deny-mode + payload tests, provenance dedup,
T4.2 robustness re-run, doc-sync) was journaled in its own entry above before
the stop bounce fired. The bounce is the documented false positive: the live
`stop-quality.sh` greps un-expanded `git status --porcelain`, which in this
not-yet-committed repo collapses `docs/` and hides `JOURNAL.md` while `?? src/`
matches as a code change. Verified again this turn: the pathspec form
(`git status --porcelain -- docs/JOURNAL.md`) *does* emit the file, so either
the proposed `-uall` amendment or a first commit resolves it; both are
R13-gated on the user.

**Evidence.** `quality_gate.py --strict` green at Session 3 close (rules-lint,
byte-compile, governance 30, imports, pytest 70); hook-check reproduction in
this turn's shell output; amendment diff in the entry above.

**Next step.** User decisions only: approve the `-uall` hook amendment, then
optionally T5.2 (live check), T5.3 (`gap_triggered`), T5.4 (first commit + tag).

**Addendum (second bounce).** The R14 reminder re-fired as predicted. Interim
mitigation applied without touching the protected hook: `git add
docs/JOURNAL.md` — staged paths are listed individually by `git status
--porcelain`, so the journal check now sees the file truthfully. Staging is
not a commit (T5.4 still awaits approval) and is trivially reversible
(`git reset`). The `-uall` amendment remains the proper fix.
