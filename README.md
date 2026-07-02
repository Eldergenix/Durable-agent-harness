# durable-agent-harness

A compact, **research-grade harness for long-horizon agents**, plus an
**ablatable memory system**, a controlled multi-session benchmark, and a
**pre-registered, machine-verdicted study** of what actually helps.

It answers two questions every applied agent team eventually hits, with
reproducible numbers instead of vibes:

1. **What is the control-plane machinery worth?** Verification loops,
   checkpoint/rollback, budget-aware planning, and failure recovery — measured
   as a *feature ladder* against a naive baseline, holding memory, model, budget,
   and failures fixed.
2. **Which memory components matter as context fills?** Working memory, lossy
   compaction, episodic retrieval, semantic consolidation — measured on
   *multi-session* tasks where working context resets between sessions, so
   cross-session questions are unanswerable without real memory.

The study is governed the way you'd want a lab artifact governed: hypotheses
**H1–H5 were pre-declared** in [`GOAL.md`](GOAL.md) with falsifiable predictions
and frozen metric definitions; [`RULES.md`](RULES.md) (determinism, no
answer-key leakage, paired seeds, no hand-edited results) is enforced by a
machine linter (`scripts/rules_lint.py`), a quality gate
(`scripts/quality_gate.py`), governance tests, and repo hooks. Verdicts are
computed by code into `results/hypotheses.csv` — including the refuted ones.

---

## Headline results

**The harness converts budget into correctness; the baseline can't.** At a
binding 18k-token budget with 12% transient step failures, switching on the full
control plane over the *same* memory and model moves QueryAccuracy **0.293 →
0.405** (paired delta **+0.112, 95% CI [+0.092, +0.133]**, +38% relative), cuts
confident-hallucination **0.301 → 0.169**, and eliminates lost ingestions
(**6.7 → 0.01 per task**). As budget grows, the baseline saturates at ~0.30
while the full harness climbs to **0.433** — it knows how to *spend*.

![budget sweep](results/fig_budget_sweep.png)

**A clean negative result on retrieval.** Episodic (similarity) retrieval is the
best single memory component with clean embeddings — and collapses from **0.398
to 0.114** cross-session accuracy as retrieval noise rises, **dropping below
dumb-but-robust persistent compaction (flat 0.291) at noise ≈0.4–0.5**.
Configurations that pair retrieval with a retrieval-independent channel (`full`)
hold ~0.36–0.38 across the whole range. If you can't vouch for your embeddings,
retrieval-only memory is the wrong default.

![noise sweep](results/fig_noise_sweep.png)

**Pre-registration kept us honest.** Of five pre-declared hypotheses: H5
(recovery ≥60% at ≤25% overhead) passed with an order of magnitude to spare
(~99% step recovery at ~0% overhead); H2 (compaction most valuable at tight
windows) held with an overlap caveat; H1 was **directional but missed its
pre-registered +15pp bar** (+5.2 to +13.4pp, CIs excluding zero at every
budget); H3's predicted within-session *harm* from retrieval never materialised;
and H4 was **refuted in the opposite direction** — salience-gated semantic
memory's value *grows* with retrieval noise (fewer stored items ⇒ fewer served
distractors). Full verdict table with CIs: `results/hypotheses.csv`, narrated in
[`docs/RESULTS.md`](docs/RESULTS.md).

---

## Why a simulation (and why that's a feature, not a cop-out)

Honest ablations need (a) a controllable ground truth and (b) enough repetitions
for tight confidence intervals. Neither is practical against a live LLM: cost,
flakiness, and no oracle for *"which fact did the model actually use?"*. So the
default provider is a **deterministic model of the two best-documented
long-context failure modes** — length decay and lost-in-the-middle — and nothing
else:

![accessibility model](results/fig_accessibility.png)

The critical property: **the degradation model is fixed across every condition**
and blind to which memory policy is running. A policy can only win by placing
the right facts at accessible positions under a token budget — never by changing
the scoring rules. Parameters are **knobs, not fitted constants**; the sweeps
show how the winning policy changes along each axis rather than claiming one
true value. And the abstraction is real: an OpenAI-compatible provider runs the
identical agent/memory/planner/verifier stack against live models.

---

## Architecture

```
                         ┌─────────────────────────────────────────────┐
                         │                 Agent loop                   │
                         │ plan → (ingest | answer) → verify → recover  │
                         └─────────────────────────────────────────────┘
   budget-aware planner ─────┘        │            │            └──── recovery policy
   (allocation + focused retries)     │            │                  (retry + rollback)
                                      ▼            ▼
                            ┌──────────────┐  ┌────────────────────┐
                            │ Checkpointer  │  │ ConsistencyVerifier │
                            │ commit/rollbk │  │ oracle-free Verdict │
                            └──────────────┘  └────────────────────┘
                                      │
              ┌───────────────────────┴───────────────────────┐
              │                MemoryManager                    │
              │  working · compaction · episodic · semantic      │
              │  + retrieval policy → AssembledContext           │
              └───────────────────────┬─────────────────────────┘
                                      ▼
                     ┌──────────────────────────────┐
                     │   Provider (blind to answers) │
                     │  Simulated  |  OpenAI-compat   │
                     └──────────────────────────────┘
                                      ▲
                          Environment owns ground truth
                    (facts, query embeddings, ALL grading)
```

Design rules that make the numbers trustworthy (enforced, not intended — see
`RULES.md` and `tests/test_governance.py`):

- **One code path, toggled by flags.** `BaselineAgent` and `HarnessAgent` are
  configurations of the same `Agent` loop — no separate "good" implementation.
- **The answer key never leaves the environment.** Agents hand raw responses to
  `env.score_query` and get graded records back; the tokens `required_fact_ids`
  / `confident_wrong` are grepped out of memory/planner/verifier/recovery/agents
  by linter *and* test.
- **The verifier is oracle-free.** It judges answers from query/recall embedding
  geometry only, so its error profile is earned, not assumed.
- **Determinism end to end.** Every RNG is explicitly seeded; RNG state rides in
  snapshots; a zero-rate fault injector is provably bit-identical to none.

---

## Quickstart

```bash
make install          # venv + editable install with dev extras
make test             # 70 tests: unit + governance + reproducibility
make reproduce        # full study (~4 min): CSVs + figures + verdicts in results/
python scripts/quality_gate.py --strict   # whole-repo health in one exit code
```

Fast smoke of the entire pipeline: `make quick`.

A minimal end-to-end program:

```python
from dah.env import TaskConfig, generate_task, MultiSessionEnv
from dah.providers import SimulatedProvider, AccessibilityParams
from dah.memory import make_memory
from dah.agents import Agent
from dah.planner import StaticPlanner

task = generate_task(TaskConfig(), seed=1)
prov = SimulatedProvider(AccessibilityParams(), topic_of={f: v.topic for f, v in task.facts.items()})
env  = MultiSessionEnv(task, prov, seed=1)
mem  = make_memory("full", task.facts, seed=1)

result = Agent(mem, prov, StaticPlanner(), seed=1).run(env)
print(result.query_accuracy(), result.query_accuracy(cross=True))
```

---

## Repository layout

| Path | What |
|------|------|
| `src/dah/budget.py` | Multi-dimensional budget: hard `charge` (no refunds) + soft `pressure()` |
| `src/dah/checkpoint.py` | Checkpoint stack with deep, total rollback over a component registry |
| `src/dah/verifier.py` | Oracle-free `ConsistencyVerifier` → `Verdict`; deterministic |
| `src/dah/planner.py` | `StaticPlanner` vs `BudgetAwarePlanner` (budget allocation + focused retries) |
| `src/dah/recovery.py` | Retry/rollback recovery policy |
| `src/dah/faults.py` | Seeded `FaultInjector`; rate 0 ≡ no injector, bit-for-bit |
| `src/dah/agents.py` | One agent loop; `BaselineAgent` / `HarnessAgent` configurations |
| `src/dah/memory/` | Working / Compactor / Episodic / Semantic stores + `MemoryManager` + presets |
| `src/dah/env/` | Multi-session task generator + environment (owns *all* grading) |
| `src/dah/providers/` | `Provider` protocol, `SimulatedProvider`, optional `OpenAICompatProvider` |
| `src/dah/experiments/` | `config.py` (single source of truth), `runner.py`, drivers, `analyze.py` |
| `scripts/` | `rules_lint.py` + `quality_gate.py` — machine enforcement of RULES.md |
| `tests/` | Unit, reproducibility, and governance tests |
| `docs/` | `DESIGN.md` (method), `RESULTS.md` (findings), `JOURNAL.md` (session log) |
| `GOAL.md` / `RULES.md` / `FEATURES.md` / `TASKS.md` / `AGENTS.md` | Pre-registered goals, enforced rules, live inventory, plan, agent manual |

---

## Running against a real model

The simulated provider is the default because it makes the study reproducible.
The interface is real, though:

```bash
pip install -e ".[openai]"
export OPENAI_API_KEY=...              # optionally OPENAI_BASE_URL for vLLM/Together/Ollama
python -m dah.experiments.live_check   # one-question live smoke test
```

`OpenAICompatProvider` implements the same `Provider` protocol, so the agent,
memory stack, planner, verifier, and checkpointer run unchanged. Oracle grading
(fact-id recall) is specific to the simulation; grading a live model's free-form
answers needs a judge, which is a clearly-marked extension point in
`src/dah/providers/openai_compat.py`. The deterministic study in
`docs/RESULTS.md` intentionally does **not** depend on a live model.

---

## Limitations (read before over-trusting the numbers)

Absolute success levels come from a *model* of context degradation, not a
specific LLM — treat them as relative comparisons inside a controlled world.
The retrieval negative result is a claim about the noisy-embedding,
distractor-dense regime (mechanism: length-driven dilution); cleaner retrieval
moves the crossover without removing the trade-off. The verifier results
characterise one embedding-based self-check signal, not critics in general.
`docs/DESIGN.md` §8 lists every threat to validity and the knob that would
overturn each conclusion.
