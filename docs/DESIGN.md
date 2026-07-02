# Design & methodology

This document explains how the harness, the memory stack, and the benchmark are
built, and — most importantly — **why the ablations are honest**. If you only
read one section, read [Anti-cheating guarantees](#anti-cheating-guarantees) and
[Threats to validity](#threats-to-validity).

---

## 1. The problem

Long-horizon and multi-session agents degrade as context fills. The community
has a grab-bag of mitigations — compaction/summarisation, episodic memory,
retrieval, semantic consolidation — applied largely by intuition. Separately,
"robust agent" machinery (verification, rollback, budget management, retries) is
folklore. This repo builds a small, controlled world where both can be
*measured* rather than asserted, cheaply and reproducibly.

Two studies:

- **Study 1 — Memory ablation.** Hold the harness fixed (no failures, generous
  budget, static plan) and vary only the memory stack.
- **Study 2 — Harness ladder.** Hold the memory fixed and switch harness
  features on one at a time under a *binding* budget and injected failures.

---

## 2. The benchmark environment

A **task** is a list of **sessions**; each session is a list of **turns**; a turn
is either an *info* turn (introduces facts) or a *query* turn (asks a question
that requires a specific set of facts to answer).

Key structural choices (`src/dah/env/tasks.py`):

- **Multi-session ⇒ working context resets between sessions.** This is what makes
  external memory necessary: a query in session *s* that needs a fact from
  session *s−1* is simply unanswerable from working memory alone.
- **Cross-session dependencies** are injected on purpose (`p_cross_session`), so a
  large fraction of queries probe long-horizon memory rather than local recall.
- **Topic-clustered facts + synthetic embeddings.** Each fact embeds near one of
  `n_topics` random unit centroids with additive noise. Same-topic facts are
  natural **distractors** for similarity retrieval, so retrieval *precision*
  (not just recall) matters.
- **Multi-fact queries** (`max_required_facts`) require several facts at once
  (logical AND), so recall errors compound — a realistic stressor.
- **Tunable context fill** (`filler_tokens_per_turn`) controls how fast the window
  fills, i.e. how much long-horizon pressure the task applies.

The environment owns the **ground truth** and does all **grading**. A query is
*correct* iff every required fact is in the model's recalled set (strict AND). It
also computes a **noisy query embedding** for retrieval: the mean of the required
facts' embeddings plus Gaussian noise — so retrieval is imperfect from the query
side too, not just the store side.

---

## 3. The provider: a fixed model of context degradation

The simulated provider (`src/dah/providers/simulated.py`) models the probability
that the model *uses* a fact that is present in the assembled context. It is
deliberately minimal — two effects, both well documented:

```
accessibility(L, rel) = a_max · g(L) · h(rel)

  g(L)   = 1 / (1 + (L / l_half)^gamma)          # length decay
  h(rel) = m_mid + (1 - m_mid) · (2·rel - 1)^2   # lost-in-the-middle (U-shaped)
```

- `L` is the total assembled-context token count; `rel ∈ [0,1]` is a fact's
  relative position (0 = start, 1 = end).
- `g(L)` falls monotonically with context length; `l_half` is the length at which
  length alone halves accessibility.
- `h(rel)` is 1 at both edges and dips to `m_mid` in the middle — the
  "lost-in-the-middle" shape (primacy + recency good, middle bad).
- **A fact that is not in the context is never recalled** (probability 0). This
  single assumption is what makes memory load-bearing.

At answer time the provider rolls one Bernoulli per in-context fact and returns
the set that succeeded. It is **blind to which facts the query needs**;
randomness comes entirely from a caller-supplied RNG, so every run is
reproducible. A fact that appears in multiple chunks (e.g. still in working
memory *and* re-surfaced by retrieval) is scored at its most favourable position.

A secondary signal, `confident_wrong`, fires when a required fact was missed but
a same-topic distractor is present — a model of plausible hallucination — and
feeds the hallucination-rate metric. It does **not** affect the primary
success/AND grade.

Default parameters (`l_half=1200, gamma=1.5, m_mid=0.60`) are **knobs**: the
figures show behaviour *across* them (the noise and context-fill sweeps), so no
conclusion rests on one magic value.

---

## 4. The memory stack

All components are toggled by one `MemoryConfig`; the ablation grid is just a set
of configs sharing one implementation (`src/dah/memory/`).

- **WorkingMemory** — bounded, ordered in-context window for the *current*
  session; evicts oldest turns past `working_window_tokens`; **reset at every
  session boundary**.
- **Compactor** — lossy summarisation of evicted/old turns. Each fact survives
  with probability `compaction_recall` (rolled once), contributing a few tokens
  to one compact summary chunk. Trades **recall** (facts can be dropped) for
  **shorter context** (better `g(L)`). Optionally **persists across sessions**,
  giving compaction a *retrieval-independent* cross-session channel.
- **EpisodicStore** — append-only raw events with embeddings; **persists
  forever**; retrieval returns top-k by cosine similarity, injected at the **end**
  of context (recency). Surfaces old facts but also distractors, and every
  retrieved item costs tokens (diluting `g(L)`).
- **SemanticStore** — salience-gated, compact, deduplicated knowledge. Only facts
  above `sem_salience_threshold` are consolidated, stored with a smaller token
  footprint. Fewer, higher-value items ⇒ higher **precision**, at the cost of
  **recall** on low-salience facts.

The **retrieval policy** is itself ablatable: `off | always | budget_aware |
gap_triggered` (retrieve only when working memory looks irrelevant to the query).

**Assembly order** encodes position effects on purpose: `[compacted summary]
(primacy) → [working memory oldest→newest] → [retrieved] (recency)`. When a hard
context cap binds, trimming follows information density per token: stale raw
working turns go first, then retrieved items (deepest hit first), and the
compacted summary only as a last resort — it is the densest chunk and the only
carrier of the compressed past. (An earlier draft trimmed retrieved items first;
that single choice silently crippled retrieval under tight caps and is written up
as finding #7 in RESULTS.md — eviction policy is as load-bearing as storage.)

---

## 5. The harness (control plane)

- **Budget** (`budget.py`) — monotonic multi-dimensional accounting
  (tokens/steps/tool-calls/wall-time) with a hard `charge` path and a soft
  `pressure()`/`remaining_fraction()` signal the planner reads. Negative charges
  raise; there is no refund path (R7).
- **BudgetAwarePlanner** (`planner.py`) — the core idea is **allocation**: divide
  the *remaining* token budget across the *remaining* queries so the agent never
  blows it all up front and abandons the tail. It also scales retrieval depth
  with budget fraction, suppresses retries under high pressure, and issues
  **focused retries** (a tighter context cap on retry attempts, squeezing stale
  working turns out in favour of summary + retrieved facts). `StaticPlanner` is
  the naive contrast: one fixed, expensive strategy regardless of budget.
- **Verifier** (`verifier.py`, T2.1) — **oracle-free** post-condition checks
  returning a `Verdict(ok, reason, severity)`. The checks use only signals a
  real agent has: *empty recall* (the model used nothing) and *low relevance*
  (nothing it used looks semantically related to the query — max cosine below a
  threshold `tau`). Verification quality therefore *emerges* from the same
  embedding geometry retrieval lives in; with noisy embeddings the verifier
  degrades exactly as a real self-check would. Verdicts are deterministic. On
  reject the agent retries with a **sharper (reformulated) query, deeper
  retrieval, and a focused context**; if the verifier still rejects after all
  retries, the agent **abstains** rather than submit an answer it believes is
  wrong. The reformulation shrinks query-embedding noise per retry (a model of
  HyDE / query rewriting / step-back prompting) but never reaches the oracle.
- **Checkpointer** (`checkpoint.py`) — a stack of deep, total snapshots over a
  registry of components. Rollback is what makes retry *safe*: it rewinds
  corrupt/partial state before the retry instead of compounding it.
- **Recovery** (`recovery.py`) — retry a transient failure, roll back first,
  escalate on repeat.

**Failure model** (`faults.py`, T2.4). Processing an info turn fails transiently
with probability `ingest_fail_rate`, decided by a `FaultInjector` with its own
seeded RNG stream (so arms stay paired and rate 0 is bit-identical to no
injector — enforced by a governance test). On failure the turn's facts are
**not ingested** (lost from *all* stores) and a **corrupt "poison" chunk** lands
in working memory (wasting context). The baseline eats both costs. The harness
checkpoints before ingest, **rolls back** the poison, and **retries** the
(transient) step — recovering the lost facts and keeping context clean. The
injector itself is deliberately *not* registered with the checkpointer: it
models the outside world, and rolling it back would replay the identical failure
on every retry, turning transient faults into permanent ones.

---

## 6. Experimental design

- **Sample size & pairing (R4).** `N_TASKS = 120` paired seeds per arm (floor is
  20): seed *i* generates the *same* task in every arm. Levels get a 4000-sample
  percentile bootstrap CI (`runner.bootstrap_ci`); **every reported delta gets a
  paired bootstrap over per-task deltas with 10,000 resamples**
  (`runner.paired_delta_ci`).
- **Traceability (R3).** Every CSV row carries `seed` and `config_id`, so any
  number in RESULTS.md traces to the exact code path that produced it. Whole
  studies are bit-for-bit reproducible (`tests/test_reproducibility.py`).
- **Metrics** are the frozen GOAL.md definitions, implemented on
  `EpisodeResult`: QueryAccuracy (strict AND, split cross/within-session),
  EpisodeSuccess (≥0.8), TokensPerCorrect, HallucinationRate, DegradationSlope
  (per-session OLS), plus step-level recovery counters. Secondary: answered
  rate, mean context tokens, rollbacks, verifier rejections, abstentions.
- **Sweeps.** Retrieval noise (`embed_noise` + query noise), context-fill rate,
  working-window size (context pressure), token budget, verifier operating
  point (`tau`), and fault rate — each isolates one axis.
- **Verdicts.** `dah-analyze` machine-computes H1–H5 into
  `results/hypotheses.csv`; RESULTS.md quotes, never invents.

---

## 7. Anti-cheating guarantees

These are the properties that make the comparison fair; they're enforced in code,
not just intended:

1. **One agent, toggled by flags.** `BaselineAgent` and `HarnessAgent`
   (`agents.py`) are thin configurations of the same `Agent` loop. There is no
   alternate implementation that could smuggle in an advantage.
2. **The context assembler cannot see the answer key.** `MemoryManager.assemble`
   receives only a noisy `query_embedding` and token budget. `required_fact_ids`
   never leave the environment/grader: agents hand raw responses to
   `env.score_query` and receive graded records back. The forbidden tokens are
   grepped out of `memory/`, `planner.py`, `verifier.py`, `recovery.py`, *and*
   `agents.py` by both the linter (R5) and `tests/test_governance.py`.
3. **The provider is blind and fixed.** The degradation model is identical across
   every condition and does not know which memory policy produced the context.
4. **The verifier is oracle-free, full stop.** `ConsistencyVerifier` sees only
   the query embedding and the embeddings of what the model recalled — never
   correctness. Its false accepts and false rejects are *earned* from embedding
   geometry, not assumed from a knob.
5. **Budget has no refunds.** Rollback restores *state*, never consumed budget —
   so recovery has a real cost and budget-aware planning genuinely matters
   (negative charges raise `ValueError`; enforced by a governance test).

---

## 8. Threats to validity

Be skeptical in exactly these places:

- **Simulated, not measured, absolute levels.** The success *levels* reflect the
  accessibility model. Trust the *relative* comparisons and *directions*, not the
  absolute fractions as benchmark scores. Mitigation: the interface runs against
  real models; the sweeps show robustness of the *ordering* across a parameter
  range rather than at a point.
- **Retrieval realism.** Real retrievers use rerankers, hybrid lexical+dense
  search, and chunking that this synthetic-embedding model omits. The negative
  result is therefore a claim about the **noisy-embedding / distractor-dense
  regime**, and it names the mechanism (length-driven dilution + query-side
  noise). Cleaner retrieval moves the crossover; it does not remove the tradeoff.
- **Verifier realism.** The relevance check is one specific (embedding-based)
  self-verification signal; real critics (unit tests, judges, self-consistency)
  have different error profiles. The tau sweep maps the whole operating curve of
  this signal so you can read conclusions at your own operating point — but a
  categorically better verification signal would strengthen the verify arm
  beyond anything shown here.
- **Compaction model.** Compaction loss is modelled as independent per-fact
  survival; real summarisers lose *correlated* structure. This likely *flatters*
  compaction, which is worth remembering when it wins the noisy regime.
- **Single provider family.** All results use one degradation functional form.
  The shapes (monotone length decay, U-shaped position) are robust in the
  literature; the exact curvature is not load-bearing thanks to the sweeps.

Each conclusion in `RESULTS.md` names the knob that would overturn it.
