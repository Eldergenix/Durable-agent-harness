# Failure-mode gallery (T4.3)

Three concrete, reproducible episodes behind the aggregate numbers. Regenerate with `python -m dah.experiments.gallery`; every episode names its seeds.

## (a) Rollback + retry saves the episode

**Episode:** task seed 0, agent seed 20240601, memory `episodic`, fault rate 0.25, recovery **on**.
- Injected ingestion failures this episode: **28**; rollbacks: **28**; recovered ingestions: **19**; permanently lost: **1**.
- Query `q45` (session 3, cross-session) needed fact(s) ['f16'] introduced at turn(s) [13].
- At answer time the context (445 tokens, sources ['retrieved', 'working']) contained the needed fact(s) and the model recalled them -> **correct**.
- Counterfactual: with recovery off, every one of those failures loses its facts permanently (see `results/recovery.csv`: accuracy 0.249 vs 0.330 at this fault rate).

## (b) Noisy retrieval surfaces distractors and hurts

**Episode:** task seed 1 at retrieval noise 1.2, memory `episodic` (k=8, similarity retrieval only).
- Query `q19` needed ['f11', 'f16']; retrieval injected **8** items but the needed fact(s) ['f16'] were missing from context.
- Context held **1** same-topic distractor(s) (['f26']) in 540 tokens -> the model answered from a plausible neighbour: **confidently wrong** (`confident_wrong=True`).
- Same seed, memory `compaction` (no retrieval at all): query answered **correctly** from the persisted summary.
- This is the noise-sweep collapse (`results/ablation_noise.csv`) at the level of a single decision: retrieval quality is the ceiling on retrieval value.

## (c) Budget pressure forces a degraded - but successful - plan

**Episode:** task seed 2, budget **13,000 tokens** (binding), memory `episodic`, planner `BudgetAwarePlanner`.
- By session 5 the agent had spent ~12289 query tokens; the planner divided the remainder across the remaining queries and capped this query's context at a degraded **328 tokens** (full window would be 1,600).
- Query `q75` was still answered **correctly** - the cap squeezes out stale working turns first, keeping the summary/retrieved facts that matter.
- Same seed with the static planner (fixed k=8, uncapped context): budget exhausted earlier and this query was **never answered** (baseline answered 74% of queries vs 87% for budget-aware).
- This is the budget-sweep gap (`results/harness_budget_sweep.csv`) at the level of a single decision: allocation beats greed.
