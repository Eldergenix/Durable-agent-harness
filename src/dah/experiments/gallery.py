"""Failure-mode gallery (T4.3): three annotated, reproducible traces.

Writes ``results/traces.md`` containing concrete episodes (with their seeds) that
show the harness mechanisms doing their jobs - or memory failing in exactly the
way the aggregate numbers say it fails:

* (a) **rollback + retry saving an episode** - an info turn suffers injected
  transient failures, is rolled back and re-ingested, and a later query that
  needs precisely those facts is answered correctly.
* (b) **retrieval surfacing distractors and hurting** - under noisy embeddings a
  query retrieves same-topic neighbours instead of the needed fact and the model
  answers confidently wrong; the same seed under compaction-only answers it.
* (c) **budget pressure forcing a degraded-but-successful plan** - late in a
  tight-budget episode the budget-aware planner shrinks the context cap and
  retrieval depth, and the query is still answered correctly - while the static
  baseline has already abandoned it.

Every episode is found by a deterministic seed scan, so the gallery reproduces
bit-for-bit:

    python -m dah.experiments.gallery
"""

from __future__ import annotations

import os
from dataclasses import replace

import numpy as np

from ..agents import Agent
from ..budget import Budget
from ..env.multisession import MultiSessionEnv
from ..env.tasks import generate_task
from ..faults import FaultConfig, FaultInjector
from ..memory.manager import make_memory
from ..planner import BudgetAwarePlanner, StaticPlanner
from ..providers.simulated import SimulatedProvider
from ..recovery import RecoveryPolicy
from ..verifier import NullVerifier
from . import config as C

RESULTS_DIR = os.path.join(os.getcwd(), "results")


class RecordingProvider:
    """Wraps SimulatedProvider to log per-query context contents (diagnostics)."""

    def __init__(self, inner: SimulatedProvider):
        self.inner = inner
        self.log: dict[str, dict] = {}

    def estimate_tokens(self, context, query):
        return self.inner.estimate_tokens(context, query)

    def respond(self, query, context, rng):
        resp = self.inner.respond(query, context, rng)
        self.log[query.query_id] = {
            "in_context": sorted(context.all_fact_ids()),
            "recalled": sorted(resp.recalled),
            "tokens": context.total_tokens,
            "sources": sorted({it.source for it in context.items}),
        }
        return resp


def _episode(task, seed, memory_name, *, planner, fault_rate=0.0, budget=None,
             recovery=None, checkpointing=False, noise=None):
    topic = {f: v.topic for f, v in task.facts.items()}
    provider = RecordingProvider(SimulatedProvider(C.ACCESS, topic_of=topic))
    qn = noise if noise is not None else C.TASKS.embed_noise
    env = MultiSessionEnv(task, provider, query_noise=qn, seed=seed)
    memory = make_memory(memory_name, task.facts, seed=seed, overrides={"retrieval_k": 8})
    injector = (
        FaultInjector(FaultConfig(ingest_fail_rate=fault_rate), seed=seed + 7_919)
        if fault_rate > 0
        else None
    )
    agent = Agent(
        memory=memory,
        provider=provider,
        planner=planner,
        verifier=NullVerifier(),
        budget=budget or Budget(),
        recovery=recovery or RecoveryPolicy.none(),
        use_checkpointing=checkpointing,
        fault_injector=injector,
        seed=seed,
    )
    res = agent.run(env)
    return res, env, provider, agent


def _trace_rollback_save() -> list[str]:
    """(a) rollback+retry recovers an ingest whose facts a later query needs."""
    for seed in range(60):
        task = generate_task(C.TASKS, seed=seed)
        res, env, prov, agent = _episode(
            task, C.GLOBAL_SEED + seed, "episodic",
            planner=StaticPlanner(k=8), fault_rate=0.25,
            recovery=RecoveryPolicy.robust(), checkpointing=True,
        )
        # exact turn indices that failed at least once and were then recovered,
        # straight from the telemetry event log
        recovered_turns = {
            ev.data["turn"] for ev in agent.trace.events if ev.kind == "ingest.recovered"
        }
        if not recovered_turns:
            continue
        for qr in res.results:
            if not (qr.correct and qr.answered):
                continue
            turn = next(t for t in task.queries if t.query_id == qr.query_id)
            src_turns = {task.facts[f].turn_introduced for f in turn.required_fact_ids}
            if not src_turns <= recovered_turns:
                continue
            # require the fact to have been introduced in an *earlier* session
            # (so it demonstrably travelled through the recovered ingest into
            # the episodic store and back out through retrieval)
            if not qr.cross_session:
                continue
            rolls = res.trace.get("checkpoint.rollback", 0)
            fails = res.trace.get("ingest.fail", 0)
            recov = res.trace.get("ingest.recovered", 0)
            log = prov.log[qr.query_id]
            return [
                f"**Episode:** task seed {seed}, agent seed {C.GLOBAL_SEED + seed}, "
                f"memory `episodic`, fault rate 0.25, recovery **on**.",
                f"- Injected ingestion failures this episode: **{fails}**; rollbacks: **{rolls}**; "
                f"recovered ingestions: **{recov}**; permanently lost: "
                f"**{res.trace.get('ingest.giveup', 0)}**.",
                f"- Query `{qr.query_id}` (session {qr.session}, cross-session) needed "
                f"fact(s) {sorted(turn.required_fact_ids)} introduced at turn(s) {sorted(src_turns)}.",
                f"- At answer time the context ({log['tokens']} tokens, sources {log['sources']}) "
                f"contained the needed fact(s) and the model recalled them -> **correct**.",
                "- Counterfactual: with recovery off, every one of those failures loses its facts "
                "permanently (see `results/recovery.csv`: accuracy 0.249 vs 0.330 at this fault rate).",
            ]
    return ["(no qualifying episode found in seed scan - increase scan range)"]


def _trace_distractor_hurt() -> list[str]:
    """(b) noisy retrieval fills context with same-topic distractors; answer is confidently wrong."""
    noisy = replace(C.TASKS, embed_noise=1.2)
    for seed in range(60):
        task = generate_task(noisy, seed=seed)
        res, env, prov, _ = _episode(
            task, C.GLOBAL_SEED + seed, "episodic", planner=StaticPlanner(), noise=1.2
        )
        res2, _, _, _ = _episode(
            task, C.GLOBAL_SEED + seed, "compaction", planner=StaticPlanner(), noise=1.2
        )
        acc2 = {r.query_id: r for r in res2.results}
        for qr in res.results:
            if not (qr.answered and not qr.correct and qr.confident_wrong and qr.n_retrieved > 0):
                continue
            twin = acc2.get(qr.query_id)
            if twin is None or not twin.correct:
                continue  # want the compaction twin to get it right
            turn = next(t for t in task.queries if t.query_id == qr.query_id)
            log = prov.log[qr.query_id]
            required = set(turn.required_fact_ids)
            in_ctx = set(log["in_context"])
            req_topics = {task.facts[f].topic for f in required}
            distractors = [
                f for f in in_ctx - required if task.facts[f].topic in req_topics
            ]
            missing = sorted(required - in_ctx)
            return [
                f"**Episode:** task seed {seed} at retrieval noise 1.2, memory `episodic` "
                f"(k=8, similarity retrieval only).",
                f"- Query `{qr.query_id}` needed {sorted(required)}; retrieval injected "
                f"**{qr.n_retrieved}** items but the needed fact(s) {missing or '(all present, drowned)'} "
                f"were {'missing from context' if missing else 'present yet inaccessible'}.",
                f"- Context held **{len(distractors)}** same-topic distractor(s) "
                f"({sorted(distractors)[:6]}{'...' if len(distractors) > 6 else ''}) in "
                f"{log['tokens']} tokens -> the model answered from a plausible neighbour: "
                f"**confidently wrong** (`confident_wrong=True`).",
                f"- Same seed, memory `compaction` (no retrieval at all): query answered "
                f"**correctly** from the persisted summary.",
                "- This is the noise-sweep collapse (`results/ablation_noise.csv`) at the level of "
                "a single decision: retrieval quality is the ceiling on retrieval value.",
            ]
    return ["(no qualifying episode found in seed scan - increase scan range)"]


def _trace_budget_degrade() -> list[str]:
    """(c) budget-aware planner degrades the plan and still lands the answer."""
    for seed in range(60):
        task = generate_task(C.TASKS, seed=seed)
        bud = Budget(max_tokens=13_000)
        res, env, prov, _ = _episode(
            task, C.GLOBAL_SEED + seed, "episodic",
            planner=BudgetAwarePlanner(base_k=8, min_k=1, max_retries=0),
            budget=bud,
        )
        base_bud = Budget(max_tokens=13_000)
        res_b, _, _, _ = _episode(
            task, C.GLOBAL_SEED + seed, "episodic",
            planner=StaticPlanner(k=8), budget=base_bud,
        )
        base = {r.query_id: r for r in res_b.results}
        # a late query, answered correctly under a visibly tightened context,
        # that the static baseline never got to answer
        late = [r for r in res.results if r.session >= task.n_sessions - 2]
        for qr in late:
            if not (qr.answered and qr.correct and 0 < qr.context_tokens <= 700):
                continue
            twin = base.get(qr.query_id)
            if twin is None or twin.answered:
                continue
            spent_before = sum(
                r.tokens_spent for r in res.results if r.query_id != qr.query_id and r.session <= qr.session
            )
            return [
                f"**Episode:** task seed {seed}, budget **13,000 tokens** (binding), memory "
                f"`episodic`, planner `BudgetAwarePlanner`.",
                f"- By session {qr.session} the agent had spent ~{spent_before:.0f} query tokens; "
                f"the planner divided the remainder across the remaining queries and capped this "
                f"query's context at a degraded **{qr.context_tokens} tokens** "
                f"(full window would be 1,600).",
                f"- Query `{qr.query_id}` was still answered **correctly** - the cap squeezes out "
                f"stale working turns first, keeping the summary/retrieved facts that matter.",
                f"- Same seed with the static planner (fixed k=8, uncapped context): budget "
                f"exhausted earlier and this query was **never answered** "
                f"(baseline answered {res_b.answered_rate():.0%} of queries vs "
                f"{res.answered_rate():.0%} for budget-aware).",
                "- This is the budget-sweep gap (`results/harness_budget_sweep.csv`) at the level "
                "of a single decision: allocation beats greed.",
            ]
    return ["(no qualifying episode found in seed scan - increase scan range)"]


def main(argv: list[str] | None = None) -> None:
    """Generate results/traces.md with the three annotated episodes."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    sections = [
        "# Failure-mode gallery (T4.3)\n",
        "Three concrete, reproducible episodes behind the aggregate numbers. Regenerate with "
        "`python -m dah.experiments.gallery`; every episode names its seeds.\n",
        "## (a) Rollback + retry saves the episode\n",
        *_trace_rollback_save(),
        "\n## (b) Noisy retrieval surfaces distractors and hurts\n",
        *_trace_distractor_hurt(),
        "\n## (c) Budget pressure forces a degraded - but successful - plan\n",
        *_trace_budget_degrade(),
        "",
    ]
    path = os.path.join(RESULTS_DIR, "traces.md")
    with open(path, "w") as f:
        f.write("\n".join(sections))
    print(f"[gallery] wrote {path}")


if __name__ == "__main__":
    main()
