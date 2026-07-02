"""The agents: one loop, two canonical configurations (T1.3, T2.5).

:class:`BaselineAgent` and :class:`HarnessAgent` are thin configurations of the
*same* :class:`Agent` loop - never separate implementations - so the headline
comparison can only be explained by which safety mechanisms are switched on:

* baseline: static plan, no verification, no checkpointing, no recovery.
* harness:  budget-aware planning, verification loop with escalating retries
  and reformulation, checkpoint + rollback around risky steps, ingestion retry.

Per info turn the agent *ingests* (optionally checkpointed, with rollback and
retry on injected transient failures). Per query it *answers*: assemble context
-> provider -> verify -> maybe retry deeper with a reformulated query -> accept
or abstain. Budget is charged on every step with no refunds (R7); when a hard
limit is crossed the episode stops and every unreached query is scored as
unanswered - which is exactly how running out of budget should be penalised.

Grading lives in the environment: the agent hands raw responses to
``env.score_query`` / ``env.score_unanswered`` and never reads the answer key
itself (R5).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .budget import Budget, BudgetExceeded
from .checkpoint import Checkpointer
from .env.multisession import EpisodeResult, MultiSessionEnv, QueryResult
from .faults import FaultConfig, FaultInjector
from .memory.manager import MemoryManager
from .planner import BudgetAwarePlanner, QueryPlan, StaticPlanner
from .providers.base import Provider
from .recovery import RecoveryPolicy
from .telemetry import Trace
from .types import Turn
from .verifier import ConsistencyVerifier, NullVerifier, Verdict


@dataclass
class AgentCosts:
    """Fixed token costs of agent-side operations."""

    ingest_token_cost: int = 6


class Agent:
    """The single toggleable agent loop over a :class:`MultiSessionEnv`."""

    def __init__(
        self,
        memory: MemoryManager,
        provider: Provider,
        planner,
        *,
        verifier=None,
        budget: Budget | None = None,
        recovery: RecoveryPolicy | None = None,
        use_checkpointing: bool = False,
        fault_injector: FaultInjector | None = None,
        abstain_on_reject: bool = False,
        costs: AgentCosts | None = None,
        trace: Trace | None = None,
        seed: int = 0,
    ):
        self.memory = memory
        self.provider = provider
        self.planner = planner
        self.verifier = verifier or NullVerifier()
        self.budget = budget or Budget()
        self.recovery = recovery or RecoveryPolicy.none()
        self.use_checkpointing = use_checkpointing
        self.faults = fault_injector
        # When the verifier still rejects after all retries: abstain (decline to
        # answer) instead of submitting an answer it believes is wrong.
        self.abstain_on_reject = abstain_on_reject
        self.costs = costs or AgentCosts()
        self.trace = trace or Trace()
        self.rng = np.random.default_rng(seed)
        self._queries_left = 0

    # -- public API -------------------------------------------------------
    def run(self, env: MultiSessionEnv) -> EpisodeResult:
        """Play one full multi-session episode and return graded results."""
        task = env.task
        cp = Checkpointer(self.trace)
        cp.register("memory", self.memory)
        # NOTE deliberately NOT registered: the fault injector. It models the
        # *outside world* (tool flakiness), not agent state. Rolling it back
        # would rewind the noise source and replay the identical failure on
        # every retry, turning transient faults into permanent ones.
        results: list[QueryResult] = []
        stopped = False
        self._queries_left = len(task.queries)  # for budget allocation across the task

        for s, session in enumerate(task.sessions):
            self.memory.start_session(s)
            for turn in session:
                try:
                    if turn.kind == "info":
                        self._ingest(turn, cp)
                    else:
                        results.append(self._answer(turn, env))
                        self._queries_left -= 1
                except BudgetExceeded as e:
                    self.trace.emit("budget.exhausted", where=turn.kind, dim=e.dimension)
                    stopped = True
                    break
                if self._budget_dead():
                    stopped = True
                    break
            if stopped:
                break

        # Queries never reached (budget stop) are scored as unanswered failures.
        seen = {r.query_id for r in results}
        for turn in task.queries:
            if turn.query_id not in seen:
                results.append(env.score_unanswered(turn))

        return EpisodeResult(
            task_id=task.id,
            memory=self.memory.config.name,
            results=results,
            budget=self.budget.snapshot(),
            trace=self.trace.summary(),
            meta={
                "agent": type(self).__name__,
                "planner": type(self.planner).__name__,
                "verifier": type(self.verifier).__name__,
                "checkpointing": self.use_checkpointing,
                "fault_rate": (self.faults.config.ingest_fail_rate if self.faults else 0.0),
                "recovery": dict(self.recovery.__dict__),
            },
        )

    # -- ingestion (with recovery) ---------------------------------------
    def _ingest(self, turn: Turn, cp: Checkpointer) -> None:
        checkpointed = False
        if self.use_checkpointing:
            cp.commit("pre-ingest", step=turn.index)
            checkpointed = True
        attempts_left = self.recovery.ingestion_retries
        failures_this_turn = 0

        while True:
            self.budget.charge(steps=1, tokens=self.costs.ingest_token_cost)
            if self.faults is None or not self.faults.ingest_fails():
                self.memory.add_info(turn)
                if checkpointed:
                    cp.pop()
                if failures_this_turn > 0:
                    # a step that failed at least once but ultimately landed:
                    # this is what the RecoveryRate metric counts.
                    self.trace.emit("ingest.recovered", turn=turn.index, failures=failures_this_turn)
                return

            # transient failure: facts not ingested + corrupt chunk in context
            failures_this_turn += 1
            self.trace.emit("ingest.fail", turn=turn.index)
            self.memory.ingest_poison(self.faults.poison_tokens)
            if checkpointed and self.recovery.rollback_on_failure:
                cp.rollback()  # restore pre-ingest state (removes the poison)
            if attempts_left > 0:
                attempts_left -= 1
                continue
            if checkpointed:
                cp.pop()
            self.trace.emit("ingest.giveup", turn=turn.index)
            return

    # -- answering (verify -> retry deeper/reformulated -> accept|abstain) --
    def _answer(self, turn: Turn, env: MultiSessionEnv) -> QueryResult:
        plan: QueryPlan = self.planner.plan_query(self.budget, queries_remaining=self._queries_left)

        attempt = 0
        retries = 0
        response = None
        stats = None
        verdict = Verdict.accept()
        while True:
            k = plan.k
            if k is not None:
                k = int(round(k * (plan.escalate_factor ** attempt)))
            # a verified retry reformulates the query (sharper embedding),
            # retrieves deeper, *and* focuses the context (tighter cap squeezes
            # out stale working turns in favour of summary + retrieved facts).
            q_emb = env.query_embedding(turn, reformulation=attempt)
            cap = plan.context_cap if attempt == 0 else (plan.retry_context_cap or plan.context_cap)
            ctx, stats = self.memory.assemble(
                q_emb, turn.tokens, self.budget, k_override=k, context_cap=cap
            )
            est = self.provider.estimate_tokens(ctx, turn)
            tool_calls = 1 if stats.retrieval_performed else 0
            if not self.budget.can_afford(tokens=est, steps=1, tool_calls=tool_calls):
                self.trace.emit("query.unaffordable", q=turn.query_id)
                return env.score_unanswered(turn, retries=retries)
            self.budget.charge(tokens=est, steps=1, tool_calls=tool_calls)
            response = self.provider.respond(turn, ctx, self.rng)

            verdict = self._verify(q_emb, response)
            if verdict.ok or attempt >= plan.max_retries:
                break
            retries += 1
            attempt += 1
            self.trace.emit("verify.reject", q=turn.query_id, attempt=attempt, reason=verdict.reason)

        if not verdict.ok and self.abstain_on_reject:
            # the verifier still doesn't believe this answer: decline rather
            # than submit a confident guess.
            self.trace.emit("query.abstain", q=turn.query_id, reason=verdict.reason)
            return env.score_unanswered(turn, retries=retries)

        self.trace.emit("query.answered", q=turn.query_id, retries=retries)
        return env.score_query(
            turn,
            response,
            retries=retries,
            rolled_back=False,
            retrieval_performed=stats.retrieval_performed,
            n_retrieved=stats.n_retrieved,
        )

    def _verify(self, q_emb: np.ndarray, response) -> Verdict:
        recalled = list(response.recalled)
        if recalled:
            embs = np.stack([self.memory.fact_db[f].embedding for f in recalled])
        else:
            embs = None
        return self.verifier.verify(q_emb, embs)

    def _budget_dead(self) -> bool:
        return any(self.budget.remaining_fraction(d) <= 0.0 for d in self.budget.bounded_dims())


# ---------------------------------------------------------------------------
# The two canonical configurations compared in the experiments.
# ---------------------------------------------------------------------------
class BaselineAgent(Agent):
    """Naive plan-and-execute: fixed strategy, no verify/rollback/recovery."""

    def __init__(
        self,
        memory: MemoryManager,
        provider: Provider,
        *,
        budget: Budget,
        base_k: int,
        fault_rate: float = 0.0,
        seed: int = 0,
    ):
        super().__init__(
            memory=memory,
            provider=provider,
            planner=StaticPlanner(k=base_k, max_retries=0, escalate_factor=1.0),
            verifier=NullVerifier(),
            budget=budget,
            recovery=RecoveryPolicy.none(),
            use_checkpointing=False,
            fault_injector=_injector(fault_rate, seed),
            abstain_on_reject=False,
            seed=seed,
        )


class HarnessAgent(Agent):
    """Full control plane: verify + rollback + budget-aware plan + recovery."""

    def __init__(
        self,
        memory: MemoryManager,
        provider: Provider,
        *,
        budget: Budget,
        base_k: int,
        fault_rate: float = 0.0,
        verifier: ConsistencyVerifier | None = None,
        seed: int = 0,
    ):
        super().__init__(
            memory=memory,
            provider=provider,
            planner=BudgetAwarePlanner(base_k=base_k, min_k=1, max_retries=2, escalate_factor=2.0),
            verifier=verifier or ConsistencyVerifier(),
            budget=budget,
            recovery=RecoveryPolicy.robust(),
            use_checkpointing=True,
            fault_injector=_injector(fault_rate, seed),
            abstain_on_reject=True,
            seed=seed,
        )


def _injector(rate: float, seed: int) -> FaultInjector | None:
    if rate <= 0.0:
        return None
    # offset the seed so the injector stream never aliases the agent stream
    return FaultInjector(FaultConfig(ingest_fail_rate=rate), seed=seed + 7_919)
