"""Budget-aware planning.

The planner turns the current *budget pressure* into concrete per-query
decisions: how deep to retrieve and how many recovery attempts to allow. The
contrast the experiments exploit:

* :class:`StaticPlanner` commits to a fixed, expensive strategy regardless of
  remaining budget - the classic way a long-horizon agent runs out of tokens
  three-quarters of the way through and abandons the tail of the task.
* :class:`BudgetAwarePlanner` scales retrieval depth (and, under severe
  pressure, recovery attempts) down as the budget drains, trading a little
  per-query quality for finishing the whole task.
"""

from __future__ import annotations

from dataclasses import dataclass

from .budget import Budget


@dataclass(slots=True)
class QueryPlan:
    """A concrete per-query plan: retrieval depth, retry allowance, context caps.

    ``retry_context_cap`` implements *focused retries*: after a verifier
    rejection the agent re-asks with a slimmer context (summary + retrieved
    facts squeeze out stale working-memory turns). This is cheaper per attempt
    *and* mechanically more accurate for the re-surfaced facts, because a
    shorter context raises the accessibility of everything left in it.
    """

    k: int | None  # retrieval depth override; None => use memory's own policy
    max_retries: int
    escalate_factor: float
    context_cap: int | None = None  # per-query context token cap; None => full window
    retry_context_cap: int | None = None  # tighter cap for attempts > 0; None => context_cap


class StaticPlanner:
    """Fixed plan; ignores budget. ``k=None`` defers to the memory preset.

    This is the naive planner: it commits to one strategy and spends the same
    (large) amount on every query regardless of how much budget is left - so
    when the budget is binding it exhausts partway through and abandons the tail
    of the task.
    """

    def __init__(
        self,
        k: int | None = None,
        max_retries: int = 0,
        escalate_factor: float = 1.0,
        retry_context_cap: int | None = 900,
    ):
        self.k = k
        self.max_retries = max_retries
        self.escalate_factor = escalate_factor
        self.retry_context_cap = retry_context_cap

    def plan_query(self, budget: Budget, queries_remaining: int | None = None) -> QueryPlan:
        """Return the same fixed plan regardless of budget state."""
        return QueryPlan(
            k=self.k,
            max_retries=self.max_retries,
            escalate_factor=self.escalate_factor,
            context_cap=None,
            retry_context_cap=self.retry_context_cap if self.max_retries > 0 else None,
        )


class BudgetAwarePlanner:
    """Scales retrieval depth and recovery with remaining token budget.

    ``base_k`` is the depth at full budget. As the token budget drains, the
    planned depth shrinks toward ``min_k`` so the agent reserves enough budget to
    reach the end of the task. Recovery retries are suppressed once pressure is
    high (a retry you cannot afford is worse than a cheap best-effort answer).
    """

    def __init__(
        self,
        base_k: int = 6,
        min_k: int = 1,
        max_retries: int = 2,
        escalate_factor: float = 2.0,
        retry_pressure_cutoff: float = 0.85,
        full_context_cap: int = 1600,
        min_context_cap: int = 650,
        retrieval_token_cost: int = 20,
    ):
        self.base_k = base_k
        self.min_k = min_k
        self.max_retries = max_retries
        self.escalate_factor = escalate_factor
        self.retry_pressure_cutoff = retry_pressure_cutoff
        self.full_context_cap = full_context_cap
        self.min_context_cap = min_context_cap
        self.retrieval_token_cost = retrieval_token_cost

    def plan_query(self, budget: Budget, queries_remaining: int | None = None) -> QueryPlan:
        frac = budget.remaining_fraction("tokens")
        remaining = budget.remaining("tokens")

        # Core idea: divide the *remaining* token budget evenly across the
        # *remaining* queries, so we never blow it all on the first few and
        # abandon the tail. Falls back to fractional scaling when unbounded.
        if queries_remaining and queries_remaining > 0 and remaining != float("inf"):
            allowance = remaining / queries_remaining
            cap = int(max(self.min_context_cap, min(self.full_context_cap, allowance)))
        else:
            cap = int(self.min_context_cap + (self.full_context_cap - self.min_context_cap) * frac)

        # retrieval depth tracks how much context we can afford after the working
        # set, and shrinks with overall budget fraction.
        k = max(self.min_k, round(self.base_k * min(1.0, frac + 0.15)))

        pressure = budget.pressure()
        max_retries = self.max_retries if pressure < self.retry_pressure_cutoff else 0
        return QueryPlan(
            k=int(k),
            max_retries=max_retries,
            escalate_factor=self.escalate_factor,
            context_cap=cap,
            retry_context_cap=min(cap, 900) if max_retries > 0 else None,
        )
