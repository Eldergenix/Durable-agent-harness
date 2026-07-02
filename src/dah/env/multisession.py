"""The multi-session environment: ground truth, query embeddings, and grading.

The environment is the *only* place (besides ``dah.experiments``) allowed to
read a query's answer key (R5). Agents hand their raw responses to
:meth:`MultiSessionEnv.score_query` and get back a fully-graded
:class:`QueryResult`; they never construct graded records themselves. This keeps
the separation auditable: grep the agent/memory/planner/verifier code and the
answer-key tokens simply do not appear.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..providers.base import ModelResponse, Provider
from ..types import Task, Turn


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


@dataclass(slots=True)
class QueryResult:
    """One graded query: correctness, provenance, and cost bookkeeping."""

    query_id: str
    session: int
    n_required: int
    n_recalled_required: int
    correct: bool
    recall_fraction: float
    cross_session: bool
    confident_wrong: bool
    context_tokens: int
    retrieval_performed: bool
    n_retrieved: int
    tokens_spent: float
    retries: int
    rolled_back: bool
    answered: bool  # False if the agent abstained or ran out of budget


@dataclass
class EpisodeResult:
    """All graded queries for one (task, agent-configuration) episode.

    Aggregation methods implement the *frozen* metric definitions from GOAL.md:
    QueryAccuracy, EpisodeSuccess, TokensPerCorrect, HallucinationRate, and
    DegradationSlope. (RecoveryRate needs cross-arm information, so it lives in
    the experiment layer.)
    """

    task_id: str
    memory: str
    results: list[QueryResult] = field(default_factory=list)
    budget: dict = field(default_factory=dict)
    trace: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    # -- frozen GOAL.md metrics -------------------------------------------
    def _subset(self, cross: bool | None) -> list[QueryResult]:
        if cross is None:
            return self.results
        return [r for r in self.results if r.cross_session == cross]

    def query_accuracy(self, cross: bool | None = None) -> float:
        """QueryAccuracy: fraction of queries with *all* required facts recalled."""
        rs = self._subset(cross)
        return float(np.mean([r.correct for r in rs])) if rs else float("nan")

    def episode_success(self, threshold: float = 0.8) -> bool:
        """EpisodeSuccess: QueryAccuracy >= ``threshold`` for this episode."""
        acc = self.query_accuracy()
        return bool(acc >= threshold) if not np.isnan(acc) else False

    def tokens_per_correct(self) -> float:
        """TokensPerCorrect: total tokens spent / number of correct queries.

        NaN when nothing was correct (undefined ratio; aggregated with nanmean
        and called out in the analysis rather than silently coerced).
        """
        n_correct = sum(r.correct for r in self.results)
        if n_correct == 0:
            return float("nan")
        return float(self.total_tokens() / n_correct)

    def hallucination_rate(self) -> float:
        """HallucinationRate: fraction of queries flagged confident-wrong."""
        return float(np.mean([r.confident_wrong for r in self.results])) if self.results else float("nan")

    def per_session_accuracy(self) -> list[tuple[int, float]]:
        """Mean QueryAccuracy per session index (sessions with >= 1 query)."""
        by: dict[int, list[bool]] = {}
        for r in self.results:
            by.setdefault(r.session, []).append(r.correct)
        return [(s, float(np.mean(v))) for s, v in sorted(by.items())]

    def degradation_slope(self) -> float:
        """DegradationSlope: OLS slope of per-session accuracy vs session index."""
        pts = self.per_session_accuracy()
        if len(pts) < 2:
            return float("nan")
        x = np.array([p[0] for p in pts], dtype=float)
        y = np.array([p[1] for p in pts], dtype=float)
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)

    # -- convenience aggregates (used by figures/tables) --------------------
    def success_rate(self, cross: bool | None = None) -> float:
        """Alias of :meth:`query_accuracy` kept for readability in plots/tests."""
        return self.query_accuracy(cross)

    def mean_recall(self, cross: bool | None = None) -> float:
        """Mean fraction of required facts recalled (graded, per query)."""
        rs = self._subset(cross)
        return float(np.mean([r.recall_fraction for r in rs])) if rs else float("nan")

    def answered_rate(self) -> float:
        """Fraction of queries actually answered (no abstain / budget-abandon)."""
        return float(np.mean([r.answered for r in self.results])) if self.results else float("nan")

    def mean_context_tokens(self) -> float:
        """Mean context size over answered queries."""
        rs = [r for r in self.results if r.answered]
        return float(np.mean([r.context_tokens for r in rs])) if rs else float("nan")

    def total_tokens(self) -> float:
        """Total tokens charged to the budget over the whole episode."""
        return float(self.budget.get("tokens", 0.0))


class MultiSessionEnv:
    """Ground truth + grading for one task.

    The environment computes a *noisy* embedding for every query (so retrieval
    is realistically imperfect from the query side too), exposes it to the
    agent, and never exposes ``required_fact_ids``.
    """

    def __init__(
        self,
        task: Task,
        provider: Provider,
        query_noise: float = 0.30,
        reformulation_gain: float = 0.55,
        seed: int = 0,
    ):
        self.task = task
        self.provider = provider
        self.query_noise = query_noise
        # Each verified retry lets the agent "reformulate" the query, shrinking
        # the query-embedding noise by this factor (models HyDE / query rewriting
        # / step-back prompting). Never reaches the oracle: noise only shrinks.
        self.reformulation_gain = reformulation_gain
        self.rng = np.random.default_rng(seed)
        self._query_base: dict[str, np.ndarray] = {}
        self._query_noise_vec: dict[str, np.ndarray] = {}
        self._build_query_embeddings()

    def _build_query_embeddings(self) -> None:
        dim = self.task.meta["config"]["embed_dim"]
        for turn in self.task.queries:
            base = np.mean(
                np.stack([self.task.facts[f].embedding for f in turn.required_fact_ids]), axis=0
            )
            self._query_base[turn.query_id] = base
            self._query_noise_vec[turn.query_id] = self.rng.standard_normal(dim)

    # -- accessors used by the agent -------------------------------------
    def query_embedding(self, turn: Turn, reformulation: int = 0) -> np.ndarray:
        """Return the (noisy) query embedding, optionally reformulated.

        ``reformulation`` is the number of prior verified retries; each one
        shrinks the effective noise by ``reformulation_gain``, so a persistent
        agent that keeps failing verification gradually sharpens its query.
        """
        base = self._query_base[turn.query_id]
        noise = self._query_noise_vec[turn.query_id]
        scale = self.query_noise * (self.reformulation_gain ** max(0, reformulation))
        return _unit(base + scale * noise)

    def is_cross_session(self, turn: Turn) -> bool:
        """True if the query depends on at least one earlier-session fact."""
        return any(self.task.facts[f].session_introduced < turn.session for f in turn.required_fact_ids)

    def topic_map(self) -> dict[str, int]:
        """fact-id -> topic, used by the provider's confusion model."""
        return {fid: f.topic for fid, f in self.task.facts.items()}

    # -- grading (env-only; R5) -------------------------------------------
    def grade(self, turn: Turn, response: ModelResponse) -> tuple[bool, float, int]:
        """Strict-AND grade: (all required recalled?, fraction recalled, n hit)."""
        required = set(turn.required_fact_ids)
        hit = required & response.recalled
        correct = len(hit) == len(required)
        frac = len(hit) / max(1, len(required))
        return correct, frac, len(hit)

    def score_query(
        self,
        turn: Turn,
        response: ModelResponse,
        *,
        retries: int = 0,
        rolled_back: bool = False,
        retrieval_performed: bool = False,
        n_retrieved: int = 0,
    ) -> QueryResult:
        """Grade an answered query into a :class:`QueryResult` record."""
        correct, frac, hit = self.grade(turn, response)
        return QueryResult(
            query_id=turn.query_id,
            session=turn.session,
            n_required=len(turn.required_fact_ids),
            n_recalled_required=hit,
            correct=correct,
            recall_fraction=frac,
            cross_session=self.is_cross_session(turn),
            confident_wrong=response.confident_wrong,
            context_tokens=response.context_tokens,
            retrieval_performed=retrieval_performed,
            n_retrieved=n_retrieved,
            tokens_spent=response.context_tokens,
            retries=retries,
            rolled_back=rolled_back,
            answered=True,
        )

    def score_unanswered(self, turn: Turn, *, retries: int = 0) -> QueryResult:
        """Record a query the agent abstained on or never reached (budget)."""
        return QueryResult(
            query_id=turn.query_id,
            session=turn.session,
            n_required=len(turn.required_fact_ids),
            n_recalled_required=0,
            correct=False,
            recall_fraction=0.0,
            cross_session=self.is_cross_session(turn),
            confident_wrong=False,
            context_tokens=0,
            retrieval_performed=False,
            n_retrieved=0,
            tokens_spent=0.0,
            retries=retries,
            rolled_back=False,
            answered=False,
        )
