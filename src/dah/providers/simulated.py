"""A deterministic simulated model of context-length degradation.

Why simulate? To run honest *ablations* you need (a) a controllable ground truth
and (b) enough repetitions for tight confidence intervals - both impractical
with a live LLM (cost, flakiness, no oracle for "which fact did it use"). The
simulated provider models the two robustly-documented failure modes of long
context:

1. **Length decay.** As total context grows, the probability of using any given
   in-context fact falls. Modelled by ``g(L) = 1 / (1 + (L / L_half) ** gamma)``.
2. **Lost-in-the-middle.** Facts near the start or end of the context are more
   accessible than facts buried in the middle. Modelled by a U-shaped position
   term ``h(rel)``.

A fact that is **not in the context at all** is never recalled (probability 0):
you cannot use what the model cannot see. That single assumption is what makes
memory load-bearing in this benchmark.

The parameters below are *tunable knobs*, not fitted constants. The point of the
harness is to show how the winning memory policy changes as you move along these
axes (see ``docs/RESULTS.md``), not to claim a specific real-world value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..types import AssembledContext, Turn
from .base import ModelResponse


@dataclass
class AccessibilityParams:
    """Knobs for the accessibility model: ``a_max * g(length) * h(position)``."""

    a_max: float = 0.98  # ceiling recall for a short, well-placed fact
    l_half: float = 900.0  # context tokens at which length halves accessibility
    gamma: float = 1.6  # steepness of length decay
    m_mid: float = 0.55  # accessibility floor in the middle (lost-in-the-middle depth)
    confusion_rate: float = 0.5  # P(confident-wrong | missed a fact w/ same-topic distractor present)
    token_overhead: int = 8  # per-call framing tokens

    def g_length(self, total_tokens: float) -> float:
        return 1.0 / (1.0 + (total_tokens / self.l_half) ** self.gamma)

    @staticmethod
    def h_position(rel: float) -> float:
        # returns the *shape* multiplier in [m_mid, 1]; caller scales by (1-m_mid)
        return (2.0 * rel - 1.0) ** 2  # 1 at edges, 0 at centre

    def accessibility(self, total_tokens: float, rel: float) -> float:
        g = self.g_length(total_tokens)
        h = self.m_mid + (1.0 - self.m_mid) * self.h_position(rel)
        return float(np.clip(self.a_max * g * h, 0.0, 1.0))


class SimulatedProvider:
    """Provider that rolls fact-level recall from the accessibility model.

    The provider is *blind* to which facts a query needs. It rolls a recall
    Bernoulli for every fact present in the assembled context and returns the
    set that succeeded. Randomness comes entirely from the caller-supplied RNG,
    so runs are reproducible.
    """

    def __init__(self, params: AccessibilityParams | None = None, topic_of: dict[str, int] | None = None):
        self.p = params or AccessibilityParams()
        # optional map fact_id -> topic, used to model same-topic confusion
        self.topic_of = topic_of or {}

    def estimate_tokens(self, context: AssembledContext, query: Turn) -> int:
        return context.total_tokens + query.tokens + self.p.token_overhead

    def respond(self, query: Turn, context: AssembledContext, rng: np.random.Generator) -> ModelResponse:
        total = context.total_tokens + query.tokens + self.p.token_overhead
        positions = context.fact_positions()
        recalled: set[str] = set()
        per_fact: dict[str, float] = {}
        for fid, rel in positions.items():
            prob = self.p.accessibility(total, rel)
            per_fact[fid] = prob
            if rng.random() < prob:
                recalled.add(fid)

        # Secondary metric: confident-wrong when a required fact was missed but a
        # same-topic neighbour is sitting in the context (a plausible distractor).
        confident_wrong = self._confusion(query, context, recalled, rng)

        return ModelResponse(
            recalled=recalled,
            per_fact_prob=per_fact,
            confident_wrong=confident_wrong,
            context_tokens=total,
        )

    def _confusion(
        self,
        query: Turn,
        context: AssembledContext,
        recalled: set[str],
        rng: np.random.Generator,
    ) -> bool:
        if not query.required_fact_ids or not self.topic_of:
            return False
        present = context.all_fact_ids()
        missed = [f for f in query.required_fact_ids if f not in recalled]
        for f in missed:
            topic = self.topic_of.get(f)
            # a distractor = a different fact of the same topic present in context
            has_distractor = any(
                self.topic_of.get(other) == topic and other != f for other in present
            )
            if has_distractor and rng.random() < self.p.confusion_rate:
                return True
        return False
