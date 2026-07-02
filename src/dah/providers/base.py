"""Provider interface and response type."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from ..types import AssembledContext, Turn


@dataclass(slots=True)
class ModelResponse:
    """What a provider returns for one query.

    ``recalled`` is the set of fact-ids the model successfully attended to and
    could use. The *environment* (never the provider) decides whether that set
    covers the query's required facts. ``per_fact_prob`` is exposed purely for
    diagnostics/plots. ``confident_wrong`` flags a hallucinated answer (used for
    the secondary hallucination-rate metric).
    """

    recalled: set[str]
    per_fact_prob: dict[str, float] = field(default_factory=dict)
    confident_wrong: bool = False
    context_tokens: int = 0


class Provider(Protocol):
    """A minimal model interface: turn assembled context into a response."""

    def respond(self, query: Turn, context: AssembledContext, rng: np.random.Generator) -> ModelResponse:
        ...

    def estimate_tokens(self, context: AssembledContext, query: Turn) -> int:
        ...
