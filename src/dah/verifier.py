"""Verification loops: oracle-free post-condition checks (T2.1).

A verifier decides whether to *accept* an answer or *reject* it and trigger
recovery. The checks here use only signals a real agent legitimately has -
what it recalled, how relevant that looks to the query, what the context
assembly reported - never the environment's answer key (R5). Verification
quality therefore *emerges* from the same embedding geometry retrieval lives
in, rather than being assumed: with noisy embeddings the verifier degrades in
exactly the way a real self-check would.

Checks (each returns a :class:`Verdict`):

* ``empty_recall``  - the model recalled nothing at all on a non-trivial query.
* ``low_relevance`` - nothing the model recalled looks semantically related to
  the query (max cosine below ``relevance_threshold``). This is the workhorse:
  it fires when the needed fact was missing from context and the model is about
  to answer from vibes.

Verdicts are **deterministic**: identical inputs always produce the identical
verdict (no RNG anywhere in this module).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class Verdict:
    """Outcome of a verification pass: accept (``ok``) or reject with a reason."""

    ok: bool
    reason: str = ""
    severity: str = "none"  # none | warn | error

    @classmethod
    def accept(cls) -> "Verdict":
        """The passing verdict."""
        return cls(ok=True)

    @classmethod
    def reject(cls, reason: str, severity: str = "error") -> "Verdict":
        """A failing verdict carrying a machine-readable reason."""
        return cls(ok=False, reason=reason, severity=severity)


@dataclass
class ConsistencyVerifier:
    """Answer-key-free verification from recall/relevance signals.

    ``relevance_threshold`` is the operating point: raise it and the verifier
    rejects more (catching more wrong answers but also flagging correct ones and
    burning retry budget). The tau sweep in the experiments maps this trade-off
    so a team can pick the point matching their cost of a confident error.
    """

    relevance_threshold: float = 0.35

    def verify(
        self,
        query_embedding: np.ndarray,
        recalled_embeddings: np.ndarray | None,
    ) -> Verdict:
        """Judge one answer attempt.

        ``recalled_embeddings`` is an ``(n, d)`` array of embeddings for the
        facts the model actually used (empty/None when it used none).
        """
        if recalled_embeddings is None or len(recalled_embeddings) == 0:
            return Verdict.reject("empty_recall", "error")
        sims = recalled_embeddings @ query_embedding
        best = float(np.max(sims))
        if best < self.relevance_threshold:
            return Verdict.reject(f"low_relevance:{best:.3f}", "error")
        return Verdict.accept()


class NullVerifier:
    """Accepts everything (baseline: no verification loop)."""

    def verify(
        self,
        query_embedding: np.ndarray,
        recalled_embeddings: np.ndarray | None,
    ) -> Verdict:
        """Always returns the passing verdict."""
        return Verdict.accept()
