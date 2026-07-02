"""Shared data model for tasks, facts, turns, and assembled context.

These types are deliberately provider-agnostic. The *environment* creates
:class:`Task` / :class:`Fact` / :class:`Turn` objects and owns the ground truth
(which facts a query needs). The *memory* stack decides what ends up in an
:class:`AssembledContext`. The *provider* looks only at the assembled context
and decides which facts it can actually use. This separation is what keeps the
simulation honest: no component that assembles context can see the answer key.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

TurnKind = Literal["info", "query"]


@dataclass(slots=True)
class Fact:
    """An atomic piece of knowledge the agent may need later.

    ``embedding`` is a synthetic vector (topic signal + noise). ``tokens`` is the
    fact's length in the abstract token unit used for budgeting and for the
    context-length degradation model.
    """

    id: str
    topic: int
    content: str
    embedding: np.ndarray
    tokens: int = 20
    salience: float = 1.0
    session_introduced: int = 0
    turn_introduced: int = 0


@dataclass(slots=True)
class Turn:
    """One event in a session: either new information or a question."""

    index: int  # global turn index across the whole task
    session: int
    kind: TurnKind
    text: str
    tokens: int
    fact_ids: list[str] = field(default_factory=list)  # facts introduced (info turns)
    required_fact_ids: list[str] = field(default_factory=list)  # ground truth (query turns)
    query_id: str | None = None


@dataclass
class Task:
    """A multi-session task: a list of sessions, each a list of turns."""

    id: str
    sessions: list[list[Turn]]
    facts: dict[str, Fact]
    meta: dict = field(default_factory=dict)

    @property
    def n_sessions(self) -> int:
        return len(self.sessions)

    @property
    def queries(self) -> list[Turn]:
        return [t for s in self.sessions for t in s if t.kind == "query"]

    def fact(self, fid: str) -> Fact:
        return self.facts[fid]


@dataclass(slots=True)
class ContextItem:
    """A chunk placed into the model's context window at assembly time.

    ``source`` records which memory subsystem produced it (working / summary /
    retrieved / semantic), which lets experiments attribute where a recalled
    fact came from. ``fact_ids`` are the facts carried by this chunk; they are
    treated as sitting at the chunk's center for position purposes.
    """

    source: str
    text: str
    tokens: int
    fact_ids: tuple[str, ...] = ()


@dataclass
class AssembledContext:
    """The ordered list of context items handed to the provider."""

    items: list[ContextItem] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return sum(it.tokens for it in self.items)

    def add(self, item: ContextItem) -> None:
        self.items.append(item)

    def fact_positions(self) -> dict[str, float]:
        """Map fact-id -> best (most accessible) relative position in ``[0, 1]``.

        A fact may appear in several items (e.g. still in working memory *and*
        re-surfaced by retrieval). We record every occurrence's center position;
        the provider will use the most favourable one.
        """
        total = max(1, self.total_tokens)
        best: dict[str, float] = {}
        cursor = 0
        for it in self.items:
            center = cursor + it.tokens / 2.0
            rel = center / total
            for fid in it.fact_ids:
                # "best" = closest to an edge (primacy/recency); smaller edge
                # distance is better, so keep the position with min |rel-0.5|-ish
                # edge score handled by provider; here we just keep all and let
                # the provider pick. We store the one with the highest edge
                # proximity to keep the map compact.
                prev = best.get(fid)
                if prev is None or _edge_score(rel) > _edge_score(prev):
                    best[fid] = rel
            cursor += it.tokens
        return best

    def all_fact_ids(self) -> set[str]:
        out: set[str] = set()
        for it in self.items:
            out.update(it.fact_ids)
        return out


def _edge_score(rel: float) -> float:
    """1.0 at either edge, lower toward the middle (lost-in-the-middle shape)."""
    return (2.0 * rel - 1.0) ** 2
