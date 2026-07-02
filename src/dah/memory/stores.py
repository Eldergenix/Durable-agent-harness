"""Concrete memory stores.

All stores keep only *light* mutable state (ids, counters, token totals, RNG
state) and look up heavy immutable data (embeddings, token sizes, topics) in a
shared read-only ``fact_db``. This keeps checkpoint snapshots small: rolling
back does not clone every embedding, only the bookkeeping.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from ..types import ContextItem, Fact


def _cosine_topk(
    query: np.ndarray,
    ids: list[str],
    fact_db: dict[str, Fact],
    k: int,
    exclude: set[str],
) -> list[tuple[str, float]]:
    """Return up to ``k`` (fact_id, score) pairs by cosine similarity.

    Embeddings are unit-norm, so a dot product is the cosine. Excluded ids
    (already present in context) are skipped so retrieval spends its budget on
    genuinely new information.
    """
    cands = [fid for fid in ids if fid not in exclude]
    if not cands or k <= 0:
        return []
    mat = np.stack([fact_db[fid].embedding for fid in cands])
    scores = mat @ query
    order = np.argsort(-scores)[:k]
    return [(cands[i], float(scores[i])) for i in order]


@dataclass
class WItem:
    """One working-memory entry: a turn's contribution to the live window."""

    source: str
    text: str
    tokens: int
    fact_ids: tuple[str, ...]


class WorkingMemory:
    """Bounded, ordered in-context window for the current session."""

    def __init__(self, window_tokens: int):
        self.window_tokens = window_tokens
        self.items: list[WItem] = []

    @property
    def tokens(self) -> int:
        return sum(it.tokens for it in self.items)

    def add(self, item: WItem) -> list[WItem]:
        """Append an item and evict the oldest items until within the window.

        Returns the evicted items so a compactor can absorb them (or, for the
        naive policy, they are simply discarded by the caller).
        """
        self.items.append(item)
        evicted: list[WItem] = []
        while self.tokens > self.window_tokens and len(self.items) > 1:
            evicted.append(self.items.pop(0))
        return evicted

    def reset(self) -> None:
        self.items = []

    def fact_ids(self) -> set[str]:
        out: set[str] = set()
        for it in self.items:
            out.update(it.fact_ids)
        return out

    def snapshot(self) -> dict:
        return {"window_tokens": self.window_tokens, "items": copy.deepcopy(self.items)}

    def restore(self, snap: dict) -> None:
        self.window_tokens = snap["window_tokens"]
        self.items = copy.deepcopy(snap["items"])


class Compactor:
    """Lossy summarisation of evicted turns.

    Each fact that flows in survives with probability ``recall`` (rolled once,
    the first time it is compacted). Surviving facts contribute
    ``tokens_per_fact`` to a single summary chunk - much smaller than the raw
    turns, which is the point: it keeps the context short. The RNG lives inside
    the compactor so that snapshot/rollback reproduces the same survival draws.
    """

    def __init__(self, recall: float, tokens_per_fact: int, base_tokens: int, seed: int):
        self.recall = recall
        self.tokens_per_fact = tokens_per_fact
        self.base_tokens = base_tokens
        self.retained: list[str] = []
        self._seen: set[str] = set()
        self.rng = np.random.default_rng(seed)

    def ingest(self, evicted: list[WItem]) -> None:
        for it in evicted:
            for fid in it.fact_ids:
                if fid in self._seen:
                    continue
                self._seen.add(fid)
                if self.rng.random() < self.recall:
                    self.retained.append(fid)

    def tokens(self) -> int:
        if not self.retained:
            return 0
        return self.base_tokens + self.tokens_per_fact * len(self.retained)

    def summary_item(self) -> ContextItem | None:
        if not self.retained:
            return None
        return ContextItem(
            source="summary",
            text=f"summary of {len(self.retained)} facts",
            tokens=self.tokens(),
            fact_ids=tuple(self.retained),
        )

    def reset(self, persist: bool) -> None:
        if not persist:
            self.retained = []
            self._seen = set()

    def snapshot(self) -> dict:
        return {
            "recall": self.recall,
            "tokens_per_fact": self.tokens_per_fact,
            "base_tokens": self.base_tokens,
            "retained": list(self.retained),
            "seen": set(self._seen),
            "rng": self.rng.bit_generator.state,
        }

    def restore(self, snap: dict) -> None:
        self.recall = snap["recall"]
        self.tokens_per_fact = snap["tokens_per_fact"]
        self.base_tokens = snap["base_tokens"]
        self.retained = list(snap["retained"])
        self._seen = set(snap["seen"])
        self.rng.bit_generator.state = snap["rng"]


class EpisodicStore:
    """Append-only raw event store with similarity retrieval (persists forever)."""

    def __init__(self, tokens_per_fact: int):
        self.tokens_per_fact = tokens_per_fact
        self.ids: list[str] = []

    def add(self, fact: Fact) -> None:
        self.ids.append(fact.id)

    def retrieve(
        self, query: np.ndarray, k: int, fact_db: dict[str, Fact], exclude: set[str]
    ) -> list[ContextItem]:
        hits = _cosine_topk(query, self.ids, fact_db, k, exclude)
        return [
            ContextItem(
                source="retrieved",
                text=f"episodic:{fid}",
                tokens=self.tokens_per_fact,
                fact_ids=(fid,),
            )
            for fid, _ in hits
        ]

    def snapshot(self) -> dict:
        return {"tokens_per_fact": self.tokens_per_fact, "ids": list(self.ids)}

    def restore(self, snap: dict) -> None:
        self.tokens_per_fact = snap["tokens_per_fact"]
        self.ids = list(snap["ids"])


class SemanticStore:
    """Salience-gated, compact knowledge store.

    Only facts at or above ``salience_threshold`` are consolidated, and they are
    stored with a smaller token footprint. The upside is *precision*: the store
    contains fewer, higher-value items, so retrieval surfaces fewer distractors.
    The downside is *recall*: a query that needs a low-salience fact will never
    find it here. Whether that trade is worth it is an empirical question - which
    is exactly what the ablation measures.
    """

    def __init__(self, tokens_per_fact: int, salience_threshold: float):
        self.tokens_per_fact = tokens_per_fact
        self.salience_threshold = salience_threshold
        self.ids: list[str] = []

    def add(self, fact: Fact) -> None:
        if fact.salience >= self.salience_threshold:
            self.ids.append(fact.id)

    def retrieve(
        self, query: np.ndarray, k: int, fact_db: dict[str, Fact], exclude: set[str]
    ) -> list[ContextItem]:
        hits = _cosine_topk(query, self.ids, fact_db, k, exclude)
        return [
            ContextItem(
                source="semantic",
                text=f"semantic:{fid}",
                tokens=self.tokens_per_fact,
                fact_ids=(fid,),
            )
            for fid, _ in hits
        ]

    def snapshot(self) -> dict:
        return {
            "tokens_per_fact": self.tokens_per_fact,
            "salience_threshold": self.salience_threshold,
            "ids": list(self.ids),
        }

    def restore(self, snap: dict) -> None:
        self.tokens_per_fact = snap["tokens_per_fact"]
        self.salience_threshold = snap["salience_threshold"]
        self.ids = list(snap["ids"])
