"""The memory manager: composition, retrieval policy, and context assembly."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace

import numpy as np

from ..budget import Budget
from ..types import AssembledContext, Fact, Turn
from .stores import Compactor, EpisodicStore, SemanticStore, WItem, WorkingMemory


@dataclass
class MemoryConfig:
    """Feature switches + hyper-parameters for the memory stack."""

    name: str = "custom"

    working_window_tokens: int = 700
    context_window_tokens: int = 1600  # hard model cap on assembled context

    use_compaction: bool = False
    compaction_recall: float = 0.80  # P(a fact survives summarisation)
    summary_tokens_per_fact: int = 6
    summary_base_tokens: int = 10
    persist_summary: bool = True  # carry the summary across sessions

    use_episodic: bool = False
    episodic_tokens_per_fact: int = 20

    use_semantic: bool = False
    sem_tokens_per_fact: int = 8
    sem_salience_threshold: float = 0.70

    retrieval_policy: str = "off"  # off | always | budget_aware | gap_triggered
    retrieval_k: int = 5
    gap_sim_threshold: float = 0.45  # gap_triggered: skip retrieval if working already this similar


@dataclass
class AssembleStats:
    """Bookkeeping about one context assembly (for telemetry and budgeting)."""

    total_tokens: int = 0
    n_working_items: int = 0
    summary_present: bool = False
    retrieval_performed: bool = False
    n_retrieved: int = 0
    retrieved_sources: dict[str, int] = field(default_factory=dict)
    trimmed: int = 0


class MemoryManager:
    """Owns all memory subsystems and builds the context handed to the provider.

    Only ``fact_db`` is shared/immutable; everything else is snapshot-safe light
    state. The manager never reads a query's answer key - retrieval uses only the
    ``query_embedding`` supplied by the environment.
    """

    def __init__(self, config: MemoryConfig, fact_db: dict[str, Fact], seed: int = 0):
        self.config = config
        self.fact_db = fact_db  # shared, read-only
        self.session_idx = -1
        self.working = WorkingMemory(config.working_window_tokens)
        self.compactor = Compactor(
            recall=config.compaction_recall,
            tokens_per_fact=config.summary_tokens_per_fact,
            base_tokens=config.summary_base_tokens,
            seed=seed,
        )
        self.episodic = EpisodicStore(config.episodic_tokens_per_fact)
        self.semantic = SemanticStore(config.sem_tokens_per_fact, config.sem_salience_threshold)

    # -- ingestion --------------------------------------------------------
    def start_session(self, session_idx: int) -> None:
        """New session = fresh conversation: working memory resets."""
        self.session_idx = session_idx
        evicted = self.working.items
        if self.config.use_compaction and evicted:
            self.compactor.ingest(evicted)
        self.working.reset()
        if self.config.use_compaction:
            self.compactor.reset(persist=self.config.persist_summary)

    def add_info(self, turn: Turn) -> None:
        facts = [self.fact_db[f] for f in turn.fact_ids]
        item = WItem(
            source="working",
            text=turn.text,
            tokens=turn.tokens,
            fact_ids=tuple(turn.fact_ids),
        )
        evicted = self.working.add(item)
        if self.config.use_compaction and evicted:
            self.compactor.ingest(evicted)
        # persistent stores capture every fact regardless of window pressure
        for fact in facts:
            if self.config.use_episodic:
                self.episodic.add(fact)
            if self.config.use_semantic:
                self.semantic.add(fact)

    def ingest_poison(self, tokens: int) -> None:
        """Simulate a corrupt observation landing in working memory.

        A failed tool/parse step can inject junk that carries no useful facts but
        still consumes the window and evicts good turns. This is the concrete
        state corruption that checkpoint+rollback exists to undo.
        """
        evicted = self.working.add(WItem(source="poison", text="<corrupt>", tokens=tokens, fact_ids=()))
        if self.config.use_compaction and evicted:
            self.compactor.ingest(evicted)

    # -- retrieval policy -------------------------------------------------
    def _decide_k(self, query_embedding: np.ndarray, budget: Budget) -> int:
        policy = self.config.retrieval_policy
        base_k = self.config.retrieval_k
        if policy == "off":
            return 0
        if policy == "always":
            return base_k
        if policy == "budget_aware":
            frac = budget.remaining_fraction("tokens")
            if frac <= 0.0:
                return 0
            return max(1, int(round(base_k * frac)))
        if policy == "gap_triggered":
            # if working memory already looks relevant to the query, don't spend
            # retrieval budget; otherwise retrieve.
            wfacts = [self.fact_db[f].embedding for f in self.working.fact_ids()]
            if wfacts:
                sims = np.stack(wfacts) @ query_embedding
                if float(sims.max()) >= self.config.gap_sim_threshold:
                    return 0
            return base_k
        raise ValueError(f"unknown retrieval_policy {policy!r}")

    # -- assembly ---------------------------------------------------------
    def assemble(
        self,
        query_embedding: np.ndarray,
        query_tokens: int,
        budget: Budget,
        k_override: int | None = None,
        context_cap: int | None = None,
    ) -> tuple[AssembledContext, AssembleStats]:
        ctx = AssembledContext()
        stats = AssembleStats()

        # 1) compacted summary goes first (older context => primacy position)
        if self.config.use_compaction:
            summ = self.compactor.summary_item()
            if summ is not None:
                ctx.add(summ)
                stats.summary_present = True

        # 2) working memory, oldest -> newest
        for it in self.working.items:
            ctx.add(_witem_to_ctx(it))
        stats.n_working_items = len(self.working.items)

        # 3) retrieval appended last (recency position). A planner may override
        #    the depth ``k`` (budget-aware); otherwise defer to the memory policy.
        if not (self.config.use_episodic or self.config.use_semantic):
            k = 0
        elif k_override is not None:
            k = max(0, int(k_override))
        else:
            k = self._decide_k(query_embedding, budget)
        if k > 0:
            exclude = ctx.all_fact_ids()
            retrieved = []
            if self.config.use_semantic:
                sem = self.semantic.retrieve(query_embedding, k, self.fact_db, exclude)
                retrieved.extend(sem)
                exclude |= {fid for it in sem for fid in it.fact_ids}
            if self.config.use_episodic:
                epi = self.episodic.retrieve(query_embedding, k, self.fact_db, exclude)
                retrieved.extend(epi)
            for it in retrieved:
                ctx.add(it)
                stats.retrieved_sources[it.source] = stats.retrieved_sources.get(it.source, 0) + 1
            stats.n_retrieved = len(retrieved)
            stats.retrieval_performed = len(retrieved) > 0

        # 4) enforce the hard context-window cap. Trim in order of information
        #    density per token: raw working turns first (oldest first, they are
        #    mostly filler), then retrieved items (deepest hit last), and the
        #    compacted summary only as an absolute last resort - it is the
        #    densest chunk and the only carrier of the compressed past.
        #    A budget-aware planner can pass a tighter per-query cap.
        cap = self.config.context_window_tokens
        if context_cap is not None:
            cap = min(cap, context_cap)
        while ctx.total_tokens + query_tokens > cap and ctx.items:
            drop_idx = _first_index(ctx.items, "working", "poison")
            if drop_idx is None:
                drop_idx = _last_index(ctx.items, "retrieved", "semantic")
            if drop_idx is None:
                drop_idx = 0  # only the summary remains
            ctx.items.pop(drop_idx)
            stats.trimmed += 1

        stats.total_tokens = ctx.total_tokens
        return ctx, stats

    # -- snapshot / restore (checkpointing) -------------------------------
    def snapshot(self) -> dict:
        return {
            "config": copy.deepcopy(self.config),
            "session_idx": self.session_idx,
            "working": self.working.snapshot(),
            "compactor": self.compactor.snapshot(),
            "episodic": self.episodic.snapshot(),
            "semantic": self.semantic.snapshot(),
        }

    def restore(self, snap: dict) -> None:
        self.config = copy.deepcopy(snap["config"])
        self.session_idx = snap["session_idx"]
        self.working.restore(snap["working"])
        self.compactor.restore(snap["compactor"])
        self.episodic.restore(snap["episodic"])
        self.semantic.restore(snap["semantic"])


def _witem_to_ctx(it: WItem):
    from ..types import ContextItem

    return ContextItem(source=it.source, text=it.text, tokens=it.tokens, fact_ids=it.fact_ids)


def _last_index(items, *sources: str) -> int | None:
    for i in range(len(items) - 1, -1, -1):
        if items[i].source in sources:
            return i
    return None


def _first_index(items, *sources: str) -> int | None:
    for i, it in enumerate(items):
        if it.source in sources:
            return i
    return None


# ---------------------------------------------------------------------------
# Presets: the cells of the memory ablation grid.
# ---------------------------------------------------------------------------
PRESETS: dict[str, MemoryConfig] = {
    "none": MemoryConfig(name="none"),
    "compaction": MemoryConfig(name="compaction", use_compaction=True, persist_summary=True),
    "episodic": MemoryConfig(name="episodic", use_episodic=True, retrieval_policy="always"),
    "semantic": MemoryConfig(name="semantic", use_semantic=True, retrieval_policy="always"),
    "episodic+semantic": MemoryConfig(
        name="episodic+semantic",
        use_episodic=True,
        use_semantic=True,
        retrieval_policy="always",
    ),
    "compaction+episodic": MemoryConfig(
        name="compaction+episodic",
        use_compaction=True,
        persist_summary=True,
        use_episodic=True,
        retrieval_policy="always",
    ),
    "compaction+semantic": MemoryConfig(
        name="compaction+semantic",
        use_compaction=True,
        persist_summary=True,
        use_semantic=True,
        retrieval_policy="always",
    ),
    "full": MemoryConfig(
        name="full",
        use_compaction=True,
        persist_summary=True,
        use_episodic=True,
        use_semantic=True,
        retrieval_policy="always",
    ),
    "full_budget_aware": MemoryConfig(
        name="full_budget_aware",
        use_compaction=True,
        persist_summary=True,
        use_episodic=True,
        use_semantic=True,
        retrieval_policy="budget_aware",
    ),
    "full_gap_triggered": MemoryConfig(
        name="full_gap_triggered",
        use_compaction=True,
        persist_summary=True,
        use_episodic=True,
        use_semantic=True,
        retrieval_policy="gap_triggered",
    ),
}


def make_memory(
    name: str,
    fact_db: dict[str, Fact],
    seed: int = 0,
    overrides: dict | None = None,
) -> MemoryManager:
    """Instantiate a memory manager from a preset name, with optional overrides."""
    if name not in PRESETS:
        raise KeyError(f"unknown memory preset {name!r}; choices: {sorted(PRESETS)}")
    cfg = replace(PRESETS[name], **(overrides or {}))
    return MemoryManager(cfg, fact_db=fact_db, seed=seed)
