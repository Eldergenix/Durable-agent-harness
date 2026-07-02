"""An ablatable memory stack.

Every component can be toggled independently through ``MemoryConfig``, so
the ablation grid is just a set of configs sharing one implementation. The
components:

* **WorkingMemory** - the in-context window for the *current* session. Resets at
  session boundaries (multi-session = separate conversations).
* **Compactor** - lossy summarisation of overflowed turns. Trades recall
  (facts can be dropped) for shorter context (better attention). Can persist
  across sessions.
* **EpisodicStore** - append-only raw events with embeddings; similarity
  retrieval re-surfaces old facts (and, inevitably, some distractors).
* **SemanticStore** - salience-gated, compact, deduplicated knowledge; fewer
  distractors and cheaper tokens, at the cost of dropping low-salience facts.
"""

from .manager import MemoryConfig, MemoryManager, PRESETS, make_memory
from .stores import Compactor, EpisodicStore, SemanticStore, WorkingMemory

__all__ = [
    "MemoryConfig",
    "MemoryManager",
    "PRESETS",
    "make_memory",
    "WorkingMemory",
    "Compactor",
    "EpisodicStore",
    "SemanticStore",
]
