"""Failure types and recovery policy.

Recovery in this harness is deliberately boring and explicit - the three moves
that actually work in practice:

1. **Retry** a transient failure (with a cap), because most tool/step failures
   are transient.
2. **Roll back** first, so the retry starts from clean state instead of
   compounding a half-applied mutation.
3. **Escalate** the strategy on repeated failure (retrieve deeper, spend more),
   rather than repeating the identical failing action.
"""

from __future__ import annotations

from dataclasses import dataclass


class TransientToolError(RuntimeError):
    """A recoverable, transient failure while processing a step (tool/parse)."""


@dataclass
class RecoveryPolicy:
    """How hard to try before giving up on a *step* (ingestion).

    Query-side recovery (verify -> retry deeper) is owned by the planner
    (``max_retries`` / ``escalate_factor``); this policy governs the *state*
    side: whether to rewind before retrying a failed ingestion, and how many
    times to retry it.
    """

    ingestion_retries: int = 0  # extra attempts to ingest a failed info turn
    rollback_on_failure: bool = False  # rewind state before retrying

    @classmethod
    def none(cls) -> "RecoveryPolicy":
        return cls()

    @classmethod
    def robust(cls) -> "RecoveryPolicy":
        return cls(ingestion_retries=3, rollback_on_failure=True)
