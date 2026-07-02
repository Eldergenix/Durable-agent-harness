"""Seeded failure injection for the recovery study (T2.4).

Long-horizon agents don't fail only because context degrades; they fail because
*steps* fail - a tool times out, a parse breaks, a retrieved blob is corrupt -
and the agent carries the damage forward. Measuring recovery honestly requires
failures that can be switched on, reproduced exactly, and switched off leaving a
bit-identical control run.

:class:`FaultInjector` owns its own explicitly-seeded RNG (R1), separate from
the agent's, so injection draws never perturb provider/compaction randomness -
which keeps arms paired. At rate ``<= 0`` it performs **no draws at all**, so a
zero-rate injector is byte-for-byte identical to having no injector: exactly the
control arm the recovery study needs. Snapshot/restore covers the RNG state (R6)
so an injector can ride inside checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FaultConfig:
    """Rates and magnitudes for injected failures."""

    ingest_fail_rate: float = 0.0  # P(an info turn fails to ingest transiently)
    poison_tokens: int = 100  # size of the corrupt chunk left in working memory on failure


class FaultInjector:
    """Decides, deterministically from a seed, when injected failures fire."""

    def __init__(self, config: FaultConfig | None = None, seed: int = 0):
        self.config = config or FaultConfig()
        self.rng = np.random.default_rng(seed)

    def ingest_fails(self) -> bool:
        """Whether the current ingestion step suffers a transient failure.

        Short-circuits at rate ``<= 0`` *without drawing*, guaranteeing a
        zero-rate injector is indistinguishable from no injector at all.
        """
        if self.config.ingest_fail_rate <= 0.0:
            return False
        return bool(self.rng.random() < self.config.ingest_fail_rate)

    @property
    def poison_tokens(self) -> int:
        """Token size of the corrupt chunk an injected failure leaves behind."""
        return self.config.poison_tokens

    def snapshot(self) -> dict:
        """Deep-copyable state, including RNG position (R6)."""
        return {
            "ingest_fail_rate": self.config.ingest_fail_rate,
            "poison_tokens": self.config.poison_tokens,
            "rng": self.rng.bit_generator.state,
        }

    def restore(self, snap: dict) -> None:
        """Restore config and RNG position captured by :meth:`snapshot`."""
        self.config = FaultConfig(
            ingest_fail_rate=snap["ingest_fail_rate"], poison_tokens=snap["poison_tokens"]
        )
        self.rng.bit_generator.state = snap["rng"]
