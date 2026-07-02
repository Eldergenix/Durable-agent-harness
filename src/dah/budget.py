"""Multi-dimensional budget accounting.

Long-horizon agents fail in two opposite ways around resources:

1. They run *out* of budget mid-task (context/token limit, wall-clock, tool
   quota) and produce nothing usable.
2. They *hoard* budget defensively and never attempt the ambitious plan that
   would have succeeded.

A good harness treats budget as a first-class signal the planner can *see* and
*adapt to*, not just a wall it hits. :class:`Budget` therefore exposes both a
hard ``charge`` path (raises when a limit is crossed) and a soft ``pressure``
signal in ``[0, 1]`` that the planner uses to choose cheaper strategies before
anything fails.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator


class BudgetExceeded(RuntimeError):
    """Raised when a hard budget limit is crossed by :meth:`Budget.charge`."""

    def __init__(self, dimension: str, spent: float, limit: float) -> None:
        super().__init__(
            f"budget exceeded on '{dimension}': spent {spent:g} > limit {limit:g}"
        )
        self.dimension = dimension
        self.spent = spent
        self.limit = limit


_DIMS = ("tokens", "steps", "tool_calls", "wall_seconds")


@dataclass
class Budget:
    """Tracks spend against optional per-dimension limits.

    A ``None`` limit means "unbounded" for that dimension. All spend is
    monotonically increasing; there is no refund path by design (rollback
    restores *state*, not consumed budget - you don't get tokens back for work
    you already paid for, which is what makes budget-aware planning matter).
    """

    max_tokens: int | None = None
    max_steps: int | None = None
    max_tool_calls: int | None = None
    max_wall_seconds: float | None = None

    tokens: float = 0.0
    steps: float = 0.0
    tool_calls: float = 0.0
    wall_seconds: float = 0.0

    def _limit(self, dim: str) -> float | None:
        return getattr(self, f"max_{dim}")

    def _spent(self, dim: str) -> float:
        return getattr(self, dim)

    def can_afford(
        self,
        *,
        tokens: float = 0.0,
        steps: float = 0.0,
        tool_calls: float = 0.0,
        wall_seconds: float = 0.0,
    ) -> bool:
        """True iff charging this amount would not cross any hard limit."""
        prospective = {
            "tokens": self.tokens + tokens,
            "steps": self.steps + steps,
            "tool_calls": self.tool_calls + tool_calls,
            "wall_seconds": self.wall_seconds + wall_seconds,
        }
        for dim in _DIMS:
            limit = self._limit(dim)
            if limit is not None and prospective[dim] > limit:
                return False
        return True

    def charge(
        self,
        *,
        tokens: float = 0.0,
        steps: float = 0.0,
        tool_calls: float = 0.0,
        wall_seconds: float = 0.0,
    ) -> None:
        """Record spend; raise :class:`BudgetExceeded` if a hard limit is crossed.

        The spend is applied *before* the check so that the ``spent`` figure in
        the exception reflects reality (useful for post-mortem telemetry).
        Spend is monotone: negative amounts are rejected because rollback must
        restore state, never refund budget (RULES.md R7).
        """
        for name, amount in (
            ("tokens", tokens),
            ("steps", steps),
            ("tool_calls", tool_calls),
            ("wall_seconds", wall_seconds),
        ):
            if amount < 0:
                raise ValueError(f"negative charge on '{name}': {amount!r}")
        self.tokens += tokens
        self.steps += steps
        self.tool_calls += tool_calls
        self.wall_seconds += wall_seconds
        for dim in _DIMS:
            limit = self._limit(dim)
            if limit is not None and self._spent(dim) > limit:
                raise BudgetExceeded(dim, self._spent(dim), limit)

    def remaining(self, dim: str) -> float:
        """Absolute remaining budget on ``dim`` (``inf`` if unbounded)."""
        limit = self._limit(dim)
        if limit is None:
            return float("inf")
        return max(0.0, limit - self._spent(dim))

    def remaining_fraction(self, dim: str) -> float:
        """Fraction of ``dim`` still available in ``[0, 1]`` (1.0 if unbounded)."""
        limit = self._limit(dim)
        if limit is None or limit <= 0:
            return 1.0
        return max(0.0, min(1.0, 1.0 - self._spent(dim) / limit))

    def pressure(self) -> float:
        """Worst-case resource pressure across all bounded dimensions.

        ``0.0`` means "plenty of headroom everywhere"; ``1.0`` means "at least
        one dimension is exhausted". This is the scalar the planner reads to
        decide how aggressive a plan it can afford.
        """
        bounded = [1.0 - self.remaining_fraction(d) for d in _DIMS if self._limit(d) is not None]
        return max(bounded) if bounded else 0.0

    def snapshot(self) -> dict[str, float]:
        """Serialisable view of current spend (for checkpoints / reporting)."""
        return {d: self._spent(d) for d in _DIMS}

    def bounded_dims(self) -> Iterator[str]:
        for d in _DIMS:
            if self._limit(d) is not None:
                yield d

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        parts = []
        for d in _DIMS:
            limit = self._limit(d)
            if limit is None:
                parts.append(f"{d}={self._spent(d):g}/inf")
            else:
                parts.append(f"{d}={self._spent(d):g}/{limit:g}")
        return "Budget(" + ", ".join(parts) + f", pressure={self.pressure():.2f})"
