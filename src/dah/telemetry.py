"""Lightweight, dependency-free telemetry.

Every component writes structured events into a :class:`Trace`. Experiments read
the trace back to compute metrics (tokens spent, rollbacks taken, retrievals
issued, ...). Keeping this in one place means the harness has a single, testable
source of truth for "what happened" during a run, rather than scattering
``print`` statements and ad-hoc counters.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Event:
    """A single structured telemetry event."""

    kind: str
    ts: float
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trace:
    """An append-only event log plus derived counters.

    The trace is intentionally cheap: appending an event is O(1) and the counter
    bump is O(1). Aggregation happens lazily via :meth:`count` / :meth:`sum`.
    """

    events: list[Event] = field(default_factory=list)
    counters: Counter[str] = field(default_factory=Counter)
    _t0: float = field(default_factory=time.perf_counter)

    def emit(self, kind: str, **data: Any) -> Event:
        """Record an event of ``kind`` with arbitrary structured ``data``."""
        ev = Event(kind=kind, ts=time.perf_counter() - self._t0, data=data)
        self.events.append(ev)
        self.counters[kind] += 1
        return ev

    def count(self, kind: str) -> int:
        """Number of events of a given ``kind``."""
        return self.counters.get(kind, 0)

    def sum(self, kind: str, key: str) -> float:
        """Sum a numeric ``data[key]`` across all events of ``kind``."""
        return float(sum(e.data.get(key, 0) or 0 for e in self.events if e.kind == kind))

    def last(self, kind: str) -> Event | None:
        for e in reversed(self.events):
            if e.kind == kind:
                return e
        return None

    def summary(self) -> dict[str, int]:
        """A dict of event-kind -> count, handy for assertions and logging."""
        return dict(self.counters)
