"""Checkpointing and rollback for agent state.

Rollback is the single most under-used reliability primitive in agent loops.
When a step corrupts working state (a bad tool result poisons the context, a
plan branch turns out to be a dead end, a verifier rejects an answer), the
cheapest correct recovery is often to *rewind to a known-good state* and try a
different branch - not to patch forward from a corrupted state.

This module implements a small, explicit checkpoint stack over any set of
components that implement the :class:`Snapshotable` protocol. Snapshots are
deep, immutable copies, so restoring is total (no aliasing back into live
state).
"""

from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .telemetry import Trace


@runtime_checkable
class Snapshotable(Protocol):
    """A component whose state can be captured and restored.

    Implementations MUST return a value from :meth:`snapshot` that is safe to
    hold onto (i.e. a deep copy or otherwise immutable) and that
    :meth:`restore` can consume to fully reconstruct state.
    """

    def snapshot(self) -> Any: ...

    def restore(self, snap: Any) -> None: ...


class DictSnapshotMixin:
    """Default deep-copy snapshotting over ``self.__dict__``.

    Convenient for plain state-holding objects. Components with non-copyable
    fields (open handles, RNGs you want to share) should implement the protocol
    explicitly instead.
    """

    def snapshot(self) -> Any:
        return copy.deepcopy(self.__dict__)

    def restore(self, snap: Any) -> None:
        self.__dict__.clear()
        self.__dict__.update(copy.deepcopy(snap))


@dataclass
class Checkpoint:
    """One committed snapshot: a labeled bundle of per-component state blobs."""

    id: int
    label: str
    step: int
    blobs: dict[str, Any]
    extra: dict[str, Any] = field(default_factory=dict)


class Checkpointer:
    """A stack of checkpoints over a registry of named components.

    Typical use::

        cp = Checkpointer(trace)
        cp.register("memory", memory_manager)
        cp.register("progress", progress_tracker)
        cp.commit("after-plan", step=3)
        ...                     # do risky work
        if verifier_failed:
            cp.rollback()       # restore to "after-plan"
    """

    def __init__(self, trace: Trace | None = None) -> None:
        self._components: dict[str, Snapshotable] = {}
        self._stack: list[Checkpoint] = []
        self._ids = itertools.count()
        self._trace = trace or Trace()

    # -- registration -----------------------------------------------------
    def register(self, name: str, component: Snapshotable) -> None:
        if not isinstance(component, Snapshotable):
            raise TypeError(f"component '{name}' is not Snapshotable")
        self._components[name] = component

    # -- checkpoint / rollback -------------------------------------------
    def commit(self, label: str, *, step: int = -1, **extra: Any) -> Checkpoint:
        """Capture a snapshot of every registered component."""
        blobs = {name: comp.snapshot() for name, comp in self._components.items()}
        cp = Checkpoint(id=next(self._ids), label=label, step=step, blobs=blobs, extra=extra)
        self._stack.append(cp)
        self._trace.emit("checkpoint.commit", id=cp.id, label=label, step=step, depth=len(self._stack))
        return cp

    def rollback(self, *, to: int | None = None) -> Checkpoint:
        """Restore state to a checkpoint.

        With ``to=None`` restores the most recent checkpoint (and leaves it on
        the stack so you can retry from the same point). With an explicit ``to``
        id, restores that checkpoint and discards everything committed after it.
        """
        if not self._stack:
            raise RuntimeError("nothing to roll back to")
        if to is None:
            cp = self._stack[-1]
        else:
            idx = next((i for i, c in enumerate(self._stack) if c.id == to), None)
            if idx is None:
                raise KeyError(f"no checkpoint with id {to}")
            cp = self._stack[idx]
            del self._stack[idx + 1 :]
        for name, blob in cp.blobs.items():
            self._components[name].restore(blob)
        self._trace.emit("checkpoint.rollback", id=cp.id, label=cp.label, depth=len(self._stack))
        return cp

    def pop(self) -> Checkpoint | None:
        """Discard the most recent checkpoint without restoring (commit success)."""
        if not self._stack:
            return None
        cp = self._stack.pop()
        self._trace.emit("checkpoint.pop", id=cp.id, label=cp.label, depth=len(self._stack))
        return cp

    @property
    def depth(self) -> int:
        return len(self._stack)

    def labels(self) -> list[str]:
        return [c.label for c in self._stack]
