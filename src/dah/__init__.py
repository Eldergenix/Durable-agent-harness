"""durable-agent-harness (dah).

A compact, research-grade harness for *long-horizon* agents. It provides the
control-plane machinery that separates a robust agent from a naive
plan-and-execute loop:

* ``budget``      - multi-dimensional budget accounting (tokens/steps/tools/time)
* ``checkpoint``  - serialisable state snapshots with rollback
* ``verifier``    - oracle-free post-condition verification (``Verdict``)
* ``recovery``    - retry / rollback-first policies for failure recovery
* ``planner``     - budget-aware planning that degrades gracefully under pressure
* ``faults``      - seeded failure injection (rate 0 == no injector, bit-for-bit)
* ``agents``      - one toggleable loop; ``BaselineAgent`` vs ``HarnessAgent``
* ``memory``      - an ablatable memory stack (working / compaction / episodic /
                    semantic) behind a single ``MemoryManager`` interface
* ``env``         - a controlled multi-session benchmark whose context-degradation
                    model is *independent* of the memory policy under test; the
                    environment owns ALL grading
* ``providers``   - a thin LLM interface with a deterministic simulated backend
                    and an optional OpenAI-compatible adapter

The design goal is that every claim in ``docs/RESULTS.md`` is reproduced by code
in this package with a fixed seed.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
