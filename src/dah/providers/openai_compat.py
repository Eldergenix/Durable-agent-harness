"""Optional OpenAI-compatible provider.

This adapter proves the :class:`~dah.providers.base.Provider` interface is not
tied to the simulation: the *same* agent, memory stack, planner, verifier, and
checkpointer run unchanged against a real chat-completions endpoint (OpenAI,
Together, vLLM, Ollama's OpenAI shim, etc.).

It is intentionally *not* imported by the default install and is **not** used by
any experiment in ``docs/RESULTS.md`` - those must stay deterministic and
oracle-graded. Use it to sanity-check the plumbing on a live model:

    pip install -e ".[openai]"
    export OPENAI_API_KEY=...    # and optionally OPENAI_BASE_URL
    python -m dah.providers.openai_compat  # tiny self-check

Grading a real model requires a real answer key or a judge; the environment in
this repo grades by *fact-id recall*, which only the simulated provider reports
directly. For live use you would swap in a judge/verifier that inspects answer
text. That wiring is left as a clearly-marked extension point below.

The interactive smoke test lives in ``dah.experiments.live_check`` (library code
is print-free per R9):

    python -m dah.experiments.live_check
"""

from __future__ import annotations

import os
import re

import numpy as np

from ..types import AssembledContext, Turn
from .base import ModelResponse


class OpenAICompatProvider:
    """Calls a chat-completions endpoint and extracts which facts it used.

    The environment tags each context chunk with a stable marker (the fact id).
    We ask the model to answer and to list the marker(s) it relied on, then parse
    those back into a ``recalled`` set so the *same* fact-id grader works. This is
    a pragmatic bridge for smoke-testing; a production setup would use a proper
    judge over free-form answers.
    """

    def __init__(self, model: str = "gpt-4o-mini", base_url: str | None = None, api_key: str | None = None):
        try:
            from openai import OpenAI  # noqa: F401
        except ImportError as e:  # pragma: no cover - optional dep
            raise ImportError("install the optional dependency: pip install -e '.[openai]'") from e
        from openai import OpenAI

        self.model = model
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
        )

    def estimate_tokens(self, context: AssembledContext, query: Turn) -> int:
        # ~4 chars/token heuristic; replace with tiktoken for accuracy
        chars = sum(len(it.text) for it in context.items) + len(query.text)
        return chars // 4 + 8

    def respond(self, query: Turn, context: AssembledContext, rng: np.random.Generator) -> ModelResponse:  # pragma: no cover - needs network
        blocks = []
        for it in context.items:
            tag = ",".join(it.fact_ids) if it.fact_ids else it.source
            blocks.append(f"[{tag}] {it.text}")
        system = (
            "You answer using only the provided context blocks. Each block is "
            "prefixed with an id in square brackets. After your answer, output a "
            "line 'USED: <comma-separated ids you relied on>'."
        )
        user = "CONTEXT:\n" + "\n".join(blocks) + f"\n\nQUESTION: {query.text}"
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.0,
        )
        text = resp.choices[0].message.content or ""
        used = set()
        m = re.search(r"USED:\s*(.+)", text)
        if m:
            used = {tok.strip() for tok in m.group(1).split(",") if tok.strip()}
        return ModelResponse(recalled=used, context_tokens=self.estimate_tokens(context, query))


def self_check() -> set[str]:  # pragma: no cover - needs a live endpoint
    """One-question live smoke test; returns the recalled id set.

    Invoked by ``python -m dah.experiments.live_check`` (which owns the printing;
    library code stays silent per R9).
    """
    from ..types import ContextItem

    prov = OpenAICompatProvider()
    ctx = AssembledContext()
    ctx.add(ContextItem(source="working", text="The launch code is ZULU-7.", tokens=8, fact_ids=("f_code",)))
    q = Turn(index=0, session=0, kind="query", text="What is the launch code?", tokens=8,
             required_fact_ids=["f_code"], query_id="q0")
    r = prov.respond(q, ctx, np.random.default_rng(0))
    return r.recalled
