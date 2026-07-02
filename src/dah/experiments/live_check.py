"""Manual live-endpoint smoke test for the OpenAI-compatible provider (T5.2).

This is the *only* place the optional real-LLM path is exercised, and it is
never part of the deterministic study::

    pip install -e ".[openai]"
    export OPENAI_API_KEY=...        # and optionally OPENAI_BASE_URL
    python -m dah.experiments.live_check
"""

from __future__ import annotations


def main() -> int:
    """Ask a live model one traceable question; report the ids it used."""
    from ..providers.openai_compat import self_check

    recalled = self_check()
    print(f"live check recalled: {sorted(recalled)}")
    ok = "f_code" in recalled
    print("live check:", "OK" if ok else "FAIL (model did not cite the planted fact)")
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
