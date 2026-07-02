"""LLM provider abstraction.

The harness never calls a model directly; it goes through :class:`Provider`.
This is what lets the *same* agent/memory code run against:

* :class:`~dah.providers.simulated.SimulatedProvider` - a deterministic model of
  context-length degradation, used for reproducible ablations, and
* :class:`~dah.providers.openai_compat.OpenAICompatProvider` - a real
  OpenAI-compatible endpoint (optional, requires the ``openai`` extra + a key).
"""

from .base import ModelResponse, Provider
from .simulated import AccessibilityParams, SimulatedProvider

__all__ = ["Provider", "ModelResponse", "SimulatedProvider", "AccessibilityParams"]
