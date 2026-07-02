"""Controlled multi-session benchmark environment.

The environment is the *ground truth*: it generates tasks, owns the answer key
(which facts each query needs), and grades answers. Crucially, its
context-degradation model lives in :mod:`dah.providers.simulated` and is
**independent of the memory policy under test** - the same accessibility
function scores every configuration, so a memory policy can only win by placing
the right facts in accessible positions, never by changing the rules.
"""

from .multisession import EpisodeResult, MultiSessionEnv, QueryResult
from .tasks import TaskConfig, generate_suite, generate_task

__all__ = [
    "TaskConfig",
    "generate_task",
    "generate_suite",
    "MultiSessionEnv",
    "QueryResult",
    "EpisodeResult",
]
