"""Reproducible experiments.

* ``config``       - the single source of truth for all experiment settings.
* ``runner``       - pure functions that turn configs into tidy result frames.
* ``run_ablation`` - the memory ablation grid + noise / context-fill sweeps.
* ``run_baseline`` - the harness-feature ladder (baseline -> full harness).
* ``analyze``      - renders every figure and table in ``docs/RESULTS.md``.
"""
