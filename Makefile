.PHONY: install test gate ablation baseline analyze gallery reproduce quick clean

VENV ?= .venv
PY := $(VENV)/bin/python

install:
	python3 -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest

gate:
	$(PY) scripts/quality_gate.py --strict

ablation:
	$(PY) -m dah.experiments.run_ablation

baseline:
	$(PY) -m dah.experiments.run_baseline

analyze:
	$(PY) -m dah.experiments.analyze

gallery:
	$(PY) -m dah.experiments.gallery

# Full study: ~4 min on a laptop, fully deterministic.
reproduce: ablation baseline analyze gallery
	@echo "Done. See results/ for CSVs + figures + verdicts, docs/RESULTS.md for the writeup."

# Fast smoke of the whole pipeline (few tasks).
quick:
	$(PY) -m dah.experiments.run_ablation --quick
	$(PY) -m dah.experiments.run_baseline --quick
	$(PY) -m dah.experiments.analyze

clean:
	rm -rf results/*.png results/*.csv results/summary.md results/traces.md
	rm -rf .pytest_cache **/__pycache__
