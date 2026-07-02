#!/usr/bin/env python3
"""The single exit code that means "this repo is healthy" (RULES.md R15).

Modes:
  --fast    rules lint + byte-compile + governance tests. Runs in seconds;
            required before an agent declares any task done. Used by the
            Cursor stop hook and the git pre-commit hook.
  --strict  everything in --fast, plus the full pytest suite and package
            import checks. Required to close a milestone or tag a release.

The gate prefers the repo venv's interpreter when present, so pytest and
third-party imports resolve without requiring an activated shell.
"""

from __future__ import annotations

import argparse
import compileall
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
VENV_PY = REPO / ".venv" / "bin" / "python"


def _python() -> str:
    return str(VENV_PY) if VENV_PY.exists() else sys.executable


def _run(name: str, cmd: list[str]) -> tuple[str, bool, float, str]:
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    out = (proc.stdout + proc.stderr).strip()
    return name, proc.returncode == 0, dt, out


def _compile_check() -> tuple[str, bool, float, str]:
    t0 = time.perf_counter()
    targets = [REPO / "src", REPO / "scripts"]
    if (REPO / "tests").is_dir():
        targets.append(REPO / "tests")
    ok = all(
        compileall.compile_dir(str(t), quiet=2, force=False) for t in targets if t.exists()
    )
    return "byte-compile", ok, time.perf_counter() - t0, "" if ok else "syntax errors above"


def main() -> int:
    """Run the selected checks; print a scoreboard; exit non-zero on any failure."""
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--fast", action="store_true", help="lint + compile + governance tests")
    mode.add_argument("--strict", action="store_true", help="fast + full pytest + import checks")
    args = ap.parse_args()
    strict = bool(args.strict)

    py = _python()
    checks: list[tuple[str, bool, float, str]] = []

    checks.append(_run("rules-lint", [py, str(REPO / "scripts" / "rules_lint.py")]))
    checks.append(_compile_check())

    gov = REPO / "tests" / "test_governance.py"
    if gov.exists():
        checks.append(_run("governance-tests", [py, "-m", "pytest", str(gov), "-q", "--no-header"]))
    else:
        checks.append(("governance-tests", False, 0.0, "tests/test_governance.py is missing"))

    if strict:
        checks.append(
            _run(
                "imports",
                [
                    py,
                    "-c",
                    "import dah, dah.memory, dah.env, dah.providers; "
                    "import dah.budget, dah.checkpoint, dah.telemetry, dah.types; "
                    "print('imports ok')",
                ],
            )
        )
        checks.append(_run("pytest-full", [py, "-m", "pytest", "-q", "--no-header"]))

    width = max(len(n) for n, *_ in checks)
    failed = False
    print(f"quality_gate ({'strict' if strict else 'fast'}):")
    for name, ok, dt, out in checks:
        status = "PASS" if ok else "FAIL"
        print(f"  {name:<{width}}  {status}  ({dt:.2f}s)")
        if not ok:
            failed = True
            if out:
                indented = "\n".join(f"      {line}" for line in out.splitlines()[-40:])
                print(indented)

    print(f"quality_gate: {'FAIL' if failed else 'OK'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
