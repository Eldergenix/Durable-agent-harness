#!/bin/bash
# stop hook (RULES.md R15 + R14): an agent may not declare itself done while
# the fast quality gate is red, and code-changing sessions must leave a
# journal entry.
#
#   1. If the working tree has no relevant changes -> no action.
#   2. Run `quality_gate.py --fast`; if it fails  -> bounce the agent back
#      with the failing output (followup_message).
#   3. If code changed but docs/JOURNAL.md did not -> remind once to journal.
#
# DAH_SKIP_GATE=1 short-circuits to "no action" (used by the governance tests
# to validate JSON plumbing without recursive gate runs).
set -u

cat > /dev/null  # payload unused; decision is derived from repo state

if [ "${DAH_SKIP_GATE:-0}" = "1" ]; then
  printf '{}\n'
  exit 0
fi

status=$(git status --porcelain 2>/dev/null || true)

changed_relevant=$(printf '%s\n' "$status" | grep -E '^\s*[AM?R ][MDA?R ]?\s+(src/|scripts/|tests/|docs/|[A-Z]+\.md)' || true)
if [ -z "$changed_relevant" ]; then
  printf '{}\n'
  exit 0
fi

PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"

gate_out=$("$PY" scripts/quality_gate.py --fast 2>&1)
gate_rc=$?

if [ $gate_rc -ne 0 ]; then
  printf '%s' "$gate_out" | tail -n 30 | /usr/bin/jq -cRs \
    '{followup_message: ("The fast quality gate is RED — the task is not done (RULES.md R15). Fix these failures, rerun `python scripts/quality_gate.py --fast`, and only then finish:\n" + .)}'
  exit 0
fi

code_changed=$(printf '%s\n' "$status" | grep -E '^\s*\S+\s+(src/|scripts/|tests/)' || true)
journal_changed=$(printf '%s\n' "$status" | grep -F 'docs/JOURNAL.md' || true)

if [ -n "$code_changed" ] && [ -z "$journal_changed" ]; then
  printf '{"followup_message": "Gate is green, but this session changed code without journaling (RULES.md R14). Append an entry to docs/JOURNAL.md — date, task IDs, what changed, evidence, next step — then finish."}\n'
  exit 0
fi

printf '{}\n'
exit 0
