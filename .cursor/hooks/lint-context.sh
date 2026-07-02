#!/bin/bash
# postToolUse feedback loop: after any file edit that touches Python under
# src/, scripts/ or tests/, run the rules linter and feed violations straight
# back to the agent as additional context. This closes the loop between
# RULES.md and the code *while it is being written*, instead of at gate time.
set -u

input=$(cat)

path=$(printf '%s' "$input" | /usr/bin/jq -r '
  (.tool_input // .input // .params // .arguments // {}) as $i
  | ($i.path // $i.file_path // $i.filePath // $i.target_file // $i.targetFile
     // $i.target_notebook // .path // .file_path // empty)' 2>/dev/null || true)

case "$path" in
  *src/*.py|*scripts/*.py|*tests/*.py) : ;;
  *) printf '{}\n'; exit 0 ;;
esac

PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"

lint_out=$("$PY" scripts/rules_lint.py 2>&1)
lint_rc=$?

if [ $lint_rc -eq 0 ]; then
  printf '{}\n'
  exit 0
fi

printf '%s' "$lint_out" | tail -n 40 | /usr/bin/jq -cRs \
  '{additional_context: ("rules_lint found violations after your last edit (RULES.md is enforced; fix before proceeding):\n" + .)}'
exit 0
