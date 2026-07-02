#!/bin/bash
# preToolUse guard for RULES.md R13 (frozen governance) and R3 (generated results).
#
# Denies agent edits/deletes that target GOAL.md, RULES.md, the hooks themselves,
# the rules linter, the quality gate, .githooks, or anything under results/.
# `permission: "ask"` is NOT enforced on preToolUse (it is schema-accepted but
# ignored), so deny + an instructive agent_message is the only real guard here.
# On legitimate amendments the *user* edits the file directly (hooks do not
# intercept the human editor) or temporarily lifts this guard.
#
# Stdin payload: {"tool_name": ..., "tool_input": {...}} (fallbacks kept for
# schema drift). DAH_BOOTSTRAP=1 downgrades deny to allow so the governance
# test-suite can exercise both modes explicitly.
set -u

input=$(cat)

tool=$(printf '%s' "$input" | /usr/bin/jq -r '.tool_name // .tool // .toolName // empty' 2>/dev/null || true)

# Only file-mutating tools are of interest; everything else passes through.
case "$(printf '%s' "$tool" | tr '[:upper:]' '[:lower:]')" in
  *write*|*edit*|*replace*|*delete*|*patch*|*notebook*|*multiedit*) : ;;
  *) printf '{ "permission": "allow" }\n'; exit 0 ;;
esac

path=$(printf '%s' "$input" | /usr/bin/jq -r '
  (.tool_input // .input // .params // .arguments // {}) as $i
  | ($i.path // $i.file_path // $i.filePath // $i.target_file // $i.targetFile
     // $i.target_notebook // .path // .file_path // empty)' 2>/dev/null || true)

if [ -z "$path" ] || [ "$path" = "null" ]; then
  printf '{ "permission": "allow" }\n'
  exit 0
fi

deny() {
  local user_msg="$1" agent_msg="$2"
  if [ "${DAH_BOOTSTRAP:-0}" = "1" ]; then
    printf '{ "permission": "allow" }\n'
    exit 0
  fi
  /usr/bin/jq -cn --arg u "$user_msg" --arg a "$agent_msg" \
    '{permission: "deny", user_message: $u, agent_message: $a}'
  exit 0
}

case "$path" in
  *GOAL.md|*RULES.md)
    deny "Blocked: agent edit to $(basename "$path") — a frozen governance file (RULES.md R13)." \
         "GOAL.md and RULES.md are frozen (R13). Do not edit them directly. Propose the amendment to the user with a rationale; the user applies it (or lifts the guard for one change) and it gets recorded in docs/JOURNAL.md per the amendment protocol."
    ;;
  *.cursor/hooks*|*.githooks*)
    deny "Blocked: agent edit to guardrail infrastructure ($path) (RULES.md R13)." \
         "Hook/guardrail files are protected (R13): an agent must not alter its own enforcement. Ask the user to make or approve this change."
    ;;
  *scripts/rules_lint.py|*scripts/quality_gate.py)
    deny "Blocked: agent edit to $(basename "$path") — rules linter / quality gate (RULES.md R13)." \
         "The linter and gate are protected enforcement machinery (R13). Propose the change to the user instead of editing directly."
    ;;
  *results/*)
    deny "Blocked: hand-edit of '$path' — results/ artifacts are generated-only (RULES.md R3)." \
         "Never hand-edit results/. Regenerate artifacts with 'make reproduce' (or the specific dah-* CLI). If the file is stale or wrong, fix the producing code and rerun."
    ;;
esac

printf '{ "permission": "allow" }\n'
exit 0
