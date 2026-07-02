#!/bin/bash
# beforeShellExecution guard (RULES.md R15/R13/R3).
#
# Denies: bypassing verification (git commit --no-verify), force pushes
# (without lease), catastrophic rm -rf targets.
# Asks:   git config writes (hook-path tampering), shell redirection into
#         results/, in-place edits of governance files via sed/perl.
# Allows: everything else, including the sanctioned
#         `git config core.hooksPath .githooks` setup command and config reads.
set -u

input=$(cat)
cmd=$(printf '%s' "$input" | /usr/bin/jq -r '.command // .tool_input.command // empty' 2>/dev/null || true)

if [ -z "$cmd" ]; then
  printf '{ "permission": "allow" }\n'
  exit 0
fi

deny() {
  /usr/bin/jq -cn --arg u "$1" --arg a "$2" '{permission: "deny", user_message: $u, agent_message: $a}'
  exit 0
}

ask() {
  /usr/bin/jq -cn --arg u "$1" --arg a "$2" '{permission: "ask", user_message: $u, agent_message: $a}'
  exit 0
}

# --- hard denies -----------------------------------------------------------
if printf '%s' "$cmd" | grep -Eq 'git[^|;&]*commit[^|;&]*--no-verify'; then
  deny "Blocked: 'git commit --no-verify' bypasses the pre-commit quality gate (RULES.md R15)." \
       "Committing with --no-verify is forbidden (R15). Fix the gate failures (python scripts/quality_gate.py --fast) and commit normally."
fi

if printf '%s' "$cmd" | grep -Eq 'git[^|;&]*push[^|;&]*(--force([^-]|$)|-f([[:space:]]|$))' \
   && ! printf '%s' "$cmd" | grep -q -- '--force-with-lease'; then
  deny "Blocked: force push (RULES.md safety rails)." \
       "Force pushes are blocked. If a rewrite is truly needed, ask the user and use --force-with-lease."
fi

if printf '%s' "$cmd" | grep -Eq 'rm[[:space:]]+(-[a-zA-Z]*r[a-zA-Z]*f|-[a-zA-Z]*f[a-zA-Z]*r)[[:space:]]+("?\$HOME|~|/([[:space:]]|$)|/[a-zA-Z]+|\.\.)'; then
  deny "Blocked: 'rm -rf' aimed at a home/root/parent path." \
       "This rm -rf targets a path outside the safe project scope. Narrow the target to an explicit path inside the repo."
fi

# --- asks ------------------------------------------------------------------
if printf '%s' "$cmd" | grep -Eq '(^|[|;&][[:space:]]*)git[[:space:]]+config' ; then
  if ! printf '%s' "$cmd" | grep -Eq 'git[[:space:]]+config[[:space:]]+(--get|--list|-l)([[:space:]]|$)' \
     && ! printf '%s' "$cmd" | grep -Eq 'git[[:space:]]+config[[:space:]]+core\.hooksPath([[:space:]]+\.githooks)?[[:space:]]*$'; then
    ask "Agent wants to change git config: $cmd" \
        "git config writes are gated (they can disable the pre-commit gate, R15). Awaiting user approval."
  fi
fi

if printf '%s' "$cmd" | grep -Eq '(>>?|[[:space:]]tee([[:space:]]+-a)?)[[:space:]]*["'"'"']?(\./)?results/'; then
  ask "Agent wants to write into results/ via the shell: $cmd — results are generated-only (RULES.md R3)." \
      "Writing into results/ by shell redirection violates R3. Regenerate artifacts with dah-baseline / dah-ablation / dah-analyze instead."
fi

if printf '%s' "$cmd" | grep -Eq '(sed[[:space:]][^|;&]*-i|perl[[:space:]][^|;&]*-i)[^|;&]*(GOAL\.md|RULES\.md|\.cursor/hooks|\.githooks|rules_lint\.py|quality_gate\.py)'; then
  ask "Agent wants to edit a protected governance file in-place via the shell: $cmd (RULES.md R13)." \
      "In-place shell edits of frozen governance files are gated (R13). Awaiting user approval."
fi

printf '{ "permission": "allow" }\n'
exit 0
