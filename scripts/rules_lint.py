#!/usr/bin/env python3
"""Machine-checkable enforcement of RULES.md.

Each check is named after the rule it enforces (R1, R2, R5, R8, R9, R10, R11,
R12, plus the lintable slice of R3). The linter is deliberately dependency-free
(stdlib only) so it runs anywhere - hooks, pre-commit, CI - without the venv.

Exit codes: 0 = clean, 1 = violations found, 2 = linter itself broke.
"""

from __future__ import annotations

import ast
import io
import re
import sys
import tokenize
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src" / "dah"
TESTS = REPO / "tests"
FEATURES = REPO / "FEATURES.md"
TASKS = REPO / "TASKS.md"
RESULTS_DIR = REPO / "results"

# R2/R8/R1 import bans. `openai` is allowed only in the sanctioned egress module.
NETWORK_MODULES = {"requests", "httpx", "aiohttp", "urllib", "socket", "http", "openai"}
NETWORK_EXEMPT = "providers/openai_compat.py"
SERIALIZER_MODULES = {"pickle", "cPickle", "dill", "shelve", "marshal"}
RANDOM_MODULES = {"random"}

# R5: answer-key tokens banned from decision-making code.
LEAKAGE_TOKENS = ("required_fact_ids", "confident_wrong")
LEAKAGE_PATHS = ("memory/", "planner.py", "verifier.py", "recovery.py")

# R12: FEATURES.md status enum.
STATUSES = {"planned", "in-progress", "implemented", "verified"}

GOVERNANCE_FILES = (
    "GOAL.md",
    "RULES.md",
    "FEATURES.md",
    "TASKS.md",
    "AGENTS.md",
    "README.md",
    "docs/JOURNAL.md",
)

RESULTS_SECTIONS = (
    "Headline",
    "Ablation matrix",
    "Failure recovery",
    "What didn't work",
    "Reproduce",
)


class Findings:
    """Accumulates violations grouped by rule id."""

    def __init__(self) -> None:
        self.items: list[tuple[str, str]] = []

    def add(self, rule: str, msg: str) -> None:
        self.items.append((rule, msg))

    def report(self) -> int:
        if not self.items:
            print("rules_lint: OK (all checks passed)")
            return 0
        by_rule: dict[str, list[str]] = {}
        for rule, msg in self.items:
            by_rule.setdefault(rule, []).append(msg)
        for rule in sorted(by_rule):
            print(f"\n[{rule}] {len(by_rule[rule])} violation(s):")
            for msg in by_rule[rule]:
                print(f"  - {msg}")
        print(f"\nrules_lint: FAIL ({len(self.items)} violation(s))")
        return 1


def _src_files() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


def _rel(p: Path) -> str:
    return str(p.relative_to(REPO))


def _code_only(text: str) -> str:
    """Return source with comments and string literals blanked out.

    Used by regex checks so that docstrings/comments mentioning a forbidden
    pattern do not trip the linter. Falls back to raw text on tokenize errors
    (fail toward more checking, not less).
    """
    try:
        out: list[str] = []
        prev_end = (1, 0)
        lines = text.splitlines(keepends=True)
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            (srow, scol), (erow, ecol) = tok.start, tok.end
            # gap since previous token: preserve as-is (whitespace/newlines)
            if (srow, scol) > prev_end:
                if srow == prev_end[0]:
                    out.append(lines[srow - 1][prev_end[1] : scol])
                else:
                    out.append(lines[prev_end[0] - 1][prev_end[1] :])
                    out.extend(lines[r] for r in range(prev_end[0], srow - 1))
                    out.append(lines[srow - 1][:scol])
            out.append(tok.string)
            prev_end = (erow, ecol)
        return "".join(out)
    except (tokenize.TokenError, IndentationError, IndexError):
        # IndexError: ENDMARKER/NEWLINE tokens can point one row past the last
        # physical line (e.g. files without a trailing newline). Fall back to
        # raw text - fail toward more checking, not less.
        return text


def _top_module(name: str) -> str:
    return name.split(".")[0]


def check_imports_and_calls(f: Findings) -> None:
    """R1 (random module), R2 (network), R8 (pickle/eval/exec), R9 (print)."""
    for path in _src_files():
        rel = _rel(path)
        text = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text)
        except SyntaxError as e:
            f.add("R10", f"{rel}: does not parse: {e}")
            continue

        is_network_exempt = rel.endswith(NETWORK_EXEMPT)
        is_experiments = "/experiments/" in rel

        for node in ast.walk(tree):
            mods: list[str] = []
            if isinstance(node, ast.Import):
                mods = [_top_module(a.name) for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                mods = [_top_module(node.module)]
            for mod in mods:
                if mod in RANDOM_MODULES:
                    f.add("R1", f"{rel}:{node.lineno}: import of '{mod}' (use seeded np.random.default_rng)")
                if mod in NETWORK_MODULES and not is_network_exempt:
                    f.add("R2", f"{rel}:{node.lineno}: import of '{mod}' (network I/O banned in library)")
                if mod in SERIALIZER_MODULES:
                    f.add("R8", f"{rel}:{node.lineno}: import of '{mod}' (no pickle-style serialization)")

            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in ("eval", "exec"):
                    f.add("R8", f"{rel}:{node.lineno}: call to {node.func.id}()")
                if node.func.id == "print" and not is_experiments:
                    f.add("R9", f"{rel}:{node.lineno}: print() in library code (use Trace.emit)")


DETERMINISM_PATTERNS = (
    (re.compile(r"\bnp\.random\.seed\s*\("), "np.random.seed() mutates global RNG state"),
    (re.compile(r"\bnumpy\.random\.seed\s*\("), "numpy.random.seed() mutates global RNG state"),
    (re.compile(r"\bdefault_rng\s*\(\s*\)"), "default_rng() without an explicit seed"),
    (re.compile(r"\btime\.time\s*\(\s*\)"), "time.time() as an input is nondeterministic (perf_counter for telemetry only)"),
    (re.compile(r"\bdatetime\.(?:now|utcnow)\s*\("), "datetime.now()/utcnow() is nondeterministic"),
)


def check_determinism(f: Findings) -> None:
    """R1: nondeterminism patterns, scanned over code with strings/comments removed."""
    for path in _src_files():
        rel = _rel(path)
        code = _code_only(path.read_text(encoding="utf-8"))
        for lineno, line in enumerate(code.splitlines(), start=1):
            for pat, why in DETERMINISM_PATTERNS:
                if pat.search(line):
                    f.add("R1", f"{rel}:{lineno}: {why}")


def check_leakage(f: Findings) -> None:
    """R5: answer-key tokens must not appear in decision-making code (raw text)."""
    for path in _src_files():
        rel = _rel(path)
        sub = rel.removeprefix("src/dah/")
        if not any(sub.startswith(p) or sub == p for p in LEAKAGE_PATHS):
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for token in LEAKAGE_TOKENS:
                if token in line:
                    f.add("R5", f"{rel}:{lineno}: forbidden token '{token}' (answer-key leakage)")


def check_api_hygiene(f: Findings) -> None:
    """R10: module docstring, future annotations, public docstrings (non-__init__)."""
    for path in _src_files():
        if path.name == "__init__.py":
            continue
        rel = _rel(path)
        text = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue  # already reported by check_imports_and_calls
        if not ast.get_docstring(tree):
            f.add("R10", f"{rel}: missing module docstring")
        has_future = any(
            isinstance(n, ast.ImportFrom) and n.module == "__future__"
            and any(a.name == "annotations" for a in n.names)
            for n in tree.body
        )
        if not has_future:
            f.add("R10", f"{rel}: missing 'from __future__ import annotations'")
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                if not ast.get_docstring(node):
                    f.add("R10", f"{rel}:{node.lineno}: public class '{node.name}' missing docstring")
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
                if not ast.get_docstring(node):
                    f.add("R10", f"{rel}:{node.lineno}: public function '{node.name}' missing docstring")


def check_tests_reference_modules(f: Findings) -> None:
    """R11: every non-__init__ src module's dotted path appears somewhere in tests/."""
    test_text = ""
    if TESTS.is_dir():
        test_text = "\n".join(
            p.read_text(encoding="utf-8") for p in sorted(TESTS.rglob("*.py"))
        )
    for path in _src_files():
        if path.name == "__init__.py":
            continue
        dotted = ".".join(path.relative_to(SRC.parent).with_suffix("").parts)  # e.g. dah.memory.stores
        if dotted not in test_text:
            f.add("R11", f"{_rel(path)}: no test references '{dotted}' (tests land with code)")


_ROW = re.compile(r"^\|\s*F\d+\s*\|")


def check_doc_sync(f: Findings) -> None:
    """R12: governance files exist; FEATURES rows valid; TASKS ids unique; module coverage."""
    for name in GOVERNANCE_FILES:
        if not (REPO / name).exists():
            f.add("R12", f"missing governance file: {name}")

    if FEATURES.exists():
        feat_text = FEATURES.read_text(encoding="utf-8")
        # every existing src module is mentioned
        for path in _src_files():
            if path.name == "__init__.py":
                continue
            if _rel(path) not in feat_text:
                f.add("R12", f"{_rel(path)}: not listed in FEATURES.md inventory")
        # rows: valid status; verified rows carry live evidence
        for lineno, line in enumerate(feat_text.splitlines(), start=1):
            if not _ROW.match(line):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) < 5:
                f.add("R12", f"FEATURES.md:{lineno}: malformed feature row")
                continue
            fid, status, evidence = cells[0], cells[3], cells[4]
            if status not in STATUSES:
                f.add("R12", f"FEATURES.md:{lineno}: {fid} has invalid status '{status}'")
            if status == "verified" and not _evidence_ok(evidence):
                f.add("R12", f"FEATURES.md:{lineno}: {fid} is 'verified' but evidence '{evidence}' has no live path or command")

    if TASKS.exists():
        ids = re.findall(r"\*\*(T\d+\.\d+)\*\*", TASKS.read_text(encoding="utf-8"))
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        for d in dupes:
            f.add("R12", f"TASKS.md: duplicate task id {d}")

    results_md = REPO / "docs" / "RESULTS.md"
    if results_md.exists():
        rt = results_md.read_text(encoding="utf-8")
        for section in RESULTS_SECTIONS:
            if section.lower() not in rt.lower():
                f.add("R12", f"docs/RESULTS.md: missing required section '{section}'")


def _evidence_ok(evidence: str) -> bool:
    """Evidence must contain a backticked path that exists, or a runnable command."""
    for token in re.findall(r"`([^`]+)`", evidence):
        if (REPO / token).exists():
            return True
        if token.startswith(("python ", "pytest", "git ", "dah-", "run ")):
            return True
    return "tests/" in evidence and any(
        (REPO / t).exists() for t in re.findall(r"tests/[\w./-]+", evidence)
    )


def check_results_integrity(f: Findings) -> None:
    """R3 (lintable slice): results CSVs carry seed and config_id columns."""
    if not RESULTS_DIR.is_dir():
        return
    for csv in sorted(RESULTS_DIR.rglob("*.csv")):
        try:
            header = csv.read_text(encoding="utf-8").splitlines()[0]
        except (IndexError, UnicodeDecodeError):
            f.add("R3", f"{_rel(csv)}: empty or unreadable results CSV")
            continue
        cols = [c.strip() for c in header.split(",")]
        for required in ("seed", "config_id"):
            if required not in cols:
                f.add("R3", f"{_rel(csv)}: missing required column '{required}'")


def main() -> int:
    """Run every check and report grouped findings."""
    if not SRC.is_dir():
        print(f"rules_lint: source tree not found at {SRC}", file=sys.stderr)
        return 2
    f = Findings()
    check_imports_and_calls(f)
    check_determinism(f)
    check_leakage(f)
    check_api_hygiene(f)
    check_tests_reference_modules(f)
    check_doc_sync(f)
    check_results_integrity(f)
    return f.report()


if __name__ == "__main__":
    sys.exit(main())
