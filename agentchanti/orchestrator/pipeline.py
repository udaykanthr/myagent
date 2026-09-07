"""
Pipeline execution — wave-based parallel/sequential step execution.
"""

import logging
import re

from concurrent.futures import ThreadPoolExecutor, as_completed

from ..cli_display import CLIDisplay, log

from .memory import FileMemory
from .classification import _classify_step, _TEST_CMD_RE, _TEST_CONFIG_RE
from .plan_step import PlanStep, build_step_context, update_step_after_execution
from .step_handlers import (
    _handle_cmd_step, _handle_code_step, _handle_test_step,
    _handle_search_step,
    _build_scoped_test_cmd,
    _detect_subproject_root,
    _prefix_subproject_paths,
    MAX_STEP_RETRIES,
)
from .diagnosis import _diagnose_failure, _apply_fix
from .error_router import classify_error

_logger = logging.getLogger(__name__)


# Outer retries: diagnose failure → fix → re-run step.
#
# Raised from 2 once attempts stopped discarding each other's work (see the
# progress check in _run_diagnosis_loop). While every attempt began by
# restoring the pre-diagnosis snapshot, a third was near-worthless — each
# one could only ever land a single fix. Now that a fix which moves the
# error on is KEPT, attempts compound, and a gate built from a chain of
# asserts needs about one attempt per condition it checks.
#
# Attempts that change nothing still revert and still burn budget, so this
# stays small: the cost of a wrong step is bounded by this number times one
# diagnosis round trip.
MAX_DIAGNOSIS_RETRIES = 3


_RESOLVE_SKIP_DIRS = frozenset({
    "node_modules", "venv", ".venv", "__pycache__", ".git",
    ".agentchanti", "dist", "build",
})


def _resolve_existing_by_basename(declared: str, memory) -> str | None:
    """Find the real on-disk file a plan-declared path refers to.

    Planners routinely declare paths for the wrong scaffold layout (e.g.
    ``project/project/urls.py`` when ``django-admin startproject X .``
    actually created ``project/urls.py``). When the declared path does
    not exist, look for existing files with the same basename — in
    session memory first, then on disk. Returns the single unambiguous
    match, or None (missing or ambiguous — caller must not guess).
    """
    import os

    basename = os.path.basename(declared.replace("\\", "/"))
    candidates: set[str] = set()

    # Memory and disk must be considered TOGETHER: session memory only
    # knows files this run wrote, so a memory-first shortcut retargets to
    # the session file even when a scaffold-created file with the same
    # basename (e.g. the real root urls.py) also exists on disk — exactly
    # the ambiguity that must refuse instead of guess.
    for mem_path in memory.all_files():
        norm = mem_path.replace("\\", "/")
        if norm.rsplit("/", 1)[-1] == basename and os.path.isfile(mem_path):
            candidates.add(norm)

    for dirpath, dirnames, filenames in os.walk("."):
        dirnames[:] = [d for d in dirnames
                       if d not in _RESOLVE_SKIP_DIRS
                       and not d.startswith(".")]
        if basename in filenames:
            rel = os.path.relpath(os.path.join(dirpath, basename))
            candidates.add(rel.replace("\\", "/").removeprefix("./"))

    if len(candidates) == 1:
        return candidates.pop()
    return None


def _try_trivial_close(
    partial: dict[str, str],
    language: str | None,
) -> dict[str, str] | None:
    """Attempt to close trivially truncated inline code without LLM.

    If each file in *partial* has ≤2 unmatched opening braces and ≤2
    unmatched opening parens, append the missing closing tokens.
    Returns the closed dict on success, or None if any file is too
    complex to close deterministically.
    """
    result: dict[str, str] = {}
    for path, content in partial.items():
        open_braces = content.count('{') - content.count('}')
        open_parens = content.count('(') - content.count(')')
        # Only attempt closure when the gap is tiny (likely a cut-off tail)
        if open_braces < 0 or open_parens < 0 or open_braces > 2 or open_parens > 2:
            return None
        tail = ('}\n' * open_braces) + (')\n' * open_parens)
        result[path] = content + ('\n' + tail if tail else '')
    return result


# ── Test file detection ───────────────────────────────────────
# Patterns that indicate a file is a test file (used for CODE→TEST
# auto-correction when the planner marks a test-editing step as CODE).
_TEST_FILE_RE = re.compile(
    r'(?:'
    # JS/TS: *.test.js, *.spec.tsx, etc.
    r'\.(?:test|spec)\.(?:js|jsx|ts|tsx|mjs|cjs)$'
    # Python: test_*.py or *_test.py, plus Django's conventional tests.py
    r'|(?:^|[/\\])test_\w+\.py$'
    r'|\w+_test\.py$'
    r'|(?:^|[/\\])tests\.py$'
    # Go: *_test.go
    r'|\w+_test\.go$'
    # Ruby: *_spec.rb
    r'|\w+_spec\.rb$'
    r')',
    re.IGNORECASE,
)
# Directories that indicate test files
_TEST_DIR_RE = re.compile(
    r'(?:^|[/\\])(?:__tests__|tests?|specs?|test_\w+)[/\\]',
    re.IGNORECASE,
)


_SOURCE_EXTS = frozenset({
    ".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
    ".go", ".rb", ".java", ".rs", ".cs", ".cpp", ".c", ".php",
})

# ── Wiring verification: fix scope parsing ────────────────────────────────
# Matches a "Fix scope:" section in the task description.
_FIX_SCOPE_SECTION_RE = re.compile(
    r'fix\s+scope\s*:?\s*(.+?)(?=\nDo not touch:|\nConstraints:|\nInterface|\nKB topics:|\Z)',
    re.IGNORECASE | re.DOTALL,
)
# Matches backtick-quoted strings that look like file paths.
_BACKTICK_PATH_RE = re.compile(r'`([^`]+\.[a-zA-Z]{1,6}[^`]*)`')
# Matches bare file names with common source/config extensions.
_BARE_FILENAME_RE = re.compile(
    r'\b([\w.\-/]+\.(?:jsx|tsx|js|ts|mjs|cjs|py|css|html|json|go|rb|rs|java|cs|vue|svelte))\b'
)

def _has_code_ext(file_path: str) -> bool:
    """True if *file_path* has a source-code extension (see _SOURCE_EXTS)."""
    name = file_path.replace("\\", "/").rsplit("/", 1)[-1]
    return ("." in name
            and "." + name.rsplit(".", 1)[-1].lower() in _SOURCE_EXTS)


def _is_test_file(file_path: str) -> bool:
    """Return True if *file_path* looks like a test file."""
    import os
    basename = os.path.basename(file_path)
    _, ext = os.path.splitext(basename)
    # Never treat non-source files (HTML, CSS, JSON, …) as test files even
    # when they live inside a __tests__ directory.
    if ext.lower() not in _SOURCE_EXTS:
        return False
    return bool(_TEST_FILE_RE.search(basename) or _TEST_DIR_RE.search(file_path))


def _syntax_gate(path: str, content: str) -> str | None:
    """Cheap syntactic validity check before content reaches disk.

    Returns an error string when *content* is not valid for *path*'s file
    type, else None. Python → ast.parse, JSON → json.loads (a minimal-diff
    patch once wrote a trailing comma into package.json — valid-looking
    text that broke every subsequent npm/vitest invocation), other code
    files → the offline tree-sitter lint.
    """
    import os
    ext = os.path.splitext(path)[1].lower()
    if ext in (".py", ".pyw"):
        from ..py_syntax import check_python_syntax
        err = check_python_syntax(content, path)
        return f"python syntax error: {err}" if err else None
    if ext == ".json":
        # tsconfig*.json is JSONC (comments/trailing commas allowed) —
        # strict parsing would reject legitimate content.
        if os.path.basename(path).lower().startswith("tsconfig"):
            return None
        import json
        try:
            json.loads(content)
        except ValueError as exc:
            return f"invalid JSON: {exc}"
        return None
    from .step_handlers import _quick_offline_lint
    errs = _quick_offline_lint({path: content})
    return errs or None


_TEST_INFRA_STEMS = (
    "vitest.config", "vitest.setup", "jest.config", "jest.setup",
    "setuptests", "conftest",
)


def _is_test_infra_file(file_path: str) -> bool:
    """True for test *infrastructure* files (vitest/jest config + setup).

    The BulkTest source-protection guard exists to stop the fix loop from
    rewriting production code, but these files serve the tests themselves
    — and the guard's existing-file requirement made it impossible to
    CREATE a missing vitest.config.js at all (observed: the fix loop was
    blocked on it three times in one run while the suite failed around
    it). Treat them like test files: the fix loop may create or modify
    them freely.
    """
    import os
    stem = os.path.basename(file_path).lower()
    return stem.startswith(_TEST_INFRA_STEMS)


# ── Router-mount mismatch detection ───────────────────────────────────────
# Production source files that use react-router primitives (Link,
# NavLink, useNavigate, …) require a Router (BrowserRouter, HashRouter,
# RouterProvider, …) mounted somewhere in the tree.  When the source
# uses primitives but no source file mounts a Router, the app crashes
# at runtime with:
#
#   TypeError: Cannot destructure property 'basename' of
#              'React.useContext(...)' as it is null.
#
# Tests can pass anyway because the BulkTest fix loop's test-only
# retry frequently wraps the failing render in <MemoryRouter> as a
# workaround — green tests then mask a broken production binary.
# This deterministic check is the canary that re-enables wiring
# verification when the workaround pattern is detected.

_ROUTER_PRIMITIVE_NAMES = (
    "Link", "NavLink", "Outlet", "Routes", "Route",
    "useNavigate", "useLocation", "useParams", "useMatch",
    "useRoutes", "useSearchParams", "useNavigationType",
)
_ROUTER_MOUNT_NAMES = (
    "BrowserRouter", "HashRouter", "MemoryRouter",
    "Router", "RouterProvider", "StaticRouter",
)
_ROUTER_PRIMITIVE_RE = re.compile(
    r"\b(?:" + "|".join(_ROUTER_PRIMITIVE_NAMES) + r")\b"
)
_ROUTER_MOUNT_JSX_RE = re.compile(
    r"<\s*(?:" + "|".join(_ROUTER_MOUNT_NAMES) + r")\b"
)
_REACT_ROUTER_IMPORT_RE = re.compile(
    r"""(?:from|require\s*\()\s*['"]react-router(?:-dom)?['"]"""
)
# Filenames that are typical app entry points; flagged so a downstream
# fixer can prefer mounting <BrowserRouter> here over editing leaf
# components.
_ENTRY_POINT_BASENAMES = frozenset({
    "main.jsx", "main.tsx", "main.js", "main.ts",
    "index.jsx", "index.tsx", "index.js", "index.ts",
    "App.jsx", "App.tsx",
})


def _detect_router_mount_missing(memory: "FileMemory") -> dict | None:
    """Return a description of the mismatch when production source
    uses react-router primitives but no source file mounts a Router,
    otherwise ``None``.

    The detector is **source-side only** — test files are intentionally
    ignored because the failure mode is "tests use ``<MemoryRouter>``
    to mask a missing production mount."  Including tests in the scan
    would let that workaround pretend the app is wired.

    Detection rules (all must hold):
      1. At least one production (non-test) source file imports
         from ``react-router`` or ``react-router-dom``.
      2. That same import surface is actually used (a router primitive
         like ``Link`` / ``useNavigate`` appears in source).
      3. No production source file mounts a Router via JSX
         (``<BrowserRouter>``, ``<RouterProvider>``, etc.).

    Returns a dict with the data the LLM-based wiring verification
    needs to fix the issue, or ``None``.
    """
    import os as _os_rm

    files_using_primitives: list[str] = []
    entry_candidates: list[str] = []
    any_source_imports_router = False
    any_source_mounts_router = False

    for fp, content in memory.all_files().items():
        if not content:
            continue
        if _is_test_file(fp):
            continue

        # Entry-point candidates are tracked unconditionally — main.jsx
        # typically does NOT import react-router (that's the bug we're
        # detecting), so it would be missed if we only looked inside
        # the react-router import block.
        base = _os_rm.path.basename(fp.replace("\\", "/"))
        if base in _ENTRY_POINT_BASENAMES:
            entry_candidates.append(fp)

        if not _REACT_ROUTER_IMPORT_RE.search(content):
            continue
        any_source_imports_router = True
        if _ROUTER_PRIMITIVE_RE.search(content):
            files_using_primitives.append(fp)
        if _ROUTER_MOUNT_JSX_RE.search(content):
            any_source_mounts_router = True

    if not any_source_imports_router:
        return None
    if not files_using_primitives:
        return None
    if any_source_mounts_router:
        return None

    return {
        "kind": "router_mount_missing",
        "files_using_primitives": sorted(files_using_primitives),
        "entry_candidates": sorted(entry_candidates),
    }


def should_run_wiring_verification(
    memory: "FileMemory",
    *,
    pipeline_success: bool,
    bulk_test_verif_ok: bool,
    wiring_enabled: bool,
) -> bool:
    """Decide whether the post-pipeline wiring verification should run.

    Wiring verification is an expensive LLM call (60–90s with cloud models)
    that checks all fix-scope files for cross-file integration issues like
    missing entry-point mounts, broken imports, default-vs-named export
    mismatches, and wrong prop shapes.

    It is REDUNDANT after a successful bulk test run because every failure
    mode it looks for would have crashed the test runner.  A green bulk
    test run is therefore implicit proof of correct wiring.

    EXCEPTION: when :func:`_detect_router_mount_missing` finds that
    production source uses react-router primitives without mounting a
    Router, bulk-test green is **not** a wiring proof — the BulkTest
    fix loop's test-only retry routinely wraps tests in
    ``<MemoryRouter>`` as a workaround, masking the missing mount.
    Force verification to run in that case so the LLM-based fixer can
    either remove the primitives from source or mount
    ``<BrowserRouter>`` in the entry point.

    Returns True when ALL of:
      • The pipeline as a whole succeeded.
      • Wiring verification is enabled in config.
    AND any of:
      • No test files exist (bulk test was skipped — so wiring is the
        only integration check we have).
      • The bulk test run did not succeed (so we cannot trust it as a
        wiring proof).
      • A router-mount mismatch was detected.
    """
    if not (pipeline_success and wiring_enabled):
        return False

    if _detect_router_mount_missing(memory) is not None:
        return True

    bulk_tests_existed = any(
        _is_test_file(f) and not f.startswith("_")
        for f in memory.all_files()
    )
    bulk_tests_passed = bulk_tests_existed and bulk_test_verif_ok
    return not bulk_tests_passed


def _diff_stats(orig: str, new_content: str) -> dict:
    """Return per-opcode line-level diff stats between two file contents.

    Shared by :func:`_is_additive_source_fix` (strict 10% additive check)
    and :func:`_attempt_targeted_source_fix` (relaxed 30% cap for the
    escape hatch) so both use identical diff arithmetic.

    Returned keys:
        ``added``     — line count for insert-only opcodes
        ``removed``   — line count for delete-only opcodes
        ``changed``   — max side of replace opcodes
        ``total_delta`` — ``added + changed``
        ``ratio``     — ``total_delta / max(orig_len, 1)``
    """
    import difflib
    orig_lines = orig.splitlines()
    new_lines = new_content.splitlines()
    orig_len = max(len(orig_lines), 1)
    sm = difflib.SequenceMatcher(None, orig_lines, new_lines, autojunk=False)
    added = 0
    removed = 0
    changed = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'insert':
            added += j2 - j1
        elif tag == 'delete':
            removed += i2 - i1
        elif tag == 'replace':
            changed += max(i2 - i1, j2 - j1)
            removed += max(0, (i2 - i1) - (j2 - j1))
    total_delta = added + changed
    return {
        'added': added,
        'removed': removed,
        'changed': changed,
        'total_delta': total_delta,
        'ratio': total_delta / orig_len,
    }


def _is_additive_source_fix(file_path: str, new_content: str,
                            memory) -> bool:
    """Return True if the proposed source change is small and additive.

    Allows minor test-supportive tweaks (adding data-testid, aria-label,
    role attributes) without altering component functionality.

    Criteria — ALL must be met:
      1. Original file exists in memory (not a new file)
      2. No lines were removed (only additions or in-place attribute additions)
      3. Total change is ≤10% of original line count
      4. Changed lines only add attributes / props, not logic
    """
    orig = memory.get(file_path)
    if orig is None:
        return False

    orig_lines = orig.splitlines()
    orig_len = len(orig_lines)
    if orig_len < 3:
        return False

    stats = _diff_stats(orig, new_content)

    # No lines may be removed — only additions or in-place edits
    if stats['removed'] > 0:
        return False

    # Total change must be ≤10% of original
    if stats['total_delta'] > max(orig_len * 0.10, 3):
        return False

    return True


def _is_safe_source_fix(file_path: str, new_content: str, memory) -> bool:
    """Relaxed gate for source fixes that fail the strict additive check.

    Full-file responses from smaller models rarely reproduce untouched
    lines byte-for-byte, so a one-line intended fix can arrive as a
    "large" diff and :func:`_is_additive_source_fix` vetoes it (observed:
    a correct one-line ``{% load static %}`` fix was blocked on five
    consecutive rounds while the run failed around it). Judge the change
    itself, not the response format: accept when the file already
    exists, every top-level export survives, and the line-level diff
    stays under the escape hatch's relaxed cap.
    """
    orig = memory.get(file_path)
    if orig is None:
        try:
            with open(file_path, 'r', encoding='utf-8',
                      errors='replace') as f:
                orig = f.read()
        except OSError:
            return False
    if not orig or not new_content.strip():
        return False

    dropped = (_extract_top_level_exports(orig)
               - _extract_top_level_exports(new_content))
    if dropped:
        _logger.warning(
            "[BulkTest] Source fix for %s drops top-level exports %s "
            "— rejected", file_path, sorted(dropped))
        return False

    stats = _diff_stats(orig, new_content)
    if stats['ratio'] > _ESCAPE_HATCH_DIFF_RATIO:
        return False
    _logger.info(
        "[BulkTest] Source fix for %s within hatch cap "
        "(added=%d removed=%d changed=%d ratio=%.0f%%)",
        file_path, stats['added'], stats['removed'], stats['changed'],
        stats['ratio'] * 100)
    return True


# ── Escape hatch: targeted source-file fix helpers ────────────────────────
# When test-only retries can't resolve a BulkTest failure (same error
# signature across consecutive attempts), the escape hatch allows ONE
# narrowly-scoped source-file fix.  Safety is enforced by:
#   1. Stack-trace scope  (cannot touch files the error doesn't reference)
#   2. Single-file scope  (multi-file fixes are rejected)
#   3. Export preservation (all top-level public names must survive)
#   4. Diff size cap      (≤30% of original lines changed)
#   5. Snapshot & revert  (caller reverts if the target test still fails)

# Normalisation regexes used to compute a stable error "shape" hash —
# stripping line/column numbers, timings, absolute paths and hex pointers
# so cosmetic churn across attempts doesn't hide a repeating error.
# How much of each end of a normalised error the signature hashes. Errors
# shorter than both windows together are hashed whole.
_SIG_WINDOW = 600

_ERROR_SIG_STRIP_RE = re.compile(
    r'(?:'
    r':\d+(?::\d+)?'              # :line or :line:col
    r'|\b\d+\s*ms\b'              # test timings
    r'|[A-Z]:\\[\w\-./\\]+'       # Windows absolute paths
    r'|(?<=[\s\'"(])/[\w\-./]+'   # POSIX absolute paths
    r'|0x[0-9a-fA-F]+'            # hex addresses
    r')'
)


def _error_signature(err: str) -> str:
    """Compute a stable short hash of the *shape* of an error string.

    Used to detect "the fix loop is making no progress": if two
    consecutive attempts produce the same signature, the test-only
    rewrites are not addressing the root cause and the escape hatch
    may fire.
    """
    import hashlib
    if not err:
        return ""
    norm = _ERROR_SIG_STRIP_RE.sub('', err)
    norm = re.sub(r'\s+', ' ', norm).strip()
    # Head AND tail, because a test runner front-loads whatever is
    # invariant and leaves the discriminating part until last. A TEST
    # step's error_info opens with a constant summary line, then the
    # verbose listing of test names in alphabetical order; what actually
    # differs between two attempts — which assertion blew up, and the
    # `FAILED (failures=F, errors=E)` tally — is at the very end.
    #
    # Hashing only `norm[:600]` therefore collided on genuinely different
    # failures. Measured live: two consecutive attempts of one step
    # produced error_info of 1692 and 1038 characters — provably not the
    # same string — and both hashed to 1a3d09c05029, so the loop reported
    # "previous fix changed nothing" twice about fixes that had changed
    # the error. For a CODE step the signature is the ONLY signal the
    # diagnosis loop has, since a bare traceback carries no test counts
    # for `_diagnosis_score` to read, and an earlier run reverted a
    # correct fix on exactly this comparison.
    if len(norm) <= 2 * _SIG_WINDOW:
        material = norm
    else:
        material = f"{norm[:_SIG_WINDOW]}…{norm[-_SIG_WINDOW:]}"
    return hashlib.sha1(
        material.encode('utf-8', errors='replace')).hexdigest()[:12]


def _diagnosis_score(err: str) -> int | None:
    """How bad is this failure? Lower is better; ``None`` means unknown.

    A signature only says whether two failures are *different*, never which
    one is worse — and both halves of that blindness have shipped a broken
    artifact. Inequality kept two fixes that took a suite from 4 failures to
    39 errors; equality discarded a fix that removed nine errors. Counting
    the failing tests gives the comparison a direction.

    Returns ``None`` for anything that is not recognisable test-runner
    output (a bare traceback from a CODE step's gate, say), because an
    unknown score must never be treated as an improvement.
    """
    if not err:
        return None
    from .step_handlers import _parse_test_counts
    try:
        passed, total, _ = _parse_test_counts(err)
    except Exception:
        return None
    # (0, 1) is the parser's "something failed, no counts available"
    # fallback — real information would name a total above one.
    if total <= 1:
        return None
    return max(0, total - passed)


# Stack-trace patterns.  Catch both parenthesised JS frames
# (`at Fn (src/Foo.jsx:12:3)`), bare JS frames (`src/Foo.jsx:12`),
# and Python `File "src/foo.py"` frames.
_STACK_JS_RE = re.compile(
    r'(?:at\s[^()]*\(|[\s(])'
    r'((?:[A-Za-z]:[\\/])?[\w\-./\\]+\.(?:jsx?|tsx?|mjs|cjs|vue|svelte))'
    r':\d+(?::\d+)?'
)
_STACK_JS_BARE_RE = re.compile(
    r'(?:^|[\s"\'(<>])'
    r'((?:[A-Za-z]:[\\/])?[\w\-./\\]+\.(?:jsx?|tsx?|mjs|cjs|vue|svelte))'
    r':\d+(?::\d+)?'
)
_STACK_PY_RE = re.compile(
    r'File\s+"((?:[A-Za-z]:[\\/])?[\w\-./\\]+\.py)"'
)


def _extract_stack_trace_files(err: str) -> set[str]:
    """Extract source file paths referenced in a test-runner error output.

    Normalises backslashes to forward slashes so comparisons with memory
    keys work on Windows.  Filters out library files (``node_modules``,
    ``site-packages``, ``dist/``) — the agent cannot fix those.
    """
    if not err:
        return set()
    paths: set[str] = set()
    for m in _STACK_JS_RE.finditer(err):
        paths.add(m.group(1).replace('\\', '/'))
    for m in _STACK_JS_BARE_RE.finditer(err):
        paths.add(m.group(1).replace('\\', '/'))
    for m in _STACK_PY_RE.finditer(err):
        paths.add(m.group(1).replace('\\', '/'))
    return {
        p for p in paths
        if 'node_modules' not in p
        and 'site-packages' not in p
        and '/dist/' not in p
        and not p.startswith('dist/')
    }


# Top-level export / definition patterns used to check that a targeted
# source fix preserves the file's public API.  Best-effort — recognises
# common JS/TS/CJS/Python patterns without a full parser.
_EXPORT_PATTERNS: list[re.Pattern] = [
    # "export default function Foo" / "export default class Foo"
    re.compile(
        r'^export\s+default\s+(?:async\s+)?(?:function|class)\s+(\w+)',
        re.MULTILINE),
    # "export function Foo" / "export class Foo"
    re.compile(
        r'^export\s+(?:async\s+)?(?:function|class)\s+(\w+)',
        re.MULTILINE),
    # "export const/let/var Foo"
    re.compile(r'^export\s+(?:const|let|var)\s+(\w+)', re.MULTILINE),
    # CJS "module.exports.Foo ="
    re.compile(r'^module\.exports\.(\w+)\s*=', re.MULTILINE),
    # Python top-level "def Foo(" / "async def Foo(" / "class Foo"
    re.compile(r'^(?:async\s+)?def\s+(\w+)\s*\(', re.MULTILINE),
    re.compile(r'^class\s+(\w+)\s*[:\(]', re.MULTILINE),
]
# "export { A, B as C }" — body captured then split
_EXPORT_NAMED_BLOCK_RE = re.compile(r'^export\s*\{([^}]+)\}', re.MULTILINE)
_EXPORT_NAMED_ITEM_RE = re.compile(r'(\w+)(?:\s+as\s+\w+)?')
# "export default Foo;" — identifier-only default export
_EXPORT_DEFAULT_IDENT_RE = re.compile(
    r'^export\s+default\s+(\w+)\s*;?\s*$', re.MULTILINE)


def _extract_top_level_exports(content: str) -> set[str]:
    """Return the set of top-level export / definition names in *content*.

    Used to verify that a targeted source fix doesn't silently drop the
    file's public API (which would break every importer).  Recognises
    common JS/TS/CJS/Python forms — intentionally best-effort, not a
    full AST walk.
    """
    names: set[str] = set()
    if not content:
        return names
    for pat in _EXPORT_PATTERNS:
        for m in pat.finditer(content):
            names.add(m.group(1))
    for m in _EXPORT_NAMED_BLOCK_RE.finditer(content):
        body = m.group(1)
        for nm in _EXPORT_NAMED_ITEM_RE.finditer(body):
            names.add(nm.group(1))
    for m in _EXPORT_DEFAULT_IDENT_RE.finditer(content):
        names.add(m.group(1))
    return names


# Relaxed diff cap for escape-hatch source fixes — loose enough to allow
# real bug fixes (remove an import + swap a JSX element) but strict
# enough to block "LLM rewrote the whole file" regressions.
_ESCAPE_HATCH_DIFF_RATIO = 0.30


def _should_trigger_escape_hatch(
    *,
    used_escape_hatch: bool,
    did_test_only_retry: bool,
    error_sig_history: list[str],
    route,
) -> bool:
    """Decide whether the BulkTest escape hatch may fire this attempt.

    Pure-function form so it can be unit-tested without standing up a
    whole loop, and called from both Loop 1 and Loop 2 to keep the
    trigger logic identical.

    All conditions MUST hold:

    1. The hatch hasn't already been used for this test file (one
       shot only — see commit message for the rationale).
    2. The test-only retry has been tried; this rules out attempt 1
       and forces the cheaper rewrite path to run first.
    3. We've recorded at least 2 error signatures and the most
       recent two are identical and non-empty — i.e. the test-only
       retries are demonstrably not converging on a fix.
    4. ErrorRouter classified the failure as ``source_type='code'``.
       Environment / data / network errors get a different remedy
       and the hatch is not designed to handle them.

    Note: there is intentionally no ``fix_attempt < MAX`` guard.
    Verification of the hatch fix happens via the immediate re-run
    inside the same loop iteration, and the snapshot revert handles
    failure on any attempt — including the last one.  An earlier
    version of this guard prevented the hatch from ever firing on
    the final attempt, which is exactly when test-only retries
    have most clearly failed.
    """
    if used_escape_hatch:
        return False
    if not did_test_only_retry:
        return False
    if len(error_sig_history) < 2:
        return False
    if not error_sig_history[-1]:
        return False
    if error_sig_history[-1] != error_sig_history[-2]:
        return False
    if route is None or getattr(route, 'source_type', None) != 'code':
        return False
    return True


def _attempt_targeted_source_fix(
    *,
    test_path: str,
    file_error: str,
    source_ctx: str,
    coder,
    executor,
    memory,
    subproject_cwd,
    lang_tag: str,
    task: str,
) -> dict[str, str] | None:
    """Last-resort source-file fix when test-only retries cannot help.

    Prompts the LLM to make the SMALLEST possible change to exactly ONE
    source file referenced in the test's error stack trace, then applies
    these post-LLM safety rails before returning:

      * Only files in the stack trace are accepted.
      * Single-file scope — multi-file responses are rejected.
      * Every top-level export in the original file must still be
        present in the new content.
      * Diff size ≤ ``_ESCAPE_HATCH_DIFF_RATIO`` (30%) of original lines.

    Returns a ``{path: content}`` dict ready to write, or ``None`` if
    the LLM response was empty, unparseable, or failed any safety
    check.  The caller is responsible for snapshotting the file(s)
    before write and restoring on regression.
    """
    # 1. Derive the in-scope source files from the stack trace.
    trace_files = _extract_stack_trace_files(file_error)
    _test_path_norm = test_path.replace('\\', '/')
    trace_files = {
        p for p in trace_files
        if not _is_test_file(p) and p != _test_path_norm
    }
    if not trace_files:
        _logger.info(
            "[BulkTest/Hatch] No source files referenced in stack trace "
            "for %s — cannot target a fix", test_path)
        return None

    _logger.info(
        "[BulkTest/Hatch] Candidate source files from stack: %s",
        sorted(trace_files))

    # 2. Build a tightly constrained prompt.
    trace_list = ", ".join(f"`{p}`" for p in sorted(trace_files))
    prompt = (
        f"Task: {task}\n\n"
        f"Test file `{test_path}` has failed repeatedly and the test "
        f"itself cannot be rewritten to work around the problem — the "
        f"ROOT CAUSE is in the SOURCE code.\n\n"
        f"Error output:\n{file_error}\n\n"
        f"Source files in scope:\n{source_ctx}\n\n"
        f"CRITICAL RULES:\n"
        f"1. Modify EXACTLY ONE source file — one of: {trace_list}\n"
        f"2. Make the SMALLEST possible change to fix the root cause.\n"
        f"3. DO NOT rewrite the file from scratch. DO NOT drop, rename, "
        f"or re-order existing top-level exports/functions/classes.\n"
        f"4. DO NOT modify the test file — the test encodes the "
        f"intended behaviour.\n"
        f"5. Output the COMPLETE updated source file using:\n"
        f"   #### [FILE]: <path>\n"
        f"   ```{lang_tag}\n   ...full file content...\n   ```\n"
    )

    try:
        response = coder.llm_client.generate_response(prompt)
    except Exception as exc:
        _logger.warning("[BulkTest/Hatch] LLM call failed: %s", exc)
        return None

    fix_files = executor.parse_code_blocks(response)
    if not fix_files:
        fix_files = executor.parse_code_blocks_fuzzy(response)
    if not fix_files:
        _logger.info(
            "[BulkTest/Hatch] LLM response had no parseable code blocks")
        return None

    if subproject_cwd:
        fix_files = _prefix_subproject_paths(
            fix_files, subproject_cwd, memory)

    # 3a. Single-file scope
    if len(fix_files) != 1:
        _logger.warning(
            "[BulkTest/Hatch] Rejected — LLM proposed %d files, "
            "expected 1: %s", len(fix_files), list(fix_files.keys()))
        return None

    fp, new_content = next(iter(fix_files.items()))
    fp_norm = fp.replace('\\', '/')

    # 3b. Must be a source file, not a test
    if _is_test_file(fp):
        _logger.warning(
            "[BulkTest/Hatch] Rejected — LLM returned a test file "
            "(%s); hatch is source-only", fp)
        return None

    # 3c. Must be in the stack-trace scope (suffix-match tolerates
    #     subproject-prefixed vs. relative paths).
    in_scope = any(
        fp_norm == tp
        or fp_norm.endswith('/' + tp)
        or tp.endswith('/' + fp_norm)
        for tp in trace_files
    )
    if not in_scope:
        _logger.warning(
            "[BulkTest/Hatch] Rejected — %s not in stack-trace scope "
            "(%s)", fp_norm, sorted(trace_files))
        return None

    # 3d. Load original content for diff + export checks.
    orig_content = memory.get(fp)
    if orig_content is None:
        try:
            with open(fp, 'r', encoding='utf-8', errors='replace') as _f:
                orig_content = _f.read()
        except OSError:
            orig_content = ""
    if not orig_content:
        _logger.warning(
            "[BulkTest/Hatch] Rejected — cannot read original %s for "
            "safety checks", fp)
        return None

    # 3e. Top-level exports must all be preserved.
    orig_exports = _extract_top_level_exports(orig_content)
    new_exports = _extract_top_level_exports(new_content)
    dropped = orig_exports - new_exports
    if dropped:
        _logger.warning(
            "[BulkTest/Hatch] Rejected — dropped top-level exports "
            "%s in %s", sorted(dropped), fp)
        return None

    # 3f. Diff size within relaxed cap.
    stats = _diff_stats(orig_content, new_content)
    if stats['ratio'] > _ESCAPE_HATCH_DIFF_RATIO:
        _logger.warning(
            "[BulkTest/Hatch] Rejected — diff ratio %.0f%% exceeds "
            "%.0f%% cap for %s (added=%d removed=%d changed=%d)",
            stats['ratio'] * 100,
            _ESCAPE_HATCH_DIFF_RATIO * 100, fp,
            stats['added'], stats['removed'], stats['changed'])
        return None

    _logger.info(
        "[BulkTest/Hatch] Validated source fix for %s "
        "(added=%d removed=%d changed=%d ratio=%.0f%%)",
        fp, stats['added'], stats['removed'], stats['changed'],
        stats['ratio'] * 100)
    return {fp: new_content}


def _find_tests_impacted_by_sources(
    modified_sources: list[str],
    all_test_files: dict[str, str],
    exclude: str,
    already_queued: set[str],
    kb_context_builder=None,
) -> list[str]:
    """Return test files (not already queued) that import any of the modified source files.

    Prefers the KB code graph (``CodeGraph.impact_analysis``) when available —
    it already tracks IMPORTS edges via tree-sitter parsing so no re-scanning is
    needed.  Falls back to a lightweight regex scan only when the graph is absent
    (e.g. KB not initialised for this project).
    """
    candidates: set[str] = set()

    graph = getattr(kb_context_builder, "_graph", None) if kb_context_builder else None

    if graph is not None:
        # Graph path: reverse-BFS over IMPORTS edges for each changed source.
        for src in modified_sources:
            for affected in graph.impact_analysis(src):
                if _is_test_file(affected):
                    candidates.add(affected)
    else:
        # Fallback: stem-match imports in test file content via regex.
        import re as _re
        _JS_IMPORT_RE = _re.compile(
            r'''(?:from\s+['"](.+?)['"]|require\s*\(\s*['"](.+?)['"]\s*\))''')
        _PY_IMPORT_RE = _re.compile(
            r'(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))')

        src_stems: set[str] = set()
        for src in modified_sources:
            stem = src.replace("\\", "/").rsplit('/', 1)[-1].rsplit('.', 1)[0].lower()
            src_stems.add(stem)

        for tpath, tcontent in all_test_files.items():
            imports: set[str] = set()
            for m in _JS_IMPORT_RE.finditer(tcontent):
                rel = m.group(1) or m.group(2)
                if rel:
                    imports.add(rel)
            for m in _PY_IMPORT_RE.finditer(tcontent):
                mod = m.group(1) or m.group(2)
                if mod:
                    imports.add(mod.replace('.', '/'))
            for imp in imports:
                imp_stem = imp.replace("\\", "/").rsplit('/', 1)[-1].rsplit('.', 1)[0].lower()
                if imp_stem in src_stems:
                    candidates.add(tpath)
                    break

    return [
        t for t in candidates
        if t != exclude and t not in already_queued
    ]


# ── External service dependency detection ─────────────────────
# Patterns that indicate the command failed because an external
# service (database, cache, message broker, etc.) is unavailable.
# These failures cannot be fixed by the agent — the user must
# ensure the service is running.

_EXTERNAL_SERVICE_PATTERNS: list[tuple[str, str]] = [
    # MongoDB
    (r'MongoServerSelectionError|MongoNetworkError|ECONNREFUSED.*27017',
     'MongoDB (default port 27017)'),
    # PostgreSQL
    (r'ECONNREFUSED.*5432|could not connect to server.*5432|pg_hba\.conf|'
     r'SequelizeConnectionRefusedError.*5432',
     'PostgreSQL (default port 5432)'),
    # MySQL / MariaDB
    (r'ECONNREFUSED.*3306|ER_ACCESS_DENIED_ERROR|PROTOCOL_CONNECTION_LOST.*3306',
     'MySQL/MariaDB (default port 3306)'),
    # Redis
    (r'ECONNREFUSED.*6379|Redis connection.*failed|NOAUTH',
     'Redis (default port 6379)'),
    # RabbitMQ
    (r'ECONNREFUSED.*5672|amqp.*connection.*refused',
     'RabbitMQ (default port 5672)'),
    # Elasticsearch
    (r'ECONNREFUSED.*9200|ConnectionError.*9200',
     'Elasticsearch (default port 9200)'),
    # Generic connection refused (with port)
    (r'ECONNREFUSED\s+\d+\.\d+\.\d+\.\d+:\d+',
     'an external service'),
    # Generic connection timeout to localhost
    (r'connect ETIMEDOUT\s+127\.0\.0\.1:\d+|'
     r'connection timed out.*localhost',
     'an external service on localhost'),
]


def _detect_external_service_failure(error_info: str) -> str | None:
    """Check if an error is caused by an unavailable external service.

    Returns a human-readable service name if detected, ``None`` otherwise.
    """
    for pattern, service_name in _EXTERNAL_SERVICE_PATTERNS:
        if re.search(pattern, error_info, re.IGNORECASE):
            return service_name
    return None


# ── System-level / environment issue detection ────────────────
# Patterns that indicate the failure is due to missing system tools,
# runtimes, or project setup files — NOT a code bug.  The agent
# cannot fix these by editing source files.

_SYSTEM_LEVEL_PATTERNS: list[tuple[str, str]] = [
    # Ruby / Bundler
    (r'Could not locate Gemfile', 'Bundler (no Gemfile found — run `bundle init` or create a Gemfile)'),
    (r'bundler:?\s+command not found|bundle:?\s+command not found',
     'Bundler (install with `gem install bundler`)'),
    (r"ruby:?\s+command not found|ruby:?\s+is not recognized",
     'Ruby runtime (install Ruby from https://www.ruby-lang.org)'),
    # Python
    (r'python3?:?\s+command not found|python3?:?\s+is not recognized',
     'Python runtime'),
    (r'pip3?:?\s+command not found|pip3?:?\s+is not recognized',
     'pip (Python package manager)'),
    # Node.js / npm
    (r'node:?\s+command not found|node:?\s+is not recognized',
     'Node.js runtime (install from https://nodejs.org)'),
    (r'npm:?\s+command not found|npm:?\s+is not recognized',
     'npm (install Node.js from https://nodejs.org)'),
    # Java
    (r'javac?:?\s+command not found|javac?:?\s+is not recognized',
     'Java SDK (install JDK)'),
    (r'mvn:?\s+command not found', 'Maven (install Apache Maven)'),
    (r'gradle:?\s+command not found', 'Gradle (install Gradle)'),
    # .NET
    (r'dotnet:?\s+command not found|dotnet:?\s+is not recognized',
     '.NET SDK (install from https://dotnet.microsoft.com)'),
    # Docker
    (r'docker:?\s+command not found|docker:?\s+is not recognized',
     'Docker (install Docker Desktop)'),
    # Generic: "X is not recognized as an internal or external command" (Windows)
    (r"'[^']+' is not recognized as an internal or external command",
     'a required system tool (see error message above)'),
]


def _detect_system_level_failure(error_info: str) -> str | None:
    """Check if an error is caused by a missing system tool or environment setup.

    Returns a human-readable description if detected, ``None`` otherwise.
    """
    for pattern, description in _SYSTEM_LEVEL_PATTERNS:
        if re.search(pattern, error_info, re.IGNORECASE):
            return description
    return None


def _infer_test_file_path(src_path: str, language: str | None) -> str:
    """Return the conventional test-file path for *src_path*.

    Examples:
        src/components/Footer.jsx  ->  src/components/__tests__/Footer.test.jsx
        src/utils/math.ts          ->  src/utils/__tests__/math.test.ts
        api/views.py               ->  api/tests/test_views.py
    """
    src_path_norm = src_path.replace("\\", "/")
    src_dir = src_path_norm.rsplit("/", 1)[0] if "/" in src_path_norm else "."
    basename = src_path_norm.rsplit("/", 1)[-1]
    stem, _, ext = basename.rpartition(".")

    _ext_map: dict[str, str] = {
        "jsx": "test.jsx", "tsx": "test.tsx",
        "js": "test.js",   "ts": "test.ts",
        "py": "py",        "rb": "spec.rb",
    }
    test_suffix = _ext_map.get(ext, f"test.{ext}")

    if ext == "py":
        return f"{src_dir}/tests/test_{stem}.{test_suffix}"
    return f"{src_dir}/__tests__/{stem}.{test_suffix}"


def _source_covered_by_test_step(
    src_path: str,
    test_step: "PlanStep",
) -> bool:
    """Return True if *test_step* explicitly references *src_path*.

    Checks plan-declared imports and inline test-file content so that
    both plan-parsed and LLM-inlined test specs are handled.
    """
    src_path_norm = src_path.replace("\\", "/")
    src_basename = src_path_norm.rsplit("/", 1)[-1]
    src_stem = src_basename.rsplit(".", 1)[0]

    # Plan-declared imports (e.g. imports: src/components/Footer.jsx:default)
    for imp_path in (test_step.imports_from or {}):
        if src_stem in imp_path or src_basename in imp_path or src_path in imp_path:
            return True

    # Inline test code content
    for content in (test_step.inline_code or {}).values():
        if src_stem in content or src_basename in content:
            return True

    return False


def _generate_test_coverage_for_inline_changes(
    *,
    uncovered_files: list[str],
    before_files: dict[str, str],
    memory: "FileMemory",
    executor,
    coder,
    display: "CLIDisplay",
    step_idx: int,
    language: str | None,
    plan_step: "PlanStep | None",
) -> None:
    """Ask the coder LLM to generate/update test files for source files
    that have no corresponding TEST step in the plan.

    Called from the Tier A inline-code path when reviewer is skipped because
    a TEST step follows — but the specific source files changed here are NOT
    imported by any of those TEST steps.  Without this function those changes
    would reach the bulk test run untested.

    For each uncovered source file:
      1. Find the best matching test file already in memory (by imports or name).
      2. Ask the LLM to add tests for the new/changed behaviour (diff context).
      3. Write the result to memory — it will be executed by run_bulk_test_execution_and_fix.
    """
    from ..language import get_code_block_lang

    all_files = memory.all_files()
    lang_tag = get_code_block_lang(language) if language else "python"

    for src_path in uncovered_files:
        old_content = before_files.get(src_path, "")
        new_content = all_files.get(src_path, "")
        if not new_content:
            continue

        src_path_norm = src_path.replace("\\", "/")
        src_basename = src_path_norm.rsplit("/", 1)[-1]
        src_stem = src_basename.rsplit(".", 1)[0]

        # ── Find best existing test file ──
        # Two strict heuristics, in order:
        #   1. Canonical naming convention — `test_<stem>.py` for Python or
        #      `<stem>.test.<ext>` / `<stem>.spec.<ext>` for JS/TS.
        #   2. Explicit import statement referencing the source module.
        # A loose substring match (the previous behaviour) clobbered
        # unrelated test files when the source basename happened to appear
        # anywhere in their content — e.g. matching `src/game.py` against
        # `test_renderer_base.py` because the latter referenced `GameState`,
        # then overwriting the renderer test with the game tests.
        existing_test_path: str | None = None
        existing_test_content: str = ""

        _src_ext = src_basename.rsplit(".", 1)[-1] if "." in src_basename else ""
        _canonical_basenames: set[str] = set()
        if _src_ext == "py":
            _canonical_basenames.add(f"test_{src_stem}.py")
            _canonical_basenames.add(f"{src_stem}_test.py")
        elif _src_ext:
            _canonical_basenames.add(f"{src_stem}.test.{_src_ext}")
            _canonical_basenames.add(f"{src_stem}.spec.{_src_ext}")
            _canonical_basenames.add(f"test_{src_stem}.{_src_ext}")

        # Pass 1 — canonical filename match.
        for fpath, content in all_files.items():
            if not _is_test_file(fpath):
                continue
            fbasename = fpath.replace("\\", "/").rsplit("/", 1)[-1]
            if fbasename in _canonical_basenames:
                existing_test_path = fpath
                existing_test_content = content
                break

        # Pass 2 — explicit import statement match.
        if existing_test_path is None:
            _src_no_ext = src_path_norm.rsplit(".", 1)[0]      # "src/game"
            _src_dotted = _src_no_ext.replace("/", ".")        # "src.game"
            _import_patterns = (
                f"from {_src_dotted} import",
                f"import {_src_dotted}",
                f"from .{src_stem} import",
                f"from '{_src_no_ext}'",
                f'from "{_src_no_ext}"',
                f"from './{src_stem}'",
                f'from "./{src_stem}"',
                f"from '../{src_stem}'",
                f'from "../{src_stem}"',
            )
            for fpath, content in all_files.items():
                if not _is_test_file(fpath):
                    continue
                if any(p in content for p in _import_patterns):
                    existing_test_path = fpath
                    existing_test_content = content
                    break

        target_test_path = existing_test_path or _infer_test_file_path(src_path, language)
        target_test_basename = target_test_path.replace("\\", "/").rsplit("/", 1)[-1]
        action = "update" if existing_test_path else "create"

        display.step_info(
            step_idx,
            f"[Inline] No test coverage for {src_basename} — "
            f"{'updating' if action == 'update' else 'generating'} "
            f"{target_test_basename}...",
        )
        _logger.info(
            "[Inline] Generating test coverage for uncovered source %s -> %s",
            src_path, target_test_path,
        )

        ctx = ""
        if existing_test_content:
            ctx = (
                f"Existing test file (DO NOT remove any existing tests):\n"
                f"#### [FILE]: {existing_test_path}\n"
                f"```{lang_tag}\n{existing_test_content}\n```\n\n"
            )

        old_block = ""
        if old_content:
            old_block = (
                f"Previous version of source (for reference — only test "
                f"new/changed behaviour):\n"
                f"```{lang_tag}\n{old_content}\n```\n\n"
            )

        # Detect subproject CWD: if source and test share a common top-level
        # directory (e.g. responsive-web-page/), the test runner's CWD when
        # invoked as `cd {subproject} && npm test` is already that directory.
        # process.cwd() calls inside the test must NOT re-include the prefix.
        _subproject_note = ""
        _src_parts = src_path_norm.split("/")
        if len(_src_parts) > 1:
            _subproject = _src_parts[0]
            _target_norm = target_test_path.replace("\\", "/")
            if _target_norm.startswith(_subproject + "/"):
                _subproject_note = (
                    f"\n\nIMPORTANT — Subproject CWD: This test runs inside the "
                    f"`{_subproject}/` directory (vitest/jest CWD when npm test "
                    f"is run as `cd {_subproject} && npm test`). "
                    f"Do NOT prefix file paths with `{_subproject}/` in "
                    f"`process.cwd()` calls.\n"
                    f"CORRECT: `resolve(process.cwd(), 'vitest.config.js')`\n"
                    f"WRONG:   `resolve(process.cwd(), '{_subproject}/vitest.config.js')`\n"
                )

        prompt = (
            f"Source file `{src_path}` was just written/updated and has "
            f"no corresponding test step in the plan.\n\n"
            f"New source content:\n"
            f"#### [FILE]: {src_path}\n```{lang_tag}\n{new_content}\n```\n\n"
            f"{old_block}"
            f"{ctx}"
            f"{'Add tests for the new or changed behaviour to the existing test file.'  if action == 'update' else 'Create a new test file covering the key functionality.'}\n\n"
            f"Requirements:\n"
            f"- Target file: `{target_test_path}`\n"
            f"- Do NOT remove existing tests\n"
            f"- Only test observable behaviour, not implementation details\n"
            f"{_subproject_note}\n"
            f"Output ONLY the complete test file:\n"
            f"#### [FILE]: {target_test_path}\n```{lang_tag}\n...full content...\n```"
        )

        try:
            response = coder.llm_client.generate_response(prompt)
            gen_files = executor.parse_code_blocks(response)
            if not gen_files:
                gen_files = executor.parse_code_blocks_fuzzy(response)
            # Accept only test files
            gen_files = {p: c for p, c in gen_files.items() if _is_test_file(p)}
            if gen_files:
                executor.write_files(gen_files)
                memory.update(gen_files)
                _logger.info(
                    "[Inline] Test coverage written for %s: %s",
                    src_path, list(gen_files.keys()),
                )
                display.step_info(
                    step_idx,
                    f"[Inline] Test coverage written: "
                    f"{', '.join(p.rsplit('/', 1)[-1] for p in gen_files)}",
                )
            else:
                _logger.warning(
                    "[Inline] LLM produced no test files for %s", src_path
                )
        except Exception as exc:
            _logger.warning(
                "[Inline] Test coverage generation failed for %s: %s", src_path, exc
            )


def build_step_waves(steps: list[str], dependencies: dict[int, set[int]]) -> list[list[int]]:
    """Group step indices into execution waves using topological ordering.

    Each wave is a list of step indices that can execute in parallel.
    Waves execute sequentially.
    """
    n = len(steps)
    remaining: set[int] = set(range(n))
    completed: set[int] = set()
    waves: list[list[int]] = []

    while remaining:
        # Find all steps whose dependencies are satisfied
        wave = [i for i in sorted(remaining)
                if dependencies.get(i, set()).issubset(completed)]
        if not wave:
            # Circular dependency or missing deps — execute remaining sequentially
            wave = [min(remaining)]
        waves.append(wave)
        for i in wave:
            remaining.discard(i)
            completed.add(i)

    return waves


def _execute_step(step_idx: int, step_text: str, *,
                  steps: list[str],
                  llm_client, executor, coder, reviewer, tester,
                  task: str, memory: FileMemory, display: CLIDisplay,
                  language: str | None, cfg=None,
                  auto: bool = False,
                  search_agent=None,
                  kb_context_builder=None,
                  code_graph=None,
                  project_profile=None,
                  knowledge_base=None,
                  project_context=None,
                  plan_step: PlanStep | None = None,
                  all_plan_steps: list[PlanStep] | None = None,
                  intent_spec=None,
                  ) -> tuple[int, bool, str]:
    """Execute a single step. Returns ``(step_idx, success, error_info)``.

    When *plan_step* is provided (structured plan), step type and
    dependencies are taken from the object — no LLM classification call.

    Catches all exceptions so that a crash inside any handler never
    kills the whole pipeline — the step is marked as failed instead.
    """
    try:
        # --- Project Orientation + KB Context Injection (Phase 4+) ---
        #
        # Project grounding ALWAYS comes first — before KB symbols,
        # before task description, before everything.  It is the LLM's
        # "north star" for the entire session.

        context_parts: list[str] = []

        # 1. Project orientation grounding (always first)
        if project_profile is not None:
            try:
                context_parts.append(project_profile.format_for_prompt())
            except Exception as orient_exc:
                _logger.warning(
                    "[KB] Project orientation formatting failed: %s",
                    orient_exc,
                )

        # 2. Project knowledge (installed packages, file purposes, tech stack)
        if knowledge_base is not None:
            try:
                kb_agent_ctx = knowledge_base.format_for_agents()
                if kb_agent_ctx:
                    context_parts.append(kb_agent_ctx)
            except Exception as kb_fmt_exc:
                _logger.warning(
                    "[KB] format_for_agents failed: %s", kb_fmt_exc,
                )

        # 3. KB context (Phase 4 — symbols, error fixes, patterns)
        #
        # Use a clean short description as the KB search query.
        # step_text may contain inline code blocks from the plan (e.g. a full
        # JSX component), which explodes the keyword-score denominator and
        # dilutes all meaningful matches.  plan_step.description is the
        # one-line human description; fall back to the first line of step_text.
        _kb_query = (
            (plan_step.description if plan_step and plan_step.description else None)
            or step_text.split("\n")[0].strip()
        )
        # Augment _kb_query with project tech stack so framework docs are
        # found for steps whose description doesn't mention the framework
        # by name (e.g. "Replace main.jsx" doesn't mention "tailwindcss").
        # Uses installed_packages from knowledge_base (already in memory —
        # no I/O or LLM calls) filtered to recognised tech keywords only.
        if knowledge_base is not None:
            try:
                _pk = knowledge_base.load()
                _pkgs = getattr(_pk, "installed_packages", [])
                if _pkgs:
                    from ..orchestrator.plan_optimizer import _TECH_KEYWORDS
                    _tech_hits = _TECH_KEYWORDS.findall(" ".join(_pkgs[:50]))
                    _query_lower = _kb_query.lower()
                    _tech_extras = [
                        t for t in dict.fromkeys(t.lower() for t in _tech_hits)
                        if t.lower() not in _query_lower
                    ][:8]
                    if _tech_extras:
                        _kb_query = f"{_kb_query} {' '.join(_tech_extras)}"
            except Exception:
                pass
        # Skip KB context (and its embedding API call) for inline-code steps —
        # the coder is never invoked for inline steps so the context is wasted.
        _has_inline = (
            plan_step is not None
            and getattr(plan_step, "inline_code", None)
            and len(plan_step.inline_code) > 0
        )
        if kb_context_builder is not None and not _has_inline:
            try:
                from ..kb.context_builder import ContextBuilder
                kb_ctx = kb_context_builder.build_context(
                    task_description=_kb_query,
                    current_file=None,
                    max_tokens=getattr(cfg, "KB_MAX_CONTEXT_TOKENS", 4000) if cfg else 4000,
                    language=getattr(project_context, "language", None) if project_context else None,
                    # Without this the builder's own "Skip for CMD steps —
                    # install commands don't need them" guard is dead code on
                    # the one path that runs for EVERY step: the parameter
                    # defaults to None, and `None != "CMD"` is True, so the
                    # check concludes the opposite of what it was written for.
                    # Observed: a `python -m venv` step was handed 1,392
                    # tokens of Python test-generation instructions.
                    step_type=getattr(plan_step, "step_type", None),
                )
                if kb_ctx.kb_available or kb_ctx.behavioral_instructions or kb_ctx.global_patterns:
                    kb_text = kb_context_builder.format_context_for_prompt(kb_ctx)
                    if kb_text:
                        context_parts.append(kb_text)
                _logger.debug(
                    "[KB] Injected context: %d tokens, sources: %s, "
                    "symbols: %d, errors: %d",
                    kb_ctx.token_count, kb_ctx.sources_used,
                    len(kb_ctx.local_symbols), len(kb_ctx.error_fixes),
                )
            except Exception as kb_exc:
                _logger.warning("[KB] Context injection failed: %s", kb_exc)

        # Combine and store in memory for downstream handlers
        if context_parts:
            memory._kb_context = "\n\n".join(context_parts)

        # --- Load KB content fixes once per pipeline run ---
        if not hasattr(memory, '_content_fixes') or memory._content_fixes is None:
            try:
                # Reuse the global store from kb_context_builder if available,
                # avoiding a redundant GlobalKBStore instantiation.
                _gkb = (getattr(kb_context_builder, '_global_store', None)
                        if kb_context_builder is not None else None)
                if _gkb is None:
                    from ..kb.global_kb.store import GlobalKBStore
                    _gkb = GlobalKBStore()
                memory._content_fixes = _gkb.get_content_fixes(language=language)
                if memory._content_fixes:
                    _logger.debug(
                        "[KB] Loaded %d content-fix rules",
                        len(memory._content_fixes),
                    )
            except Exception as exc:
                _logger.debug("[KB] Failed to load content fixes: %s", exc)
                memory._content_fixes = []

        # --- KB-guided file scoping (Option A) ---
        # Use KB to identify most relevant files for this step,
        # so the coder only sees focused context instead of everything.
        if kb_context_builder is not None:
            try:
                changed = list(memory.all_files().keys())[:10]
                relevant_files = kb_context_builder.get_relevant_files(
                    task_description=_kb_query,
                    changed_files=changed,
                    max_files=15,
                )
                if relevant_files:
                    memory._kb_relevant_files = relevant_files
                    _logger.debug(
                        "[KB] Scoped to %d relevant files for step %d",
                        len(relevant_files), step_idx + 1,
                    )
            except Exception as kb_scope_exc:
                _logger.debug("[KB] File scoping failed: %s", kb_scope_exc)

        log.info(f"\n{'='*60}\nTask {step_idx+1}: {step_text}\n"
                 f"Memory: {memory.summary()}\n{'='*60}")

        display.start_step(step_idx)

        # --- Structured plan: use declared type (skip LLM classification) ---
        if plan_step is not None and plan_step.step_type != "UNCLASSIFIED":
            step_type = plan_step.step_type
            display.step_info(step_idx, f"Type: [{step_type}] (from plan)")
            display.step_tokens(step_idx, 0, 0)
            plan_step.status = "in_progress"
        elif plan_step is not None and plan_step.step_type == "UNCLASSIFIED":
            # Infer type from plan_step fields before falling back to LLM
            if plan_step.command:
                step_type = "CMD"
                plan_step.step_type = "CMD"
                _logger.info(
                    "[PlanStep] step %d was UNCLASSIFIED but has command — "
                    "inferred CMD (0 LLM tokens)", step_idx,
                )
                display.step_info(step_idx, f"Type: [{step_type}] (inferred from plan command)")
                display.step_tokens(step_idx, 0, 0)
                plan_step.status = "in_progress"
            else:
                _logger.warning(
                    "[PlanStep] step %d has type UNCLASSIFIED — "
                    "falling back to LLM classification. "
                    "This happens when structured metadata is lost "
                    "(e.g. plan was edited in TUI).",
                    step_idx,
                )
                display.step_info(step_idx, "Loading context and classifying...")
                step_type = _classify_step(step_text, llm_client, display, step_idx)
                # Persist classified type back on PlanStep for checkpoint
                plan_step.step_type = step_type
        else:
            _logger.warning(
                "[PlanStep] plan_step is None for step %d — "
                "falling back to LLM classification (tokens wasted). "
                "Check if plan_steps_parsed is intact at execution time.",
                step_idx,
            )
            display.step_info(step_idx, "Loading context and classifying...")
            step_type = _classify_step(step_text, llm_client, display, step_idx)

        display.steps[step_idx]["type"] = step_type
        display.render()

        # ── Step Type Auto-Correction ──
        # Heuristic: If type is CODE/TEST but looks like CMD, verify with LLM.
        if plan_step is not None and step_type in ("CODE", "TEST"):
            has_targets = "target:" in step_text.lower()
            has_inline = plan_step.inline_code and len(plan_step.inline_code) > 0
            has_cmd_markers = re.search(r'^[ \t]*[>$][ \t]+', step_text, re.MULTILINE)

            if not has_targets and not has_inline and has_cmd_markers:
                _logger.info("[Pipeline] Step %s misclassification suspected (%s -> CMD). Re-classifying...",
                             plan_step.id, step_type)
                display.step_info(step_idx, f"Suspicious {step_type} classification, verifying...")

                try:
                    # Reuse the same classification flow as fallback
                    new_type = _classify_step(step_text, llm_client, display, step_idx)
                    if new_type != step_type:
                        _logger.info("[Pipeline] Step %s re-classified: %s -> %s",
                                     plan_step.id, step_type, new_type)
                        step_type = new_type
                        plan_step.step_type = step_type
                        display.steps[step_idx]["type"] = step_type
                        display.render()
                except Exception as e:
                    _logger.warning("[Pipeline] Re-classification failed for %s: %s", plan_step.id, e)

        # ── CMD → TEST Auto-Correction (deterministic, 0 LLM cost) ──
        # If planner labelled a step as CMD but the description/command
        # contains a test runner invocation (e.g. "npx vitest run",
        # "pytest", "npm test"), reclassify to TEST so the test handler
        # (with retry-and-fix logic) is used instead of the plain CMD handler.
        if step_type == "CMD":
            _check_text = step_text
            if plan_step is not None and plan_step.command:
                _check_text = f"{step_text} {plan_step.command}"
            if _TEST_CMD_RE.search(_check_text) and not _TEST_CONFIG_RE.search(_check_text):
                _logger.info(
                    "[Pipeline] Step %s reclassified CMD -> TEST "
                    "(test runner detected in description/command, 0 LLM tokens)",
                    plan_step.id if plan_step else step_idx,
                )
                step_type = "TEST"
                if plan_step is not None:
                    plan_step.step_type = "TEST"
                display.steps[step_idx]["type"] = step_type
                display.render()

        # ── CODE → TEST Auto-Correction (deterministic, 0 LLM cost) ──
        # If planner labelled a step as CODE but ALL target files are test
        # files (e.g. __tests__/App.test.jsx, test_main.py), reclassify to
        # TEST.  The TEST handler validates the fix by running the tests and
        # retries on failure, while CODE just writes the file without running.
        if step_type == "CODE" and plan_step is not None and plan_step.target_files:
            if all(_is_test_file(f) for f in plan_step.target_files):
                _logger.info(
                    "[Pipeline] Step %s reclassified CODE -> TEST "
                    "(all target files are test files: %s, 0 LLM tokens)",
                    plan_step.id, plan_step.target_files,
                )
                step_type = "TEST"
                plan_step.step_type = "TEST"
                display.steps[step_idx]["type"] = step_type
                display.render()

        log.info(f"Task {step_idx+1}: Classified as [{step_type}]")

        # --- Structured plan: inject plan-aware context (thread-local) ---
        # Use thread-local storage so parallel wave steps don't overwrite each
        # other's context on the shared memory object (race condition fix).
        if plan_step is not None and all_plan_steps is not None:
            try:
                from .memory import set_plan_context_files
                plan_ctx_files = build_step_context(
                    plan_step, all_plan_steps, memory,
                    read_from_disk=lambda p: executor.read_file(p)
                    if hasattr(executor, 'read_file') else None,
                )
                if plan_ctx_files:
                    set_plan_context_files(plan_ctx_files)
                    _logger.debug(
                        "[PlanStep] Injected %d plan-context files for step %s",
                        len(plan_ctx_files), plan_step.id,
                    )
            except Exception as pctx_exc:
                _logger.debug("[PlanStep] Context build failed: %s", pctx_exc)

        success, error_info = True, ""

        # ── Dependency check: before-snapshot ─────────────────────
        _dep_check_enabled = cfg is None or getattr(cfg, "DEPENDENCY_CHECK_ENABLED", True)
        _before_files = dict(memory.all_files()) if _dep_check_enabled else None

        if step_type == "IGNORE":
            display.step_info(step_idx, "Not actionable, skipping.")
            display.complete_step(step_idx, "skipped")

        elif step_type == "CMD":
            success, error_info = _handle_cmd_step(
                step_text, executor, llm_client, memory, display, step_idx,
                language=language, project_context=project_context,
                plan_step=plan_step, intent_spec=intent_spec, cfg=cfg)

        elif step_type == "CODE":
            # ── Inline edit fast path ──
            # If the planner provided find/replace edit blocks, apply them
            # surgically to the existing files and promote the result into
            # inline_code so the existing quality gate handles it naturally.
            # Falls through to coder if any edit fails to apply.
            if (plan_step is not None
                    and plan_step.inline_edits
                    and not plan_step.inline_code):
                _edit_subproject = _detect_subproject_root(memory)
                _edit_all_ok = True
                _patched = None
                _cur = None

                for _edit_fpath, _edit_pairs in plan_step.inline_edits.items():
                    import os as _os_edit
                    _resolved = _edit_fpath
                    if _edit_subproject and not _edit_fpath.startswith(_edit_subproject):
                        _candidate = f"{_edit_subproject}/{_edit_fpath}"
                        if _os_edit.path.exists(_candidate):
                            _resolved = _candidate

                    if not _os_edit.path.exists(_resolved):
                        _logger.warning(
                            "[InlineEdit] Target not found: %s — skipping edit path",
                            _resolved,
                        )
                        _edit_all_ok = False
                        break

                    try:
                        with open(_resolved, "r", encoding="utf-8") as _ef:
                            _cur = _ef.read()
                    except OSError as _oe:
                        _logger.warning("[InlineEdit] Cannot read %s: %s", _resolved, _oe)
                        _edit_all_ok = False
                        break

                    # Pre-validate: log which FIND strings won't match so the
                    # cause is visible before any edits are attempted.
                    for _pi, (_find_pre, _) in enumerate(_edit_pairs):
                        if _find_pre not in _cur:
                            _find_lines_pre = [l.strip() for l in _find_pre.splitlines() if l.strip()]
                            _cur_stripped = [l.strip() for l in _cur.splitlines()]
                            _n_pre = len(_find_lines_pre)
                            _pre_fuzzy = any(
                                _cur_stripped[_li:_li + _n_pre] == _find_lines_pre
                                for _li in range(max(0, len(_cur_stripped) - _n_pre + 1))
                            )
                            if not _pre_fuzzy:
                                _logger.warning(
                                    "[InlineEdit] FIND block #%d in %s will not match "
                                    "(neither exact nor fuzzy). Likely cause: planner "
                                    "hallucinated content not present in the file. "
                                    "First non-matching line: %r",
                                    _pi + 1, _resolved,
                                    next(
                                        (l for l in _find_lines_pre if l not in _cur_stripped),
                                        "<all lines present but sequence mismatch>",
                                    ),
                                )

                    _patched = _cur
                    for _find_str, _repl_str in _edit_pairs:
                        # Skip pairs where the find string has no real file
                        # content (e.g. leftover ---file-content-end--- marker
                        # accidentally flushed as a phantom edit pair).
                        _find_meaningful_lines = [
                            l for l in _find_str.splitlines()
                            if l.strip() and not l.strip().startswith("---")
                        ]
                        if not _find_meaningful_lines:
                            _logger.debug(
                                "[InlineEdit] Skipping empty/marker-only FIND pair in %s",
                                _resolved,
                            )
                            continue
                        # Skip no-op edits where FIND and REPLACE are identical.
                        # These waste a step and, worse, silently claim success
                        # without adding the methods/code the plan intended.
                        if _find_str == _repl_str:
                            _logger.warning(
                                "[InlineEdit] Skipping no-op FIND/REPLACE pair in %s "
                                "(FIND and REPLACE are identical — plan likely "
                                "failed to include the intended changes)",
                                _resolved,
                            )
                            continue
                        if _find_str in _patched:
                            _find_pos = _patched.index(_find_str)
                            _after_find = _patched[_find_pos + len(_find_str):]
                            # Guard: if REPLACE starts with FIND text and the
                            # original file has content after the match, a plain
                            # str.replace would append that tail after the
                            # replacement block, duplicating it.  This happens
                            # when the LLM anchors on a single import/line but
                            # puts the entire new file in REPLACE.  Fix: treat
                            # FIND as a positional anchor and replace from its
                            # position to EOF with REPLACE.
                            #
                            # However, if REPLACE is only slightly larger than
                            # FIND (small insertion like adding one import line),
                            # this is NOT a full-file rewrite — use normal
                            # str.replace to preserve the rest of the file.
                            _repl_is_full_rewrite = (
                                _repl_str.lstrip('\n').startswith(_find_str.lstrip('\n'))
                                and _after_find.strip()
                                # REPLACE must be large enough to plausibly
                                # contain the rest of the file.  If it's
                                # shorter than the content after the FIND
                                # anchor it's clearly a small insertion, not
                                # a full-file rewrite.
                                and len(_repl_str) > len(_after_find)
                            )
                            if _repl_is_full_rewrite:
                                _patched = _patched[:_find_pos] + _repl_str
                                _logger.debug(
                                    "[InlineEdit] Anchor-to-EOF applied in %s "
                                    "(REPLACE starts with FIND — tail dedup)",
                                    _resolved,
                                )
                            else:
                                _patched = _patched.replace(_find_str, _repl_str, 1)
                            _logger.debug("[InlineEdit] Exact match applied in %s", _resolved)
                        else:
                            # Fuzzy fallback: normalize whitespace per-line and
                            # try to locate the find block in the file.
                            # Works for both single-line and multi-line finds.
                            _find_lines_stripped = [l.strip() for l in _find_str.splitlines() if l.strip()]
                            _file_lns = _patched.splitlines(keepends=True)
                            _file_stripped = [l.strip() for l in _file_lns]
                            _n_find = len(_find_lines_stripped)
                            _fuzzy_hit = False
                            if _n_find > 0:
                                for _li in range(len(_file_lns) - _n_find + 1):
                                    if _file_stripped[_li:_li + _n_find] == _find_lines_stripped:
                                        # Determine base indent from first matched line
                                        _indent = len(_file_lns[_li]) - len(_file_lns[_li].lstrip())
                                        _repl_lines = _repl_str.splitlines()
                                        _last_orig_newline = _file_lns[_li + _n_find - 1].endswith("\n")
                                        _replacement = (
                                            "\n".join(
                                                (" " * _indent + rl.lstrip()) if rl.strip() else rl
                                                for rl in _repl_lines
                                            )
                                            + ("\n" if _last_orig_newline else "")
                                        )
                                        _file_lns[_li:_li + _n_find] = [_replacement]
                                        _patched = "".join(_file_lns)
                                        _fuzzy_hit = True
                                        _logger.debug(
                                            "[InlineEdit] Fuzzy match applied in %s (lines %d-%d)",
                                            _resolved, _li + 1, _li + _n_find,
                                        )
                                        break
                            if not _fuzzy_hit:
                                # ── Minimal-diff fallback ──────────────────────
                                # When the full FIND block fails (e.g. planner
                                # assumed stale scaffold content), extract only
                                # the lines that actually change between FIND and
                                # REPLACE and try to apply each change individually.
                                # Only succeeds when every changed line appears
                                # exactly once in the file (no ambiguity).
                                import difflib as _difflib
                                _find_all_lns = _find_str.splitlines()
                                _repl_all_lns = _repl_str.splitlines()
                                _sm = _difflib.SequenceMatcher(
                                    None, _find_all_lns, _repl_all_lns, autojunk=False
                                )
                                _delta: list = []  # [(old_line, new_line | None)]
                                for _tag, _i1, _i2, _j1, _j2 in _sm.get_opcodes():
                                    if _tag == "replace":
                                        for _k in range(max(_i2 - _i1, _j2 - _j1)):
                                            _o = _find_all_lns[_i1 + _k] if _i1 + _k < _i2 else None
                                            _n = _repl_all_lns[_j1 + _k] if _j1 + _k < _j2 else None
                                            if _o is not None:
                                                _delta.append((_o, _n))
                                    elif _tag == "delete":
                                        for _k in range(_i1, _i2):
                                            _delta.append((_find_all_lns[_k], None))
                                    # 'insert' has no anchor line — skip
                                if _delta:
                                    _work_lns = _patched.splitlines(keepends=True)
                                    _work_s = [l.strip() for l in _work_lns]
                                    _min_ok = True
                                    for _old_ln, _new_ln in _delta:
                                        _old_s = _old_ln.strip()
                                        _idxs = [i for i, s in enumerate(_work_s) if s == _old_s]
                                        if len(_idxs) != 1:
                                            _min_ok = False
                                            break
                                        _idx = _idxs[0]
                                        if _new_ln is not None:
                                            _bi = len(_work_lns[_idx]) - len(_work_lns[_idx].lstrip())
                                            _work_lns[_idx] = (
                                                " " * _bi + _new_ln.lstrip()
                                                + ("\n" if _work_lns[_idx].endswith("\n") else "")
                                            )
                                            _work_s[_idx] = _new_ln.strip()
                                        else:
                                            _work_lns[_idx] = ""
                                            _work_s[_idx] = ""
                                    if _min_ok:
                                        _patched = "".join(_work_lns)
                                        _fuzzy_hit = True
                                        _logger.info(
                                            "[InlineEdit] Minimal-diff fallback applied in %s "
                                            "(%d change(s))",
                                            _resolved, len(_delta),
                                        )
                                if not _fuzzy_hit:
                                    _logger.warning(
                                        "[InlineEdit] find string not found in %s — "
                                        "falling through to coder",
                                        _resolved,
                                    )
                                    _edit_all_ok = False
                                    break

                    if not _edit_all_ok:
                        break
                    # Reject syntactically broken patches before promotion —
                    # the fuzzy/minimal-diff fallbacks can mangle docstrings
                    # (observed: an unterminated triple-quote written to
                    # disk; a trailing comma minimal-diffed into
                    # package.json). Fall through to the coder instead.
                    if _resolved.endswith((".py", ".pyw", ".json")):
                        _gate_err = _syntax_gate(_resolved, _patched)
                        if _gate_err:
                            _logger.warning(
                                "[InlineEdit] Patched %s failed validation "
                                "(%s) — falling through to coder",
                                _resolved, _gate_err)
                            _edit_all_ok = False
                            break
                    # Promote the patched content into inline_code so the
                    # existing quality gate below handles writing + validation.
                    plan_step.inline_code[_resolved] = _patched
                    # The patch applied against the file's REAL current
                    # content — mark it grounded so the protected-manifest
                    # write guard lets it through (editing package.json is
                    # sometimes the whole task; a skipped write previously
                    # reported success while changing nothing).
                    _ge = getattr(plan_step, "_grounded_edit_targets", None)
                    if _ge is None:
                        _ge = plan_step._grounded_edit_targets = set()
                    _ge.add(_resolved)
                    _logger.info(
                        "[InlineEdit] Promoted patched %s -> inline_code", _resolved,
                    )

                if not _edit_all_ok:
                    import os as _os_inline_fb
                    # ── REPLACE-as-full-file fallback ──
                    # When the FIND block doesn't match (e.g. scaffold template
                    # changed between Vite versions), check if the REPLACE block
                    # looks like a complete file replacement.  If so, use it
                    # directly instead of falling through to the coder LLM
                    # (which lacks KB context and generates garbage).
                    _used_replace_fallback = False
                    if plan_step.inline_edits:
                        for _ie_path, _ie_pairs in plan_step.inline_edits.items():
                            _ie_resolved = _ie_path
                            # Find the matching resolved path
                            for _kf in (plan_step.target_files or []):
                                if _kf.endswith(_ie_path) or _ie_path.endswith(_os_inline_fb.path.basename(_kf)):
                                    _ie_resolved = _kf
                                    break
                            # An edit: block targets an EXISTING file. If the
                            # declared path doesn't exist the planner assumed
                            # the wrong scaffold layout — writing there would
                            # create a dead file (e.g. project/project/urls.py
                            # that Django never loads) while the real target
                            # stays unedited. Retarget by basename when
                            # unambiguous; otherwise skip promotion so the
                            # step falls through to the coder/agent loop.
                            if not _os_inline_fb.path.exists(_ie_resolved):
                                _reresolved = _resolve_existing_by_basename(
                                    _ie_resolved, memory)
                                if _reresolved is not None:
                                    _logger.info(
                                        "[InlineEdit] Declared edit target %s "
                                        "does not exist — retargeting to "
                                        "existing %s", _ie_resolved, _reresolved)
                                    _ie_resolved = _reresolved
                                else:
                                    _logger.warning(
                                        "[InlineEdit] Declared edit target %s "
                                        "does not exist and no unambiguous "
                                        "same-name file found — not promoting "
                                        "REPLACE (would create a dead file at "
                                        "a phantom path)", _ie_resolved)
                                    continue
                            # Collect substantial REPLACE blocks for this
                            # file. Promotion is only safe with EXACTLY ONE:
                            # concatenating multiple blocks into one file
                            # merged a second file's content into the first
                            # (observed: main.jsx's REPLACE appended to
                            # App.jsx → duplicate `App` declaration that was
                            # valid syntax but broke the build). Multi-block
                            # edits fall through to the coder/agent loop,
                            # which works from the real file.
                            _repl_parts: list[str] = []
                            for _, _repl in _ie_pairs:
                                _repl_stripped = _repl.strip()
                                _has_structure = (
                                    'import ' in _repl_stripped
                                    or 'export ' in _repl_stripped
                                    or 'function ' in _repl_stripped
                                    or 'class ' in _repl_stripped
                                    or 'def ' in _repl_stripped
                                    or 'from ' in _repl_stripped
                                )
                                _is_substantial = len(_repl_stripped) > 50
                                if _has_structure and _is_substantial:
                                    _repl_parts.append(_repl_stripped)
                            if len(_repl_parts) > 1:
                                _logger.warning(
                                    "[InlineEdit] %d REPLACE blocks for %s — "
                                    "refusing to merge-promote (risks fusing "
                                    "two files' content); falling through to "
                                    "coder", len(_repl_parts), _ie_resolved)
                                _repl_parts = []
                            if _repl_parts:
                                _combined = _repl_parts[0]
                                # Safety check: verify the combined REPLACE
                                # text looks like a complete file, not a
                                # fragment from an incremental edit.
                                #
                                # For JS/JSX/TS/TSX files a complete module
                                # must contain an export statement.  Without
                                # one the REPLACE is just a fragment (e.g.
                                # import lines from the first edit pair)
                                # and would produce a corrupt file.
                                #
                                # For multi-block concatenations, validate the
                                # combined text with the same language-aware
                                # syntax checker used by the Tier-B static-check
                                # gate below (compile() for Python, tree-sitter
                                # via `syntax_checker` for everything else in
                                # EXTENSION_MAP). This replaces a brace/paren
                                # balance count, which is meaningless for
                                # indentation-scoped languages like Python and
                                # let syntactically-broken concatenations (e.g.
                                # dangling methods missing their class header)
                                # through to disk.
                                _ext = _os_inline_fb.path.splitext(_ie_resolved)[1].lower()
                                _js_like = _ext in {
                                    '.js', '.jsx', '.ts', '.tsx', '.mjs',
                                    '.cjs', '.mts', '.cts',
                                }
                                _is_complete = True
                                # JS/TS modules must have an export
                                if _js_like:
                                    _has_export = (
                                        'export ' in _combined
                                        or 'module.exports' in _combined
                                    )
                                    if not _has_export:
                                        _is_complete = False
                                _lint_errs = ""
                                if _is_complete:
                                    _lint_errs = _syntax_gate(
                                        _ie_resolved, _combined) or ""
                                    if _lint_errs:
                                        _is_complete = False
                                if _is_complete:
                                    plan_step.inline_code[_ie_resolved] = _combined
                                    _logger.info(
                                        "[InlineEdit] FIND failed but REPLACE "
                                        "looks like a complete file — promoting "
                                        "%d REPLACE block(s) for %s (%d chars)",
                                        len(_repl_parts),
                                        _ie_resolved, len(_combined),
                                    )
                                    _used_replace_fallback = True
                                else:
                                    _logger.warning(
                                        "[InlineEdit] FIND failed and %d REPLACE "
                                        "block(s) for %s look like fragments — "
                                        "falling through to coder LLM%s",
                                        len(_repl_parts), _ie_resolved,
                                        f" ({_lint_errs.strip()})" if _lint_errs else "",
                                    )

                    if _used_replace_fallback:
                        plan_step.inline_edits = {}
                        _logger.info(
                            "[InlineEdit] Using plan REPLACE content as "
                            "full-file replacement (skipping coder LLM)")
                    else:
                        # If some blocks applied before the failure, write the
                        # partial result to disk so the coder sees the already-
                        # patched file rather than the original.
                        if _patched is not None and _cur is not None and _patched != _cur:
                            try:
                                with open(_resolved, "w", encoding="utf-8") as _wf:
                                    _wf.write(_patched)
                                memory.update({_resolved: _patched})
                                _logger.info(
                                    "[InlineEdit] Partial edits written to %s "
                                    "before falling through to coder",
                                    _resolved,
                                )
                            except OSError as _we:
                                _logger.warning(
                                    "[InlineEdit] Could not write partial edits to %s: %s",
                                    _resolved, _we,
                                )
                        plan_step.inline_code.clear()
                        _logger.info("[InlineEdit] Falling through to coder (edit failed)")

            # ── User approval gate for inline code (pre-write) ──
            # The inline fast path skips the Coder LLM, but writes still
            # need user approval unless --auto.  Build a preview of the
            # resolved files and show the diff editor first.  On rejection,
            # clear plan_step.inline_code so the inline `if` below evaluates
            # False and execution falls through to the Coder path for a
            # fresh attempt.  Mirrors the existing rejection pattern at
            # plan_step.py:1005 (clear-inline_code-to-fall-back-to-coder).
            if (plan_step is not None
                    and plan_step.inline_code
                    and len(plan_step.inline_code) > 0
                    and not auto):
                # Build the same resolved files dict the inline path will
                # write, so the diff preview shows accurate paths/contents.
                # Resolution work is duplicated below — that is intentional:
                # it keeps the existing 325-line inline body untouched and
                # the resolution is cheap (string ops + memory scan).
                _preview_files = dict(plan_step.inline_code)
                from .classification import (
                    resolve_cmd_placeholders as _resolve_ph_pre,
                )
                _ph_task_pre = task or ''
                if any('<' in k for k in _preview_files):
                    _preview_files = {
                        _resolve_ph_pre(
                            k, step_text=step_text, task=_ph_task_pre
                        ): v
                        for k, v in _preview_files.items()
                    }
                _preview_subproject = _detect_subproject_root(memory)
                if not _preview_subproject:
                    import re as _re_pre
                    _mem_all_pre = memory.all_files()
                    _scaffold_pats_pre = [
                        _re_pre.compile(
                            r'npm\s+create\s+vite(?:@\S+)?\s+(\S+)'),
                        _re_pre.compile(
                            r'create-vite(?:@\S+)?\s+(\S+)'),
                        _re_pre.compile(
                            r'create-next-app(?:@\S+)?\s+(\S+)'),
                        _re_pre.compile(
                            r'create-react-app\s+(\S+)'),
                        _re_pre.compile(r'ng\s+new\s+(\S+)'),
                    ]
                    import os as _os_pre
                    for _fp_pre, _ct_pre in _mem_all_pre.items():
                        if not _fp_pre.startswith('_cmd_output/'):
                            continue
                        _first_pre = (
                            _ct_pre.split('\n')[0] if _ct_pre else ''
                        )
                        for _pat_pre in _scaffold_pats_pre:
                            _m_pre = _pat_pre.search(_first_pre)
                            if _m_pre:
                                _cand_pre = (
                                    _m_pre.group(1).strip().rstrip('/')
                                )
                                if _cand_pre and _os_pre.path.isdir(
                                        _cand_pre):
                                    _preview_subproject = _cand_pre
                                    break
                        if _preview_subproject:
                            break
                if _preview_subproject:
                    _preview_files = _prefix_subproject_paths(
                        _preview_files, _preview_subproject, memory)
                from .dependency_check import (
                    clean_diff_markers as _clean_diff_pre,
                )
                _preview_files = {
                    path: _clean_diff_pre(content)
                    for path, content in _preview_files.items()
                }

                # Show the diff and wait for approval.
                from ..diff_display import (
                    prompt_diff_approval as _prompt_inline_approval,
                )
                display.stop_spinner()
                _inline_user_approved = _prompt_inline_approval(
                    _preview_files, auto=False, display=display,
                    base_dir=getattr(memory, 'base_dir', "."),
                )
                display.step_info(step_idx, "Processing...")
                if not _inline_user_approved:
                    _logger.info(
                        "[Inline] User rejected inline code for step %s "
                        "— falling back to Coder for fresh attempt",
                        plan_step.id if plan_step else step_idx,
                    )
                    display.step_info(
                        step_idx,
                        "Inline code rejected — running Coder",
                    )
                    # Clear so the inline `if` below evaluates False and
                    # execution naturally falls into the Coder `else:`
                    # branch.  Coder will regenerate from scratch instead
                    # of starting from the rejected planner draft.
                    plan_step.inline_code.clear()

            # ── Inline code fast path ──
            # If the planner already provided complete code in the plan,
            # write it directly — zero Coder LLM calls needed.
            if (plan_step is not None
                    and plan_step.inline_code
                    and len(plan_step.inline_code) > 0):
                display.step_info(step_idx, "Writing inline code from plan (0 LLM calls)")
                _inline_files = dict(plan_step.inline_code)
                # Resolve <project-name> and similar placeholder tokens that the
                # planner may have left in file path keys (e.g. when a dumb LLM
                # outputs "target: <project-name>/src/App.jsx").  The same logic
                # used for CMD placeholders is applied to path strings.
                from .classification import resolve_cmd_placeholders as _resolve_ph
                _ph_task = task or ''
                if any('<' in k for k in _inline_files):
                    _inline_files = {
                        _resolve_ph(k, step_text=step_text, task=_ph_task): v
                        for k, v in _inline_files.items()
                    }
                _inline_subproject = _detect_subproject_root(memory)
                # Fallback: if memory-based detection failed (e.g. no source
                # files in memory yet, so _detect_subproject_root bails early),
                # infer from the CMD-output entries that ARE in memory.
                if not _inline_subproject:
                    import re as _re
                    _mem_all = memory.all_files()
                    _scaffold_pats = [
                        _re.compile(r'npm\s+create\s+vite(?:@\S+)?\s+(\S+)'),
                        _re.compile(r'create-vite(?:@\S+)?\s+(\S+)'),
                        _re.compile(r'create-next-app(?:@\S+)?\s+(\S+)'),
                        _re.compile(r'create-react-app\s+(\S+)'),
                        _re.compile(r'ng\s+new\s+(\S+)'),
                    ]
                    import os as _os
                    for _fpath, _content in _mem_all.items():
                        if not _fpath.startswith('_cmd_output/'):
                            continue
                        _first = _content.split('\n')[0] if _content else ''
                        for _pat in _scaffold_pats:
                            _m = _pat.search(_first)
                            if _m:
                                _cand = _m.group(1).strip().rstrip('/')
                                if _cand and _os.path.isdir(_cand):
                                    _inline_subproject = _cand
                                    _logger.info(
                                        "[Inline] Subproject from CMD "
                                        "fallback: %s/", _cand)
                                    break
                        if _inline_subproject:
                            break
                _logger.debug(
                    "[Inline] subproject=%r inline_keys=%r",
                    _inline_subproject, list(_inline_files.keys()),
                )
                if _inline_subproject:
                    _inline_files = _prefix_subproject_paths(
                        _inline_files, _inline_subproject, memory)

                # Gate: strip any pseudo-diff markers the planner may have emitted
                from .dependency_check import clean_diff_markers as _clean_diff
                _inline_files = {
                    path: _clean_diff(content)
                    for path, content in _inline_files.items()
                }

                # Capture which targets already exist before overwriting
                import os as _os_inline
                _existing_inline_targets = {
                    p for p in _inline_files if _os_inline.path.exists(p)
                }

                _grounded = getattr(plan_step, "_grounded_edit_targets", None)
                _written_files = executor.write_files(
                    _inline_files, allow_protected=_grounded)
                # A manifest this step CREATED must reach memory. The
                # protected-basename guard exists to stop a hallucinated
                # replacement clobbering a real one, but it tests
                # os.path.isfile() — which is true the moment write_files
                # creates the file, so a brand-new requirements.txt looked
                # pre-existing and was dropped. The content then stayed
                # invisible to dependency checks and context injection for
                # the rest of the run, while the log claimed a skip that
                # protected nothing. _existing_inline_targets was captured
                # BEFORE the write, so it distinguishes the two cases.
                _created_now = {p for p in _inline_files
                                if p not in _existing_inline_targets}
                memory.update(
                    _inline_files,
                    allow_protected=set(_grounded or ()) | _created_now)
                display.step_tokens(step_idx, 0, 0)
                _logger.info(
                    "[PlanStep] Inline code: wrote %d of %d file(s) for "
                    "step %s: %s",
                    len(_written_files), len(_inline_files), plan_step.id,
                    list(_inline_files.keys()),
                )
                if len(_written_files) < len(_inline_files):
                    _logger.warning(
                        "[PlanStep] %d file(s) were NOT written (protected "
                        "or blocked) for step %s — the step may not have "
                        "taken effect",
                        len(_inline_files) - len(_written_files),
                        plan_step.id,
                    )

                # ── Seed scaffold entry-point files into memory ──
                # When writing into a scaffolded subproject, the entry-point
                # file (main.jsx, index.jsx, etc.) may not be touched by any
                # plan step.  If it's not in memory, depcheck can't see it
                # and reports false "orphaned export" gaps for App.jsx.
                # Read it into memory so the import graph is complete.
                _scaff_root = getattr(memory, '_scaffolded_subproject', None)
                if _scaff_root:
                    import os as _os_scaff
                    _entry_names = (
                        'main.jsx', 'main.tsx', 'index.jsx', 'index.tsx',
                        'main.py', 'main.go', 'main.rs', 'index.js', 'index.ts',
                    )
                    for _ename in _entry_names:
                        _epath = f"{_scaff_root}/src/{_ename}"
                        if _epath not in memory.all_files():
                            _full = _os_scaff.path.join('.', _epath)
                            if _os_scaff.path.isfile(_full):
                                try:
                                    with open(_full, 'r', encoding='utf-8',
                                              errors='replace') as _ef:
                                        _econtent = _ef.read()
                                    memory.update({_epath: _econtent})
                                    _logger.info(
                                        "[Scaffold] Seeded entry-point %s "
                                        "into memory (depcheck visibility)",
                                        _epath)
                                except OSError:
                                    pass

                # Deterministic KB content-fix gate for inline code.
                #
                # The planner generates inline code WITH full KB context
                # (e.g. Tailwind v4 docs), so its output is typically correct.
                # Sending it to an LLM reviewer is counterproductive — local
                # models apply outdated training-data bias and "fix" correct
                # code back to v3 patterns.  Instead, apply the same
                # deterministic _apply_content_fixes() rules used in
                # _handle_code_step — these catch known LLM mistakes (e.g.
                # @tailwind directives, wrong plugin names) without LLM calls.
                from .step_handlers import _apply_content_fixes
                _cf = getattr(memory, "_content_fixes", None)
                if _cf:
                    _fixed_inline = _apply_content_fixes(_inline_files, _cf)
                    _changed = [
                        p for p in _inline_files
                        if _fixed_inline.get(p) != _inline_files.get(p)
                    ]
                    if _changed:
                        executor.write_files(
                            {p: _fixed_inline[p] for p in _changed})
                        memory.update(
                            {p: _fixed_inline[p] for p in _changed})
                        display.step_info(
                            step_idx,
                            f"[Inline] Content fixes applied to "
                            f"{len(_changed)} file(s): "
                            f"{', '.join(_changed)}",
                        )
                    else:
                        _logger.debug(
                            "[Inline] Content fixes checked — "
                            "no corrections needed"
                        )

                # NOTE: BrowserRouter injection was removed here — duplicate
                # provider/wrapper issues (Router, ThemeProvider, etc.) are now
                # caught by the language-agnostic post-completion wiring
                # verification (run_wiring_verification), which checks all
                # files together after all CODE steps complete.

                # ── API grounding probe (no LLM cost) ──
                # Verify that module.attr usages in the new Python code exist
                # in the *installed* package versions (probed in the project
                # venv).  Code written against the wrong major version passes
                # lint, review, and tests that never exercise the library —
                # this is the only pre-runtime gate that catches it.
                _api_issues: list[str] = []
                _py_inline = {
                    p: c for p, c in _inline_files.items() if p.endswith(".py")
                }
                if _py_inline:
                    try:
                        from .api_grounding import (
                            probe_api_usage, local_top_levels_from_files)
                        _api_issues = probe_api_usage(
                            _py_inline, executor,
                            local_top_levels=local_top_levels_from_files(
                                memory.all_files().keys()),
                        )
                    except Exception as _probe_exc:
                        _logger.debug(
                            "[ApiGrounding] Probe skipped: %s", _probe_exc)

                # ── Per-step execution check (no LLM cost) ──
                # Actually load each written file in the project environment
                # (import for Python, compile for registered compiled
                # languages).  Catches missing modules, broken relative
                # imports, and module-level crashes the moment they are
                # written instead of at the end of the pipeline.
                try:
                    from .step_verify import verify_step_files
                    _api_issues.extend(verify_step_files(
                        _inline_files, language, executor))
                except Exception as _sv_exc:
                    _logger.debug("[StepVerify] Skipped: %s", _sv_exc)

                if _api_issues:
                    display.step_info(
                        step_idx,
                        f"[ApiGrounding] {len(_api_issues)} API usage "
                        f"issue(s) in inline code — routing to fix loop",
                    )

                # ── Inline code quality gate (Phase 1) ──
                # The planner wrote this code WITH full KB context, so it is
                # typically correct.  We avoid unconditional reviewer LLM calls
                # and instead apply a tiered gate:
                #
                #   Tier A: TEST-step lookahead — tests will validate; skip all.
                #   Tier B: Static lint + import checks (free, no LLM).
                #           Fail → fall back to Coder+Reviewer loop.
                #   Tier C: Existing-file rewrite + full review mode → run
                #           Reviewer LLM to verify the overwrite is correct.
                #           Fail → fall back to Coder+Reviewer loop.
                #   Tier D: All clear → done.  The post-step dependency check
                #           (run_dependency_check at line ~830) already handles
                #           orphaned exports and wiring via its own LLM fix path.
                _has_test_after_inline = False
                if all_plan_steps is not None:
                    _has_test_after_inline = any(
                        s.step_type == "TEST" for s in all_plan_steps
                        if s.index > step_idx
                    )

                if _has_test_after_inline and not _api_issues:
                    # Tier A: TEST follows — tests will validate, skip review.
                    _logger.info(
                        "[Inline] Skipping review for step %s — TEST step follows",
                        plan_step.id if plan_step else step_idx,
                    )

                    # ── Coverage gap check ──
                    # A TEST step exists somewhere after this CODE step, but it
                    # may not import the specific source files written here.
                    # Identify any source files that no future TEST step covers
                    # and proactively generate/update their test files so the
                    # bulk run at end-of-pipeline exercises the new changes.
                    _inline_sources = [
                        p for p in _inline_files if not _is_test_file(p)
                    ]
                    if _inline_sources and all_plan_steps is not None:
                        _covered: set[str] = set()
                        for _ts in all_plan_steps:
                            if _ts.index <= step_idx or _ts.step_type != "TEST":
                                continue
                            for _sp in _inline_sources:
                                if _source_covered_by_test_step(_sp, _ts):
                                    _covered.add(_sp)
                        # Skip files that are inherently untestable or
                        # produce fragile/circular tests:
                        # - Non-code files (README.md, requirements.txt, …):
                        #   nothing to import, the LLM call is wasted
                        # - __main__.py: trivial entry points, runpy+patch fails
                        # - __init__.py: package markers / re-export hubs —
                        #   covered via the modules they re-export; importing
                        #   them directly can drag in heavy deps (GUI libs)
                        # - Config/scaffold files: importing vitest.config
                        #   inside vitest creates circular issues; postcss/
                        #   tailwind/vite configs are validated by the build
                        _CONFIG_SKIP = (
                            'vitest.config', 'vite.config', 'postcss.config',
                            'tailwind.config', 'jest.config', 'vitest.setup',
                            'setupTests', 'babel.config', 'tsconfig',
                            '.eslintrc', 'eslint.config',
                        )
                        _uncovered = [
                            p for p in _inline_sources
                            if p not in _covered
                            and _has_code_ext(p)
                            and not p.endswith('__main__.py')
                            and not p.endswith('__init__.py')
                            and not any(
                                cfg in p.replace("\\", "/").rsplit("/", 1)[-1]
                                for cfg in _CONFIG_SKIP
                            )
                        ]
                        if _uncovered and not getattr(
                                memory, '_task_requests_tests', True):
                            # The user never asked for tests. Auto-generated
                            # per-file coverage tests on such tasks only add
                            # failure surface (observed: 3 unsolicited
                            # component tests turned a passing build into an
                            # 8-turn BulkTest fix cycle over duplicate text).
                            # Plan-declared TEST steps still run — this skips
                            # only the unsolicited extras.
                            _logger.info(
                                "[Inline] Skipping auto test coverage for %s "
                                "— task does not request tests", _uncovered)
                        elif _uncovered:
                            _logger.info(
                                "[Inline] Source files with no TEST-step coverage: %s",
                                _uncovered,
                            )
                            _generate_test_coverage_for_inline_changes(
                                uncovered_files=_uncovered,
                                before_files=_before_files or {},
                                memory=memory,
                                executor=executor,
                                coder=coder,
                                display=display,
                                step_idx=step_idx,
                                language=language,
                                plan_step=plan_step,
                            )
                else:
                    # Tier B: Static lint + import checks
                    from .step_handlers import _quick_offline_lint, _validate_import_paths
                    _inline_lint = _quick_offline_lint(_inline_files)
                    _inline_import_errs = _validate_import_paths(_inline_files, memory)
                    _inline_static_errs = (
                        (_inline_lint + "\n" + _inline_import_errs).strip()
                        if _inline_import_errs else _inline_lint
                    )
                    if _api_issues:
                        _inline_static_errs = (
                            "\n".join(_api_issues) + "\n" + _inline_static_errs
                        ).strip()
                    if _inline_static_errs:
                        display.step_info(
                            step_idx,
                            "[Inline] Static errors found — falling back to Coder+Reviewer loop",
                        )
                        _logger.info(
                            "[Inline] Static check failed for step %s — "
                            "falling back to _handle_code_step: %s",
                            plan_step.id if plan_step else step_idx,
                            _inline_static_errs[:200],
                        )
                        _graph_inline = code_graph
                        if _graph_inline is None and kb_context_builder is not None:
                            _graph_inline = getattr(kb_context_builder, "_graph", None)
                        success, error_info = _handle_code_step(
                            step_text, coder, reviewer, executor,
                            task, memory, display, step_idx,
                            language=language, cfg=cfg,
                            auto=auto, code_graph=_graph_inline,
                            project_profile=project_profile,
                            skip_review=_has_test_after_inline,
                            project_context=project_context,
                            plan_step=plan_step,
                            all_plan_steps=all_plan_steps,
                            kb_context_builder=kb_context_builder,
                            initial_error=(
                                f"Syntax/lint errors in the inline-edited file "
                                f"— fix these:\n{_inline_static_errs}"
                            ),
                            intent_spec=intent_spec,
                        )
                    else:
                        # Tier C: Existing-file rewrite — run Reviewer when in
                        # full review mode so overwritten files are verified.
                        _inline_review_mode = "static"
                        if cfg is not None:
                            _inline_review_mode = getattr(
                                cfg, "REVIEW_MODE", "static"
                            )
                        _should_review_inline = (
                            _inline_review_mode == "full"
                            and bool(_existing_inline_targets)
                        )
                        if _should_review_inline:
                            display.step_info(
                                step_idx,
                                f"[Inline] Reviewing overwrite of "
                                f"{len(_existing_inline_targets)} existing "
                                f"file(s) via Reviewer...",
                            )
                            _inline_review_code = "\n\n".join(
                                f"#### {p}\n```\n{_inline_files[p]}\n```"
                                for p in _existing_inline_targets
                                if p in _inline_files
                            )
                            _kb_ctx_inline = getattr(memory, "_kb_context", "")
                            # Also load step-specific global KB docs (plan_step.kb_docs)
                            # so the reviewer has framework docs like "Tailwind CSS v4
                            # Setup Guide" and doesn't reject valid code based on older
                            # training data.
                            _gstore_inline = getattr(kb_context_builder, '_global_store', None) if kb_context_builder else None
                            _declared_kb_inline = getattr(plan_step, 'kb_docs', None) if plan_step else None
                            if _gstore_inline and _declared_kb_inline:
                                try:
                                    _step_docs_inline = _gstore_inline.get_by_titles(_declared_kb_inline)
                                    if _step_docs_inline:
                                        _step_doc_text = "\n".join(
                                            getattr(d, "content", "") or getattr(d, "title", "")
                                            for d in _step_docs_inline
                                            if getattr(d, "content", "") or getattr(d, "title", "")
                                        )
                                        _kb_ctx_inline = (_kb_ctx_inline + "\n\n" + _step_doc_text).strip()
                                except Exception:
                                    pass
                            _reviewer_kb_inline = (
                                f"\n\n[KB Documentation — trust this over your "
                                f"training data]\n{_kb_ctx_inline}\n"
                                if _kb_ctx_inline else ""
                            )
                            _inline_review_resp = reviewer.process(
                                f"Review this inline code for the step: "
                                f"{step_text}\n\n{_inline_review_code}",
                                context=(
                                    f"Step: {step_text}\n"
                                    f"This code replaces existing file(s). "
                                    f"Verify the replacement is complete and "
                                    f"correct."
                                    f"{_reviewer_kb_inline}"
                                ),
                                language=language,
                            )
                            _inline_review_lower = (
                                _inline_review_resp or ""
                            ).lower()
                            _inline_approved = any(
                                phrase in _inline_review_lower for phrase in (
                                    "code looks good", "looks good",
                                    "no issues", "no critical issues",
                                    "no bugs found", "code is correct",
                                    "functionally correct", "lgtm",
                                )
                            )
                            if _inline_approved:
                                display.step_info(
                                    step_idx,
                                    "[Inline] Reviewer approved existing-file "
                                    "rewrite ✔",
                                )
                                _logger.info(
                                    "[Inline] Reviewer approved inline rewrite "
                                    "for step %s",
                                    plan_step.id if plan_step else step_idx,
                                )
                            else:
                                display.step_info(
                                    step_idx,
                                    "[Inline] Reviewer flagged issues — "
                                    "falling back to Coder+Reviewer loop",
                                )
                                _logger.info(
                                    "[Inline] Reviewer rejected inline rewrite "
                                    "for step %s — falling back: %s",
                                    plan_step.id if plan_step else step_idx,
                                    (_inline_review_resp or "")[:200],
                                )
                                _graph_inline = code_graph
                                if _graph_inline is None and kb_context_builder is not None:
                                    _graph_inline = getattr(
                                        kb_context_builder, "_graph", None
                                    )
                                success, error_info = _handle_code_step(
                                    step_text, coder, reviewer, executor,
                                    task, memory, display, step_idx,
                                    language=language, cfg=cfg,
                                    auto=auto, code_graph=_graph_inline,
                                    project_profile=project_profile,
                                    skip_review=_has_test_after_inline,
                                    project_context=project_context,
                                    plan_step=plan_step,
                                    all_plan_steps=all_plan_steps,
                                    kb_context_builder=kb_context_builder,
                                    intent_spec=intent_spec,
                                )
                        else:
                            # Tier D: Static clean, no existing-file rewrite
                            # concern — accept inline code as-is.
                            # Dependency wiring (orphaned exports) is handled
                            # by run_dependency_check after this block.
                            _logger.info(
                                "[Inline] Static checks passed for step %s — "
                                "accepted (0 reviewer LLM calls)",
                                plan_step.id if plan_step else step_idx,
                            )
            else:
                # ── No inline code (or inline was truncated) ──
                # Phase 2: If the planner's inline code was truncated (token
                # limit), _partial_inline_code holds what was written before
                # the cut-off.  Two strategies:
                #
                #   1. Trivial close: if unmatched braces/parens are small
                #      (≤2 each), close them deterministically — 0 LLM calls.
                #   2. Partial hint: inject the partial code into coder context
                #      so the coder completes rather than regenerates cold.
                #      Skip reviewer (static-only) since the base was planner-
                #      written and only the tail needs filling.
                _partial = getattr(plan_step, '_partial_inline_code', None) if plan_step else None
                _used_trivial_close = False

                if _partial:
                    _closed = _try_trivial_close(_partial, language)
                    if _closed is not None:
                        # Strategy 1: lint first, write only if clean
                        from .dependency_check import clean_diff_markers as _clean_diff_trunc
                        _closed = {p: _clean_diff_trunc(c) for p, c in _closed.items()}
                        from .step_handlers import _quick_offline_lint, _validate_import_paths
                        _trunc_lint = _quick_offline_lint(_closed)
                        _trunc_imp = _validate_import_paths(_closed, memory)
                        if not _trunc_lint and not _trunc_imp:
                            # Lint clean — write and accept
                            _trunc_subproject = _detect_subproject_root(memory)
                            if _trunc_subproject:
                                _closed = _prefix_subproject_paths(
                                    _closed, _trunc_subproject, memory)
                            executor.write_files(_closed)
                            memory.update(_closed)
                            display.step_tokens(step_idx, 0, 0)
                            display.step_info(
                                step_idx,
                                "[Inline/trunc] Trivially closed truncated code (0 LLM calls)",
                            )
                            _logger.info(
                                "[Inline/trunc] Step %s: trivial close succeeded for %s",
                                plan_step.id if plan_step else step_idx,
                                list(_closed.keys()),
                            )
                            _used_trivial_close = True
                            success, error_info = True, ""
                        else:
                            _logger.info(
                                "[Inline/trunc] Trivial close lint failed for step %s "
                                "— falling through to coder with partial hint",
                                plan_step.id if plan_step else step_idx,
                            )

                if _partial and not _used_trivial_close:
                    # Strategy 2: inject partial code as completion hint
                    _logger.info(
                        "[Inline/trunc] Step %s: using partial code as coder hint (%d file(s))",
                        plan_step.id if plan_step else step_idx,
                        len(_partial),
                    )
                    display.step_info(
                        step_idx,
                        "[Inline/trunc] Completing truncated inline code via coder hint",
                    )

                if not _used_trivial_close:
                    # Extract code graph from kb_context_builder if available
                    _graph = code_graph
                    if _graph is None and kb_context_builder is not None:
                        _graph = getattr(kb_context_builder, "_graph", None)

                    # Look ahead: skip LLM review if a TEST step follows OR
                    # if we are completing partial planner code (base was correct)
                    if all_plan_steps is not None:
                        _has_test_after = any(
                            s.step_type == "TEST" for s in all_plan_steps
                            if s.index > step_idx
                        )
                    else:
                        _test_keywords = re.compile(
                            r'\b(test|spec|unit.test|integration.test|jest|vitest|pytest|rspec)\b',
                            re.IGNORECASE,
                        )
                        _has_test_after = any(
                            _test_keywords.search(steps[j])
                            for j in range(step_idx + 1, len(steps))
                        )

                    # Partial hint: skip reviewer — coder is only completing tail
                    _skip_review_for_partial = bool(_partial and not _used_trivial_close)

                    success, error_info = _handle_code_step(
                        step_text, coder, reviewer, executor,
                        task, memory, display, step_idx, language=language, cfg=cfg,
                        auto=auto, code_graph=_graph,
                        project_profile=project_profile,
                        skip_review=_has_test_after or _skip_review_for_partial,
                        project_context=project_context,
                        plan_step=plan_step,
                        all_plan_steps=all_plan_steps,
                        kb_context_builder=kb_context_builder,
                        partial_inline_code=_partial,
                        intent_spec=intent_spec,
                    )

        elif step_type == "TEST":
            # ── Inline test fast path ──
            # If the planner already provided test code in the plan,
            # write it directly and run — zero Tester LLM calls needed.
            if (plan_step is not None
                    and plan_step.inline_code
                    and len(plan_step.inline_code) > 0):
                display.step_info(step_idx, "Writing inline test code from plan (0 LLM calls)")
                _inline_test_files = dict(plan_step.inline_code)
                _inline_test_subproject = _detect_subproject_root(memory)
                _logger.debug(
                    "[Inline/test] subproject=%r inline_keys=%r",
                    _inline_test_subproject, list(_inline_test_files.keys()),
                )
                if _inline_test_subproject:
                    _inline_test_files = _prefix_subproject_paths(
                        _inline_test_files, _inline_test_subproject, memory)

                # Gate: strip any pseudo-diff markers
                from .dependency_check import clean_diff_markers as _clean_diff_t
                _inline_test_files = {
                    path: _clean_diff_t(content)
                    for path, content in _inline_test_files.items()
                }

                # Gate: syntax validation before writing inline test code.
                # Mirrors the Tier B lint check on the CODE inline path —
                # catches truncated files (missing `}`), bad JSX, etc. and
                # falls back to the Tester LLM to regenerate from scratch.
                from .step_handlers import _quick_offline_lint
                _inline_test_lint = _quick_offline_lint(_inline_test_files)
                if _inline_test_lint:
                    _logger.warning(
                        "[Inline/test] Syntax errors in step %s — "
                        "falling back to Tester LLM: %s",
                        plan_step.id if plan_step else step_idx,
                        _inline_test_lint[:300],
                    )
                    display.step_info(
                        step_idx,
                        "[Inline/test] Syntax errors found — falling back to Tester LLM",
                    )
                    # Clear inline code so the else branch runs _handle_test_step
                    plan_step.inline_code.clear()
                    success, error_info = _handle_test_step(
                        step_text, tester, coder, reviewer, executor,
                        task, memory, display, step_idx, language=language,
                        auto=auto, search_agent=search_agent,
                        project_context=project_context,
                        kb_context_builder=kb_context_builder,
                        plan_step=plan_step,
                        all_plan_steps=all_plan_steps,
                        project_profile=project_profile,
                        intent_spec=intent_spec, cfg=cfg)
                else:
                    executor.write_files(_inline_test_files)
                    memory.update(_inline_test_files)
                    display.step_tokens(step_idx, 0, 0)
                    _logger.info(
                        "[PlanStep] Inline test code: wrote %d file(s) for step %s: %s",
                        len(_inline_test_files), plan_step.id,
                        list(_inline_test_files.keys()),
                    )

                    # Deterministic KB content-fix gate (e.g. jest-dom → jest-dom/vitest)
                    from .step_handlers import _apply_content_fixes as _acf_test
                    _cf_test = getattr(memory, "_content_fixes", None)
                    if _cf_test:
                        _fixed_test = _acf_test(_inline_test_files, _cf_test)
                        _changed_test = [
                            p for p in _inline_test_files
                            if _fixed_test.get(p) != _inline_test_files.get(p)
                        ]
                        if _changed_test:
                            executor.write_files(
                                {p: _fixed_test[p] for p in _changed_test})
                            memory.update(
                                {p: _fixed_test[p] for p in _changed_test})
                            display.step_info(
                                step_idx,
                                f"[Inline/test] Content fixes applied to "
                                f"{len(_changed_test)} file(s)",
                            )

                    # Defer test execution — all TEST steps write their files
                    # first; a single bulk run happens after all waves complete.
                    # This avoids redundant parallel runs when multiple TEST steps
                    # are in the same wave and prevents source-fixes for one test
                    # from breaking another test that hasn't run yet.
                    display.step_info(
                        step_idx,
                        "[Inline/test] Test files written — execution deferred to bulk run",
                    )
            else:
                success, error_info = _handle_test_step(
                    step_text, tester, coder, reviewer, executor,
                    task, memory, display, step_idx, language=language,
                    auto=auto, search_agent=search_agent,
                    project_context=project_context,
                    kb_context_builder=kb_context_builder,
                    plan_step=plan_step,
                    all_plan_steps=all_plan_steps,
                    project_profile=project_profile,
                    intent_spec=intent_spec, cfg=cfg)

        elif step_type == "SEARCH":
            success, error_info = _handle_search_step(
                step_text, search_agent, memory, display, step_idx,
                language=language)

        else:
            display.step_info(step_idx, f"Unknown type '{step_type}', skipping.")
            display.complete_step(step_idx, "skipped")

        # Clear plan context for this thread so it doesn't leak into the next step
        from .memory import clear_plan_context_files
        clear_plan_context_files()

        # ── Dependency check: after-snapshot + fix ─────────────────
        # Runs BEFORE complete_step so the spinner stays active during
        # gap detection and LLM fix generation.
        if _before_files is not None and success and step_type not in ("IGNORE",):
            try:
                after_files = memory.all_files()
                new_or_changed = [
                    f for f in after_files
                    if f not in _before_files or _before_files[f] != after_files[f]
                ]
                # Only run if actual source files changed (skip metadata keys)
                new_or_changed = [
                    f for f in new_or_changed if not f.startswith("_")
                ]
                if new_or_changed:
                    display.step_info(step_idx, "Running dependency check...")
                    from .dependency_check import build_snapshot, run_dependency_check
                    dep_before = build_snapshot(_before_files, language)
                    dep_after = build_snapshot(after_files, language)
                    integration_fixes = run_dependency_check(
                        step_idx, step_text, new_or_changed,
                        dep_before, dep_after,
                        memory, llm_client, executor, display, language, cfg,
                        all_plan_steps=all_plan_steps,
                        kb_context=getattr(memory, "_kb_context", "") or "",
                    )
                    if integration_fixes:
                        executor.write_files(integration_fixes)
                        memory.update(integration_fixes)
                        display.step_info(
                            step_idx,
                            f"[DepCheck] Fixed {len(integration_fixes)} file(s) "
                            f"for dependency integration",
                        )
            except Exception as dep_exc:
                _logger.warning("[DepCheck] Post-step check failed: %s", dep_exc)

        # Complete the step AFTER dependency check so spinner stays visible.
        # IGNORE and unknown-type steps already called complete_step above.
        if step_type in ("CMD", "CODE", "TEST", "SEARCH"):
            display.complete_step(step_idx, "done" if success else "failed")

        # --- Structured plan: update step status + actual exports ---
        if plan_step is not None:
            if success:
                try:
                    # Collect files generated in this step
                    after_all = memory.all_files()
                    new_files = {
                        f: c for f, c in after_all.items()
                        if f not in (_before_files or {})
                        or (_before_files or {}).get(f) != c
                    }
                    if new_files:
                        update_step_after_execution(plan_step, new_files)
                    else:
                        plan_step.status = "completed"
                except Exception:
                    plan_step.status = "completed"
            else:
                plan_step.status = "failed"

        # Per-step knowledge upsert (lightweight, no LLM calls)
        # Runs on both success and failure — CMD packages only on success,
        # but CODE/TEST file purposes are recorded regardless.
        if knowledge_base is not None:
            try:
                knowledge_base.record_step_completion(
                    step_type, step_text, step_idx, memory.as_dict(),
                    success=success)
            except Exception as kb_exc:
                _logger.warning("[KB] Per-step upsert failed: %s", kb_exc)

        return step_idx, success, error_info

    except Exception as exc:
        log.error(f"Task {step_idx+1}: Unhandled exception: {exc}")
        display.step_info(step_idx, f"Error: {type(exc).__name__}: {exc}")
        display.complete_step(step_idx, "failed")
        return step_idx, False, f"Unhandled exception: {type(exc).__name__}: {exc}"


# A gate is only ever suspected after this many diagnosis rounds have failed
# against it. One failure is an ordinary red test; a gate that survives two
# rounds of correct-looking fixes is the thing worth doubting.
_GATE_SUSPICION_AFTER = 2


def _consider_gate_superseded(plan_step, passed_cmds, executor, step_idx,
                              diag_attempt) -> bool:
    """Record a proven stand-in when the step's own gate is the defect.

    Twice on hello-world runs the plan's gate named something that did not
    exist (`pytest test_hello.py` against a tester that wrote
    `tests/test_hello_world.py`). Diagnosis said so every round and proposed
    the working command; the pipeline RAN it, saw exit 0 — and then re-ran
    the gate and failed the step. The evidence needed to save the run was
    produced and discarded three times.

    Nothing is believed on the strength of that earlier observation:
    :func:`prove_gate_superseded` re-runs both commands now, and accepts the
    substitution only while the gate still fails and the candidate still
    passes. It must also drive the same instrument, so `echo ok` — which
    names no runner at all — can never stand in for a suite.

    Returns True when a repair was recorded; the ledger and the next gate
    execution pick it up through ``effective_gate``.
    """
    if diag_attempt < _GATE_SUSPICION_AFTER or not passed_cmds:
        return False
    gate = getattr(plan_step, "verify_cmd", None)
    if not gate:
        return False

    from .gate_integrity import (effective_gate, prove_gate_superseded,
                                 record_gate_repair)
    gate = effective_gate(gate)

    def _run(cmd: str):
        return executor.run_command(cmd)

    for candidate in passed_cmds:
        try:
            if prove_gate_superseded(gate, candidate, _run):
                record_gate_repair(
                    gate, candidate,
                    f"diagnosis-equivalent after {diag_attempt} failed rounds")
                log.warning(
                    "Task %d: the step's gate is the defect — it still fails "
                    "while `%s` passes on the same files. Using the proven "
                    "command from here on.", step_idx + 1, candidate)
                return True
        except Exception as exc:            # never let this break the loop
            log.debug("[GateSuspicion] candidate check failed: %s", exc)
    return False


def _run_diagnosis_loop(step_idx: int, step_text: str, error_info: str, *,
                        steps: list[str],
                        llm_client, executor, coder, reviewer, tester,
                        task: str, memory: FileMemory, display: CLIDisplay,
                        language: str | None, cfg=None,
                        auto: bool = False,
                        search_agent=None,
                        kb_context_builder=None,
                        project_profile=None,
                        knowledge_base=None,
                        project_context=None,
                        plan_step: PlanStep | None = None,
                        all_plan_steps: list[PlanStep] | None = None,
                        intent_spec=None,
                        ) -> bool:
    """Run diagnose → fix → retry loop. Returns ``True`` if the step was fixed.

    All exceptions are caught so that a crash during diagnosis (e.g. an
    embedding error) never kills the whole pipeline — the step is simply
    marked as failed and the pipeline halts gracefully.
    """
    # ── Early exit: external service dependency ──────────────────
    # If the failure is due to an unavailable external service (DB,
    # cache, etc.), diagnosis cannot help — inform the user instead.
    service = _detect_external_service_failure(error_info)
    if service:
        msg = (f"Step requires {service} which is not reachable. "
               f"Please ensure the service is running and accessible, "
               f"then re-run the pipeline.")
        display.step_info(step_idx, msg)
        log.warning(f"Task {step_idx+1}: External service unavailable: {service}")
        log.warning(f"Task {step_idx+1}: Skipping diagnosis — "
                    f"this is not a code issue.")
        display.complete_step(step_idx, "skipped")
        return False

    # ── Early exit: missing system tool / environment setup ─────
    # If the failure is because a runtime, package manager, or project
    # config file is missing, editing code won't help.
    sys_issue = _detect_system_level_failure(error_info)
    if sys_issue:
        msg = (f"System dependency missing: {sys_issue}. "
               f"Please install the required tool and re-run the pipeline.")
        display.step_info(step_idx, msg)
        log.warning(f"Task {step_idx+1}: System-level issue: {sys_issue}")
        log.warning(f"Task {step_idx+1}: Skipping diagnosis — "
                    f"this is an environment issue, not a code bug.")
        display.complete_step(step_idx, "failed")
        return False

    # ── Agent-loop recovery (opt-in) ─────────────────────────────
    # One bounded tool-loop attempt REPLACES the diagnose → fix → re-run
    # machinery: the model reads the real error, inspects the project
    # with tools, fixes the cause and completes the step in place.
    from .agent_loop import (
        RECOVERY_FAILED_MARKER, agent_loop_enabled, build_step_tools,
        run_recovery_loop, verify_cmd_for_language,
    )
    if agent_loop_enabled(cfg, llm_client):
        if RECOVERY_FAILED_MARKER in (error_info or ""):
            log.warning(
                f"Task {step_idx+1}: Agent-loop recovery already attempted "
                f"for this failure — not retrying.")
            display.complete_step(step_idx, "failed")
            return False
        display.step_info(step_idx, "Agent loop: recovering from failure")
        _rec_step_type = display.steps[step_idx].get("type", "CODE")
        # The plan-declared gate wins, for every step type. Recovery runs
        # AFTER the main loop failed its gate, so a recovery held to a
        # weaker gate does not recover the step — it redefines success.
        # Observed: a TEST step declaring `python -m unittest -v` (the
        # task's own stated acceptance criterion) failed, recovery was
        # handed the language default instead, pip-installed pytest, went
        # green on `python -m pytest -q`, and the ledger recorded the
        # SUBSTITUTE. `unittest` was never checked again and the run
        # reported Finished. CODE steps used to recover with no gate at
        # all, resting entirely on the model's own summary — an honest
        # "the verification is still failing" that happened not to end
        # with the RECOVERY: blocked marker counted as success.
        from .step_handlers import _declared_verify_cmd
        _rec_verify = _declared_verify_cmd(plan_step, memory, task=task)
        if _rec_verify is None and _rec_step_type == "TEST":
            # No declared gate: fall back to the language-level default so
            # a TEST recovery is still held to something deterministic.
            _rec_sub = _detect_subproject_root(memory)
            _rec_verify = verify_cmd_for_language(language, _rec_sub or ".")
            if _rec_verify and _rec_sub:
                _rec_verify = f"cd {_rec_sub} && {_rec_verify}"
        _rec_tools = build_step_tools(
            executor, memory, kb_context_builder=kb_context_builder)
        try:
            recovered, rec_info = run_recovery_loop(
                llm_client, _rec_tools, step_text, task, error_info,
                display=display, step_idx=step_idx, language=language,
                max_turns=getattr(cfg, "AGENT_LOOP_MAX_TURNS", 8),
                verify_cmd=_rec_verify,
                escalation_client=getattr(coder, "escalation_client", None))
        except Exception as rec_exc:
            # This function promises that a crash during diagnosis fails the
            # STEP, not the run — but the recovery call sat outside the
            # guard. Observed: a model that rejected every tool-calling
            # request raised LLMError out of here and killed the pipeline
            # with a bare traceback, after the very first CODE step.
            log.error(
                "Task %d: Agent-loop recovery raised %s: %s",
                step_idx + 1, type(rec_exc).__name__, rec_exc)
            display.complete_step(step_idx, "failed")
            return False
        if recovered:
            log.info(f"Task {step_idx+1}: Agent loop recovered the step: "
                     f"{rec_info[:200]}")
            # A step that completes via recovery must still put its gate in
            # the monotonic ledger, otherwise nothing rechecks it for the
            # rest of the run. Observed: a recovered CODE step whose gate
            # was never recorded, leaving the guard blind to it entirely.
            # Record exactly the command recovery enforced.
            if _rec_verify and plan_step is not None:
                setattr(plan_step, "_verified_gate_cmd", _rec_verify)
                from .step_handlers import _record_passed_gate
                _record_passed_gate(True, plan_step, memory, task=task)
            display.complete_step(step_idx, "done")
            return True
        log.warning(f"Task {step_idx+1}: Agent-loop recovery failed: "
                    f"{rec_info[:300]}")
        display.complete_step(step_idx, "failed")
        return False

    last_diagnosis_content = None

    # Classify the error once — result is reused across all retry attempts.
    # Uses regex (0 tokens) for common cases; falls back to a tiny single-shot
    # LLM call only for ambiguous errors.
    _step_type_for_route = display.steps[step_idx].get("type", "CODE")
    _kb_matched = False  # refined inside _diagnose_failure after KB lookup
    error_route = classify_error(
        error_info=error_info,
        step_type=_step_type_for_route,
        project_context=project_context,
        kb_matched=_kb_matched,
        llm_client=llm_client if search_agent is not None else None,
    )
    log.info(
        "Step %d: ErrorRouter → source=%s skip_web=%s confidence=%s reason=%s",
        step_idx + 1, error_route.source_type, error_route.skip_web,
        error_route.confidence, error_route.reason,
    )

    # Error the loop is currently working against, used to tell a fix that
    # ADVANCED the step from one that achieved nothing.
    _prev_error_sig = _error_signature(error_info)

    # Best state seen anywhere in this loop, by failing-test count. The
    # signature logic below decides what the NEXT attempt builds on; this
    # decides what the step is left holding if every attempt fails, and the
    # two are not the same question. Measured: a run whose attempt-1 fix
    # took the suite from 9 errors + 1 failure down to 1 failure had that
    # fix reverted, diagnosed two more times from the worse state, and then
    # shipped the 9-error file — a state it had already improved on and
    # committed. Keeping the best snapshot costs one dict copy.
    _best_snapshot = memory.snapshot()
    _best_score = _diagnosis_score(error_info)
    _prev_score = _best_score

    # ── Escalation for the final attempt ─────────────────────────
    # `models.escalation` used to reach the model only through
    # `run_agent_loop_with_escalation`, so with `agent_loop: false` it was
    # dead config: the startup banner announced the stronger model, the
    # classic loop spent all three attempts on the base one, and the run
    # halted having never asked. That reads as "even the strong model could
    # not fix it" when the strong model was never called.
    #
    # Only the LAST attempt escalates. The earlier ones are cheap and often
    # right (one measured run root-caused a typo'd attribute on attempt 1);
    # escalating from the start would pay the premium on every failure.
    _escalation_client = (getattr(coder, "escalation_client", None)
                          or getattr(llm_client, "escalation_client", None))

    for diag_attempt in range(1, MAX_DIAGNOSIS_RETRIES + 1):
        try:
            # Restore snapshot at the start of each attempt so that a
            # bad fix from the previous attempt doesn't compound —
            # but ONLY when that attempt achieved nothing.
            #
            # A gate is usually a chain of asserts, so each fix uncovers
            # the next failing condition. Restoring unconditionally threw
            # away every good fix: observed on a Pac-Man run where attempt
            # 1 correctly fixed `Map.is_walkable`'s arity, attempt 2
            # reverted it and fixed the *next* error instead, and the step
            # halted having never held both fixes at once. Worse, the
            # revert left the file in state A while error_info still
            # described state B, so the second diagnosis reasoned from a
            # premise that no longer matched the disk.
            #
            # FEWER failing tests means the previous fix moved the step
            # forward — keep it and build on it. As many or more means it
            # did not, so revert before trying again. The count is the
            # comparison that has a direction; the signature only says
            # "different", which is why it was wrong in both directions
            # (it kept fixes that took a suite from 4 failures to 39
            # errors, and discarded one that removed nine errors). The
            # signature stays as the fallback for errors no test-runner
            # parser can score, such as a bare traceback from a gate.
            if diag_attempt > 1:
                _cur_sig = _error_signature(error_info)
                _cur_score = _diagnosis_score(error_info)

                # A measured run reverted a correct fix under "changed
                # nothing" while the same two states, reconstructed offline,
                # hashed differently — so the inputs to this decision are
                # worth having in the log the next time it looks wrong.
                log.debug(
                    "Task %d: diagnosis progress check — sig %s→%s, "
                    "score %s→%s, error_info %d chars",
                    step_idx + 1, _prev_error_sig, _cur_sig,
                    _prev_score, _cur_score, len(error_info or ""))

                if _cur_score is not None and _prev_score is not None:
                    _advanced = _cur_score < _prev_score
                    _why = f"{_prev_score} → {_cur_score} failing"
                else:
                    _advanced = _cur_sig != _prev_error_sig
                    _why = "error signature changed" if _advanced else \
                           "error signature unchanged"

                # Record the best state seen BEFORE any revert below can
                # discard it. What the next attempt builds on and what the
                # step is left holding are different questions.
                if _cur_score is not None and (_best_score is None
                                               or _cur_score < _best_score):
                    _best_score = _cur_score
                    _best_snapshot = memory.snapshot()

                if _advanced:
                    log.info(
                        "Task %d: diagnosis attempt %d moved the error on "
                        "(%s) — keeping the fix and building on it",
                        step_idx + 1, diag_attempt - 1, _why)
                else:
                    memory.restore(_best_snapshot, executor=executor)
                    log.info(
                        "Task %d: Restored best file snapshot before "
                        "diagnosis attempt %d (previous fix did not improve "
                        "on it: %s)", step_idx + 1, diag_attempt, _why)
                _prev_error_sig = _cur_sig
                _prev_score = _cur_score

            display.step_info(
                step_idx, f"Diagnosing failure ({diag_attempt}/{MAX_DIAGNOSIS_RETRIES})...")
            log.info(f"Task {step_idx+1}: Diagnosis attempt "
                     f"{diag_attempt}/{MAX_DIAGNOSIS_RETRIES}")

            step_type = display.steps[step_idx].get("type", "CODE")

            _diag_client = llm_client
            if (_escalation_client is not None
                    and diag_attempt == MAX_DIAGNOSIS_RETRIES):
                _diag_client = _escalation_client
                _esc_model = getattr(_escalation_client, "model", "?")
                log.info(
                    "Task %d: final diagnosis attempt — escalating to %s",
                    step_idx + 1, _esc_model)
                display.step_info(
                    step_idx,
                    f"Escalating final diagnosis to {_esc_model}...")

            diagnosis = _diagnose_failure(
                step_text, step_type, error_info,
                memory, _diag_client, display, step_idx,
                search_agent=search_agent, language=language,
                previous_diagnosis=last_diagnosis_content,
                kb_context_builder=kb_context_builder,
                error_route=error_route,
                intent_spec=intent_spec,
                executor=executor)

            # Extract the original failing command from error_info so
            # _apply_fix can filter it out (prevents re-running the same
            # broken command extracted from diagnosis inline backticks).
            import re as _re_diag
            _orig_cmd_match = _re_diag.search(
                r"Command `(.+?)` failed\.", error_info or "")
            _orig_cmd = _orig_cmd_match.group(1) if _orig_cmd_match else None

            _task_goal = getattr(project_context, 'goal_summary', '') if project_context else ''
            _step_targets = plan_step.target_files if plan_step else None
            (fix_applied, cmds_succeeded, has_fix_commands, _fix_cmds_run,
             _fix_cmds_passed) = _apply_fix(
                diagnosis, executor, memory, display, step_idx,
                step_type=step_type,
                original_error_cmd=_orig_cmd,
                step_text=step_text,
                task=_task_goal,
                step_target_files=_step_targets,
                final_attempt=(diag_attempt == MAX_DIAGNOSIS_RETRIES))

            if not fix_applied:
                # ── Test-only retry for TEST steps ──
                # When the diagnosis produced code (it has [EDIT]: or
                # [FILE]: markers) but _apply_fix returned False, the
                # likely cause is the diff guard blocking destructive
                # source rewrites.  For TEST steps, retry with a
                # test-only prompt.
                #
                # We do NOT retry when:
                #  - The diagnosis had no code at all (prose-only) — the
                #    LLM genuinely didn't know the fix
                #  - step_type != "TEST" — source bugs need source fixes
                #  - The diagnosis was a CMD fix (has_fix_commands=True)
                _diag_had_code = (
                    "#### [EDIT]:" in diagnosis
                    or "#### [FILE]:" in diagnosis
                    or "```" in diagnosis
                )
                if step_type == "TEST" and _diag_had_code:
                    _test_targets = (
                        plan_step.target_files if plan_step else None
                    ) or []
                    _test_paths = [
                        t for t in _test_targets
                        if any(seg in t for seg in (
                            '.test.', '.spec.', '__tests__', 'test_'))
                    ]
                    # Fallback: when plan_step has no target_files
                    # (e.g. CMD step reclassified to TEST), discover
                    # test files from memory that are failing.
                    if not _test_paths:
                        _test_paths = [
                            fp for fp in memory.all_files()
                            if any(seg in fp for seg in (
                                '.test.', '.spec.',
                                '__tests__', 'test_'))
                        ]
                    if _test_paths:
                        log.info(
                            "Task %d: Diagnosis had code but nothing "
                            "applied (likely diff guard) — retrying "
                            "with test-only constraint",
                            step_idx + 1)

                        # Build context: step intent, source files
                        # (read-only), and existing test content
                        _dt_step_desc = ""
                        if plan_step and plan_step.description:
                            _dt_step_desc = (
                                f"STEP INTENT: {plan_step.description}\n"
                            )
                        if step_text:
                            _dt_step_desc += (
                                f"STEP DESCRIPTION: {step_text[:1000]}\n"
                            )

                        _dt_briefing = ""
                        if intent_spec:
                            _brief = getattr(
                                intent_spec, 'briefing', '') or ''
                            if _brief:
                                _dt_briefing = (
                                    f"TASK BRIEFING (preserve these "
                                    f"constraints):\n{_brief[:1000]}\n\n"
                                )

                        # Include source files the tests import so the
                        # LLM knows what the components render
                        _dt_source_ctx = ""
                        _all_mem = memory.all_files()
                        for _tp in _test_paths:
                            _tc = _all_mem.get(_tp, "")
                            if _tc:
                                _dt_source_ctx += (
                                    f"CURRENT TEST FILE (to fix):\n"
                                    f"#### [FILE]: {_tp}\n"
                                    f"```\n{_tc}\n```\n\n"
                                )
                        # Add source components as read-only context
                        _dt_imports = (
                            plan_step.imports_from
                            if plan_step else {}
                        )
                        # Fallback: when plan_step has no imports_from
                        # (e.g. CMD step), include non-test source files
                        # from memory so the LLM knows what the
                        # components actually render.
                        if not _dt_imports:
                            _dt_imports = {
                                fp: ""
                                for fp in _all_mem
                                if not any(seg in fp for seg in (
                                    '.test.', '.spec.',
                                    '__tests__', 'test_',
                                    '_cmd_output/',
                                    'node_modules/'))
                                and fp.endswith((
                                    '.jsx', '.tsx', '.js', '.ts',
                                    '.py', '.vue'))
                            }
                        for _imp_path in _dt_imports:
                            _imp_content = _all_mem.get(_imp_path, "")
                            if _imp_content:
                                _dt_source_ctx += (
                                    f"SOURCE FILE (READ-ONLY — do NOT "
                                    f"modify):\n"
                                    f"#### [FILE]: {_imp_path}\n"
                                    f"```\n{_imp_content[:3000]}\n"
                                    f"```\n\n"
                                )

                        _diag_test_prompt = (
                            f"{_dt_briefing}"
                            f"{_dt_step_desc}\n"
                            f"A test step failed. The previous fix "
                            f"attempt was rejected because it tried to "
                            f"rewrite source files too aggressively.\n\n"
                            f"Error:\n{error_info[:3000]}\n\n"
                            f"{_dt_source_ctx}"
                            f"CRITICAL RULES:\n"
                            f"1. Fix ONLY the test file(s). Source "
                            f"files are correct — do NOT modify them.\n"
                            f"2. Do NOT remove or weaken test "
                            f"assertions — the intended functionality "
                            f"must still be verified.\n"
                            f"3. Adapt assertions to match what the "
                            f"source components ACTUALLY render.\n\n"
                            f"Common test fixes:\n"
                            f"- Wrap renders in <MemoryRouter> when "
                            f"components use react-router Link/NavLink\n"
                            f"- Use getAllByRole/getAllByText when "
                            f"multiple elements match (desktop + mobile)\n"
                            f"- Scope queries with within(container)\n\n"
                            f"Test files to fix: {_test_paths}\n\n"
                            f"Return the COMPLETE fixed test file(s) "
                            f"using #### [FILE]: format."
                        )
                        try:
                            _dt_resp = _diag_client.generate_response(
                                _diag_test_prompt)
                            _dt_files = executor.parse_code_blocks(
                                _dt_resp)
                            if not _dt_files:
                                _dt_files = (
                                    executor.parse_code_blocks_fuzzy(
                                        _dt_resp))
                            if _dt_files:
                                # Only accept test files
                                _dt_files = {
                                    fp: fc
                                    for fp, fc in _dt_files.items()
                                    if any(seg in fp for seg in (
                                        '.test.', '.spec.',
                                        '__tests__', 'test_'))
                                }
                            if _dt_files:
                                executor.write_files(_dt_files)
                                memory.update(_dt_files)
                                fix_applied = True
                                log.info(
                                    "Task %d: Diagnosis test-only "
                                    "retry produced fix for: %s",
                                    step_idx + 1,
                                    list(_dt_files.keys()))
                        except Exception as _dt_exc:
                            log.warning(
                                "Task %d: Diagnosis test-only retry "
                                "failed: %s", step_idx + 1, _dt_exc)

            if not fix_applied:
                last_diagnosis_content = diagnosis
                display.step_info(step_idx, "No actionable fix found in diagnosis.")
                log.warning(f"Task {step_idx+1}: Diagnosis produced no actionable fix.")
                continue

            # For CMD steps: if the diagnosis both wrote code fixes AND ran
            # new commands successfully, AND the diagnosis signals that the
            # original command is deprecated/removed, treat the step as
            # resolved.  Re-running a deprecated command will never succeed
            # regardless of how many fixes are applied.
            import re as _re_depr
            _DEPRECATION_RE = _re_depr.compile(
                r'\b(deprecated|removed|no longer|discontinued|obsolete|'
                r'replaced by|use instead|not supported|not available)\b',
                _re_depr.IGNORECASE,
            )
            if (step_type == "CMD"
                    and has_fix_commands and cmds_succeeded and fix_applied
                    and _DEPRECATION_RE.search(diagnosis)):
                display.step_info(
                    step_idx,
                    "Original command is deprecated — fix applied, step resolved.")
                log.info(
                    f"Task {step_idx+1}: CMD step resolved via deprecation-aware fix "
                    f"(code fixes + replacement commands succeeded). "
                    f"Skipping re-run of deprecated original command.")
                return True

            # For CMD steps: if the fix command is a corrected replacement of
            # the original (same core operation, different path/syntax), skip
            # re-running the original — it would just fail again.
            # Detection: strip leading "cd X &&" prefixes and compare the
            # first two words (e.g. "npm install"). If they match, the fix
            # *is* the corrected command, not a prerequisite.
            if (step_type == "CMD"
                    and has_fix_commands and cmds_succeeded and fix_applied
                    and _fix_cmds_run and _orig_cmd):
                import re as _re_repl

                def _cmd_keys(c: str) -> set[str]:
                    segments = [s.strip() for s in _re_repl.split(r'&&|;', c)]
                    keys = set()
                    for seg in segments:
                        parts = seg.split()
                        if not parts: continue
                        cmd = parts[0].lower()
                        if cmd in ('cd', 'mkdir', 'echo', 'set', 'export'):
                            continue
                        keys.add(' '.join(parts[:2]).lower() if len(parts) >= 2 else cmd)
                    return keys

                _orig_keys = _cmd_keys(_orig_cmd)
                if _orig_keys and any(
                        _orig_keys.intersection(_cmd_keys(fc)) for fc in _fix_cmds_run):
                    display.step_info(
                        step_idx,
                        "Fix command replaced original — step resolved.")
                    log.info(
                        f"Task {step_idx+1}: CMD step resolved via corrected replacement "
                        f"command (same operation, fix succeeded). "
                        f"Skipping re-run of original command.")
                    return True

            # Always re-run the original step after applying fixes.
            # Fix commands may be prerequisites (e.g. `npm install` for a
            # missing dependency) rather than replacements for the original
            # command.  Re-running verifies the original intent is satisfied
            # (e.g. tests actually pass, build actually succeeds).
            display.step_info(step_idx, "Fix applied — retrying step...")
            # The plan's inline content is by definition what failed —
            # re-applying it on the re-run clobbers the diagnosis fix
            # (observed: a correct probe-verified rewrite overwritten by
            # the stale inline content seconds later). Clear it so the
            # re-run builds on the fixed files via the coder path.
            if plan_step is not None:
                try:
                    if plan_step.inline_code:
                        log.info(
                            "Task %d: Clearing stale inline content before "
                            "post-diagnosis re-run", step_idx + 1)
                        plan_step.inline_code.clear()
                    if plan_step.inline_edits:
                        plan_step.inline_edits = {}
                except Exception:
                    pass
            _, success, error_info = _execute_step(
                step_idx, step_text,
                steps=steps,
                llm_client=llm_client, executor=executor,
                coder=coder, reviewer=reviewer, tester=tester,
                task=task, memory=memory, display=display,
                language=language, cfg=cfg, auto=auto,
                search_agent=search_agent,
                kb_context_builder=kb_context_builder,
                project_profile=project_profile,
                knowledge_base=knowledge_base,
                project_context=project_context,
                plan_step=plan_step,
                all_plan_steps=all_plan_steps,
                intent_spec=intent_spec,
            )

            if success:
                return True
            else:
                log.warning(f"Task {step_idx+1}: Still failing after "
                            f"diagnosis attempt {diag_attempt}")
                _consider_gate_superseded(
                    plan_step, _fix_cmds_passed, executor, step_idx,
                    diag_attempt)

        except Exception as exc:
            log.error(f"Task {step_idx+1}: Exception during diagnosis "
                      f"attempt {diag_attempt}: {exc}")
            display.step_info(step_idx, f"Diagnosis error: {type(exc).__name__}")
            continue

    # The final attempt's result is never seen by the top-of-loop check
    # above (there is no attempt N+1), and the final attempt is the
    # escalated one — the most likely of the three to be right. A measured
    # run had gpt-5.6-sol root-cause both real defects on attempt 3 and
    # lose the fix here.
    _final_score = _diagnosis_score(error_info)
    if _final_score is not None and (_best_score is None
                                     or _final_score < _best_score):
        _best_score = _final_score
        _best_snapshot = memory.snapshot()

    # Restore files to the BEST state this loop reached, so that
    # destructive edits from failed diagnosis attempts (e.g. overwriting
    # package.json with downgraded versions) don't persist on disk and
    # corrupt future runs or resume attempts. Best, not pre-diagnosis:
    # the step still failed, but shipping a state the loop had already
    # improved on — and in one measured run had already committed — is a
    # loss the run creates for itself.
    memory.restore(_best_snapshot, executor=executor)
    log.info("Task %d: Restored best file snapshot after all diagnosis "
             "attempts failed (%s).", step_idx + 1,
             "no test counts available, pre-diagnosis state"
             if _best_score is None else f"{_best_score} failing test(s)")

    display.step_info(
        step_idx, "Step failed after all fix attempts. Halting pipeline.")
    log.error(f"Task {step_idx+1}: Failed after {MAX_DIAGNOSIS_RETRIES} "
              f"diagnosis attempts. Halting pipeline.")
    return False


# ── Final cross-step test verification ────────────────────────────────────────

_MAX_FINAL_VERIFY_ATTEMPTS = 3


def _lazify_display_imports(content: str) -> str:
    """
    Post-process a source file to prevent test-collection failures caused by
    display-requiring imports (e.g. pygame) at module level.

    Strategy:
    1. Strip any unindented 'import pygame' / 'from pygame import ...' lines.
    2. For every function that references 'pygame.' but has no local
       'import pygame', inject one as the first statement of the function body.

    This is a deterministic guard — the LLM frequently ignores the KB
    instruction to use lazy imports, so we enforce it here instead.
    """
    _DISPLAY_PKGS = ("pygame",)

    def _is_display_import(line: str) -> bool:
        s = line.strip()
        return any(
            s.startswith(f"import {pkg}") or s.startswith(f"from {pkg}")
            for pkg in _DISPLAY_PKGS
        )

    lines = content.splitlines()

    # Fast-exit: nothing to do if there are no display imports at all
    if not any(not ln.startswith((" ", "\t")) and _is_display_import(ln) for ln in lines):
        return content

    # Pass 1 — remove module-level display imports
    lines = [
        ln for ln in lines
        if not (not ln.startswith((" ", "\t")) and _is_display_import(ln))
    ]

    # Pass 2 — inject lazy imports inside functions that use 'pygame.'
    result: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        is_toplevel_def = (
            not line.startswith((" ", "\t"))
            and stripped.startswith("def ")
            and stripped.rstrip().endswith(":")
        )
        if not is_toplevel_def:
            result.append(line)
            i += 1
            continue

        # Collect the entire function block (until next unindented non-empty line)
        func: list[str] = [line]
        i += 1
        while i < len(lines):
            nxt = lines[i]
            if nxt.strip() and not nxt.startswith((" ", "\t")):
                break
            func.append(nxt)
            i += 1

        func_text = "\n".join(func)
        needs_pygame = "pygame." in func_text

        if needs_pygame:
            # Determine body indentation from the first non-empty body line
            body_indent = "    "
            for fl in func[1:]:
                if fl.strip():
                    body_indent = " " * (len(fl) - len(fl.lstrip()))
                    break
            lazy_import = f"{body_indent}import pygame"
            if lazy_import not in func_text:
                # Insert after the def line (skip past any docstring)
                insert_at = 1
                in_docstring = False
                quote = None
                for j, fl in enumerate(func[1:], start=1):
                    s = fl.strip()
                    if not in_docstring:
                        if s.startswith(('"""', "'''")):
                            quote = s[:3]
                            in_docstring = not s.endswith(quote) or len(s) == 3
                            if not in_docstring:
                                insert_at = j + 1
                        else:
                            insert_at = j
                            break
                    else:
                        if s.endswith(quote):
                            in_docstring = False
                            insert_at = j + 1
                func.insert(insert_at, lazy_import)

        result.extend(func)

    return "\n".join(result)


def _extract_failing_test_imports(error_output: str, all_files: dict) -> str:
    """
    Parse ERROR lines from pytest output, find those test files in memory,
    and return their import statements so the LLM knows exactly which symbols
    each failing test needs from the source files.
    """
    import re as _re
    # Match lines like "ERROR collecting test_foo.py"
    failing = _re.findall(r"ERROR collecting (\S+\.py)", error_output)
    if not failing:
        return ""

    lines = []
    for test_fname in failing:
        # Normalise path separators and find the file in memory
        needle = test_fname.replace("\\", "/")
        content = None
        for fpath, fcontent in all_files.items():
            if fpath.replace("\\", "/").endswith(needle):
                content = fcontent
                break
        if content is None:
            continue
        # Collect import lines only
        imports = [
            ln.strip()
            for ln in content.splitlines()
            if ln.strip().startswith(("import ", "from "))
        ]
        if imports:
            lines.append(f"  {needle} imports:\n    " + "\n    ".join(imports))

    if not lines:
        return ""
    return "\n\nSymbols required by failing tests (you MUST export all of these from the source):\n" + "\n".join(lines)


def run_final_test_verification(
    *,
    memory: FileMemory,
    executor,
    coder,
    display: CLIDisplay,
    language: str | None,
    task: str,
    cfg=None,
    project_context=None,
    kb_context_builder=None,
) -> tuple[bool, str]:
    """Re-run all test files generated in this session as a final regression gate.

    Individual TEST steps only verify their own test files in isolation.  When a
    source fix in step N causes tests from step M to regress, the pipeline would
    naively declare success.  This function catches those cross-step regressions
    by running every test file written during the session together, after all
    steps have completed.

    Only runs when there are 2+ distinct test files (a single test file was
    already verified by its own step — no cross-step regression is possible).

    Returns ``(success, error_info)``.
    """
    from ..language import get_test_framework, detect_language_from_files

    # Collect session test files
    all_files = memory.all_files()
    test_files = {
        fpath: content
        for fpath, content in all_files.items()
        if _is_test_file(fpath) and not fpath.startswith("_")
    }

    if len(test_files) <= 1:
        _logger.info(
            "[FinalVerify] %d test file(s) — no cross-step regression possible, skipping.",
            len(test_files),
        )
        return True, ""

    _logger.info(
        "[FinalVerify] Running final regression check on %d test file(s): %s",
        len(test_files), list(test_files.keys()),
    )
    print(f"\n  [FinalVerify] Re-running {len(test_files)} test file(s) for cross-step regression check...")

    # Determine test command — read package.json scripts.test for JS/TS projects
    lang = language
    if lang is None:
        lang = detect_language_from_files(list(test_files.keys()))

    # Detect subproject root (needed for reading package.json)
    subproject_cwd = _detect_subproject_root(memory)

    _test_runner_fv = None
    if lang in ("javascript", "typescript"):
        from .step_handlers import _read_js_project_env
        _js_env_fv = _read_js_project_env(subproject_cwd)
        _test_runner_fv = _js_env_fv.get("test_runner")
        if _test_runner_fv and _test_runner_fv != "jest":
            _logger.info("[FinalVerify] Detected test runner from package.json: %s", _test_runner_fv)

    fw = get_test_framework(lang, test_runner=_test_runner_fv) if lang else get_test_framework("python")
    base_cmd = fw["command"]

    # Vitest fallback override: catch cases where test files import vitest directly
    uses_vitest = False
    if "jest" in base_cmd.lower():
        uses_vitest = any(
            "from 'vitest'" in c or 'from "vitest"' in c
            for c in test_files.values()
        )
        if not uses_vitest:
            _vitest_configs = (
                "vitest.config.js", "vitest.config.ts",
                "vitest.config.mjs", "vitest.config.mts",
            )
            uses_vitest = any(
                any(f.endswith(cfg) for cfg in _vitest_configs)
                for f in all_files
            )
        if uses_vitest:
            base_cmd = "npx vitest run"
            _logger.info("[FinalVerify] Overriding to vitest (import/config fallback)")

    test_cmd = _build_scoped_test_cmd(base_cmd, test_files, subproject_cwd)
    _logger.info("[FinalVerify] Test command: %s", test_cmd)

    last_output = ""
    for attempt in range(1, _MAX_FINAL_VERIFY_ATTEMPTS + 1):
        ok, output = executor.run_command(test_cmd, cwd=subproject_cwd)
        last_output = output
        if ok:
            _logger.info("[FinalVerify] All tests passed on attempt %d.", attempt)
            print(f"  [FinalVerify] All tests passed.")
            return True, ""

        _logger.warning(
            "[FinalVerify] Attempt %d/%d failed:\n%s",
            attempt, _MAX_FINAL_VERIFY_ATTEMPTS, output[:800],
        )

        if attempt == _MAX_FINAL_VERIFY_ATTEMPTS:
            break

        # Ask coder to fix source files only (test files are already correct —
        # they passed during their own steps; only source regressions are at fault)
        print(f"  [FinalVerify] Tests failed — asking coder to fix source files (attempt {attempt})...")
        source_files = {
            fpath: content
            for fpath, content in all_files.items()
            if not _is_test_file(fpath) and not fpath.startswith("_")
        }
        if not source_files:
            _logger.warning("[FinalVerify] No source files to fix — aborting fix loop.")
            break

        context_parts = [
            f"#### [FILE]: {fpath}\n```\n{content}\n```"
            for fpath, content in list(source_files.items())[:6]
        ]
        # Optionally inject KB behavioral instructions into the fix prompt
        kb_instructions = ""
        if kb_context_builder is not None:
            try:
                from ..kb.context_builder import ContextBuilder
                kb_ctx = kb_context_builder.build_context(
                    task_description=task,
                    current_file=None,
                    max_tokens=getattr(cfg, "KB_MAX_CONTEXT_TOKENS", 2000) if cfg else 2000,
                    language=language,
                    step_type="TEST",
                )
                kb_text = kb_context_builder.format_context_for_prompt(kb_ctx)
                if kb_text:
                    kb_instructions = f"\n\nKnowledge base guidance:\n{kb_text}"
            except Exception:
                pass

        failing_imports = _extract_failing_test_imports(output, all_files)

        _fv_briefing = getattr(memory, '_task_briefing', '')
        _fv_briefing_block = (
            "TASK BRIEFING (overall goal — respect Preserve and Key constraint):\n"
            f"{_fv_briefing}\n\n"
        ) if _fv_briefing else ""

        fix_prompt = (
            f"{_fv_briefing_block}"
            f"Task: {task}\n\n"
            f"All individual test steps passed, but running the full test suite together "
            f"revealed a cross-step regression: a source fix for one test broke another.\n\n"
            f"Test command: {test_cmd}\n\n"
            f"Failure output:\n{output[:8000]}\n\n"
            f"Source files (do NOT modify test files — they are correct):\n"
            + "\n\n".join(context_parts)
            + failing_imports
            + kb_instructions
            + "\n\nFix the source file(s) so ALL tests pass."
            + "\n\nIMPORTANT: Preserve ALL existing public symbols (classes, functions, constants) — only add or modify, never remove."
            + "\n\nCRITICAL: NEVER abbreviate or summarize existing code with comments like `// existing code` or `/* unchanged */`. If you are editing a chunk or a file, you MUST write out the ENTIRE content of that chunk or file. Abbreviating code will cause it to be permanently deleted!"
            + "\n\nPrefer CHUNK FORMAT for surgical fixes:\n"
            + "#### [EDIT]: path/to/file.py:function_name (lines start-end)\n```\n// replacement chunk\n```\n"
            + "Use full-file [FILE]: format only when the whole file must be rewritten."
        )
        try:
            fix_response = coder.llm_client.generate_response(fix_prompt)
            # Try chunk edits first (surgical), fall back to full-file
            fix_files = {}
            try:
                from ..editing.chunk_editor import ChunkEditor as _FvCE
                _fv_ce = _FvCE()
                _fv_edits = _fv_ce.parse_chunk_response(fix_response)
                if _fv_edits:
                    for _fv_edit in _fv_edits:
                        _fv_fp = _fv_edit.file_path
                        _fv_existing = memory.get(_fv_fp)
                        if _fv_existing is None:
                            try:
                                with open(_fv_fp, "r", encoding="utf-8", errors="replace") as _f:
                                    _fv_existing = _f.read()
                            except OSError:
                                pass
                        if _fv_existing:
                            try:
                                fix_files[_fv_fp] = _fv_ce.apply_chunk_edits(_fv_existing, [_fv_edit])
                            except Exception:
                                pass
            except ImportError:
                pass
            if not fix_files:
                fix_files = executor.parse_code_blocks(fix_response)
            if not fix_files:
                fix_files = executor.parse_code_blocks_fuzzy(fix_response)
            # Strictly filter: only apply fixes to non-test source files
            fix_files = {
                fpath: content for fpath, content in fix_files.items()
                if not _is_test_file(fpath)
            }
            # Post-process: enforce lazy display imports regardless of LLM output
            fix_files = {
                fpath: _lazify_display_imports(content)
                for fpath, content in fix_files.items()
            }
            if fix_files:
                executor.write_files(fix_files)
                memory.update(fix_files)
                _logger.info("[FinalVerify] Applied source fixes: %s", list(fix_files.keys()))
            else:
                _logger.warning("[FinalVerify] Coder produced no source-only fixes.")
                continue
        except Exception as exc:
            _logger.warning("[FinalVerify] Fix generation failed: %s", exc)
            break

    error_msg = (
        f"Final cross-step test verification failed: {len(test_files)} test file(s) "
        f"did not all pass together after source fixes.\n{last_output[:600]}"
    )
    print(f"  [FinalVerify] FAILED — cross-step regression detected.")
    return False, error_msg


# ---------------------------------------------------------------------------
# Bulk test execution and per-file fix (replaces per-step inline test runs)
# ---------------------------------------------------------------------------

_MAX_BULK_TEST_FIX_ATTEMPTS = 3


def _resolve_django_failed_files(
    output: str,
    subproject_cwd: str | None = None,
) -> list[str]:
    """Resolve Django ERROR:/FAIL: module paths from test output to file paths.

    Django test output reports failures as:
        ERROR: test_foo (app.tests.MyClass.test_foo)
        FAIL:  test_bar (app.tests.test_views.MyClass.test_bar)

    This extracts the dotted module path inside the parentheses, converts it
    to a file path (e.g. ``app/tests.py`` or ``app/tests/test_views.py``),
    and checks whether that file exists on disk.  Returns paths relative to
    the project root (prefixed with *subproject_cwd* when given).

    Useful when the failing test file was NOT written during the current
    session (not in *known_test_files*) so the normal matcher misses it.
    """
    import os as _os
    from .step_handlers import _ANSI_RE
    clean = _ANSI_RE.sub('', output)
    base = subproject_cwd.rstrip("/\\") if subproject_cwd else "."

    # Extract all dotted module paths from Django ERROR/FAIL lines
    modules = re.findall(
        r'(?:ERROR|FAIL):\s+\S+\s+\(([^)]+)\)',
        clean,
        re.MULTILINE | re.IGNORECASE,
    )

    resolved: list[str] = []
    seen: set[str] = set()
    for dotted in modules:
        parts = dotted.split('.')
        # Try progressively shorter prefixes: the class name and method name
        # are the last 1-2 parts; the module is the remainder.
        # e.g. home.tests.HomePageTests.test_foo → try home/tests.py first
        for n in range(len(parts) - 1, 0, -1):
            candidate_rel = '/'.join(parts[:n]) + '.py'
            candidate_abs = _os.path.join(base, candidate_rel)
            if _os.path.isfile(candidate_abs):
                # Return relative to project root (not subproject)
                full_rel = (
                    subproject_cwd.rstrip('/\\') + '/' + candidate_rel
                    if subproject_cwd and subproject_cwd != '.'
                    else candidate_rel
                )
                if full_rel not in seen:
                    seen.add(full_rel)
                    resolved.append(full_rel)
                break

    return resolved


def _django_settings_context(subproject_cwd: str | None) -> str:
    """Return a source context block containing Django settings.py and the
    actual ROOT_URLCONF file for the project.

    When the LLM only sees app-level urls.py files it cannot tell that
    ROOT_URLCONF = "pkg.urls" resolves to ``pkg/urls.py`` — a different file
    from the one it has been editing.  Injecting both files into the fix prompt
    lets the LLM find and fix the real URLconf instead of a decoy.

    Returns an empty string when the project is not Django or files can't be
    found.
    """
    import os as _os
    import re as _re

    base = subproject_cwd or "."

    # Resolve the REAL settings file via DJANGO_SETTINGS_MODULE in manage.py.
    # Django projects often have TWO settings.py files: a root-level stub
    # (ignored by Django) and the real one inside the project package, e.g.
    # bootstrap_homepage/bootstrap_homepage/settings.py.  Reading manage.py
    # is the only reliable way to know which one Django actually loads.
    settings_path: str | None = None
    manage_py = _os.path.join(base, "manage.py")
    if _os.path.isfile(manage_py):
        try:
            with open(manage_py, "r", encoding="utf-8", errors="replace") as _mf:
                manage_content = _mf.read()
            _dsm_m = _re.search(
                r"DJANGO_SETTINGS_MODULE['\"]?\s*,\s*['\"]([^'\"]+)['\"]",
                manage_content,
            )
            if _dsm_m:
                # e.g. "bootstrap_homepage.settings" → bootstrap_homepage/settings.py
                _module = _dsm_m.group(1)
                _rel = _module.replace(".", _os.sep) + ".py"
                _candidate = _os.path.join(base, _rel)
                if _os.path.isfile(_candidate):
                    settings_path = _candidate
        except OSError:
            pass

    # Fallback: root-level settings.py (simple single-file layout)
    if settings_path is None:
        _fallback = _os.path.join(base, "settings.py")
        if _os.path.isfile(_fallback):
            settings_path = _fallback

    if settings_path is None:
        return ""

    try:
        with open(settings_path, "r", encoding="utf-8", errors="replace") as _f:
            settings_content = _f.read()
    except OSError:
        return ""

    ctx = (
        f"#### [FILE]: {settings_path}\n"
        f"```python\n{settings_content}\n```\n\n"
        "NOTE: ROOT_URLCONF above defines the actual Django URL entry point — "
        "make sure you edit THAT file, not a same-named file at a different path.\n\n"
    )

    # Resolve ROOT_URLCONF to a file path and include its current content
    m = _re.search(r'ROOT_URLCONF\s*=\s*["\']([^"\']+)["\']', settings_content)
    if m:
        module = m.group(1)  # e.g. "bootstrap_homepage.urls"
        rel_path = module.replace(".", _os.sep) + ".py"  # bootstrap_homepage/urls.py
        urlconf_abs = _os.path.join(base, rel_path)
        if _os.path.isfile(urlconf_abs):
            # Express the path relative to the project root for the LLM
            urlconf_rel = _os.path.join(
                subproject_cwd.rstrip("/\\"), rel_path
            ) if subproject_cwd else rel_path
            try:
                with open(urlconf_abs, "r", encoding="utf-8", errors="replace") as _uf:
                    urlconf_content = _uf.read()
                ctx += (
                    f"#### [FILE]: {urlconf_rel}\n"
                    f"```python\n{urlconf_content}\n```\n\n"
                    f"NOTE: This is the ROOT_URLCONF file ({module}). "
                    "It must include your app's URLs for reverse() to work.\n\n"
                )
            except OSError:
                pass

    return ctx


def _fix_django_startup_crashes(
    output: str,
    subproject_cwd: str | None,
    executor,
) -> str:
    """Detect and self-heal Django test-runner startup crashes.

    Returns the output of a fresh test run if a fix was applied, otherwise
    returns the original *output* unchanged.

    Currently handles:
    - ``tests.py`` stub + ``tests/`` package coexistence:
      Django raises ``ImportError: 'tests' module incorrectly imported from …``
      when both exist in the same app directory.  Fix: delete the ``.py`` stub.
    """
    import os

    # Pattern: Django incorrectly-imported module conflict
    _conflict_re = re.compile(
        r"ImportError: '(\w+)' module incorrectly imported from '([^']+)'",
        re.IGNORECASE,
    )
    m = _conflict_re.search(output)
    if not m:
        return output

    module_name = m.group(1)          # e.g. "tests"
    package_dir = m.group(2)          # e.g. "/abs/path/to/homepage/tests"

    # The conflicting stub is <parent_dir>/<module_name>.py
    parent_dir = os.path.dirname(package_dir)
    stub_rel_candidates = [
        os.path.join(parent_dir, f"{module_name}.py"),
    ]
    if subproject_cwd:
        stub_rel_candidates.append(
            os.path.join(subproject_cwd, parent_dir, f"{module_name}.py")
        )

    deleted: list[str] = []
    for stub_path in stub_rel_candidates:
        if os.path.isfile(stub_path):
            try:
                os.remove(stub_path)
                deleted.append(stub_path)
                _logger.info(
                    "[BulkTest] Removed conflicting stub '%s' "
                    "(shadowed by '%s/' package)",
                    stub_path, package_dir,
                )
            except OSError as exc:
                _logger.warning(
                    "[BulkTest] Could not remove stub '%s': %s", stub_path, exc
                )

    if not deleted:
        # Couldn't find the stub on disk — log the full error tail so
        # the diagnostic LLM gets the actual ImportError text
        _logger.warning(
            "[BulkTest] Django startup crash (could not auto-fix):\n%s",
            output[-2000:],
        )
        return output

    # Re-run to get fresh output after the fix
    from .step_handlers import get_test_framework, detect_language_from_files
    ok, new_output = executor.run_command("python manage.py test", cwd=subproject_cwd)
    _logger.info(
        "[BulkTest] Post-stub-removal run: exit=%s", "0" if ok else "1"
    )
    return new_output


def _parse_failed_test_files(
    output: str,
    known_test_files: list[str],
    subproject_cwd: str | None = None,
) -> list[str]:
    """Parse test runner output to find which test files failed.

    Matches FAIL lines from vitest/jest/pytest/Django against the known test
    files written during the session, then also resolves Django module paths
    to files on disk (catching failures in pre-existing test files that were
    not written this session).
    """
    from .step_handlers import _ANSI_RE
    clean = _ANSI_RE.sub('', output)
    failed: list[str] = []
    # Normalize to forward-slashes so mixed-separator duplicates are caught
    # (e.g. "foo/bar.jsx" and "foo\bar.jsx" refer to the same file on Windows).
    failed_set: set[str] = set()  # stores normalized paths

    def _norm(p: str) -> str:
        return p.replace("\\", "/")

    for fpath in known_test_files:
        basename = fpath.rsplit('/', 1)[-1].rsplit('\\', 1)[-1]
        stem = basename[:-3] if basename.endswith(".py") else basename
        norm_fpath = _norm(fpath)

        # vitest/jest:  " FAIL src/__tests__/Foo.test.jsx"
        # pytest:       "FAILED tests/test_foo.py::test_bar"
        if re.search(
            r'(?:^|\s)(?:FAIL(?:ED)?)\s.*' + re.escape(basename),
            clean,
            re.MULTILINE | re.IGNORECASE,
        ):
            if norm_fpath not in failed_set:
                failed.append(fpath)
                failed_set.add(norm_fpath)
            continue

        # Django manage.py test:
        #   "ERROR: test_foo (app.tests.test_views.MyTestCase.test_foo)"
        #   "FAIL: test_foo (app.tests.test_views.MyTestCase.test_foo)"
        if re.search(
            r'(?:^|\s)(?:ERROR|FAIL):\s+\S+\s+\([^)]*\b' + re.escape(stem) + r'\b',
            clean,
            re.MULTILINE | re.IGNORECASE,
        ):
            if norm_fpath not in failed_set:
                failed.append(fpath)
                failed_set.add(norm_fpath)

    # Also resolve Django module paths to actual files on disk — catches
    # failures in pre-existing test files not written during this session.
    for extra in _resolve_django_failed_files(output, subproject_cwd):
        if _norm(extra) not in failed_set:
            failed.append(extra)
            failed_set.add(_norm(extra))

    # Fallback: if we couldn't identify specific files but tests failed,
    # treat all known test files as candidates
    if not failed and output:
        failed = list(known_test_files)
    return failed


def _ensure_pytest_available(executor, cwd: str | None = None) -> None:
    """Install pytest into the target environment if it is missing.

    agentchanti runs tests via ``python -m pytest`` regardless of what the
    plan installed, so pytest is a runtime dependency of the pipeline
    itself: a fresh venv that only installed runtime packages fails every
    test run with ``No module named pytest`` — an environment error that
    the test-fix loop (which may only edit test files) can never repair.
    """
    ok, _ = executor.run_command(
        "python -m pytest --version", cwd=cwd, timeout=60)
    if ok:
        return
    _logger.info("[BulkTest] pytest missing in target environment — installing")
    print("  [BulkTest] pytest not installed — installing it now...")
    ok, out = executor.run_command(
        "python -m pip install pytest", cwd=cwd, timeout=300)
    if not ok:
        _logger.warning(
            "[BulkTest] pytest install failed: %s", (out or "")[:300])


# Vitest/Node resolution errors for uninstalled npm packages, e.g.
#   Cannot find package '@testing-library/react' imported from ...
#   Cannot find module 'axios'
# Capture the WHOLE quoted specifier, then decide whether it names a
# package. Matching a package-shaped prefix instead silently truncated
# anything else into something installable: Node says `Cannot find
# module '<absolute path>'` when a FILE is missing, and `[\w.-]+` stops
# at the colon, so `C:\Users\...\run-tests.js` yielded the package
# name "C" and the loop ran `npm install -D C`. Measured 2026-08-19
# run 14. That is the supply-chain hazard `_missing_third_party_module`
# already refuses on the Python side — a name that happens to exist on
# the registry would be installed into the project — and single-letter
# npm packages do exist.
_JS_MISSING_PKG_RE = re.compile(
    r"Cannot find (?:package|module) '([^']+)'")

# npm's own naming rules, tightened to what can appear in an import.
_NPM_NAME_RE = re.compile(r"^(?:@[a-z0-9][\w.-]*/)?[a-z0-9][\w.-]*$",
                          re.IGNORECASE)


def _npm_package_of(spec: str) -> str | None:
    r"""The package *spec* refers to, or None when it is not a package.

    A path is not a package: absolute (`/x`, `C:\x`), relative (`./x`,
    `../x`), or carrying a separator that scoping cannot explain. A deep
    import (`lodash/debounce`) resolves to its package (`lodash`), which
    is what installing it would need.
    """
    spec = (spec or "").strip()
    if not spec or spec[0] in "./~" or "\\" in spec:
        return None
    if re.match(r"^[A-Za-z]:", spec):      # Windows drive letter
        return None
    if spec.startswith("@"):
        parts = spec.split("/")
        pkg = "/".join(parts[:2]) if len(parts) >= 2 else spec
    else:
        pkg = spec.split("/")[0]
    return pkg if _NPM_NAME_RE.match(pkg) else None


def _missing_js_packages(output: str) -> list[str]:
    """npm package names whose absence made the JS test run fail.

    Relative-path imports (``./Header``) are source problems, not
    packages — the regex's character class already excludes anything
    starting with ``.`` or ``/``.
    """
    seen: list[str] = []
    for m in _JS_MISSING_PKG_RE.finditer(output):
        pkg = _npm_package_of(m.group(1))
        if pkg and pkg not in seen:
            seen.append(pkg)
    return seen


def _missing_third_party_module(output: str, project_files) -> str | None:
    """Extract a missing-module name from test output — unless it is a
    project-local module (a sys.path/code problem; pip-installing a name
    that happens to exist on PyPI would be wrong and a supply-chain risk).
    """
    m = re.search(r"No module named ['\"]?([\w\.]+)", output or "")
    if not m:
        return None
    mod = m.group(1).split(".")[0]
    from .api_grounding import local_top_levels_from_files
    if mod in local_top_levels_from_files(project_files):
        return None
    for p in project_files:
        pn = p.replace("\\", "/")
        if pn.startswith(f"{mod}/") or f"/{mod}/" in f"/{pn}":
            return None
        # A bare `import game` that resolves to a project `*/game.py` file is
        # local: failing to import it is a sys.path problem, never a missing
        # PyPI package. Pip-installing the name (a real `game` package exists
        # on PyPI) is a dependency-confusion hazard — observed clobbering a
        # project's own module with an unrelated download.
        if pn.rsplit("/", 1)[-1] == f"{mod}.py":
            return None
    return mod


def _plan_declared_suite_cmd(all_plan_steps, test_files) -> str | None:
    """The single verify command the plan declared for the collected TEST
    files, or ``None`` when zero / more than one distinct command applies.

    BulkTest otherwise substitutes the framework default runner (pytest for
    Python), which can resolve imports *differently* than the command the
    planner declared and the agent loop already passed: bare sibling imports
    inside a ``src/`` package resolve under ``unittest discover -s src`` but
    not under ``pytest`` run from the repo root. Preferring the declared
    command avoids manufacturing an import failure the suite never had.

    Only returned when one command covers *every* collected test file — a
    single per-file command would silently skip the others.
    """
    if not all_plan_steps:
        return None
    want = {f.replace("\\", "/") for f in test_files}
    cmds: set[str] = set()
    covered: set[str] = set()
    for ps in all_plan_steps:
        cmd = getattr(ps, "verify_cmd", None)
        if not cmd:
            continue
        for tf in getattr(ps, "target_files", []):
            tfn = tf.replace("\\", "/")
            if tfn in want:
                cmds.add(" ".join(cmd.split()))
                covered.add(tfn)
    if len(cmds) == 1 and covered == want:
        return next(iter(cmds))
    return None


def declared_gate_cwd(cmd: str, subproject_cwd: str | None) -> str | None:
    """Where to launch a plan-declared gate: None means the repo root.

    A gate that opens with its own ``cd`` already knows where to run, and
    handing it the sub-project cwd as WELL applies the prefix twice.
    Observed: `cd react-home && npm test -- --run` launched from
    `react-home/` looked for `react-home/react-home`, cmd.exe answered
    "The system cannot find the path specified" with exit 1, and BulkTest
    logged a spurious "Plan-declared gate did not pass" — demoting a
    perfectly good gate to the framework default, which is precisely the
    substitution this preflight exists to prevent. Every other caller ran
    the same command from the repo root and it passed.

    The test SHARES `references_subproject` with `_declared_verify_cmd`,
    which decides the mirror-image question — whether to ADD a `cd {sub}`
    prefix — so the two cannot disagree about what "self-locating" means.
    They did disagree while this checked only for a leading `cd `:
    `npm --prefix frontend test -- --run && npm --prefix backend test`
    names the sub-project without cd-ing to it, so it was launched with
    cwd=frontend, where `--prefix frontend` means `frontend/frontend`.
    Measured 2026-08-19: the gate died four times (0xFFFFF026) and
    BulkTest logged "Plan-declared gate did not pass", demoting a correct
    gate to the framework default — the substitution this preflight
    exists to prevent. The identical command passed from the repo root
    immediately before and after.
    """
    if not cmd:
        return subproject_cwd
    if cmd.lstrip().lower().startswith("cd "):
        return None
    from .step_handlers import references_subproject
    if references_subproject(cmd, subproject_cwd):
        return None
    return subproject_cwd


def run_bulk_test_execution_and_fix(
    *,
    memory: FileMemory,
    executor,
    coder,
    display: CLIDisplay,
    language: str | None,
    task: str,
    cfg=None,
    project_context=None,
    kb_context_builder=None,
    all_plan_steps=None,
    search_agent=None,
) -> tuple[bool, str]:
    """Run all session test files in a single bulk execution, then fix failures
    one test file at a time.

    This replaces the per-step inline test runs that used to fire immediately
    after each TEST step wrote its files.  By deferring execution until all
    test files are written:

      - Parallel TEST steps in the same wave no longer race to run the full
        suite simultaneously.
      - A source-file fix for one failing test cannot break another test
        before it has been verified.
      - Total LLM calls are reduced because a single diagnosis loop handles
        all failures rather than one loop per step.

    Fix strategy: run all tests → collect failed files → for each failed file
    ask the coder to fix it (or its imported source) → re-run that single file
    → move to the next.  A final run-all confirms everything passes.

    Returns ``(success, error_info)``.
    """
    from ..language import get_test_framework, detect_language_from_files
    from .step_handlers import (
        _extract_file_specific_errors,
        _extract_imported_sources,
        _ANSI_RE,
        _parse_test_counts,
    )

    # Build test-file → kb_docs mapping from planner-declared step metadata.
    # Used in the fix loop to skip broad KB search when the plan already
    # specifies exactly which docs apply to each test file.
    _step_kb_docs: dict[str, list[str]] = {}
    _step_descs: dict[str, str] = {}
    if all_plan_steps:
        for _ps in all_plan_steps:
            _declared = getattr(_ps, 'kb_docs', None)
            _desc = getattr(_ps, 'description', None)
            for _tf in getattr(_ps, 'target_files', []):
                if _declared:
                    _step_kb_docs[_tf] = _declared
                if _desc:
                    _step_descs[_tf] = _desc

    all_files = memory.all_files()
    # Deduplicate test files by normalized path — mixed separators (foo/bar vs foo\bar)
    # can produce the same basename twice, causing double-processing of one file.
    _seen_norm: set[str] = set()
    test_files: dict[str, str] = {}
    for fpath, content in all_files.items():
        if not (_is_test_file(fpath) and not fpath.startswith("_")):
            continue
        _np = fpath.replace("\\", "/")
        if _np not in _seen_norm:
            _seen_norm.add(_np)
            test_files[fpath] = content

    if not test_files:
        _logger.info("[BulkTest] No test files found — skipping bulk run.")
        return True, ""

    _logger.info(
        "[BulkTest] Running bulk test execution on %d file(s): %s",
        len(test_files), list(test_files.keys()),
    )
    print(f"\n  [BulkTest] Running all {len(test_files)} test file(s)...")

    # Detect test command
    subproject_cwd = _detect_subproject_root(memory)
    lang = language
    if lang is None:
        lang = detect_language_from_files(list(test_files.keys()))

    # For JS/TS projects, read package.json to detect the actual test runner
    # (covers Angular/ng, Karma, Mocha, Vitest, Jest, etc. without hardcoding).
    import os as _os_bt
    _test_runner = None
    if lang in ("javascript", "typescript"):
        from .step_handlers import _read_js_project_env
        _js_env = _read_js_project_env(subproject_cwd)
        _test_runner = _js_env.get("test_runner")
        if _test_runner and _test_runner != "jest":
            _logger.info("[BulkTest] Detected test runner from package.json: %s", _test_runner)

    fw = get_test_framework(lang, test_runner=_test_runner) if lang else get_test_framework("python")
    base_cmd = fw["command"]

    # Django project detection: prefer manage.py test over pytest
    if (not lang or lang == "python") and _os_bt.path.isfile(
        _os_bt.path.join(subproject_cwd or "", "manage.py")
    ):
        base_cmd = "python manage.py test"
        _logger.info("[BulkTest] Django project detected — using 'python manage.py test'")

    # Vitest fallback override: catch cases where _read_js_project_env
    # didn't detect vitest but test files import from it directly.
    uses_vitest = False
    if "jest" in base_cmd.lower():
        uses_vitest = any(
            "from 'vitest'" in c or 'from "vitest"' in c
            for c in test_files.values()
        )
        if not uses_vitest:
            _vitest_cfgs = (
                "vitest.config.js", "vitest.config.ts",
                "vitest.config.mjs", "vitest.config.mts",
            )
            uses_vitest = any(
                any(f.endswith(vc) for vc in _vitest_cfgs)
                for f in all_files
            )
        if uses_vitest:
            base_cmd = "npx vitest run"
            _logger.info("[BulkTest] Overriding to vitest (import/config fallback)")

    # Deterministic vitest environment: DOM-testing suites need jsdom, the
    # testing-library packages, and a jsdom-enabled config. Planners emit
    # this setup unreliably (or not at all) — bootstrap it here so the
    # first run doesn't fail on a missing environment.
    if "vitest" in base_cmd.lower():
        from .step_handlers import ensure_vitest_env
        ensure_vitest_env(executor, subproject_cwd, test_files, memory=memory)

    # ── Preflight: honor the plan's declared TEST verify command ─────────
    # The planner declares an acceptance gate per TEST step and the agent
    # loop already gated the step on it. Substituting the framework default
    # runner here can flip a passing suite to failing purely by changing
    # import roots (e.g. bare `from game import Game` in a src/ package
    # resolves under `unittest discover -s src` but not `pytest` from the
    # repo root). Run the declared command first; only fall back to the
    # framework runner if it genuinely does not pass.
    from .wave_snapshots import is_abnormal_exit
    _declared_suite = _plan_declared_suite_cmd(all_plan_steps, test_files)
    if _declared_suite and _declared_suite != " ".join(base_cmd.split()):
        # A gate that opens with its own `cd` already knows where to run.
        # Handing it the sub-project cwd as WELL applies the prefix twice:
        # `cd react-home && npm test` launched from `react-home/` looks for
        # `react-home/react-home`, and cmd.exe answers "The system cannot
        # find the path specified" with exit 1. Observed as a spurious
        # "Plan-declared gate did not pass", demoting a perfectly good gate
        # to the framework default — the exact substitution the preflight
        # exists to avoid. Every other caller ran this same command from
        # the repo root and it passed.
        _pf_cwd = declared_gate_cwd(_declared_suite, subproject_cwd)
        _logger.info("[BulkTest] Running plan-declared suite gate first: %s",
                     _declared_suite)
        _pf_ok, _pf_out = executor.run_command(
            _declared_suite, cwd=_pf_cwd)
        if not _pf_ok and is_abnormal_exit(
                getattr(executor, "last_exit_code", None)):
            # The process died rather than reporting failures — on Windows
            # a pygame suite fast-fails (0xC0000409) or access-violates
            # (0xC0000005) roughly one invocation in three. Believing that
            # demotes the plan's declared gate to the framework default,
            # which is a different command with different import roots;
            # observed flipping a suite that had just passed four times
            # into two failures. Retry once, exactly as the gate ledger
            # does, before falling back.
            from .wave_snapshots import (describe_abnormal_exit,
                                         log_crash_diagnostics)
            _code = getattr(executor, "last_exit_code", None)
            _logger.warning(
                "[BulkTest] Plan-declared gate terminated abnormally "
                "(%s) — retrying once before believing it: %s",
                describe_abnormal_exit(_code) or _code, _declared_suite)
            log_crash_diagnostics(_code, _declared_suite)
            _pf_ok, _pf_out = executor.run_command(
                _declared_suite, cwd=_pf_cwd)
        if _pf_ok:
            _logger.info("[BulkTest] Suite passed via plan-declared gate — "
                         "skipping framework re-run.")
            print("  [BulkTest] All tests passed (plan-declared gate).")
            for fpath in test_files:
                display.record_test_result(
                    fpath, passed=1, total=1, failures=[])
            return True, ""
        _logger.info("[BulkTest] Plan-declared gate did not pass — falling "
                     "back to framework runner (%s).", base_cmd)

    # The runner itself is a pipeline dependency, not a project one — make
    # sure it exists before the framework runner needs it. Deliberately
    # AFTER the plan-declared gate: that gate passed on every run of a
    # unittest-based project, so installing pytest up front paid a pip
    # install and a network round-trip per run for a runner never invoked.
    if "pytest" in base_cmd:
        _ensure_pytest_available(executor, cwd=subproject_cwd)

    # ── Step 1: Run all tests ──
    ok, output = executor.run_command(base_cmd, cwd=subproject_cwd)
    if not ok and is_abnormal_exit(getattr(executor, "last_exit_code", None)):
        # Same category error the plan-declared gate above guards against,
        # one command later: the runner DIED rather than reporting
        # failures, so there is no verdict to believe. Observed on a green
        # 10-test pygame suite that access-violated (0xC0000005) inside an
        # iterative BFS — no SDL, no recursion — and passed on a retry with
        # zero code changes. Without this, the crash is read as a real
        # failure and the agent-loop fix path spends a turn "fixing" code
        # that was never broken.
        from .wave_snapshots import (describe_abnormal_exit,
                                     log_crash_diagnostics)
        _code = getattr(executor, "last_exit_code", None)
        _logger.warning(
            "[BulkTest] Test runner terminated abnormally (%s) — retrying "
            "once before believing it: %s",
            describe_abnormal_exit(_code) or _code, base_cmd)
        log_crash_diagnostics(_code, base_cmd)
        ok, output = executor.run_command(base_cmd, cwd=subproject_cwd)
    if ok:
        _logger.info("[BulkTest] All tests passed on first run.")
        print("  [BulkTest] All tests passed.")
        # Record every test file as passed
        for fpath in test_files:
            display.record_test_result(fpath, passed=1, total=1, failures=[])
        return True, ""

    _logger.warning("[BulkTest] Tests failed:\n%s", output[:1000])

    # ── Environment-error self-heal ─────────────────────────────────────
    # "No module named X" is an environment problem — editing test files
    # can never fix it (observed: three LLM fix rounds, all blocked by
    # guardrails, while the real fix was one pip install). Install the
    # missing module and re-run before burning per-file fix budgets.
    if not lang or lang == "python":
        _missing_mod = _missing_third_party_module(output, list(all_files))
        if _missing_mod:
            _logger.info(
                "[BulkTest] Missing third-party module '%s' — installing "
                "into the project environment", _missing_mod)
            print(f"  [BulkTest] Installing missing module: {_missing_mod}")
            _inst_ok, _inst_out = executor.run_command(
                f"python -m pip install {_missing_mod}",
                cwd=subproject_cwd, timeout=300)
            if _inst_ok:
                try:
                    from .api_grounding import refresh_installed_versions
                    refresh_installed_versions(
                        project_context, executor=executor,
                        cwd=subproject_cwd)
                except Exception:
                    pass
                ok, output = executor.run_command(
                    base_cmd, cwd=subproject_cwd)
                if ok:
                    _logger.info(
                        "[BulkTest] All tests pass after installing '%s'.",
                        _missing_mod)
                    print(f"  [BulkTest] All tests passed after installing "
                          f"{_missing_mod}.")
                    for fpath in test_files:
                        display.record_test_result(
                            fpath, passed=1, total=1, failures=[])
                    return True, ""
                _logger.warning(
                    "[BulkTest] Still failing after installing '%s' — "
                    "continuing to fix loop", _missing_mod)
            else:
                _logger.warning(
                    "[BulkTest] Install of '%s' failed: %s",
                    _missing_mod, (_inst_out or "")[:300])

    # ── Environment-error self-heal (JS/TS) ─────────────────────────────
    # Vitest/Jest "Cannot find package 'X'" means an uninstalled npm
    # dependency — editing test code can never fix it (observed: nine LLM
    # fix rounds, some blocked at node_modules guardrails, while the real
    # fix was one `npm install -D @testing-library/react`).
    elif lang in ("javascript", "typescript"):
        _missing_pkgs = _missing_js_packages(output)
        if _missing_pkgs:
            _pkg_list = " ".join(_missing_pkgs)
            _logger.info(
                "[BulkTest] Missing npm package(s) %s — installing into "
                "the project", _missing_pkgs)
            print(f"  [BulkTest] Installing missing package(s): {_pkg_list}")
            _inst_ok, _inst_out = executor.run_command(
                f"npm install -D {_pkg_list}",
                cwd=subproject_cwd, timeout=300)
            if _inst_ok:
                try:
                    from .api_grounding import refresh_installed_versions
                    refresh_installed_versions(
                        project_context, executor=executor,
                        cwd=subproject_cwd, language=lang)
                except Exception:
                    pass
                ok, output = executor.run_command(
                    base_cmd, cwd=subproject_cwd)
                if ok:
                    _logger.info(
                        "[BulkTest] All tests pass after installing %s.",
                        _pkg_list)
                    print(f"  [BulkTest] All tests passed after installing "
                          f"{_pkg_list}.")
                    for fpath in test_files:
                        display.record_test_result(
                            fpath, passed=1, total=1, failures=[])
                    return True, ""
                _logger.warning(
                    "[BulkTest] Still failing after installing %s — "
                    "continuing to fix loop", _pkg_list)
            else:
                _logger.warning(
                    "[BulkTest] npm install of %s failed: %s",
                    _pkg_list, (_inst_out or "")[:300])

    # ── Startup crash detection (Django only) ──────────────────────────────
    # Django raises ImportError / ModuleNotFoundError at collection time when
    # there is a `tests.py` stub AND a `tests/` package in the same directory.
    # This crashes the whole run before any test executes, so _parse_failed_test_files
    # returns the unhelpful fallback (all known files).  Detect the pattern and
    # self-heal before falling into the per-file fix loop.
    if "manage.py" in base_cmd:
        output = _fix_django_startup_crashes(output, subproject_cwd, executor)

    # ── Agent-loop first-line fix (opt-in) ──────────────────────────────
    # One grounded loop attempt before the per-file coder machinery: the
    # model gets the real runner output and the exact exit criterion.
    # (Observed without this: three per-file fix rounds rewriting
    # templates and __init__ files — all blocked as destructive — while
    # the actual bug was one wrong import root in the test file.)
    from .agent_loop import (
        agent_loop_enabled, build_step_tools, run_recovery_loop,
    )
    _bt_llm = getattr(coder, "llm_client", None)
    if agent_loop_enabled(cfg, _bt_llm):
        _bt_verify = (f"cd {subproject_cwd} && {base_cmd}"
                      if subproject_cwd else base_cmd)
        _bt_tools = build_step_tools(
            executor, memory, kb_context_builder=kb_context_builder)
        _logger.info("[BulkTest] Agent-loop fix attempt before per-file loop")
        print("  [BulkTest] Agent loop attempting a grounded fix...")
        recovered, rec_info = run_recovery_loop(
            _bt_llm, _bt_tools,
            step_text=f"Make the project's test suite pass: `{_bt_verify}`",
            task=task,
            error_info=(output or "")[-4000:],
            display=None, step_idx=0, language=lang,
            max_turns=getattr(cfg, "AGENT_LOOP_MAX_TURNS", 8),
            verify_cmd=_bt_verify,
            escalation_client=getattr(coder, "escalation_client", None))
        if recovered:
            _logger.info("[BulkTest] Agent loop fixed the suite: %s",
                         rec_info[:200])
            print("  [BulkTest] All tests passed (agent-loop fix).")
            for fpath in test_files:
                display.record_test_result(fpath, passed=1, total=1,
                                           failures=[])
            return True, ""
        _logger.warning("[BulkTest] Agent-loop fix failed (%s) — "
                        "falling back to per-file fix loop",
                        rec_info[:200])
        # Refresh the failure output for the classic path below.
        ok, output = executor.run_command(base_cmd, cwd=subproject_cwd)
        if ok:
            for fpath in test_files:
                display.record_test_result(fpath, passed=1, total=1,
                                           failures=[])
            return True, ""

    # ── Step 2a: Fix shared root causes before per-file loop ─────────────────
    # When all (or most) test files fail with the same error pointing at a
    # shared config file (vitest.config.js, jest.config.js, vite.config.js,
    # setup files, etc.), fix that one file first rather than burning
    # per-file fix budgets on the same root cause 3× per test.
    failed_files = _parse_failed_test_files(output, list(test_files.keys()), subproject_cwd)
    if len(failed_files) > 1:
        _shared_fix_applied = False
        # Extract the first error line that references a non-test config file
        _CONFIG_FILE_RE = re.compile(
            r'[\w./\\-]*(vitest\.config|vite\.config|jest\.config|'
            r'webpack\.config|babel\.config|setup\w*)[\w./\\-]*',
            re.IGNORECASE,
        )
        _error_lines = output.splitlines()
        _shared_config: str | None = None
        for _el in _error_lines[:60]:  # scan first 60 lines of test output
            _cm = _CONFIG_FILE_RE.search(_el)
            if _cm:
                _candidate = _cm.group(0).strip().replace("\\", "/")
                # Must look like an actual file path (contains a dot/extension),
                # not a bare keyword like "setup" from test description prose.
                # Also exclude test files themselves (e.g. vitest.setup.test.js).
                if (
                    "." in _candidate
                    and not re.search(r'\.(test|spec)\.[a-z]+$', _candidate, re.IGNORECASE)
                ):
                    _shared_config = _candidate
                    break

        if _shared_config:
            # Count how many failing test files mention this config in the output
            _mentioning = sum(
                1 for _tf in failed_files
                if _shared_config in output or _shared_config.split("/")[-1] in output
            )
            if _mentioning >= len(failed_files):
                _logger.info(
                    "[BulkTest] Shared root cause detected — %s referenced in all "
                    "%d failing test outputs. Fixing shared config first.",
                    _shared_config, len(failed_files),
                )
                print(f"  [BulkTest] Shared config error in {_shared_config} — fixing once...")
                # Read the current config file content
                _cfg_candidates = [
                    _shared_config,
                    _os_bt.path.join(subproject_cwd or "", _shared_config.split("/")[-1]),
                    _os_bt.path.join(subproject_cwd or "", _shared_config),
                ]
                _cfg_content = ""
                _cfg_path = ""
                for _cp in _cfg_candidates:
                    try:
                        with open(_cp, "r", encoding="utf-8", errors="replace") as _cf:
                            _cfg_content = _cf.read()
                            _cfg_path = _cp
                            break
                    except OSError:
                        pass
                if _cfg_content:
                    _shared_fix_prompt = (
                        f"Task: {task}\n\n"
                        f"All test files are failing due to an error in the shared "
                        f"config file `{_cfg_path}`.\n\n"
                        f"Error output:\n{output[:2000]}\n\n"
                        f"Current content of `{_cfg_path}`:\n"
                        f"```\n{_cfg_content}\n```\n\n"
                        f"Fix ONLY `{_cfg_path}` so tests can run. "
                        f"Do NOT touch any test files or component files.\n\n"
                        f"IMPORTANT: Output only code — no prose, no markdown headers.\n"
                        f"Use full-file format:\n"
                        f"#### [FILE]: {_cfg_path}\n"
                        f"```\n// fixed content\n```"
                    )
                    try:
                        _shared_fix_resp = coder.llm_client.generate_response(_shared_fix_prompt)
                        _shared_fix_files = executor.parse_code_blocks(_shared_fix_resp)
                        if not _shared_fix_files:
                            _shared_fix_files = executor.parse_code_blocks_fuzzy(_shared_fix_resp)
                        if _shared_fix_files:
                            if subproject_cwd:
                                from .step_handlers import _prefix_subproject_paths
                                _shared_fix_files = _prefix_subproject_paths(
                                    _shared_fix_files, subproject_cwd, memory)
                            executor.write_files(_shared_fix_files)
                            memory.update(_shared_fix_files)
                            _logger.info(
                                "[BulkTest] Shared config fix applied: %s",
                                list(_shared_fix_files.keys()),
                            )
                            # Re-run all tests to see if the shared fix resolved things
                            _ok_shared, output = executor.run_command(base_cmd, cwd=subproject_cwd)
                            if _ok_shared:
                                _logger.info("[BulkTest] Shared config fix resolved all failures.")
                                print("  [BulkTest] All tests pass after shared config fix.")
                                return True, ""
                            # Update failed_files with whatever remains
                            failed_files = _parse_failed_test_files(
                                output, list(test_files.keys()), subproject_cwd)
                            _shared_fix_applied = True
                            _logger.info(
                                "[BulkTest] After shared fix, %d file(s) still failing: %s",
                                len(failed_files), failed_files,
                            )
                    except Exception as _sfe:
                        _logger.warning("[BulkTest] Shared config fix failed: %s", _sfe)

    # ── Step 2: Fix one failing test file at a time ──
    failed_files = _parse_failed_test_files(output, list(test_files.keys()), subproject_cwd)
    _logger.info("[BulkTest] Failed test files: %s", failed_files)
    print(f"  [BulkTest] {len(failed_files)} test file(s) failed — fixing one at a time...")

    # Show initial pass/fail state in TEST RESULTS immediately
    _failed_set = set(failed_files)
    for fpath in test_files:
        if fpath in _failed_set:
            _p, _t, _f = _parse_test_counts(output)
            display.record_test_result(fpath, passed=_p, total=_t, failures=_f)
        else:
            display.record_test_result(fpath, passed=1, total=1, failures=[])

    lang_tag = lang or "python"

    # Track fix content hashes per test file to prevent the loop from
    # re-applying an identical (failed) fix across attempts.
    # Keys are test file paths; values are sets of hex content digests.
    import hashlib as _hashlib
    _applied_fix_signatures: dict[str, set[str]] = {}

    # Use an index-based loop so we can append newly-impacted test files
    # to failed_files mid-iteration without losing them.
    fix_idx = 0
    while fix_idx < len(failed_files):
        test_path = failed_files[fix_idx]
        fix_idx += 1
        basename = test_path.rsplit('/', 1)[-1]
        print(f"  [BulkTest] Fixing {basename}...")

        current_output = output  # use full output for first attempt
        _no_code_last_attempt = False   # True when LLM returned prose but no code
        _test_rewrite_done = False      # True after a test-rewrite pivot was attempted

        # Escape-hatch state — see _attempt_targeted_source_fix.
        # _did_test_only_retry replaces the broken `getattr(int, ...)` flag.
        _did_test_only_retry = False
        _used_escape_hatch = False
        _hatch_snap: dict[str, str] | None = None
        _error_sig_history: list[str] = []

        # ── ErrorRouter: classify this test failure once per file ────────────
        # Determines whether web search is worth calling before fixing.
        # Computed from the initial error output (before any fix is applied).
        _initial_error = _extract_file_specific_errors(output, test_path, max_chars=2000)
        if not _initial_error:
            _initial_error = output[-2000:]
        _route = None
        _bulk_search_context = ""
        try:
            from .error_router import classify_error as _classify_error
            _route = _classify_error(
                error_info=_initial_error,
                step_type="TEST",
                project_context=project_context,
                kb_matched=False,
                llm_client=coder.llm_client if search_agent is not None else None,
            )
            log.info(
                "[BulkTest] ErrorRouter %s → source=%s skip_web=%s (%s)",
                basename, _route.source_type, _route.skip_web, _route.reason,
            )
            if search_agent is not None and not _route.skip_web:
                _lang = language or (
                    getattr(project_context, "language", None) if project_context else None)
                _kb_ctx = getattr(memory, "_kb_context", "")
                _bulk_search_context = search_agent.search_for_error(
                    _initial_error, test_path,
                    language=_lang,
                    kb_context=_kb_ctx,
                    query_override=_route.query_hint or None,
                )
                if _bulk_search_context:
                    log.info("[BulkTest] Search context injected for %s", basename)
        except Exception as _re_exc:
            log.debug("[BulkTest] ErrorRouter failed for %s: %s", basename, _re_exc)

        for fix_attempt in range(1, _MAX_BULK_TEST_FIX_ATTEMPTS + 1):
            # Extract error relevant to this file
            file_error = _extract_file_specific_errors(
                current_output, test_path, max_chars=3000)
            if not file_error:
                # Take the tail (where tracebacks and ImportErrors appear)
                # rather than the head (which is often setup noise).
                file_error = current_output[-3000:]

            # Track normalised error shape for the escape-hatch trigger:
            # if the same signature appears on consecutive attempts the
            # test-only retries are not converging.
            _error_sig_history.append(_error_signature(file_error))

            # Build source context for this test file
            current_content = memory.all_files().get(test_path, "")
            if not current_content:
                # Pre-existing file not tracked in session memory — read from disk
                try:
                    with open(test_path, "r", encoding="utf-8", errors="replace") as _tf:
                        current_content = _tf.read()
                except OSError:
                    pass
            imported_sources = _extract_imported_sources(
                {test_path: current_content}, memory,
                resolve_from_disk=True)

            # Also resolve 2nd-level imports (files imported by the direct
            # imports) so the LLM can see all relevant source components.
            # Capped at 4 extra files to avoid bloating the prompt.
            _second_level: dict[str, str] = {}
            if imported_sources:
                _second_level = _extract_imported_sources(
                    imported_sources, memory, resolve_from_disk=True)
                # Remove anything already in imported_sources or the test file
                for _k in list(_second_level.keys()):
                    if _k in imported_sources or _k == test_path:
                        del _second_level[_k]
                # Keep at most 4 extra files (shortest paths first = most local)
                _second_level = dict(
                    sorted(_second_level.items(), key=lambda kv: len(kv[0]))[:4]
                )

            source_ctx = (
                f"#### [FILE]: {test_path}\n```{lang_tag}\n{current_content}\n```\n\n"
            )
            for fp, cnt in imported_sources.items():
                source_ctx += (
                    f"#### [FILE]: {fp}\n```{lang_tag}\n{cnt}\n```\n\n"
                )
            for fp, cnt in _second_level.items():
                source_ctx += (
                    f"#### [FILE]: {fp}\n```{lang_tag}\n{cnt}\n```\n\n"
                )

            # For Django projects inject settings.py + the real ROOT_URLCONF so
            # the LLM can see the correct file to edit instead of guessing.
            source_ctx += _django_settings_context(subproject_cwd)

            # Inject KB guidance — use planner-declared docs when available
            # (exact title lookup, no semantic search) to avoid injecting
            # irrelevant docs (e.g. Django instructions for a React test).
            # Fall back to build_context() only when the plan has no kb_docs.
            kb_instructions = ""
            if kb_context_builder is not None:
                try:
                    _declared_titles = _step_kb_docs.get(test_path)
                    if _declared_titles:
                        _gstore = getattr(kb_context_builder, '_global_store', None)
                        if _gstore is not None:
                            _kb_results = _gstore.get_by_titles(_declared_titles)
                            if _kb_results:
                                _kb_parts = [
                                    getattr(r, 'content', '') or getattr(r, 'title', '')
                                    for r in _kb_results
                                ]
                                kb_instructions = (
                                    "\n\nKnowledge base guidance:\n"
                                    + "\n".join(p for p in _kb_parts if p)
                                )
                    else:
                        from ..kb.context_builder import ContextBuilder
                        kb_ctx = kb_context_builder.build_context(
                            task_description=task,
                            current_file=test_path,
                            max_tokens=getattr(cfg, "KB_MAX_CONTEXT_TOKENS", 2000) if cfg else 2000,
                            language=lang,
                            step_type="TEST",
                        )
                        kb_text = kb_context_builder.format_context_for_prompt(kb_ctx)
                        if kb_text:
                            kb_instructions = f"\n\nKnowledge base guidance:\n{kb_text}"
                except Exception:
                    pass

            _bt_briefing = getattr(memory, '_task_briefing', '')
            _bt_briefing_block = (
                "TASK BRIEFING (overall goal — respect Preserve and Key constraint):\n"
                f"{_bt_briefing}\n\n"
            ) if _bt_briefing else ""

            fix_prompt = (
                f"{_bt_briefing_block}"
                f"Task: {task}\n\n"
                f"Test file `{test_path}` failed. Fix it so the tests pass.\n\n"
                f"Error output:\n{file_error}\n\n"
                f"Relevant files:\n{source_ctx}"
                f"{kb_instructions}\n\n"
                "You may fix the test file itself OR fix a source file it imports — "
                "whichever is correct.  Do NOT remove any existing tests.\n\n"
                "CRITICAL: Do NOT remove or comment out the tested component/feature.\n\n"
                "IMPORTANT — before modifying any template or source file to satisfy "
                "an assertContains/assertIn/assertEqual test:\n"
                "1. Read the EXACT string literal from the test assertion above.\n"
                "2. Copy that exact string (correct case, spacing, punctuation) into "
                "   the template or source file.\n"
                "3. Do NOT paraphrase, guess, or change the casing of the expected string.\n\n"
                "CRITICAL: NEVER abbreviate or summarize existing code with comments like `// existing code` or `/* unchanged */`. If you are editing a chunk or a file, you MUST write out the ENTIRE content of that chunk or file. Abbreviating code will cause it to be permanently deleted!\n\n"
                "Prefer CHUNK FORMAT for surgical fixes:\n"
                f"#### [EDIT]: path/to/file:{lang_tag}:function_name (lines start-end)\n"
                f"```{lang_tag}\n// replacement chunk\n```\n"
                "Use full-file [FILE]: format only when the whole file must be rewritten."
            )
            if _bulk_search_context:
                fix_prompt += (
                    f"\n\nWeb search context (use to inform your fix):\n"
                    f"{_bulk_search_context}\n"
                )
            if _no_code_last_attempt:
                fix_prompt += (
                    "\n\nCRITICAL: Your previous response contained only explanation text "
                    "with no code changes. You MUST output actual file content using "
                    "#### [FILE]: or #### [EDIT]: markers — no prose-only replies."
                )

            try:
                fix_response = coder.llm_client.generate_response(fix_prompt)
                # Try chunk edits first (surgical), fall back to full-file
                fix_files = {}
                try:
                    from ..editing.chunk_editor import ChunkEditor as _BtCE
                    _bt_ce = _BtCE()
                    _bt_edits = _bt_ce.parse_chunk_response(fix_response)
                    if _bt_edits:
                        for _bt_edit in _bt_edits:
                            _bt_fp = _bt_edit.file_path
                            _bt_existing = memory.get(_bt_fp)
                            if _bt_existing is None:
                                try:
                                    with open(_bt_fp, "r", encoding="utf-8", errors="replace") as _f:
                                        _bt_existing = _f.read()
                                except OSError:
                                    pass
                            # If the LLM used a subproject-relative path (e.g.
                            # "src/components/Foo.jsx" instead of
                            # "react-responsive-page/src/components/Foo.jsx"),
                            # try resolving it via the subproject prefix so the
                            # existing file content can be found and the edit
                            # doesn't silently fall through.
                            if _bt_existing is None and subproject_cwd:
                                _bt_fp_prefixed = f"{subproject_cwd}/{_bt_fp}"
                                _bt_existing = memory.get(_bt_fp_prefixed)
                                if _bt_existing is None:
                                    try:
                                        with open(_bt_fp_prefixed, "r", encoding="utf-8", errors="replace") as _f:
                                            _bt_existing = _f.read()
                                    except OSError:
                                        pass
                                if _bt_existing is not None:
                                    _bt_fp = _bt_fp_prefixed
                                    _bt_edit = type(_bt_edit)(
                                        file_path=_bt_fp,
                                        chunk_id=_bt_edit.chunk_id,
                                        line_start=_bt_edit.line_start,
                                        line_end=_bt_edit.line_end,
                                        new_content=_bt_edit.new_content,
                                        is_new=_bt_edit.is_new,
                                        insert_after=_bt_edit.insert_after,
                                    )
                            if _bt_existing:
                                # Guard: reject full-file replacements that are
                                # suspiciously small compared to the existing file
                                # and lack any function/class definitions — these
                                # are almost always a lone stub like
                                # `export default Foo;` that would destroy the file.
                                _is_full_replace = (
                                    _bt_edit.line_start == 1
                                    and _bt_edit.line_end >= 99999
                                )
                                if _is_full_replace and _bt_existing:
                                    _new_lines = _bt_edit.new_content.splitlines()
                                    _old_lines = _bt_existing.splitlines()
                                    _too_small = len(_new_lines) < max(5, len(_old_lines) * 0.15)
                                    _has_def = any(
                                        kw in _bt_edit.new_content
                                        for kw in (
                                            'function ', 'const ', 'class ',
                                            'def ', 'export default function',
                                            '=>', 'return (', 'return <',
                                        )
                                    )
                                    if _too_small and not _has_def:
                                        _logger.warning(
                                            "[BulkTest] Rejected suspiciously tiny "
                                            "full-file replacement for %s "
                                            "(%d lines → %d lines, no definitions) "
                                            "— skipping to avoid destroying the file.",
                                            _bt_fp, len(_old_lines), len(_new_lines),
                                        )
                                        continue
                                try:
                                    fix_files[_bt_fp] = _bt_ce.apply_chunk_edits(_bt_existing, [_bt_edit])
                                except Exception:
                                    pass
                except ImportError:
                    pass
                if not fix_files:
                    fix_files = executor.parse_code_blocks(fix_response)
                if not fix_files:
                    fix_files = executor.parse_code_blocks_fuzzy(fix_response)

                # ── No code in response — don't re-run unchanged test ──────────
                if not fix_files:
                    _logger.warning(
                        "[BulkTest] No code extracted from LLM response for %s "
                        "(attempt %d/%d) — skipping re-run, retrying with stronger prompt.",
                        basename, fix_attempt, _MAX_BULK_TEST_FIX_ATTEMPTS,
                    )
                    _no_code_last_attempt = True
                    continue

                _no_code_last_attempt = False

                if subproject_cwd:
                    from .step_handlers import _prefix_subproject_paths
                    fix_files = _prefix_subproject_paths(
                        fix_files, subproject_cwd, memory)

                # Dedup: compute a signature of the proposed file contents
                # and skip if this exact fix was already applied for this
                # test file — prevents the loop from burning retries on an
                # identical broken fix.
                _fix_sig = _hashlib.md5(
                    "".join(sorted(fix_files.values())).encode("utf-8", errors="replace")
                ).hexdigest()
                _sigs_for_test = _applied_fix_signatures.setdefault(test_path, set())
                if _fix_sig in _sigs_for_test:
                    _logger.warning(
                        "[BulkTest] Duplicate source fix for %s (attempt %d/%d) "
                        "— pivoting to test rewrite.",
                        basename, fix_attempt, _MAX_BULK_TEST_FIX_ATTEMPTS,
                    )
                    # ── Pivot: rewrite the TEST to match actual implementation ──
                    # The source is likely correct; the test expectation is wrong.
                    if not _test_rewrite_done:
                        _test_rewrite_done = True
                        _rw_content = memory.all_files().get(test_path, "")
                        if not _rw_content:
                            try:
                                with open(test_path, "r", encoding="utf-8",
                                          errors="replace") as _rf:
                                    _rw_content = _rf.read()
                            except OSError:
                                pass
                        _rw_prompt = (
                            f"The source fix for `{test_path}` was already applied "
                            f"but the test still fails.\n\n"
                            f"Rewrite the TEST FILE ITSELF so it matches what the "
                            f"implementation actually does. Do NOT remove any tests — "
                            f"update expected values, selectors, or text to match "
                            f"reality.\n\n"
                            "IMPORTANT GUIDELINES for the rewrite:\n"
                            "- If a test asserts an exact library-internal string (e.g. a "
                            "plugin name like `'vite:react-babel'`, an internal key, or an "
                            "internal version string), use a flexible matcher instead: "
                            "`includes('react')`, `startsWith('vite:')`, or `toContain()`. "
                            "Exact internal names change across library versions.\n"
                            "- For assertions on DOM text or component output, copy the "
                            "exact string from the source/template into the test.\n"
                            "- Do NOT change what is being tested — only update the "
                            "expected values to match what the code actually produces.\n\n"
                            f"Current test file:\n"
                            f"#### [FILE]: {test_path}\n"
                            f"```{lang_tag}\n{_rw_content}\n```\n\n"
                            f"Error output:\n{file_error}\n\n"
                            f"Relevant source:\n{source_ctx}\n\n"
                            "Output the COMPLETE rewritten test file using:\n"
                            f"#### [FILE]: {test_path}\n"
                            f"```{lang_tag}\n// full file\n```"
                        )
                        try:
                            _rw_response = coder.llm_client.generate_response(_rw_prompt)
                            _rw_files = executor.parse_code_blocks(_rw_response)
                            if not _rw_files:
                                _rw_files = executor.parse_code_blocks_fuzzy(_rw_response)
                            if _rw_files:
                                if subproject_cwd:
                                    _rw_files = _prefix_subproject_paths(
                                        _rw_files, subproject_cwd, memory)
                                executor.write_files(_rw_files)
                                memory.update(_rw_files)
                                _logger.info(
                                    "[BulkTest] Test rewrite applied for %s: %s",
                                    basename, list(_rw_files.keys()),
                                )
                                _single_rw = _build_scoped_test_cmd(
                                    base_cmd, {test_path: ""}, subproject_cwd)
                                _ok_rw, current_output = executor.run_command(
                                    _single_rw, cwd=subproject_cwd)
                                if _ok_rw:
                                    _logger.info(
                                        "[BulkTest] %s passes after test rewrite.", basename)
                                    print(f"  [BulkTest] {basename} fixed via test rewrite")
                                    _p, _t, _ = _parse_test_counts(current_output)
                                    display.record_test_result(
                                        test_path, passed=_p, total=_t, failures=[])
                                    break
                                _logger.warning(
                                    "[BulkTest] Test rewrite for %s still failing.",
                                    basename)
                        except Exception as _rw_exc:
                            _logger.warning(
                                "[BulkTest] Test rewrite failed for %s: %s",
                                basename, _rw_exc)
                    break

                _sigs_for_test.add(_fix_sig)
                # ── Source file protection ──
                # BulkTest may modify test files freely. Source files
                # are only allowed through if the change is small and
                # additive (e.g. adding data-testid, aria-label) — it
                # must not alter functionality.
                if fix_files:
                    _bt_filtered = {}
                    for _bt_fp, _bt_fc in fix_files.items():
                        if _is_test_file(_bt_fp):
                            _bt_filtered[_bt_fp] = _bt_fc
                        elif _is_test_infra_file(_bt_fp):
                            _bt_filtered[_bt_fp] = _bt_fc
                            _logger.info(
                                "[BulkTest] Allowed test-infra fix "
                                "for %s", _bt_fp)
                        elif _is_additive_source_fix(
                                _bt_fp, _bt_fc, memory):
                            _bt_filtered[_bt_fp] = _bt_fc
                            _logger.info(
                                "[BulkTest] Allowed small additive "
                                "source fix for %s", _bt_fp)
                        elif _is_safe_source_fix(_bt_fp, _bt_fc, memory):
                            _bt_filtered[_bt_fp] = _bt_fc
                            _logger.info(
                                "[BulkTest] Allowed bounded source fix "
                                "for %s (diff within hatch cap)", _bt_fp)
                        else:
                            _logger.warning(
                                "[BulkTest] Blocked fix for source file "
                                "%s — change is too large or destructive",
                                _bt_fp)
                    fix_files = _bt_filtered
                if not fix_files:
                    # All fixes targeted source files — retry once with a
                    # test-only constraint instead of giving up.  Common
                    # case: Header uses <Link> but test doesn't wrap in
                    # MemoryRouter.  LLM tries to remove Link from Header
                    # instead of wrapping the test render.
                    if not _did_test_only_retry:
                        _did_test_only_retry = True
                        _logger.info(
                            "[BulkTest] All fixes were source files for %s "
                            "— retrying with test-only constraint", basename)
                        _bt_step_desc = ""
                        _ps_desc = _step_descs.get(test_path)
                        if _ps_desc:
                            _bt_step_desc = (
                                f"STEP INTENT: {_ps_desc}\n"
                            )
                        _vitest_hint = ""
                        if uses_vitest:
                            _vitest_hint = (
                                "- This project uses VITEST (not Jest). "
                                "Use `vi.mock()` instead of `jest.mock()`, "
                                "`vi.fn()` instead of `jest.fn()`, and "
                                "import `{ vi }` from 'vitest' if needed.\n"
                            )
                        _test_only_prompt = (
                            f"{_bt_briefing_block}"
                            f"Task: {task}\n\n"
                            f"{_bt_step_desc}"
                            f"Test file `{test_path}` failed.\n\n"
                            f"Error output:\n{file_error}\n\n"
                            f"Relevant source files (READ-ONLY — do NOT modify these):\n"
                            f"{source_ctx}\n\n"
                            "CRITICAL RULES:\n"
                            "1. Fix ONLY the test file. Source files are correct.\n"
                            "2. Do NOT remove or weaken test assertions — the intended "
                            "functionality must still be verified.\n"
                            "3. Adapt assertions to match what the source components "
                            "ACTUALLY render (read the source files above).\n\n"
                            "Common fixes:\n"
                            f"{_vitest_hint}"
                            "- If the error mentions Router context (useHref, useLocation, "
                            "basename is null), wrap the render in <MemoryRouter> from "
                            "'react-router-dom'.\n"
                            "- If the error mentions missing element/text, update the "
                            "test assertions to match what the component actually renders.\n"
                            "- Use getAllByText/getAllByRole if multiple elements match.\n\n"
                            f"Output the COMPLETE fixed test file using:\n"
                            f"#### [FILE]: {test_path}\n"
                            f"```{lang_tag}\n// complete test file\n```"
                        )
                        try:
                            _to_resp = coder.llm_client.generate_response(
                                _test_only_prompt)
                            _to_files = executor.parse_code_blocks(_to_resp)
                            if not _to_files:
                                _to_files = executor.parse_code_blocks_fuzzy(
                                    _to_resp)
                            if _to_files:
                                if subproject_cwd:
                                    from .step_handlers import (
                                        _prefix_subproject_paths,
                                    )
                                    _to_files = _prefix_subproject_paths(
                                        _to_files, subproject_cwd, memory)
                                # Filter again — only test files
                                _to_files = {
                                    fp: fc for fp, fc in _to_files.items()
                                    if _is_test_file(fp)
                                }
                            if _to_files:
                                fix_files = _to_files
                                _logger.info(
                                    "[BulkTest] Test-only retry produced "
                                    "fix for %s: %s",
                                    basename, list(fix_files.keys()))
                        except Exception as _to_exc:
                            _logger.warning(
                                "[BulkTest] Test-only retry failed: %s",
                                _to_exc)
                    if not fix_files:
                        # ── Escape hatch: targeted source-file fix ──
                        # See _should_trigger_escape_hatch for the
                        # full set of preconditions.
                        if _should_trigger_escape_hatch(
                            used_escape_hatch=_used_escape_hatch,
                            did_test_only_retry=_did_test_only_retry,
                            error_sig_history=_error_sig_history,
                            route=_route,
                        ):
                            _logger.info(
                                "[BulkTest] Escape hatch triggered for %s "
                                "— error signature stable (%s), attempting "
                                "targeted source fix",
                                basename, _error_sig_history[-1])
                            _used_escape_hatch = True  # one shot only
                            _hatch_files = _attempt_targeted_source_fix(
                                test_path=test_path,
                                file_error=file_error,
                                source_ctx=source_ctx,
                                coder=coder,
                                executor=executor,
                                memory=memory,
                                subproject_cwd=subproject_cwd,
                                lang_tag=lang_tag,
                                task=task,
                            )
                            if _hatch_files:
                                # Snapshot the files we're about to
                                # overwrite so we can revert on regression.
                                _hatch_snap = memory.snapshot(
                                    list(_hatch_files.keys()))
                                for _hfp in _hatch_files:
                                    if _hfp not in _hatch_snap:
                                        try:
                                            with open(_hfp, 'r',
                                                      encoding='utf-8',
                                                      errors='replace') as _f:
                                                _hatch_snap[_hfp] = _f.read()
                                        except OSError:
                                            _hatch_snap[_hfp] = ''
                                fix_files = _hatch_files
                                _logger.info(
                                    "[BulkTest] Escape hatch fix ready for "
                                    "%s: %s",
                                    basename, list(fix_files.keys()))
                    if not fix_files:
                        _logger.warning(
                            "[BulkTest] All fix files were source files — "
                            "skipping write for %s", basename)
                        break
                executor.write_files(fix_files)
                memory.update(fix_files)
                _logger.info(
                    "[BulkTest] Applied fixes for %s: %s",
                    basename, list(fix_files.keys()),
                )
            except Exception as exc:
                _logger.warning("[BulkTest] Fix generation failed for %s: %s", basename, exc)
                break

            # Re-run this single file
            single_cmd = _build_scoped_test_cmd(
                base_cmd, {test_path: ""}, subproject_cwd)
            ok_single, current_output = executor.run_command(
                single_cmd, cwd=subproject_cwd)
            if ok_single:
                _logger.info("[BulkTest] %s now passes.", basename)
                print(f"  [BulkTest] {basename} fixed ✔")
                _p, _t, _ = _parse_test_counts(current_output)
                display.record_test_result(test_path, passed=_p, total=_t, failures=[])
                # Hatch fix worked — drop the snapshot so it isn't
                # restored later if a subsequent attempt regresses.
                _hatch_snap = None

                # Check if any source files were modified and find other test
                # files that import them — they may have been broken by the fix.
                if fix_files:
                    modified_sources = [f for f in fix_files if not _is_test_file(f)]
                    if modified_sources:
                        already_queued = set(failed_files)
                        impacted = _find_tests_impacted_by_sources(
                            modified_sources,
                            test_files,
                            exclude=test_path,
                            already_queued=already_queued,
                            kb_context_builder=kb_context_builder,
                        )
                        for rt in impacted:
                            rt_cmd = _build_scoped_test_cmd(
                                base_cmd, {rt: ""}, subproject_cwd)
                            ok_rt, out_rt = executor.run_command(
                                rt_cmd, cwd=subproject_cwd)
                            if not ok_rt:
                                rt_base = rt.rsplit('/', 1)[-1]
                                _logger.warning(
                                    "[BulkTest] %s broke after fix to %s — queuing",
                                    rt_base, modified_sources,
                                )
                                print(
                                    f"  [BulkTest] {rt_base} impacted by source "
                                    f"change — queuing fix..."
                                )
                                failed_files.append(rt)
                                _p_rt, _t_rt, _f_rt = _parse_test_counts(out_rt)
                                display.record_test_result(
                                    rt, passed=_p_rt, total=_t_rt, failures=_f_rt)
                break
            _logger.warning(
                "[BulkTest] %s still failing (attempt %d/%d)",
                basename, fix_attempt, _MAX_BULK_TEST_FIX_ATTEMPTS,
            )
            # If the just-applied fix came from the escape hatch and the
            # target test still fails, the source change did not help —
            # roll it back so subsequent attempts (and downstream tests)
            # operate on the known-good source content.
            if _hatch_snap is not None:
                _logger.warning(
                    "[BulkTest] Escape-hatch source fix for %s did not "
                    "fix the test — reverting %d file(s)",
                    basename, len(_hatch_snap))
                memory.restore(_hatch_snap, executor=executor)
                _hatch_snap = None
            # Update TEST RESULTS with latest counts on each fix attempt
            _p, _t, _f = _parse_test_counts(current_output)
            display.record_test_result(test_path, passed=_p, total=_t, failures=_f)
        else:
            print(f"  [BulkTest] {basename} could not be fixed after "
                  f"{_MAX_BULK_TEST_FIX_ATTEMPTS} attempt(s).")

    # ── Step 3: Final run-all to confirm everything passes ──
    # If the run-all surfaces new failures (e.g. a pre-existing test file that
    # wasn't in the per-file fix queue, or a Django ERROR: that the initial
    # parse missed), feed them back into the fix loop rather than giving up.
    _MAX_RUNALL_ROUNDS = 2
    for _round in range(_MAX_RUNALL_ROUNDS):
        ok_final, output_final = executor.run_command(base_cmd, cwd=subproject_cwd)
        if ok_final:
            _logger.info("[BulkTest] Final run-all passed.")
            print("  [BulkTest] All tests pass after fixes.")
            # Files that were never confirmed fixed in the per-file loop
            # (their fix attempts were skipped/exhausted) can still show
            # as passing here if the combined run happens to succeed —
            # e.g. import-order-dependent test doubles that pass only in
            # a particular file collection order. Re-verify those in
            # isolation instead of trusting the aggregate result, so the
            # TEST RESULTS panel never shows a green badge for a file
            # that fails standalone.
            _unconfirmed = set(failed_files)
            for fpath in test_files:
                if fpath in _unconfirmed:
                    _verify_cmd = _build_scoped_test_cmd(
                        base_cmd, {fpath: ""}, subproject_cwd)
                    _ok_v, _out_v = executor.run_command(
                        _verify_cmd, cwd=subproject_cwd)
                    _p_v, _t_v, _f_v = _parse_test_counts(_out_v)
                    display.record_test_result(
                        fpath, passed=_p_v, total=_t_v, failures=_f_v)
                    if not _ok_v:
                        _logger.warning(
                            "[BulkTest] %s passed in the combined run-all "
                            "but fails in isolation — likely "
                            "order-dependent test pollution; leaving "
                            "marked as failing.", fpath)
                    continue
                _existing = display.get_test_result(fpath)
                if _existing and _existing.get("total", 0) > 1:
                    # Already have real per-file counts — don't clobber
                    # them with a fake 1/1.
                    continue
                display.record_test_result(fpath, passed=1, total=1, failures=[])
            return True, ""

        # Parse failures from the full run — includes files not in the original
        # per-file fix queue (e.g. pre-existing tests, Django ERROR: lines).
        if "manage.py" in base_cmd:
            output_final = _fix_django_startup_crashes(output_final, subproject_cwd, executor)
        still_failing = _parse_failed_test_files(output_final, list(test_files.keys()), subproject_cwd)
        _logger.warning(
            "[BulkTest] Run-all round %d/%d failed. Still failing: %s",
            _round + 1, _MAX_RUNALL_ROUNDS, still_failing,
        )

        # Find files that weren't already fixed in the per-file loop
        already_fixed = set(failed_files)
        new_failures = [f for f in still_failing if f not in already_fixed]

        if not new_failures:
            # Same files are still broken — no point retrying
            break

        print(
            f"  [BulkTest] Run-all found {len(new_failures)} additional "
            f"failing file(s) — fixing..."
        )
        # Append to failed_files and re-run the fix loop for just the new ones
        fix_start = len(failed_files)
        for nf in new_failures:
            if nf not in set(failed_files):
                failed_files.append(nf)
                _p_nf, _t_nf, _f_nf = _parse_test_counts(output_final)
                display.record_test_result(nf, passed=_p_nf, total=_t_nf, failures=_f_nf)

        current_output = output_final
        while fix_idx < len(failed_files):
            test_path = failed_files[fix_idx]
            fix_idx += 1
            basename = test_path.rsplit('/', 1)[-1]
            print(f"  [BulkTest] Fixing {basename} (from run-all)...")

            # Per-test escape-hatch state — see Loop 1 for details.
            _did_test_only_retry = False
            _used_escape_hatch = False
            _hatch_snap: dict[str, str] | None = None
            _error_sig_history: list[str] = []

            # Classify the failure once per test so the escape-hatch
            # trigger can require ErrorRouter source_type == 'code'.
            _route = None
            try:
                from .error_router import classify_error as _classify_error
                _initial_error = _extract_file_specific_errors(
                    current_output, test_path, max_chars=2000)
                if not _initial_error:
                    _initial_error = current_output[-2000:]
                _route = _classify_error(
                    error_info=_initial_error,
                    step_type="TEST",
                    project_context=project_context,
                    kb_matched=False,
                    llm_client=coder.llm_client if search_agent is not None else None,
                )
                log.info(
                    "[BulkTest] ErrorRouter %s → source=%s skip_web=%s (%s)",
                    basename, _route.source_type, _route.skip_web, _route.reason,
                )
            except Exception as _re_exc:
                log.debug("[BulkTest] ErrorRouter failed for %s: %s", basename, _re_exc)

            for fix_attempt in range(1, _MAX_BULK_TEST_FIX_ATTEMPTS + 1):
                file_error = _extract_file_specific_errors(
                    current_output, test_path, max_chars=3000)
                if not file_error:
                    # Take the tail (where tracebacks and ImportErrors appear)
                    # rather than the head (which is often setup noise).
                    file_error = current_output[-3000:]

                # Track normalised error shape (escape-hatch trigger).
                _error_sig_history.append(_error_signature(file_error))

                current_content = memory.all_files().get(test_path, "")
                if not current_content:
                    # Pre-existing file not tracked in session memory — read from disk
                    try:
                        with open(test_path, "r", encoding="utf-8", errors="replace") as _tf2:
                            current_content = _tf2.read()
                    except OSError:
                        pass
                imported_sources = _extract_imported_sources(
                    {test_path: current_content}, memory,
                    resolve_from_disk=True)
                source_ctx = (
                    f"#### [FILE]: {test_path}\n```{lang_tag}\n{current_content}\n```\n\n"
                )
                for fp, cnt in imported_sources.items():
                    source_ctx += (
                        f"#### [FILE]: {fp}\n```{lang_tag}\n{cnt}\n```\n\n"
                    )
                source_ctx += _django_settings_context(subproject_cwd)

                _bt_briefing = getattr(memory, '_task_briefing', '')
                _bt_briefing_block = (
                    "TASK BRIEFING (overall goal):\n"
                    f"{_bt_briefing}\n\n"
                ) if _bt_briefing else ""

                fix_prompt = (
                    f"{_bt_briefing_block}"
                    f"Task: {task}\n\n"
                    f"Test file `{test_path}` failed in the full test suite. "
                    f"Fix it so the tests pass.\n\n"
                    f"Error output:\n{file_error}\n\n"
                    f"Relevant files:\n{source_ctx}\n\n"
                    "You may fix the test file itself OR fix a source file it imports — "
                    "whichever is correct.  Do NOT remove any existing tests.\n\n"
                    "IMPORTANT — before modifying any template or source file to satisfy "
                    "an assertContains/assertIn/assertEqual test:\n"
                    "1. Read the EXACT string literal from the test assertion above.\n"
                    "2. Copy that exact string (correct case, spacing, punctuation) into "
                    "   the template or source file.\n"
                    "3. Do NOT paraphrase, guess, or change the casing of the expected string.\n\n"
                    "CRITICAL: NEVER abbreviate or summarize existing code with comments like `// existing code` or `/* unchanged */`. If you are editing a chunk or a file, you MUST write out the ENTIRE content of that chunk or file. Abbreviating code will cause it to be permanently deleted!\n\n"
                    "Prefer CHUNK FORMAT for surgical fixes:\n"
                    f"#### [EDIT]: path/to/file:{lang_tag}:function_name (lines start-end)\n"
                    f"```{lang_tag}\n// replacement chunk\n```\n"
                    "Use full-file [FILE]: format only when the whole file must be rewritten."
                )

                try:
                    fix_response = coder.llm_client.generate_response(fix_prompt)
                    fix_files = {}
                    try:
                        from ..editing.chunk_editor import ChunkEditor as _BtCE2
                        _bt_ce2 = _BtCE2()
                        _bt_edits2 = _bt_ce2.parse_chunk_response(fix_response)
                        if _bt_edits2:
                            for _bt_edit2 in _bt_edits2:
                                _bt_fp2 = _bt_edit2.file_path
                                _bt_existing2 = memory.get(_bt_fp2)
                                if _bt_existing2 is None:
                                    try:
                                        with open(_bt_fp2, "r", encoding="utf-8",
                                                  errors="replace") as _f2:
                                            _bt_existing2 = _f2.read()
                                    except OSError:
                                        pass
                                if _bt_existing2:
                                    try:
                                        fix_files[_bt_fp2] = _bt_ce2.apply_chunk_edits(
                                            _bt_existing2, [_bt_edit2])
                                    except Exception:
                                        pass
                    except ImportError:
                        pass
                    if not fix_files:
                        fix_files = executor.parse_code_blocks(fix_response)
                    if not fix_files:
                        fix_files = executor.parse_code_blocks_fuzzy(fix_response)
                    if fix_files:
                        if subproject_cwd:
                            from .step_handlers import _prefix_subproject_paths
                            fix_files = _prefix_subproject_paths(
                                fix_files, subproject_cwd, memory)
                        # ── Source file protection ──
                        _bt2_filtered = {}
                        for _bt2_fp, _bt2_fc in fix_files.items():
                            if _is_test_file(_bt2_fp):
                                _bt2_filtered[_bt2_fp] = _bt2_fc
                            elif _is_test_infra_file(_bt2_fp):
                                _bt2_filtered[_bt2_fp] = _bt2_fc
                                _logger.info(
                                    "[BulkTest] Allowed test-infra fix "
                                    "for %s", _bt2_fp)
                            elif _is_additive_source_fix(
                                    _bt2_fp, _bt2_fc, memory):
                                _bt2_filtered[_bt2_fp] = _bt2_fc
                                _logger.info(
                                    "[BulkTest] Allowed small additive "
                                    "source fix for %s", _bt2_fp)
                            elif _is_safe_source_fix(
                                    _bt2_fp, _bt2_fc, memory):
                                _bt2_filtered[_bt2_fp] = _bt2_fc
                                _logger.info(
                                    "[BulkTest] Allowed bounded source "
                                    "fix for %s (diff within hatch cap)",
                                    _bt2_fp)
                            else:
                                _logger.warning(
                                    "[BulkTest] Blocked fix for source "
                                    "file %s — too large or destructive",
                                    _bt2_fp)
                        _bt2_blocked = set(fix_files) - set(_bt2_filtered)
                        if _bt2_blocked:
                            _logger.warning(
                                "[BulkTest] Blocked source file(s): %s",
                                list(_bt2_blocked))
                        fix_files = _bt2_filtered
                        if not fix_files and not _did_test_only_retry:
                            # Retry with test-only constraint (same
                            # logic as Loop 1 — see detailed comments there)
                            _did_test_only_retry = True
                            try:
                                _to2_step_desc = ""
                                _to2_ps = _step_descs.get(test_path)
                                if _to2_ps:
                                    _to2_step_desc = (
                                        f"STEP INTENT: {_to2_ps}\n")
                                _to2_vitest_hint = ""
                                if uses_vitest:
                                    _to2_vitest_hint = (
                                        "- VITEST project: use `vi.mock()` "
                                        "not `jest.mock()`, `vi.fn()` not "
                                        "`jest.fn()`.\n"
                                    )
                                _to2_prompt = (
                                    f"{_bt_briefing_block}"
                                    f"{_to2_step_desc}"
                                    f"Test file `{test_path}` failed.\n\n"
                                    f"Error:\n{current_output[:3000]}\n\n"
                                    f"Source files (READ-ONLY):\n"
                                    f"{source_ctx}\n\n"
                                    "CRITICAL RULES:\n"
                                    "1. Fix ONLY the test file. Source "
                                    "files are correct.\n"
                                    "2. Do NOT remove or weaken "
                                    "assertions — verify the intended "
                                    "functionality.\n"
                                    "3. Adapt to match what the source "
                                    "components actually render.\n"
                                    f"{_to2_vitest_hint}"
                                    "- Use MemoryRouter for Router "
                                    "context.\n"
                                    "- Use getAllBy* for multiple "
                                    "matches.\n\n"
                                    f"#### [FILE]: {test_path}\n"
                                )
                                _to2_resp = coder.llm_client.generate_response(
                                    _to2_prompt)
                                _to2_files = (
                                    executor.parse_code_blocks(_to2_resp)
                                    or executor.parse_code_blocks_fuzzy(_to2_resp)
                                )
                                if _to2_files and subproject_cwd:
                                    from .step_handlers import _prefix_subproject_paths
                                    _to2_files = _prefix_subproject_paths(
                                        _to2_files, subproject_cwd, memory)
                                if _to2_files:
                                    _to2_files = {
                                        fp: fc for fp, fc in _to2_files.items()
                                        if _is_test_file(fp)
                                    }
                                if _to2_files:
                                    fix_files = _to2_files
                                    _logger.info(
                                        "[BulkTest] Test-only retry produced "
                                        "fix: %s", list(fix_files.keys()))
                            except Exception:
                                pass
                        if not fix_files:
                            # Escape hatch — same trigger as Loop 1.
                            if _should_trigger_escape_hatch(
                                used_escape_hatch=_used_escape_hatch,
                                did_test_only_retry=_did_test_only_retry,
                                error_sig_history=_error_sig_history,
                                route=_route,
                            ):
                                _logger.info(
                                    "[BulkTest] Escape hatch triggered for "
                                    "%s (run-all loop) — error signature "
                                    "stable (%s), attempting targeted "
                                    "source fix",
                                    basename, _error_sig_history[-1])
                                _used_escape_hatch = True
                                _hatch_files = _attempt_targeted_source_fix(
                                    test_path=test_path,
                                    file_error=file_error,
                                    source_ctx=source_ctx,
                                    coder=coder,
                                    executor=executor,
                                    memory=memory,
                                    subproject_cwd=subproject_cwd,
                                    lang_tag=lang_tag,
                                    task=task,
                                )
                                if _hatch_files:
                                    _hatch_snap = memory.snapshot(
                                        list(_hatch_files.keys()))
                                    for _hfp in _hatch_files:
                                        if _hfp not in _hatch_snap:
                                            try:
                                                with open(_hfp, 'r',
                                                          encoding='utf-8',
                                                          errors='replace') as _f:
                                                    _hatch_snap[_hfp] = _f.read()
                                            except OSError:
                                                _hatch_snap[_hfp] = ''
                                    fix_files = _hatch_files
                                    _logger.info(
                                        "[BulkTest] Escape hatch fix ready "
                                        "for %s: %s",
                                        basename, list(fix_files.keys()))
                        if not fix_files:
                            break
                        executor.write_files(fix_files)
                        memory.update(fix_files)
                        _logger.info(
                            "[BulkTest] Applied fixes for %s: %s",
                            basename, list(fix_files.keys()),
                        )
                except Exception as exc:
                    _logger.warning(
                        "[BulkTest] Fix generation failed for %s: %s", basename, exc)
                    break

                single_cmd = _build_scoped_test_cmd(
                    base_cmd, {test_path: ""}, subproject_cwd)
                ok_single, current_output = executor.run_command(
                    single_cmd, cwd=subproject_cwd)
                if ok_single:
                    _logger.info("[BulkTest] %s now passes.", basename)
                    print(f"  [BulkTest] {basename} fixed ✔")
                    _p, _t, _ = _parse_test_counts(current_output)
                    display.record_test_result(test_path, passed=_p, total=_t, failures=[])
                    _hatch_snap = None  # commit hatch fix
                    break
                _logger.warning(
                    "[BulkTest] %s still failing (attempt %d/%d)",
                    basename, fix_attempt, _MAX_BULK_TEST_FIX_ATTEMPTS,
                )
                # Revert any escape-hatch source fix that didn't help.
                if _hatch_snap is not None:
                    _logger.warning(
                        "[BulkTest] Escape-hatch source fix for %s did "
                        "not fix the test — reverting %d file(s)",
                        basename, len(_hatch_snap))
                    memory.restore(_hatch_snap, executor=executor)
                    _hatch_snap = None
                _p, _t, _f = _parse_test_counts(current_output)
                display.record_test_result(test_path, passed=_p, total=_t, failures=_f)
            else:
                print(f"  [BulkTest] {basename} could not be fixed after "
                      f"{_MAX_BULK_TEST_FIX_ATTEMPTS} attempt(s).")

        current_output = output_final  # reset for next round's error extraction

    # All rounds exhausted — update display and report failure
    still_failing_final = set(_parse_failed_test_files(output_final, list(test_files.keys()), subproject_cwd))
    for fpath in test_files:
        if fpath in still_failing_final:
            _p, _t, _f = _parse_test_counts(output_final)
            display.record_test_result(fpath, passed=_p, total=_t, failures=_f)
        else:
            display.record_test_result(fpath, passed=1, total=1, failures=[])

    error_msg = (
        f"Bulk test execution failed: some test file(s) still failing "
        f"after per-file fix attempts.\n{output_final[:600]}"
    )
    _logger.warning("[BulkTest] Final run-all failed:\n%s", output_final[:600])
    print("  [BulkTest] FAILED — some tests still failing after fixes.")
    return False, error_msg


# ---------------------------------------------------------------------------
# Wiring verification — cross-file integration check after all steps complete
# ---------------------------------------------------------------------------

def _parse_fix_scope(task: str) -> tuple[list[str], list[str]]:
    """Parse fix scope file references from the task description.

    Looks for a ``Fix scope:`` section, then extracts:
    - backtick-quoted paths / filenames  → *exact_paths*
    - remaining natural-language text    → *nl_queries* (for KB semantic search)

    Returns ``(exact_paths, nl_queries)``.
    """
    # Narrow to "Fix scope:" section when present; otherwise scan full task.
    m = _FIX_SCOPE_SECTION_RE.search(task)
    scope_text = m.group(1) if m else task

    exact_paths: list[str] = []
    seen: set[str] = set()

    # 1. Backtick-quoted paths/names
    for bm in _BACKTICK_PATH_RE.finditer(scope_text):
        p = bm.group(1).strip()
        if p and p not in seen:
            exact_paths.append(p)
            seen.add(p)

    # 2. Bare file names not already captured
    for bm in _BARE_FILENAME_RE.finditer(scope_text):
        p = bm.group(1).strip()
        if p not in seen:
            exact_paths.append(p)
            seen.add(p)

    # 3. Remaining text as NL query (strip captured paths, collapse whitespace)
    nl_text = _BACKTICK_PATH_RE.sub('', scope_text)
    nl_text = _BARE_FILENAME_RE.sub('', nl_text)
    nl_text = re.sub(r'[\s,;()]+', ' ', nl_text).strip()
    nl_queries = [nl_text] if len(nl_text) > 20 else []

    return exact_paths, nl_queries


def _resolve_fix_scope_files(
    exact_paths: list[str],
    nl_queries: list[str],
    memory: FileMemory,
    kb_context_builder=None,
    project_root: str = "",
) -> dict[str, str]:
    """Resolve fix-scope entries to ``{path: content}``.

    Fallback chain per entry:
    1. FileMemory exact key match
    2. FileMemory basename / suffix match
    3. Disk read  (file exists, was not modified this session)
    4. Filesystem glob  (``**/<basename>``)
    5. KB semantic search  (when *kb_context_builder* is available)

    Files from steps 3-5 are included as read-only context — they are NOT
    added to FileMemory so that the written-files audit trail stays clean.
    """
    import glob as _glob
    import os as _os

    root = project_root or _os.getcwd()

    def _read_disk(path: str) -> str | None:
        abs_p = path if _os.path.isabs(path) else _os.path.join(root, path)
        try:
            if _os.path.isfile(abs_p):
                with open(abs_p, encoding="utf-8", errors="replace") as fh:
                    return fh.read()
        except OSError:
            pass
        return None

    def _glob_search(name: str) -> tuple[str, str] | None:
        basename = _os.path.basename(name)
        hits = _glob.glob(_os.path.join(root, "**", basename), recursive=True)
        for hit in hits:
            content = _read_disk(hit)
            if content:
                rel = _os.path.relpath(hit, root).replace("\\", "/")
                return rel, content
        return None

    result: dict[str, str] = {}

    for path in exact_paths:
        basename = _os.path.basename(path)

        # 1. FileMemory exact
        content = memory.get(path)
        if content is not None:
            result[path] = content
            continue

        # 2. FileMemory suffix match, THEN basename match — two separate
        #    passes over ALL files. A single combined pass let a basename
        #    hit on an earlier file shadow the exact suffix match on a
        #    later one: `core/urls.py` and `config/urls.py` both resolved
        #    to accounts/urls.py (written first), so the files that
        #    actually needed review never entered the verification prompt
        #    while the LLM still "fixed" them blind. Every match is kept —
        #    same-named files are exactly the ones the wiring check must
        #    see side by side.
        stored = memory.all_files()
        p_norm = path.replace("\\", "/").lstrip("./")
        matches = [sp for sp in stored
                   if sp.replace("\\", "/") == p_norm
                   or sp.replace("\\", "/").endswith("/" + p_norm)]
        if not matches:
            matches = [sp for sp in stored
                       if _os.path.basename(sp) == basename]
        if matches:
            for sp in matches:
                result[sp] = stored[sp]
            continue

        # 3. Disk read
        content = _read_disk(path)
        if content is not None:
            result[path] = content
            continue

        # 4. Glob
        found = _glob_search(path)
        if found:
            result[found[0]] = found[1]
            continue

        # 5. KB semantic search
        resolved_via_kb = False
        if kb_context_builder is not None:
            try:
                relevant = kb_context_builder.get_relevant_files(
                    task_description=f"file {path} {basename}",
                    max_files=1,
                )
                for r_path in relevant:
                    r_content = memory.get(r_path) or _read_disk(r_path)
                    if r_content:
                        result[r_path] = r_content
                        resolved_via_kb = True
                        break
            except Exception as kb_exc:
                _logger.debug(
                    "[WiringVerification] KB search failed for %s: %s",
                    path, kb_exc,
                )
        if not resolved_via_kb:
            _logger.warning(
                "[WiringVerification] Could not resolve: %s — skipping", path
            )

    # Natural-language queries → KB semantic search (top-3 files each)
    for nl_query in nl_queries:
        if kb_context_builder is not None:
            try:
                relevant = kb_context_builder.get_relevant_files(
                    task_description=nl_query,
                    max_files=3,
                )
                for r_path in relevant:
                    if r_path not in result:
                        r_content = memory.get(r_path) or _read_disk(r_path)
                        if r_content:
                            result[r_path] = r_content
            except Exception as exc:
                _logger.debug(
                    "[WiringVerification] KB NL query failed (%r): %s", nl_query, exc
                )

    return result


# Vendor/build artifacts that must never enter an LLM verification prompt.
_WIRING_SKIP_DIRS = frozenset({
    "node_modules", "dist", "build", "coverage", ".git",
    "venv", ".venv", "__pycache__",
})
_WIRING_SKIP_FILES = frozenset({
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
})
_WIRING_MAX_FILE_CHARS = 12_000


def _sanitize_wiring_context(ctx: dict[str, str]) -> dict[str, str]:
    """Clean the wiring-verification file set before prompt assembly.

    - Normalises path separators and deduplicates (the same file can be
      resolved once with ``/`` and once with ``\\`` — it would otherwise
      be injected twice).
    - Drops vendor/build files (``node_modules``, lockfiles, minified
      assets): a single ``bootstrap.min.css`` is ~50k tokens of noise.
    - Caps per-file content so one big file cannot dominate the prompt.
    """
    clean: dict[str, str] = {}
    for path, content in ctx.items():
        norm = path.replace("\\", "/")
        if norm in clean:
            continue
        parts = norm.split("/")
        basename = parts[-1]
        if any(p in _WIRING_SKIP_DIRS for p in parts[:-1]):
            _logger.debug("[WiringVerification] Skipping vendor/build "
                          "file: %s", norm)
            continue
        if basename in _WIRING_SKIP_FILES or ".min." in basename:
            _logger.debug("[WiringVerification] Skipping artifact: %s", norm)
            continue
        if len(content) > _WIRING_MAX_FILE_CHARS:
            content = (content[:_WIRING_MAX_FILE_CHARS]
                       + "\n/* ... truncated for verification ... */")
        clean[norm] = content
    return clean


def run_wiring_verification(
    *,
    memory: FileMemory,
    executor,
    coder,
    display: "CLIDisplay",
    task: str,
    language: str | None,
    cfg=None,
    kb_context_builder=None,
    project_root: str = "",
) -> tuple[bool, str]:
    """Cross-file wiring check executed once after all pipeline steps complete.

    Algorithm
    ---------
    1. Parse ``Fix scope:`` from the task description to get target files.
    2. Resolve each target via layered fallback:
       FileMemory → disk read → filesystem glob → KB semantic search.
       Files not in FileMemory land in a read-only *verification_context*
       buffer — they are never written back.
    3. Make a single LLM call asking for cross-file integration issues
       (missing mounts, import/export mismatches, wrong prop shapes, etc.).
    4. If the LLM finds issues, parse code blocks and apply fixes via the
       executor + memory.  Returns ``(True, "")`` on success.
    5. If no fix scope is declared in the task, falls back to all session
       files tracked by FileMemory.

    Always non-fatal on LLM errors — returns ``(True, "")`` rather than
    blocking the pipeline on a transient failure.
    """
    import os as _os
    from .memory import _should_skip_for_context

    _logger.info("[WiringVerification] Starting cross-file wiring check")

    # ── 1. Parse fix scope ────────────────────────────────────────────────
    exact_paths, nl_queries = _parse_fix_scope(task)

    # ── 2. Resolve files ──────────────────────────────────────────────────
    if exact_paths or nl_queries:
        verification_context = _resolve_fix_scope_files(
            exact_paths, nl_queries, memory,
            kb_context_builder=kb_context_builder,
            project_root=project_root or _os.getcwd(),
        )
    else:
        # No explicit scope — use all session files as a broad check.
        _logger.info(
            "[WiringVerification] No fix scope in task — checking all session files"
        )
        verification_context = {
            p: c for p, c in memory.all_files().items()
            if not _should_skip_for_context(p)
        }

    verification_context = _sanitize_wiring_context(verification_context)

    if not verification_context:
        _logger.info("[WiringVerification] No files resolved — skipping")
        return True, ""

    _logger.info(
        "[WiringVerification] Resolved %d file(s): %s",
        len(verification_context), list(verification_context.keys()),
    )

    # ── 3. Build verification prompt ──────────────────────────────────────
    lang_tag = language or "code"
    context_block = "\n\n".join(
        f"#### [FILE]: {path}\n```{lang_tag}\n{content}\n```"
        for path, content in verification_context.items()
    )

    ver_line = ""
    try:
        from .api_grounding import (get_installed_package_versions,
                                    grounding_packages)
        _versions = get_installed_package_versions(
            cwd=project_root or None, executor=executor, language=language)
        _pkgs = grounding_packages(_versions, memory)
        if _pkgs:
            ver_line = (
                "Installed packages — any fix must only use APIs that exist "
                "in these EXACT versions: " + ", ".join(_pkgs) + "\n\n"
            )
    except Exception:
        pass

    prompt = (
        f"Task: {task}\n\n"
        f"{ver_line}"
        "You are performing a final cross-file wiring review. "
        "Examine ALL files below together and identify any integration "
        "issue that would cause a blank screen, runtime crash, or silent "
        "failure at startup. Common issues to check:\n"
        "  • Missing ReactDOM.createRoot / wrong entry-point mounting\n"
        "  • Missing context provider mount — e.g. a component imports "
        "`Link` / `useNavigate` from react-router-dom but no source file "
        "wraps the tree in `<BrowserRouter>` / `<RouterProvider>`. Tests "
        "may pass via `<MemoryRouter>` wrappers while the browser crashes "
        "with `Cannot destructure 'basename' of useContext(...)`. Either "
        "mount `<BrowserRouter>` in the entry point, or strip the router "
        "primitives from the source if the page uses in-page anchor links.\n"
        "  • Default-export / named-export mismatch between files\n"
        "  • Import path typo or missing file extension\n"
        "  • Component receives wrong prop shape (undefined, wrong type)\n"
        "  • Undefined variable or missing null-check at render time\n"
        "  • Module not found (package not imported / wrong casing)\n"
        "  • Duplicate context/provider wrappers (e.g. Router, ThemeProvider, "
        "Store) in both entry-point AND child — causes nested provider errors\n"
        "  • Route/URL name mismatch — a template or view references a "
        "named route (e.g. Django `{% url 'name' %}` / `reverse('name')`) "
        "that no URLconf in scope defines under that exact name; mind "
        "`app_name` namespacing\n"
        "  • Any other wiring issue that prevents the UI from rendering\n\n"
        f"Files in scope:\n{context_block}\n\n"
        "RESPONSE FORMAT:\n"
        "  • If NO issues exist, respond with exactly: NO_ISSUES_FOUND\n"
        "  • If issues exist, output the COMPLETE fixed content of every "
        "affected file using:\n"
        "    #### [FILE]: path/to/file\n"
        f"    ```{lang_tag}\n"
        "    // complete file content — never abbreviate\n"
        "    ```\n"
        "CRITICAL: Never use `// existing code` or `/* unchanged */` — "
        "write the full file content or your fix will delete existing code.\n"
        "CRITICAL: Only rewrite files from the list above. NEVER invent new "
        "files or rewrite files that are not shown — a fix touching an "
        "unlisted file will be discarded in its entirety."
    )

    display.show_status("Verifying cross-file wiring...")

    # The status footer keeps the spinner alive during the LLM call.
    # Clear it on every exit path (try/finally) so the next post-step
    # phase does not display a stale "Verifying cross-file wiring..."
    # message after this function returns.
    try:
        # ── 4. LLM call ───────────────────────────────────────────────
        try:
            response = coder.llm_client.generate_response(prompt)
        except Exception as exc:
            _logger.warning(
                "[WiringVerification] LLM call failed (non-fatal): %s", exc)
            return True, ""

        if "NO_ISSUES_FOUND" in response:
            _logger.info("[WiringVerification] No wiring issues found")
            print("  [WiringVerify] No cross-file wiring issues found.")
            return True, ""

        # ── 5. Parse and apply fixes ──────────────────────────────────
        fix_files: dict[str, str] = {}
        try:
            fix_files = executor.parse_code_blocks(response)
        except Exception:
            pass
        if not fix_files:
            try:
                fix_files = executor.parse_code_blocks_fuzzy(response)
            except Exception:
                pass

        if not fix_files:
            # LLM described an issue in prose but produced no code — log
            # and treat as informational (don't block the pipeline).
            _logger.warning(
                "[WiringVerification] LLM reported issues but no code "
                "blocks found; review manually:\n%s", response[:400],
            )
            return True, ""

        # Ground the fix set in the files the model actually saw. The
        # wiring LLM sometimes "fixes" files that never entered the
        # prompt, inventing their entire content from references it saw
        # elsewhere (observed: core/urls.py — absent from the scope —
        # rewritten with a new `app_name` namespace that broke every
        # un-namespaced {% url %} tag in templates it also invented).
        # In-scope rewrites may depend on the invented files (a base
        # template including a new partial), so the whole set is rejected
        # rather than filtered: the files on disk have already passed
        # their own step checks.
        _allowed = set(verification_context.keys())
        _strays = [p for p in fix_files
                   if p.replace("\\", "/").lstrip("./") not in _allowed]
        if _strays:
            _logger.warning(
                "[WiringVerification] Rejecting fix — it rewrites file(s) "
                "outside the verification scope: %s", _strays)
            print(
                "  [WiringVerify] Proposed fix rejected (touches files not "
                "in the reviewed scope) — keeping existing files."
            )
            return True, ""

        # The wiring LLM regenerates whole files and can reintroduce
        # removed APIs that earlier stages already fixed. A rejected
        # rewrite is strictly safer than an ungrounded one — the files
        # on disk have already passed their own step checks.
        _py_fixes = {p: c for p, c in fix_files.items()
                     if p.endswith(".py")}
        if _py_fixes:
            try:
                from .api_grounding import (local_top_levels_from_files,
                                            probe_api_usage)
                _api_errs = probe_api_usage(
                    _py_fixes, executor,
                    cwd=project_root or None,
                    local_top_levels=local_top_levels_from_files(
                        memory.all_files().keys()),
                )
            except Exception as exc:
                _logger.debug(
                    "[WiringVerification] API probe failed (non-fatal): %s",
                    exc)
                _api_errs = []
            if _api_errs:
                _logger.warning(
                    "[WiringVerification] Rejecting rewrite — it uses APIs "
                    "missing from the installed packages: %s",
                    "; ".join(e.split(" — ")[0] for e in _api_errs))
                print(
                    "  [WiringVerify] Proposed fix rejected (uses APIs not "
                    "in installed packages) — keeping existing files."
                )
                return True, ""

        _logger.info(
            "[WiringVerification] Applying fixes for: %s",
            list(fix_files.keys()),
        )
        print(
            f"  [WiringVerify] Wiring issues found — fixing: "
            f"{', '.join(fix_files.keys())}"
        )
        try:
            executor.write_files(fix_files)
            memory.update(fix_files)
        except Exception as exc:
            _logger.warning(
                "[WiringVerification] Failed to write fixes: %s", exc)
            return False, f"[WiringVerification] Fix write failed: {exc}"

        return True, ""
    finally:
        display.show_status("")
