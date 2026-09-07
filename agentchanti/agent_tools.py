"""
Agent tool registry — the agent-computer interface for tool-calling loops.

Wraps existing subsystems (Executor, KB Searcher, FileMemory) as a small
set of :class:`~agentchanti.llm.chat_types.ToolDef` tools that a model can
invoke through ``LLMClient.chat()``. Execution never raises: every outcome
(including errors) is returned as a string so it can be fed straight back
to the model as a tool-result message.
"""

from __future__ import annotations

import ast
import os
import re
from typing import Optional

from .cli_display import log
from .llm.chat_types import Message, ToolCall, ToolDef


def _protected_basenames() -> set[str]:
    """Manifest/lock filenames guarded against whole-file replacement.

    Deferred import, matching the rest of this module: Executor pulls in the
    heavier execution stack, which the tool definitions do not need.
    """
    from .executor import Executor
    return Executor._PROTECTED_FILENAMES

# Directories never listed/searched — build artifacts and VCS internals.
_IGNORED_DIRS = frozenset({
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv",
    "dist", "build", ".agentchanti", ".pytest_cache", ".mypy_cache",
    ".next", ".nuxt", "coverage", "target",
})

# Caps keep tool results within a predictable token budget.
_MAX_READ_CHARS = 40_000
_MAX_CMD_OUTPUT_CHARS = 8_000
_MAX_LIST_ENTRIES = 300

# POSIX heredoc (`python - << 'PY' ... PY`). cmd.exe parses `<<` as two
# redirects, so on Windows the command exits 1 with no useful output and
# the model retries variations of the same broken syntax.
_HEREDOC_RE = re.compile(r"<<-?\s*['\"]?\w+['\"]?")

# Trailing `| head -N` / `| tail -N` / `| more`, optionally preceded by
# `2>&1`. These only limit how much output is shown — which this module
# already does — and the binaries do not exist on Windows, so the whole
# pipeline dies before the real command's output is ever seen.
_POSIX_OUTPUT_PIPE_RE = re.compile(
    r"\s*\|\s*(?:head|tail|more)\b[^|]*$", re.IGNORECASE)


# Both `python -m unittest` and `python -m pytest` exit 5 when the runner
# COLLECTED NOTHING — a discovery problem, not a failing assertion. The
# tool result only ever said "exit: FAILED", so the model could not tell
# the two apart and debugged the wrong thing: observed a loop spending
# four consecutive run_command turns re-running a suite that had no tests
# to run, then editing source that was never the problem. 19 occurrences
# across 7 of 8 measured runs.
_NO_TESTS_EXIT = 5
# Substring match, not a regex: the command only has to LOOK like a test
# runner for exit 5 to be meaningful.
_TEST_RUNNER_TOKENS = ("pytest", "unittest", "nose2", "tox",
                       "manage.py test", "go test", "npm test")
_NO_TESTS_OUTPUT_MARKERS = ("no tests ran", "ran 0 tests",
                            "collected 0 items")


def _no_tests_collected(command: str, exit_code, output: str) -> bool:
    """True when a test runner found nothing to run.

    Exit 5 alone is not enough — it is an ordinary failure code for other
    programs — so the command must look like a test runner, or the output
    must say so outright.
    """
    low = (command or "").lower()
    if not any(tok in low for tok in _TEST_RUNNER_TOKENS):
        return False
    if (isinstance(exit_code, int) and not isinstance(exit_code, bool)
            and exit_code == _NO_TESTS_EXIT):
        return True
    low_out = (output or "").lower()
    return any(m in low_out for m in _NO_TESTS_OUTPUT_MARKERS)


# Sentinel the agent loop's exit gate greps for: a verify command that
# collected nothing has proved nothing, whatever it exited with.
NO_TESTS_MARKER = "COLLECTED NO TESTS"

_NO_TESTS_HINT = (
    f"\n\nNOTE: the runner exited having {NO_TESTS_MARKER}. This is a "
    "discovery problem, not a failing assertion — nothing was executed, so "
    "there is no bug in the code under test to chase here, and a zero exit "
    "status above is NOT evidence that anything passed. Check that the "
    "test file exists, is named test_*.py, sits in a directory with an "
    "__init__.py if you are importing it as a package, and that you are "
    "running from the project root."
)


# Distributions named by an install command that FAILED. A model whose
# `pip install pygame` fails has two honest options — fix the install or
# report the blocker — and one dishonest one: write a local `pygame/`
# package so the import succeeds. Observed: three write_file turns
# producing `pygame/__init__.py`, `display.py` and `draw.py` whose own
# docstring says the functions "perform no real rendering". That shadows
# the real library on sys.path, so every later step and test would have
# passed against a no-op renderer — a green run with no game in it.
#
# The trigger is deliberately narrow: the guard needs an install of that
# exact distribution to have failed IN THIS STEP. A run whose installs
# succeed can never reach it.
_INSTALL_RE = re.compile(
    r"\b(?:pip|pip3|uv)\s+(?:install|add)\b(?P<args>.*)", re.IGNORECASE)
# A chained command holds several independent invocations; matching `.*`
# across a separator swallows the next one's argv as if it were package
# names (`... && python -m pip install X` yielded `python` and `install`).
_CMD_SEPARATOR_RE = re.compile(r"&&|\|\||[;|]")
# Strip a version/extras specifier: `pygame==2.6.0`, `pygame[all]`,
# `pygame>=2,<3` all name the distribution `pygame`.
_DIST_SPLIT_RE = re.compile(r"[\[<>=!~;].*$")


def _normalise_dist(name: str) -> str:
    """PEP 503 style normalisation — `Foo_Bar` and `foo-bar` are one name."""
    return re.sub(r"[-_.]+", "-", name.strip().strip("'\"")).lower()


def parse_failed_install_targets(command: str) -> set[str]:
    """Distribution names an install *command* would have installed.

    Returns an empty set for anything that is not an install command, so
    the caller can record unconditionally on failure.
    """
    targets: set[str] = set()
    for segment in _CMD_SEPARATOR_RE.split(command or ""):
        m = _INSTALL_RE.search(segment)
        if not m:
            continue
        for tok in m.group("args").split():
            if tok.startswith("-"):
                continue  # flag, or the value of one we do not care about
            # Requirement files and URLs name no single distribution.
            if "/" in tok or "\\" in tok or tok.endswith(".txt"):
                continue
            dist = _normalise_dist(_DIST_SPLIT_RE.sub("", tok))
            if dist and re.fullmatch(r"[a-z0-9][a-z0-9-]*", dist):
                targets.add(dist)
    return targets


# Executables whose name, if written into the project, would be found
# before the real tool. On Windows the current directory is searched
# first and `.CMD`/`.BAT` are on PATHEXT, so a `node.cmd` beside the code
# silently replaces the interpreter for everything run from there.
_TOOLCHAIN_NAMES = frozenset({
    "node", "npm", "npx", "yarn", "pnpm", "deno", "bun",
    "python", "python3", "pip", "pip3", "py",
    "git", "sh", "bash", "cmd", "powershell", "pwsh",
})
_EXECUTABLE_SUFFIXES = frozenset({
    "", ".cmd", ".bat", ".exe", ".com", ".ps1", ".sh", ".pl",
})


def toolchain_shim(rel_path: str) -> Optional[str]:
    """The tool *rel_path* would shadow when executed, if any.

    Measured 2026-08-19. A gate asked bare Node to import a `.jsx`
    module, which no Node can parse. After three loops and 30 turns the
    agent stopped trying to satisfy the gate with code and replaced the
    interpreter instead, writing `frontend/node.cmd`::

        @echo off
        "%ProgramFiles%\nodejs\node.exe" --experimental-loader ... %*

    Every later `node` invocation from that directory would have run the
    shim. Nothing in the project needs a file named after its toolchain,
    and the one case that produced it was an agent escaping a gate it
    could not otherwise pass - which is precisely when the guard should
    hold. Judged on the basename alone: `tools/node.cmd` shadows just as
    well as `node.cmd` for anything run from `tools/`.
    """
    name = re.split(r"[\\/]+", (rel_path or "").strip("\\/"))[-1]
    if not name:
        return None
    stem, dot, ext = name.rpartition(".")
    suffix = ("." + ext).lower() if dot else ""
    base = (stem if dot else name).lower()
    if base in _TOOLCHAIN_NAMES and suffix in _EXECUTABLE_SUFFIXES:
        return base
    return None


_BARE_NPM_INSTALL_RE = re.compile(
    r"^\s*npm\s+(?:i|install|add)\s+(?!-)(?P<rest>\S.*)$", re.IGNORECASE)


def rootless_npm_install_reason(command: str, project_root: str) -> str | None:
    """Why a bare `npm install <pkg>` here would create a phantom package.

    `npm install <pkg>` in a directory with no `package.json` does not
    fail — it CREATES one, plus a `node_modules` and a lockfile. In a
    repo whose real packages live in sub-directories, that manufactures a
    top-level package belonging to no project, and the app only keeps
    working because Node resolution walks upward. Shipping the
    sub-project alone then breaks.

    Measured across three runs of one benchmark. Two came from healers
    and were fixed at their source; the third was the agent itself
    running::

        npm install jsonwebtoken && npm install jsonwebtoken --save --prefix backend

    — the second half correct, the first half leaving a root
    `package.json` containing a single stray dependency. This is the
    `run_command` gap the architecture notes already name: tool
    sandboxing, not a gate check.

    Narrow by construction. It fires only when the working directory has
    **no** manifest of its own *and* some immediate subdirectory does,
    which is precisely the multi-root shape — a genuine greenfield root
    install has no sibling manifests yet and is left alone, as is any
    install in a directory that legitimately owns a `package.json`.
    """
    # Every segment, not just the first. A chain holds several
    # independent invocations, which is why `parse_failed_install_targets`
    # already splits on the same separators. Measured 2026-08-19 run 25:
    #     npm install --prefix backend jsonwebtoken --save-prod
    #       && npm install jsonwebtoken
    # The anchored match saw only the correctly-directed first half, found
    # `--prefix`, and allowed the whole line — so the bare root install
    # after the `&&` was never examined and created the phantom package
    # this guard exists to prevent.
    args: list[str] = []
    for segment in _CMD_SEPARATOR_RE.split(command or ""):
        segment = segment.strip()
        # Segments share sequential state, so scanning them independently
        # loses the very context that makes an install correct. A `cd`
        # moves everything after it out of this directory, and `npm init`
        # gives the destination a manifest — after either, the question
        # this guard asks no longer applies and it cannot answer it
        # statically. Measured 2026-08-19 run 26, where the segment-wise
        # check added for the previous defect refused
        #     mkdir backend && cd backend && npm init -y
        #       && npm install express cors dotenv jsonwebtoken bcryptjs
        # which is the ordinary way to stand a sub-project up.
        if re.match(r"^cd\s+\S", segment, re.IGNORECASE) or                 re.match(r"^npm\s+init", segment, re.IGNORECASE):
            return None
        m = _BARE_NPM_INSTALL_RE.match(segment)
        if not m:
            continue
        rest = m.group("rest")
        if "--prefix" in rest or "-C " in rest:
            continue                     # this one is directed somewhere
        seg_args = [a for a in rest.split() if not a.startswith("-")]
        if seg_args:
            args = seg_args
            break
    if not args:
        return None                      # nothing undirected to install
    if os.path.isfile(os.path.join(project_root, "package.json")):
        return None                      # the root really is a package

    subs = []
    try:
        for entry in sorted(os.scandir(project_root), key=lambda e: e.name):
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            if entry.name == "node_modules":
                continue
            if os.path.isfile(os.path.join(entry.path, "package.json")):
                subs.append(entry.name)
    except OSError:
        return None
    if not subs:
        return None                      # nothing better to suggest

    pkgs = " ".join(args)
    return (
        f"ERROR: refusing to run '{command.strip()}'. There is no "
        f"package.json here, so npm would CREATE one along with a "
        f"node_modules and a lockfile - a top-level package belonging to "
        f"no project. The real package(s) are: {', '.join(subs)}.\n"
        f"Install into the one that needs it, e.g. "
        f"`npm --prefix {subs[0]} install {pkgs}`.")


def phantom_root_manifest_reason(rel_path: str, project_root: str,
                                 declared: "set[str] | None" = None) -> "str | None":
    r"""Why writing this root ``package.json`` would manufacture a phantom package.

    `rootless_npm_install_reason` refuses `npm install <pkg>` at a root
    that owns no manifest, because npm would CREATE one and leave a
    top-level package belonging to no project. That guard has an escape
    hatch -- if the root really is a package, the install is fine -- and
    the escape hatch was **agent-writable**.

    Measured 2026-08-19 run 28. Step 5.1's gate did `require('jsonwebtoken')`
    from the repo root, where it cannot resolve: the package lives in
    `backend/node_modules`. The env self-heal correctly installed into
    `backend/`, which did not help a root-level require, and the gate
    then failed six times over two artifacts. Everything upstream got it
    right -- `gate STALLED ... the gate is not measuring the artifact`,
    then `NOT escalating - the gate is the defect, not the code`, and
    `refused rootless npm install: npm install jsonwebtoken --no-save`.

    On the very next turn the agent wrote a root `package.json` with
    `write_file`, and the recovery loop then ran
    `npm install jsonwebtoken --save`, which the npm guard permitted
    because a root manifest now existed. The file it wrote --
    `{"name": "fullstack-auth-project", "private": true, "dependencies":
    {"jsonwebtoken": "^9.0.3"}}` -- is plainly hand-authored: `npm init
    -y` names the directory and emits version/main/scripts.

    The harm was not hypothetical and nothing reported it. The stray root
    manifest SHADOWED the frontend's, so `[SmokeTest] JS build check: npm
    run build (cwd=frontend)` became `[SmokeTest] No build script in
    package.json - skipping`, silently disabling the build and
    style-coupling checks that had run in the previous run. Both
    acceptance instruments still passed, because neither looks at the
    repo root.

    This is the sixth recorded instance of an unsatisfiable gate being
    answered by deforming the project rather than the code, after
    frontend/frontend/package.json, frontend/node.cmd,
    frontend/backend/.env.example, frontend/frontend/src/pages/*.jsx and
    the `mklink /J node_modules` junction.

    Narrow by construction, and by the same shape as the npm guard: it
    fires only when the root owns no manifest *and* some immediate
    subdirectory does -- the multi-root layout. A greenfield root install
    has no sibling manifests and is left alone; so is a root that already
    owns a manifest. And a target the PLAN declared is always allowed,
    because a workspaces root someone planned is a real decision, not an
    agent working around a measurement.
    """
    import os

    norm = (rel_path or "").replace("\\", "/").lstrip("/")
    while norm.startswith("./"):
        norm = norm[2:]
    if norm.lower() != "package.json":
        return None                      # only the ROOT manifest
    if norm in (declared or set()):
        return None                      # the plan asked for it
    if os.path.isfile(os.path.join(project_root, "package.json")):
        return None                      # editing an existing root package

    subs = []
    try:
        for entry in sorted(os.scandir(project_root), key=lambda e: e.name):
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            if entry.name == "node_modules":
                continue
            if os.path.isfile(os.path.join(entry.path, "package.json")):
                subs.append(entry.name)
    except OSError:
        return None
    if not subs:
        return None                      # nothing better to suggest

    return (
        "ERROR: refusing to create a top-level package.json. The real "
        "package(s) are: " + ", ".join(subs) + ", and no plan step "
        "declares a root manifest. A package.json here belongs to no "
        "project: it shadows the sub-projects' own manifests (a root one "
        "with no build script silently disables the frontend build "
        "check), and it makes a root `npm install` look legitimate when "
        "it is not.\n"
        "If a gate cannot resolve a package because it runs in the wrong "
        "directory, that is a defect in the GATE - run it where the "
        "manifest that owns the dependency lives, e.g. "
        "`npm --prefix " + subs[0] + " ...`, or say so in your summary. "
        "Do not give the root a manifest to make the gate pass.")


_DEP_DIR_NAMES = frozenset({"node_modules", "site-packages", "vendor",
                            "bower_components"})

# `mklink /J link target`, `mklink /D link target`, `ln -s target link`,
# and PowerShell's `New-Item -ItemType SymbolicLink|Junction -Path link`.
_LINK_CMD_RES = (
    re.compile(r"\bmklink\s+(?:/[A-Za-z]+\s+)*(?P<link>\S+)", re.IGNORECASE),
    re.compile(r"\bln\s+-s\w*\s+\S+\s+(?P<link>\S+)"),
    re.compile(r"New-Item\b.*?-ItemType\s+(?:SymbolicLink|Junction)"
               r".*?-(?:Path|Name)\s+(?P<link>\S+)",
               re.IGNORECASE | re.DOTALL),
)


def fabricated_dependency_link(command: str) -> Optional[str]:
    """The dependency directory *command* would fake by linking, if any.

    A dependency tree is created by a package manager, in the directory
    that owns the manifest. Aliasing one into another location makes
    resolution depend on a filesystem link that survives no clone, copy,
    archive or deploy — the project appears to work here and nowhere
    else.

    Measured 2026-08-19 run 23. A gate ran from the repo root and did a
    bare `require('jsonwebtoken')`, which cannot resolve there in a
    multi-root layout: the package lives in `backend/node_modules`.
    Rather than report the gate as wrongly scoped, the agent ran::

        mklink /J node_modules backend\\node_modules

    and the gate went green. Removing the junction afterwards left the
    application working exactly as before, so it carried no weight — it
    existed only to satisfy the measurement. That is the fifth recorded
    instance of an unsatisfiable gate being answered by deforming the
    project instead of the code, after frontend/frontend/package.json,
    frontend/node.cmd, frontend/backend/.env.example and
    frontend/frontend/src/pages/*.jsx.

    Judged on the link NAME only. Linking ordinary project directories is
    left alone; it is aliasing the dependency tree that turns a
    resolution failure into a hidden one.
    """
    for pattern in _LINK_CMD_RES:
        m = pattern.search(command or "")
        if not m:
            continue
        link = m.group("link").strip().strip("\"'")
        name = re.split(r"[\\/]+", link.rstrip("\\/"))[-1].lower()
        if name in _DEP_DIR_NAMES:
            return name
    return None


def shadowed_dist(rel_path: str, failed: set[str]) -> Optional[str]:
    """The failed distribution *rel_path* would shadow, if any.

    Only a NEW TOP-LEVEL module or package can shadow an import, so this
    looks at the first path component and nothing deeper: `pygame/draw.py`
    and `pygame.py` shadow, `src/pygame/draw.py` does not.
    """
    if not failed:
        return None
    parts = re.split(r"[\\/]+", rel_path.strip("\\/"))
    if not parts or not parts[0]:
        return None
    head = parts[0]
    if len(parts) == 1:
        if not head.endswith(".py"):
            return None
        head = head[:-3]
    candidate = _normalise_dist(head)
    return candidate if candidate in failed else None


def _truncate(text: str, limit: int, what: str = "output") -> str:
    if len(text) <= limit:
        return text
    return (text[:limit]
            + f"\n... [{what} truncated at {limit} chars"
              f" of {len(text)} total]")


class AgentTools:
    """Executable tool set scoped to one project root.

    Parameters
    ----------
    project_root:
        Directory all file paths resolve against; access outside it is
        rejected.
    executor:
        :class:`~agentchanti.executor.Executor` for ``run_command``.
        Created lazily when omitted.
    searcher:
        Optional KB :class:`~agentchanti.kb.local.searcher.Searcher` backing
        ``search_code``. Without it the tool degrades to a hint message.
    memory:
        Optional :class:`~agentchanti.orchestrator.memory.FileMemory`;
        writes/edits are recorded so the rest of the pipeline sees them.
    command_timeout:
        Seconds before ``run_command`` gives up.
    """

    def __init__(self, project_root: str = ".", executor=None,
                 searcher=None, memory=None, command_timeout: int = 120):
        self.project_root = os.path.abspath(project_root)
        self._executor = executor
        self._searcher = searcher
        self._memory = memory
        self._command_timeout = command_timeout
        # Distributions whose install failed during this step. Scoped to
        # the instance, and build_step_tools() builds one per step, so the
        # window closes when the step ends.
        self._failed_installs: set[str] = set()
        # Protected manifests this instance created. Only covers the current
        # step — build_step_tools() makes one instance per step — so the
        # cross-step answer comes from FileMemory, which is run-scoped.
        self._created_manifests: set[str] = set()
        # Files the run's acceptance commands invoke. See _acceptance_refusal.
        self._acceptance_files: set[str] = set()

    def protect_acceptance_files(self, paths) -> None:
        """Mark *paths* as the run's independent acceptance instruments.

        Normalised to project-root-relative POSIX form so a later write
        matches however the model spells the path.
        """
        for p in paths or ():
            if p:
                self._acceptance_files.add(
                    str(p).replace("\\", "/").lstrip("./").strip("/"))

    # ── Definitions ──

    def definitions(self) -> list[ToolDef]:
        return [
            ToolDef(
                name="list_files",
                description=(
                    "List files under a directory (recursive), relative to "
                    "the project root. Build artifacts and VCS directories "
                    "are skipped."),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string",
                                 "description": "Directory to list, relative "
                                                "to project root. Default: "
                                                "project root."},
                    },
                },
            ),
            ToolDef(
                name="read_file",
                description=(
                    "Read a file's content. Returns numbered lines. "
                    "Optionally restrict to a line range."),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string",
                                 "description": "File path relative to "
                                                "project root."},
                        "start_line": {"type": "integer",
                                       "description": "First line (1-based)."},
                        "end_line": {"type": "integer",
                                     "description": "Last line (inclusive)."},
                    },
                    "required": ["path"],
                },
            ),
            ToolDef(
                name="write_file",
                description=(
                    "Create or fully overwrite a file with the given "
                    "content. Use edit_file for partial changes to an "
                    "existing file."),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string",
                                 "description": "File path relative to "
                                                "project root."},
                        "content": {"type": "string",
                                    "description": "Complete file content."},
                    },
                    "required": ["path", "content"],
                },
            ),
            ToolDef(
                name="edit_file",
                description=(
                    "Replace one exact occurrence of old_text with new_text "
                    "in a file. old_text must match exactly (including "
                    "whitespace) and be unique in the file — include enough "
                    "surrounding lines to make it unique."),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string",
                                 "description": "File path relative to "
                                                "project root."},
                        "old_text": {"type": "string",
                                     "description": "Exact text to replace."},
                        "new_text": {"type": "string",
                                     "description": "Replacement text."},
                    },
                    "required": ["path", "old_text", "new_text"],
                },
            ),
            ToolDef(
                name="run_command",
                description=(
                    "Run a shell command in the project root and return its "
                    "combined output. Non-interactive; commands that prompt "
                    "for input will fail."),
                parameters={
                    "type": "object",
                    "properties": {
                        "command": {"type": "string",
                                    "description": "Shell command to run."},
                    },
                    "required": ["command"],
                },
            ),
            ToolDef(
                name="search_code",
                description=(
                    "Semantic search over the project's code (knowledge "
                    "base). Returns matching symbols with file, line range "
                    "and snippet."),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string",
                                  "description": "Natural-language query, "
                                                 "e.g. 'where is user auth "
                                                 "validated'."},
                    },
                    "required": ["query"],
                },
            ),
        ]

    # ── Execution ──

    def execute(self, call: ToolCall,
                allowed: "frozenset[str] | set[str] | None" = None) -> str:
        """Execute one tool call; always returns a string result.

        *allowed* is the set of tool names offered for this turn. A call to
        anything outside it is refused rather than run.

        Narrowing the offered tool list is not enough on its own: the loop
        withholds the read-only tools when a model spends turn after turn
        inspecting, but the model is free to ignore the list and ask anyway.
        Observed on gpt-oss:120b-cloud — the offer dropped to three acting
        tools and the next FOUR turns were still `read_file`, so a step
        burned its whole budget on seven reads and zero writes before
        escalating to a much more expensive model. The intervention had
        silently done nothing.

        The refusal comes back as an ordinary tool result, so the model
        reads "this is disabled" and can act on it, which is the same
        contract every other error here uses.
        """
        handler = getattr(self, f"_tool_{call.name}", None)
        # Unknown outranks withheld: a name this class has never had is a
        # different mistake from one deliberately taken away this turn, and
        # "disabled" would send the model looking for a way to re-enable it.
        if handler is None:
            names = ", ".join(t.name for t in self.definitions())
            return f"ERROR: unknown tool '{call.name}'. Available: {names}"
        if allowed is not None and call.name not in allowed:
            log.info("[AgentTools] refused withheld tool '%s' "
                     "(offered: %s)", call.name, ", ".join(sorted(allowed)))
            return (f"ERROR: '{call.name}' is disabled for this turn. "
                    f"Available: {', '.join(sorted(allowed))}. "
                    f"Use one of those to change something now.")
        try:
            return handler(**call.arguments)
        except TypeError as e:
            return f"ERROR: bad arguments for {call.name}: {e}"
        except Exception as e:
            log.warning(f"[AgentTools] {call.name} failed: {e}")
            return f"ERROR: {call.name} failed: {e}"

    def execute_all(self, calls: list[ToolCall],
                    allowed: "frozenset[str] | set[str] | None" = None
                    ) -> list[Message]:
        """Execute tool calls and wrap results as ``role="tool"`` messages.

        *allowed* restricts which tools may run this turn; see :meth:`execute`.
        Left as None every known tool runs, which is the behaviour for every
        turn that withholds nothing.
        """
        return [
            Message(role="tool", content=self.execute(c, allowed=allowed),
                    tool_call_id=c.id, tool_name=c.name)
            for c in calls
        ]

    # ── Helpers ──

    def _resolve(self, path: str) -> str:
        """Resolve *path* inside the project root; reject escapes."""
        full = os.path.abspath(os.path.join(self.project_root, path))
        if os.path.commonpath([full, self.project_root]) != self.project_root:
            raise ValueError(f"path '{path}' is outside the project root")
        return full

    def _acceptance_refusal(self, path: str, rel: str) -> str | None:
        """Refuse a write to a file the acceptance commands invoke, else None.

        `acceptance_cmds` is described throughout this project as the one
        instrument the model neither wrote nor can edit, and so the only
        check allowed to fail a run on its own. The first half was always
        true; the second half was not enforced for a command that invokes
        a FILE — only the command string in config was out of reach, and
        the script it runs sat in the project root like any other source.

        Observed 2026-08-19: a planner declared `target: acceptance_check.cjs`
        on a TEST step. It behaved — the step's description said "run the
        supplied unchanged acceptance checker" and the bytes were identical
        afterwards — but nothing had made that the only possible outcome,
        and a run that rewrites its own acceptance check reports independent
        evidence for a contract it authored. That is the oldest cheat in
        this codebase's history, already documented for seeded tests.

        Reading is untouched: the model may inspect what it must satisfy.
        """
        if rel.replace("\\", "/") not in self._acceptance_files:
            return None
        log.warning("[AgentTools] refused write to acceptance instrument "
                    "'%s'", rel)
        return (
            f"ERROR: refusing to modify '{path}'. It is an acceptance "
            f"check supplied by the operator via `acceptance_cmds`, and it "
            f"is the only verification in this run that you did not write. "
            f"Editing it would make the run's independent evidence "
            f"self-authored, which is worth less than no evidence at all.\n"
            f"Read it as often as you like and change the PROJECT until it "
            f"passes. If you believe the check itself is wrong, say so in "
            f"your summary — do not edit it.")

    def _record(self, rel_path: str, content: str) -> None:
        """Record a write that has ALREADY landed on disk.

        ``allow_protected`` matters here. FileMemory refuses to track a
        protected manifest basename (requirements.txt, package.json, …) that
        exists on disk, to stop a hallucinated replacement clobbering a real
        one. But this method is only ever called *after* the write
        succeeded, so the guard cannot prevent anything — it can only make
        memory disagree with the filesystem, while logging a WARNING that
        claims a skip which did not happen.

        Observed in 4 of 6 benchmark runs: the loop wrote requirements.txt,
        the gate read 'pygame' back off disk and passed, and the content was
        absent from FileMemory for the rest of the run — invisible to
        dependency checks, context injection and the checkpoint's
        file_memory.
        """
        if self._memory is not None:
            try:
                self._memory.update({rel_path: content},
                                    allow_protected={rel_path})
            except Exception as e:
                log.debug(f"[AgentTools] FileMemory update failed: {e}")

    # ── Tool implementations ──

    def _tool_list_files(self, path: str = ".") -> str:
        root = self._resolve(path)
        if not os.path.isdir(root):
            return f"ERROR: '{path}' is not a directory"
        entries: list[str] = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames
                                 if d not in _IGNORED_DIRS
                                 and not d.startswith("."))
            for fname in sorted(filenames):
                rel = os.path.relpath(os.path.join(dirpath, fname),
                                      self.project_root)
                entries.append(rel.replace("\\", "/"))
                if len(entries) >= _MAX_LIST_ENTRIES:
                    entries.append(f"... [listing truncated at "
                                   f"{_MAX_LIST_ENTRIES} entries]")
                    return "\n".join(entries)
        return "\n".join(entries) if entries else "(empty directory)"

    def _tool_read_file(self, path: str, start_line: Optional[int] = None,
                        end_line: Optional[int] = None) -> str:
        full = self._resolve(path)
        if not os.path.isfile(full):
            return f"ERROR: file not found: {path}"
        with open(full, "r", encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
        lo = max(1, start_line or 1)
        hi = min(len(lines), end_line or len(lines))
        if lo > len(lines):
            return f"ERROR: start_line {lo} beyond end of file ({len(lines)} lines)"
        numbered = "\n".join(
            f"{i}: {lines[i - 1]}" for i in range(lo, hi + 1))
        header = f"{path} (lines {lo}-{hi} of {len(lines)})\n"
        return header + _truncate(numbered, _MAX_READ_CHARS, "file content")

    def _tool_write_file(self, path: str, content: str) -> str:
        full = self._resolve(path)
        rel = os.path.relpath(full, self.project_root)
        refusal = self._acceptance_refusal(path, rel)
        if refusal is not None:
            return refusal
        phantom = phantom_root_manifest_reason(
            rel, self.project_root,
            getattr(self._memory, "_plan_declared_files", None))
        if phantom is not None:
            log.warning("[AgentTools] refused phantom root manifest '%s'", rel)
            return phantom
        shim = toolchain_shim(rel)
        if shim is not None:
            log.warning("[AgentTools] refused toolchain shim '%s' (would "
                        "shadow '%s')", rel, shim)
            return (
                f"ERROR: refusing to write '{path}'. A file named after "
                f"'{shim}' is found before the real tool when anything runs "
                f"from that directory, so this would silently replace the "
                f"{shim} the rest of the run - and the user - depends on.\n"
                f"If a gate cannot be satisfied because a tool lacks a "
                f"capability (Node cannot parse JSX, for instance), that is "
                f"a defect in the GATE. Say so in your summary; do not "
                f"replace the tool.")
        dist = shadowed_dist(rel, self._failed_installs)
        if dist is not None:
            log.warning(f"[AgentTools] refused shadow write '{rel}' "
                        f"(install of '{dist}' failed in this step)")
            return (
                f"ERROR: refusing to write '{path}'. Installing '{dist}' "
                f"failed earlier in this step, and a top-level '{dist}' "
                f"module here would shadow the real package on sys.path — "
                f"imports would resolve to this file instead of the "
                f"library. A local stub makes the import succeed while the "
                f"functionality stays missing, so every later test would "
                f"pass against code that does nothing.\n"
                f"Fix the installation instead (check the version actually "
                f"has a wheel for this Python, or install without pinning "
                f"a version), or report that the dependency cannot be "
                f"installed. If you genuinely need a project module with "
                f"this name, put it under a package directory rather than "
                f"at the project root.")
        guard = self._protected_overwrite_error(rel, full, path)
        if guard is not None:
            return guard
        os.makedirs(os.path.dirname(full) or self.project_root, exist_ok=True)
        with open(full, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
        rel_key = os.path.relpath(full, self.project_root).replace("\\", "/")
        if os.path.basename(rel_key) in _protected_basenames():
            self._created_manifests.add(rel_key)
        self._record(os.path.relpath(full, self.project_root), content)
        return f"OK: wrote {len(content)} chars to {path}"

    def _protected_overwrite_error(self, rel: str, full: str,
                                   path: str) -> "str | None":
        """Refuse a whole-file overwrite of a manifest this run did not write.

        The classic writer has always had this guard; the loop's write_file
        went straight to disk, so a model could replace a real
        requirements.txt or package.json with a shorter regenerated one and
        silently drop dependencies. It is a whole-file rewrite that does the
        damage, so edit_file — exact-match, single-occurrence, grounded in
        the current content — stays available and is the right way to add a
        dependency.

        Creating a manifest is legitimate and common (5 of 8 benchmark runs
        did it), so the test is create-versus-overwrite, not
        existence. FileMemory answers it across steps because
        build_step_tools() makes a fresh AgentTools per step.
        """
        rel_key = rel.replace("\\", "/")
        if os.path.basename(rel_key) not in _protected_basenames():
            return None
        if not os.path.isfile(full):
            return None                      # creating it — allowed
        if rel_key in self._created_manifests:
            return None                      # this step wrote it
        if self._memory is not None and self._memory.get(rel_key) is not None:
            return None                      # an earlier step in this run did
        log.warning(f"[AgentTools] refused overwrite of pre-existing "
                    f"manifest '{rel_key}'")
        return (
            f"ERROR: refusing to overwrite '{path}'. It already existed "
            f"before this run, so it is the project's real manifest, and a "
            f"regenerated replacement almost always drops dependencies that "
            f"are still needed — every later step would then build against "
            f"a different dependency set than the project actually has.\n"
            f"Use edit_file to change one entry at a time: it matches the "
            f"current content exactly, so it cannot silently discard the "
            f"rest of the file. Read the file first if you need to see what "
            f"is in it.")

    def _tool_edit_file(self, path: str, old_text: str, new_text: str) -> str:
        full = self._resolve(path)
        rel = os.path.relpath(full, self.project_root)
        refusal = self._acceptance_refusal(path, rel)
        if refusal is not None:
            return refusal
        # Creation is already blocked in _tool_write_file, so this covers
        # the other direction: a shim the project legitimately ships (or
        # one an earlier run left behind) is not the agent's to rewrite.
        shim = toolchain_shim(rel)
        if shim is not None:
            log.warning("[AgentTools] refused edit of toolchain shim '%s' "
                        "(would shadow '%s')", rel, shim)
            return (
                f"ERROR: refusing to edit '{path}'. It shadows the real "
                f"'{shim}' for anything run from that directory, so changing "
                f"it changes the tool the rest of the run depends on. If a "
                f"gate cannot be satisfied because a tool lacks a "
                f"capability, that is a defect in the GATE — say so in your "
                f"summary rather than altering the tool.")
        if not os.path.isfile(full):
            return f"ERROR: file not found: {path}"
        with open(full, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        count = content.count(old_text)
        if count == 0:
            return ("ERROR: old_text not found in file. It must match "
                    "exactly, including whitespace and indentation. "
                    "Re-read the file and try again.")
        if count > 1:
            return (f"ERROR: old_text matches {count} locations. Include "
                    "more surrounding lines to make it unique.")

        updated = content.replace(old_text, new_text, 1)

        # Syntax-validate Python edits before committing them to disk.
        # compile(), not ast.parse: the latter accepts a `from __future__
        # import ...` that is no longer at the top of the file, which the
        # interpreter rejects outright.
        if full.endswith(".py"):
            from .py_syntax import check_python_syntax
            err = check_python_syntax(updated, full)
            if err:
                return (f"ERROR: edit rejected — resulting Python has a "
                        f"syntax error: {err}")
        # Same for JSON: a structurally broken package.json breaks every
        # subsequent npm/node invocation with a confusing downstream error.
        # tsconfig*.json is JSONC (comments/trailing commas allowed) — skip.
        if (full.endswith(".json")
                and not os.path.basename(full).startswith("tsconfig")):
            import json as _json
            try:
                _json.loads(updated)
            except ValueError as e:
                return (f"ERROR: edit rejected — resulting JSON is invalid: "
                        f"{e}. Re-read the file and fix commas/braces.")

        with open(full, "w", encoding="utf-8", newline="\n") as f:
            f.write(updated)
        self._record(os.path.relpath(full, self.project_root), updated)
        return f"OK: replaced 1 occurrence in {path}"

    def _tool_run_command(self, command: str) -> str:
        faked = fabricated_dependency_link(command)
        if faked is not None:
            log.warning("[AgentTools] refused fabricated %s link: %s",
                        faked, command.strip()[:120])
            return (
                f"ERROR: refusing to run '{command.strip()}'. Linking a "
                f"'{faked}' directory into another location fakes module "
                f"resolution: it survives no clone, copy, archive or "
                f"deploy, so the project would appear to work here and "
                f"nowhere else.\n"
                f"If a gate cannot resolve a package, the gate is running "
                f"in the wrong directory — run it where the manifest lives "
                f"(`npm --prefix <dir> ...`, or require the package by its "
                f"real path). Say so in your summary rather than aliasing "
                f"the dependency tree.")
        rootless = rootless_npm_install_reason(command, self.project_root)
        if rootless is not None:
            log.warning("[AgentTools] refused rootless npm install: %s",
                        command.strip()[:120])
            return rootless
        stripped_pipe = ""
        if os.name == "nt":
            _clean = _POSIX_OUTPUT_PIPE_RE.sub("", command).strip()
            if _clean and _clean != command.strip():
                # `head`/`tail` do not exist on Windows, so the whole
                # pipeline fails and the model learns nothing about the
                # command it was actually trying to run — observed on a
                # Pygame run, three turns spent on `... | head -100` and
                # `... | head -150` against a test suite whose output it
                # never saw. Drop the pipe and run the real command; the
                # output is length-capped here anyway, which is all the
                # pipe was for.
                stripped_pipe = command.strip()[len(_clean):].strip()
                command = _clean
        if os.name == "nt" and _HEREDOC_RE.search(command):
            return ("ERROR: POSIX heredoc syntax (<<) does not work on "
                    "Windows cmd — the command would fail without a useful "
                    "error. Write the script to a file with write_file and "
                    "run that file, or use python -c \"...\" for one-liners.")
        if self._executor is None:
            from .executor import Executor
            self._executor = Executor()
        success, output = self._executor.run_command(
            command, timeout=self._command_timeout, cwd=self.project_root)
        if not success:
            # Arm the shadow guard for whatever this install was after.
            self._failed_installs |= parse_failed_install_targets(command)
        status = "exit: success" if success else "exit: FAILED"
        body = _truncate(output or "(no output)", _MAX_CMD_OUTPUT_CHARS)
        # "exit: FAILED" reads identically whether an assertion failed or
        # the runner never found a test to run. Say which.
        #
        # NOT gated on failure. Whether a zero-test run exits non-zero is
        # CPython policy: unittest only gained that status in 3.12, so on
        # 3.10 and 3.11 discovering nothing reports "exit: success" — a
        # green result backed by zero executed tests, which is the more
        # dangerous of the two cases and the one that most needs saying.
        hint = ""
        if _no_tests_collected(
                command, getattr(self._executor, "last_exit_code", None),
                output):
            hint = _NO_TESTS_HINT
        if stripped_pipe:
            hint += (f"\n[note] Dropped `{stripped_pipe}` — head/tail/more do "
                     f"not exist on Windows and the pipeline would have "
                     f"failed before running anything. Output is length-"
                     f"capped here already; do not add output pipes.")
        return f"{status}\n{body}{hint}"

    def _tool_search_code(self, query: str) -> str:
        if self._searcher is None:
            return ("search_code unavailable (no knowledge base index). "
                    "Use list_files and read_file to explore instead.")
        results = self._searcher.search(query, top_k=5)
        if not results:
            return f"No results for: {query}"
        parts = []
        for r in results:
            snippet = _truncate(r.code_snippet or "", 1_200, "snippet")
            parts.append(
                f"{r.file}:{r.line_start}-{r.line_end} "
                f"[{r.symbol_type}] {r.symbol_name} (score {r.score:.2f})\n"
                f"{snippet}")
        return "\n\n".join(parts)
