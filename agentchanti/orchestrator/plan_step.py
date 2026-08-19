"""
PlanStep — structured execution plan data model and parser.

Replaces the old numbered-text plan format with a line-based structured
format that encodes step type, dependencies, target files, and
import/export relationships in a single LLM output.

Format
------
::

    ==PLAN==

    --STEP 1.1 [CMD] depends:none
    Create React project with Vite
    > npm create vite@latest my-app -- --template react-ts
    produces: package.json, vite.config.ts, src/main.tsx, src/App.tsx

    --STEP 2.1 [CODE] depends:1.1
    Create Header component
    target: src/components/Header.tsx
    exports: Header
    imports: none

    ==END==
"""

from __future__ import annotations

import logging
import ast
import posixpath
import re
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class PlanStep:
    """Structured representation of a single pipeline step."""

    id: str                                     # e.g. "1.1", "2.1", "3.2"
    step_type: str                              # CMD, CODE, TEST, IGNORE
    description: str = ""                       # human-readable description
    depends_on: list[str] = field(default_factory=list)
    command: Optional[str] = None               # shell command for CMD steps
    target_files: list[str] = field(default_factory=list)
    exports: list[str] = field(default_factory=list)
    imports_from: dict[str, list[str]] = field(default_factory=dict)  # file -> [symbols]
    status: str = "pending"                     # pending, in_progress, completed, failed, skipped
    actual_exports: list[str] = field(default_factory=list)  # filled after step execution
    inline_code: dict[str, str] = field(default_factory=dict)  # file -> code from plan
    inline_edits: dict[str, list[tuple[str, str]]] = field(default_factory=dict)  # file -> [(find, replace), ...]
    kb_docs: list[str] = field(default_factory=list)  # KB doc titles used when writing inline code
    verify_cmd: Optional[str] = None  # plan-declared acceptance command for this step

    # Which files should import this step's target file (plan-declared or derived)
    imported_by: list[str] = field(default_factory=list)

    # Legacy compat: 0-based integer index assigned after parsing
    index: int = -1

    def to_dict(self) -> dict:
        """Serialize for checkpoint / JSON."""
        d = {
            "id": self.id,
            "step_type": self.step_type,
            "description": self.description,
            "depends_on": self.depends_on,
            "command": self.command,
            "target_files": list(self.target_files),
            "exports": list(self.exports),
            "imports_from": {k: list(v) for k, v in self.imports_from.items()},
            "status": self.status,
            "actual_exports": list(self.actual_exports),
            "index": self.index,
        }
        if self.inline_code:
            d["inline_code"] = dict(self.inline_code)
        if self.inline_edits:
            d["inline_edits"] = {k: list(v) for k, v in self.inline_edits.items()}
        if self.kb_docs:
            d["kb_docs"] = list(self.kb_docs)
        if self.imported_by:
            d["imported_by"] = list(self.imported_by)
        if self.verify_cmd:
            d["verify_cmd"] = self.verify_cmd
        return d

    @classmethod
    def from_dict(cls, d: dict) -> PlanStep:
        """Deserialize from checkpoint / JSON."""
        return cls(
            id=d.get("id", "0"),
            step_type=d.get("step_type", "CODE"),
            description=d.get("description", ""),
            depends_on=d.get("depends_on", []),
            command=d.get("command"),
            target_files=d.get("target_files", []),
            exports=d.get("exports", []),
            imports_from={k: list(v) for k, v in d.get("imports_from", {}).items()},
            status=d.get("status", "pending"),
            actual_exports=d.get("actual_exports", []),
            inline_code=d.get("inline_code", {}),
            inline_edits={k: [tuple(p) for p in v] for k, v in d.get("inline_edits", {}).items()},
            kb_docs=d.get("kb_docs", []),
            imported_by=d.get("imported_by", []),
            index=d.get("index", -1),
            verify_cmd=d.get("verify_cmd"),
        )


def plan_looks_truncated(plan_text: str,
                         steps: Optional[list["PlanStep"]] = None
                         ) -> tuple[bool, str]:
    """Heuristically detect a plan whose generation was cut off mid-way.

    Returns ``(True, reason)`` when the plan is very likely incomplete. A
    truncated plan can still parse cleanly and run (observed: a 6-step stub
    for a ~15-file task, cut mid-step, that shipped only because downstream
    recovery backfilled the missing files) — so it must be caught before
    execution rather than trusted.

    Mechanical signals only; the caller folds in the provider's
    ``_last_truncated`` (output-token-cap) flag separately:

      * a structured plan opened with ``==PLAN==`` but never closed with
        the mandatory ``==END==`` marker;
      * the last parsed CODE/TEST step has no body at all (no target,
        inline code/edit, verify, or command) — its content block was cut.
    """
    text = (plan_text or "").rstrip()
    if not text:
        return False, ""
    if "==PLAN==" in text and "==END==" not in text:
        return True, "structured plan was cut off before the ==END== marker"
    if steps:
        last = steps[-1]
        if last.step_type in ("CODE", "TEST") and not (
                last.target_files or last.inline_code or last.inline_edits
                or last.verify_cmd or last.command):
            return True, (f"last step {last.id} has no body — its content "
                          f"block appears truncated")
    return False, ""


def plan_salvageable(steps: Optional[list["PlanStep"]]) -> bool:
    """True when a plan flagged truncated only by the missing ``==END==``
    marker is safe to run anyway.

    A full re-plan costs a complete second generation and churns every
    path in the plan (observed: a re-plan renamed the project dir, so
    steps from the two plans referenced different trees). When every
    parsed step is structurally complete — the cut, if any, landed on the
    marker itself — the plan is almost certainly whole and salvaging it
    is cheaper and safer than regenerating. Callers must still require
    that the provider's output-token-cap flag did NOT fire: a genuine cap
    hit means later steps may be missing entirely.
    """
    if not steps or len(steps) < 3:
        return False
    last = steps[-1]
    # Mirrors the bodyless-last-step truncation signal above: a complete
    # final step means the generation reached a step boundary.
    return bool(last.target_files or last.inline_code or last.inline_edits
                or last.verify_cmd or last.command)


# ---------------------------------------------------------------------------
# Echo command parser
# ---------------------------------------------------------------------------

# Regex for redirect operators in echo commands. Whitespace around the
# operator is optional: planners emit both ``echo x > f`` and the compact
# ``echo x>f`` / ``echo x>>f`` forms (the latter is what Windows one-liners
# use).
_ECHO_REDIR_RE = re.compile(r'\s*(>{1,2})\s*([^>]+)$')
# Regex for "type nul > file" (Windows) or "touch file" (Unix)
_NUL_RE = re.compile(r'^(?:type\s+nul|touch)\s+>\s+(.+)$')
_TOUCH_RE = re.compile(r'^touch\s+(.+)$')
# Regex for a cmd.exe blank-line echo: ``echo.>> file`` / ``echo. > file``
_ECHO_DOT_RE = re.compile(r'^echo\.\s*(>{1,2})\s*(.+)$')


def _parse_echo_commands(lines: list[str]) -> dict[str, str]:
    """Parse shell-style file creation commands from plan step command lines.

    Extracts file paths and contents from patterns like::

        > cd my-app && echo "import React from 'react'" >> src/App.jsx
        > cd my-app && type nul > src/index.css
        > echo "export default {}" >> config.js

    Handles:
    - ``echo "content" >> file`` (append) and ``echo "content" > file`` (overwrite)
    - ``type nul > file`` / ``touch file`` (empty file creation)
    - ``cd dir && ...`` chains (strips the cd prefix)
    - Escaped quotes inside content (``\\"`` → ``"``)
    - ``\\n`` escape sequences → actual newlines
    - Common LLM typo ``src.App.jsx`` → ``src/App.jsx``

    Parameters
    ----------
    lines : list[str]
        Raw command lines (with ``> `` prefix already stripped).

    Returns
    -------
    dict[str, str]
        Mapping of file path → assembled content.  Empty dict if no
        echo commands were found.
    """
    files: dict[str, str] = {}

    for raw_line in lines:
        # Split on ' && ' to handle chained commands
        parts = raw_line.split(' && ')
        for part in parts:
            part = part.strip()
            # Planners often wrap each redirection in parentheses, e.g.
            # ``(echo import x > f)`` — strip one wrapping pair so the
            # ``echo``/redirect matching below still fires.
            if part.startswith('(') and part.endswith(')'):
                part = part[1:-1].strip()
            # Skip bare 'cd' commands and 'mkdir' commands
            if not part or part.startswith('cd ') or part.startswith('mkdir '):
                continue

            # ── Blank line: cmd.exe ``echo.>> f`` / ``echo. > f`` ──
            m_blank = _ECHO_DOT_RE.match(part)
            if m_blank:
                operator = m_blank.group(1)
                fpath = _fix_dot_paths(m_blank.group(2).strip().strip('"\''))
                if operator == '>':
                    files[fpath] = "\n"
                else:
                    files[fpath] = files.get(fpath, "") + "\n"
                continue

            # ── Empty file creation: type nul > file / touch file ──
            m_nul = _NUL_RE.match(part)
            if not m_nul:
                m_nul = _TOUCH_RE.match(part)
            if m_nul:
                fpath = m_nul.group(1).strip().strip('"\'')
                fpath = _fix_dot_paths(fpath)
                if fpath not in files:
                    files[fpath] = ""
                continue

            # ── Echo with redirect ──
            m_redir = _ECHO_REDIR_RE.search(part)
            if m_redir and part.startswith('echo '):
                operator = m_redir.group(1)
                fpath = m_redir.group(2).strip().strip('"\'')
                fpath = _fix_dot_paths(fpath)

                # Extract the content between 'echo ' and the redirect
                content_echo = part[5:m_redir.start()].strip()
                content_echo = _unescape_echo(content_echo)

                if operator == '>':
                    files[fpath] = content_echo + "\n"
                else:  # '>>' append
                    if fpath not in files:
                        files[fpath] = ""
                    files[fpath] += content_echo + "\n"

    return files


def _fix_dot_paths(fpath: str) -> str:
    """Fix common LLM typo: ``src.App.jsx`` → ``src/App.jsx``.

    Only fixes when the first segment looks like a directory name
    (e.g. ``src``, ``lib``, ``app``, ``components``), so legitimate
    dotted filenames like ``postcss.config.mjs`` or ``vite.config.ts``
    are preserved.
    """
    # If path already has slashes, it's fine
    if '/' in fpath or '\\' in fpath:
        return fpath
    # Split on dots; if >2 parts and last part is an extension, fix
    parts = fpath.split('.')
    if len(parts) > 2:
        ext = parts[-1]
        first = parts[0].lower()
        # Common source extensions
        if ext in ('js', 'jsx', 'ts', 'tsx', 'css', 'mjs', 'cjs',
                   'py', 'html', 'json', 'yaml', 'yml', 'toml',
                   'md', 'txt', 'cfg', 'ini', 'xml', 'svg'):
            # Only fix if the first segment looks like a directory
            _dir_prefixes = {
                'src', 'lib', 'app', 'components', 'pages', 'views',
                'utils', 'helpers', 'hooks', 'services', 'api',
                'routes', 'middleware', 'models', 'controllers',
                'public', 'static', 'assets', 'styles', 'tests',
                '__tests__', 'test', 'spec',
            }
            if first in _dir_prefixes:
                return '/'.join(parts[:-1]) + '.' + ext
    return fpath


def _unescape_echo(content: str) -> str:
    """Unescape echo content: strip surrounding quotes and decode escapes."""
    # cmd.exe caret escapes: ``^X`` → ``X`` (used to preserve leading
    # spaces / escape special chars in one-line echo redirections, e.g.
    # ``echo ^  plugins:``). Do this first so the quote/whitespace handling
    # below sees the real content.
    if '^' in content:
        content = re.sub(r'\^(.)', r'\1', content)
    if content.startswith('"') and content.endswith('"'):
        content = content[1:-1]
        content = content.replace('\\"', '"')
    elif content.startswith("'") and content.endswith("'"):
        content = content[1:-1]
        content = content.replace("\\'", "'")
    # Decode \n escape sequences to actual newlines
    content = content.replace('\\n', '\n')
    return content


def _flush_echo_inline(step: PlanStep, cmd_lines: list[str]) -> None:
    """Parse echo commands from a step's command lines and populate inline_code.

    Only populates ``inline_code`` if it is currently empty (i.e.
    ``---file-content-start---`` was not already used).
    """
    if step.inline_code:
        return  # ---file-content-start--- takes priority
    if not cmd_lines:
        return
    echo_files = _parse_echo_commands(cmd_lines)
    if echo_files:
        step.inline_code.update(echo_files)


def _looks_like_file_write_segment(seg: str) -> bool:
    """True if *seg* is a single file-creation redirection segment."""
    seg = seg.strip()
    if seg.startswith('(') and seg.endswith(')'):
        seg = seg[1:-1].strip()
    if _NUL_RE.match(seg) or _TOUCH_RE.match(seg) or _ECHO_DOT_RE.match(seg):
        return True
    return bool(seg.startswith('echo ') and _ECHO_REDIR_RE.search(seg))


def _cmd_is_pure_file_creation(command: str) -> tuple[bool, Optional[str]]:
    """Return ``(True, cd_dir)`` when *command* does nothing but write file
    content via shell redirection (``echo``/``type nul``/``touch``), aside
    from ``cd``/``mkdir`` preamble.

    A single real command anywhere in the chain (``npm``, ``node``, ...)
    returns ``(False, None)`` — such steps must keep running as CMD.
    ``cd_dir`` is the directory a leading ``cd`` switched to, so the
    reconstructed relative paths can be re-prefixed.
    """
    cd_dir: Optional[str] = None
    saw_write = False
    for seg in command.split('&&'):
        s = seg.strip()
        if s.startswith('(') and s.endswith(')'):
            s = s[1:-1].strip()
        if not s:
            continue
        if s.startswith('cd '):
            if cd_dir is None:
                cd_dir = s[3:].strip().strip('"\'')
            continue
        if s.startswith('mkdir '):
            continue
        if _looks_like_file_write_segment(s):
            saw_write = True
            continue
        return False, None  # a genuine command — leave the step as CMD
    return saw_write, cd_dir


_JS_LIKE_EXTS = ('.js', '.jsx', '.ts', '.tsx', '.mjs', '.cjs', '.mts', '.cts')

_BARE_CD_RE = re.compile(r'^cd\s+("[^"]+"|\S+)$')


def dedupe_redundant_cd(command: Optional[str]) -> Optional[str]:
    """Drop `cd X` segments that re-enter the directory the chain is
    already in.

    Multi-line CMD steps are written by the planner as independent lines,
    each assuming the project root — but they are joined into one
    ``&&`` chain where the cwd persists. The second ``cd app`` then runs
    from *inside* ``app`` and fails with "The system cannot find the path
    specified", killing an otherwise-correct install chain.
    """
    if not command or "&&" not in command:
        return command
    segments = [s.strip() for s in command.split("&&")]
    current: Optional[str] = None
    kept: list[str] = []
    dropped = 0
    for seg in segments:
        m = _BARE_CD_RE.match(seg)
        if m:
            target = m.group(1).strip('"')
            if current is not None and target == current:
                dropped += 1
                continue  # already effectively in this directory
            current = target
        kept.append(seg)
    if not dropped:
        return command
    return " && ".join(kept)


def route_blind_edits(steps: list[PlanStep],
                      project_root: str = ".") -> list[str]:
    """Neutralize ``edit:`` blocks written against files that do not exist
    at plan time (they will be created by an earlier step, e.g. a scaffold
    command) — their FIND text is a guess that cannot be trusted.

    Called by the CLI after parsing (not inside the parser, whose other
    callers run outside the project directory). Per edit target that is
    absent on disk:

      * exactly one substantial REPLACE that looks like a complete file
        (JS-likes must carry an export) → promoted to ``inline_code`` as a
        deterministic full-file overwrite;
      * anything else → the pairs are dropped, so the step reaches the
        grounded coder/agent-loop path instead of attempting doomed FIND
        matching against content the planner never saw.

    Existing files are left alone — their edits are legitimately grounded
    in the [FILES TO MODIFY] source the planner was shown. Returns a list
    of human-readable notes for logging.
    """
    import os
    notes: list[str] = []
    for step in steps:
        if not step.inline_edits:
            continue
        for path in list(step.inline_edits):
            if os.path.exists(os.path.join(project_root, path)):
                continue  # grounded edit — planner saw this file
            pairs = step.inline_edits.pop(path)
            substantial = [r.strip() for _, r in pairs if len(r.strip()) > 50]
            _ext = os.path.splitext(path)[1].lower()
            complete = (
                len(substantial) == 1
                and (_ext not in _JS_LIKE_EXTS
                     or 'export ' in substantial[0]
                     or 'module.exports' in substantial[0])
            )
            if complete and path not in step.inline_code:
                step.inline_code[path] = substantial[0] + "\n"
                notes.append(
                    f"Step {step.id}: edit on plan-created file {path} — "
                    "converted single complete REPLACE to full-file write")
            else:
                notes.append(
                    f"Step {step.id}: dropped {len(pairs)} blind edit "
                    f"pair(s) for plan-created file {path} — step will be "
                    "implemented against the real file")
    return notes


def _reclassify_file_creation_cmds(steps: list[PlanStep]) -> None:
    """Convert CMD steps that only write file *content* via echo/redirect
    chains into CODE steps.

    Planners sometimes emit a source file as ``(echo ... > f) && (echo ...
    >> f)``. On Windows cmd.exe this dies on the first ``[``/``{``/``(`` in
    the content (``] was unexpected at this time``), and it is brittle even
    on POSIX. The file content is already reconstructed into ``inline_code``
    by :func:`_flush_echo_inline`, so writing it directly as a CODE step is
    reliable and cross-platform.
    """
    for step in steps:
        if step.step_type != "CMD" or not step.command or not step.inline_code:
            continue
        ok, cd_dir = _cmd_is_pure_file_creation(step.command)
        if not ok:
            continue
        # Reconstructed paths are relative to any leading ``cd`` dir.
        prefix = f"{cd_dir}/" if cd_dir else ""
        step.inline_code = {
            _norm_target_path(prefix + p): c
            for p, c in step.inline_code.items()
        }
        step.target_files = list(step.inline_code.keys())
        step.step_type = "CODE"
        step.command = None


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

_STEP_RE = re.compile(
    r"^--STEP\s+([\d.]+)\s+\[(\w+)\]\s*(?:depends?:(.*))?",
    re.IGNORECASE,
)


def _norm_target_path(path: str) -> str:
    """Normalise a planner-emitted file path.

    Planners sometimes emit doubled backslashes (``main\\\\templates\\...``)
    or mixed separators. Left raw, these survive into memory keys as
    ``main//templates//...`` and poison every downstream consumer (the
    Django probe derived template names like ``/main//base.html`` and
    failed verification against a working app).
    """
    return re.sub(r"[\\/]+", "/", path.strip()).lstrip("./")


def parse_structured_plan(text: str) -> list[PlanStep]:
    """Parse the line-based structured plan format into PlanStep objects.

    Resilient to minor LLM formatting errors — each line is parsed
    independently. Unknown lines are treated as description continuation.

    Inline code between ``---file-content-start---`` and
    ``---file-content-end---`` markers is captured into
    ``PlanStep.inline_code``.  When the step has a single target file,
    the code is mapped to that file.  When multiple targets exist, the
    parser looks for ``// FileName.jsx`` comment headers to split the
    code into per-file blocks.

    When a reasoning model emits multiple draft plans in its thinking
    output, only the LAST ``==PLAN==...==END==`` block is used so that
    intermediate drafts don't bloat the step count.
    """
    # Extract the last ==PLAN== ... ==END== block (handles reasoning models
    # that include multiple draft plans in their thinking output).
    # Use rfind so partial/unclosed ==PLAN== blocks in thinking text don't
    # cause the regex to merge everything into one giant block.
    _upper = text.upper()
    _last_plan = _upper.rfind("==PLAN==")
    if _last_plan >= 0:
        # Use rfind so that small LLMs that emit ==END== after EVERY step
        # (instead of only once at the end) don't cause premature truncation.
        _end_after = _upper.rfind("==END==", _last_plan)
        if _end_after > _last_plan:
            text = text[_last_plan + len("==PLAN=="):_end_after]
        else:
            # No ==END== found after last ==PLAN== — use everything after it
            text = text[_last_plan + len("==PLAN=="):]

    steps: list[PlanStep] = []
    current: Optional[PlanStep] = None
    desc_lines: list[str] = []
    cmd_lines: list[str] = []        # all '> ...' lines for echo parsing
    in_code_block = False
    code_lines: list[str] = []

    in_markdown_fence = False  # track ```...``` inside inline blocks

    # edit: block state
    in_edit_block = False       # between edit: and <<<END>>>
    _edit_section = "find"      # "find" or "replace"
    _edit_find_lines: list[str] = []
    _edit_replace_lines: list[str] = []
    _edit_target: Optional[str] = None  # file path for current edit block
    # Ordinal of the current bare `edit:` block within the step. Multi-target
    # steps emit one bare `edit:` block per target file, in target order —
    # defaulting every block to target_files[0] mis-keyed the 2nd file's
    # edits onto the 1st (observed: main.jsx's REPLACE merged into App.jsx,
    # producing a duplicate `App` declaration that broke the build).
    _edit_block_ord = -1

    def _default_edit_target() -> str:
        if current is None or not current.target_files:
            return ""
        idx = min(max(_edit_block_ord, 0), len(current.target_files) - 1)
        return current.target_files[idx]

    for raw_line in text.splitlines():
        line = raw_line.strip()

        # ── Edit block handling (find/replace patches) ──
        # Format:
        #   edit:
        #   <<<FIND>>>
        #   old text
        #   <<<REPLACE>>>
        #   new text
        #   <<<END>>>
        if in_edit_block:
            _lnorm = line.upper().replace(" ", "")
            # Structural markers (new step header or plan end) terminate the
            # edit block — fall through to be processed normally.
            if line.upper() in ("==END==",) or _STEP_RE.match(line):
                in_edit_block = False
                _edit_find_lines = []
                _edit_replace_lines = []
                _edit_target = None
                # Fall through to process this line as a step header / ==END==
            elif line.lower().startswith("edit:"):
                # A new edit: header while already inside an edit block —
                # the planner emits one bare block per target file. Without
                # this, the header is swallowed as FIND content and every
                # pair defaults to target_files[0].
                _rest = line[5:].strip()
                if _rest and not _rest.startswith("<<<"):
                    _edit_target = _rest
                else:
                    _edit_target = None
                    _edit_block_ord += 1
                _edit_section = "find"
                _edit_find_lines = []
                _edit_replace_lines = []
                continue
            elif _lnorm in ("<<<FIND>>>", "<<FIND>>", "<FIND>"):
                _edit_section = "find"
                _edit_find_lines = []
                continue
            elif _lnorm in ("<<<REPLACE>>>", "<<REPLACE>>", "<REPLACE>"):
                _edit_section = "replace"
                _edit_replace_lines = []
                continue
            elif _lnorm in ("<<<END>>>", "<<END>>", "<END>", "<<<EDITEND>>>"):
                # Commit this find/replace pair
                if current is not None and (_edit_find_lines or _edit_replace_lines):
                    _find_str = "\n".join(_edit_find_lines)
                    _repl_str = "\n".join(_edit_replace_lines)
                    _tgt = _norm_target_path(
                        _edit_target or _default_edit_target())
                    if _tgt:
                        current.inline_edits.setdefault(_tgt, []).append((_find_str, _repl_str))
                # Stay in edit block — there may be more <<<FIND>>>...<<<END>>> pairs.
                # The block ends naturally when a new --STEP header or ==END== is seen.
                _edit_section = "find"
                _edit_find_lines = []
                _edit_replace_lines = []
                continue
            else:
                # Accumulate lines
                if _edit_section == "find":
                    _edit_find_lines.append(raw_line)
                else:
                    _edit_replace_lines.append(raw_line)
                continue

        # ── Inline code block handling ──
        # Accept multiple LLM output variants:
        #   ---file-content-start---  /  ---file-content-end---
        #   --- content ---  (followed by markdown fence)
        _norm = line.lower().replace(" ", "").rstrip("-")
        if _norm.startswith("---file-content-start") or _norm == "---content":
            in_code_block = True
            in_markdown_fence = False
            code_lines = []
            continue
        if _norm.startswith("---file-content-end") or (
            in_code_block and _norm == "---contentend"
        ):
            in_code_block = False
            in_markdown_fence = False
            if current is not None and code_lines:
                _assign_inline_code(current, code_lines)
            code_lines = []
            continue
        if in_code_block:
            # Structural markers end the code block implicitly
            # (handles LLM omitting ---file-content-end---)
            # Mark as truncated — proper close uses ---file-content-end---
            if line.upper() in ("==END==",) or _STEP_RE.match(line):
                in_code_block = False
                in_markdown_fence = False
                if current is not None and code_lines:
                    _assign_inline_code(current, code_lines)
                    current._inline_truncated = True  # type: ignore[attr-defined]
                code_lines = []
                # Fall through to process this line normally
            else:
                # Strip markdown fences (```js, ```) that wrap inline code
                if line.startswith("```"):
                    if in_markdown_fence:
                        # Closing fence — mark boundary so multi-target steps
                        # can split blocks even without // filename.ext headers.
                        code_lines.append(_FENCE_BOUNDARY)
                    in_markdown_fence = not in_markdown_fence
                    continue
                # Strip "> " prefix from code lines — the LLM sometimes
                # carries over the CMD "> command" format into code blocks
                code_line = raw_line
                if code_line.lstrip().startswith("> "):
                    code_line = code_line.lstrip()[2:]
                code_lines.append(code_line)  # preserve original indentation
                continue

        # Re-open code capture: multi-file steps often close each block
        # with ---file-content-end--- and start the next with a bare
        # ``` fence, without repeating `content:`. Without this, every
        # block after the first silently leaks into the step description
        # (observed: 8 of 9 templates in a step never written). Only
        # re-open while the step still has unassigned target files, so
        # fenced snippets in prose never get captured by accident.
        if (line.startswith("```") and not in_code_block
                and current is not None
                and len(current.target_files) > len(current.inline_code)):
            in_code_block = True
            in_markdown_fence = True
            code_lines = []
            continue

        # Skip plan boundary markers
        if line.upper() in ("==PLAN==", "==END==", ""):
            continue

        # New step header
        m = _STEP_RE.match(line)
        if m:
            # Flush previous step
            if current is not None:
                current.description = " ".join(desc_lines).strip()
                _flush_echo_inline(current, cmd_lines)
                steps.append(current)

            step_id = m.group(1)
            step_type = m.group(2).upper()
            deps_raw = (m.group(3) or "").strip()

            # Parse depends
            depends: list[str] = []
            if deps_raw and deps_raw.lower() != "none":
                depends = [d.strip() for d in deps_raw.split(",") if d.strip()]

            current = PlanStep(id=step_id, step_type=step_type, depends_on=depends)
            desc_lines = []
            _edit_block_ord = -1  # per-step ordinal for bare edit: blocks
            cmd_lines = []
            continue

        if current is None:
            continue

        # Command line (for CMD steps)
        if line.startswith("> "):
            cmd_text = line[2:].strip()
            # Strip backtick wrapping added by LLMs (e.g. `> `npm install`` -> `npm install`)
            if len(cmd_text) >= 2 and cmd_text[0] == "`" and cmd_text[-1] == "`":
                cmd_text = cmd_text[1:-1]
            # Skip markdown metadata annotations that the LLM sometimes
            # prefixes with ">" (e.g. "> **produces:** ..." or "> **note:** ...")
            _bare = cmd_text.lstrip("*_ \t")
            _bare_lower = _bare.lower()
            # Handle CODE/TEST metadata that some LLMs incorrectly prefix with "> "
            # (they see CMD steps use "> command" and copy the pattern to metadata)
            if _bare_lower.startswith("target:"):
                raw = _bare[7:].strip()
                if raw:
                    current.target_files = [
                        _norm_target_path(f) for f in raw.split(",")
                        if f.strip()]
                continue
            elif _bare_lower.startswith("exports:"):
                raw = _bare[8:].strip()
                if raw and raw.lower() != "none":
                    current.exports = [e.strip() for e in raw.split(",") if e.strip()]
                continue
            elif _bare_lower.startswith("imports:"):
                raw = _bare[8:].strip()
                if raw and raw.lower() != "none":
                    for entry in raw.split(","):
                        entry = entry.strip()
                        if ":" in entry:
                            file_path, symbol = entry.rsplit(":", 1)
                            current.imports_from.setdefault(
                                _norm_target_path(file_path), []
                            ).append(symbol.strip())
                continue
            elif _bare_lower.startswith("imported_by:"):
                raw = _bare[12:].strip()
                if raw and raw.lower() != "none":
                    current.imported_by = [f.strip() for f in raw.split(",") if f.strip()]
                continue
            elif _bare_lower.startswith("content:"):
                # Inline code block follows — enter code-block mode
                in_code_block = True
                in_markdown_fence = False
                code_lines = []
                rest = _bare[8:].strip()
                # Strip opening markdown fence (```jsx, ```, etc.)
                if rest.startswith("```"):
                    rest = rest[3:].lstrip("abcdefghijklmnopqrstuvwxyz").strip()
                if rest:
                    code_lines.append(rest)
                continue
            elif _bare_lower.startswith("produces:"):
                raw = _bare[9:].strip()
                if raw:
                    current.target_files.extend(
                        f.strip() for f in raw.split(",") if f.strip()
                    )
                continue
            elif _bare_lower.startswith("verify:"):
                raw = _bare[7:].strip().strip("`")
                if raw and raw.lower() != "none":
                    current.verify_cmd = raw
                continue
            _meta_prefixes = ("note:", "output:", "creates:",
                              "result:", "generates:", "returns:")
            if cmd_text.startswith("**") or any(
                _bare_lower.startswith(p) for p in _meta_prefixes
            ):
                continue  # metadata annotation, not a shell command
            # Check for "content:" appearing mid-line (e.g.
            # "> prop-types:default content: ```" where content: is not
            # at the start).  Only trigger when followed by ``` or nothing.
            _content_pos = _bare_lower.find(" content:")
            if _content_pos >= 0:
                _after = _bare[_content_pos + 9:].strip()
                if not _after or _after.startswith("```"):
                    in_code_block = True
                    in_markdown_fence = False
                    code_lines = []
                    if _after.startswith("```"):
                        _after = _after[3:].lstrip(
                            "abcdefghijklmnopqrstuvwxyz").strip()
                    if _after:
                        code_lines.append(_after)
                    continue
            # Join multiple commands per step with && so all run sequentially
            if current.command:
                current.command = current.command + " && " + cmd_text
            else:
                current.command = cmd_text
            cmd_lines.append(cmd_text)

        # Target files
        elif line.lower().startswith("target:"):
            raw = line[7:].strip()
            if raw:
                current.target_files = [
                    _norm_target_path(f) for f in raw.split(",")
                    if f.strip()]

        # Exports
        elif line.lower().startswith("exports:"):
            raw = line[8:].strip()
            if raw and raw.lower() != "none":
                current.exports = [e.strip() for e in raw.split(",") if e.strip()]

        # Imports: src/file.py:Symbol, src/other.py:OtherSymbol
        elif line.lower().startswith("imports:"):
            raw = line[8:].strip()
            if raw and raw.lower() != "none":
                for entry in raw.split(","):
                    entry = entry.strip()
                    if ":" in entry:
                        file_path, symbol = entry.rsplit(":", 1)
                        current.imports_from.setdefault(
                            _norm_target_path(file_path), []
                        ).append(symbol.strip())

        # Produces (alias for target_files, used by CMD steps)
        elif line.lower().startswith("produces:"):
            raw = line[9:].strip()
            if raw:
                produced = [f.strip() for f in raw.split(",") if f.strip()]
                current.target_files.extend(produced)

        # KB docs declared by planner for reviewer context
        elif line.lower().startswith("kb_docs:"):
            raw = line[8:].strip()
            if raw and raw.lower() != "none":
                current.kb_docs = [t.strip() for t in raw.split(",") if t.strip()]

        # Per-step acceptance command — the deterministic gate for this step
        elif line.lower().startswith("verify:"):
            raw = line[7:].strip().strip("`")
            if raw and raw.lower() != "none":
                current.verify_cmd = raw

        # imported_by: which files should import this step's target file
        elif line.lower().startswith("imported_by:"):
            raw = line[12:].strip()
            if raw and raw.lower() != "none":
                current.imported_by = [f.strip() for f in raw.split(",") if f.strip()]

        # edit: block — find/replace patch for an existing file.
        # Supports optional "edit: path/to/file" to specify a different target.
        elif line.lower().startswith("edit:"):
            rest = line[5:].strip()
            # If a file path is given on the same line, use it; otherwise the
            # ordinal-th target file is used (resolved when <<<END>>> is hit):
            # one bare edit: block per target, in target order.
            if rest and not rest.startswith("<<<"):
                _edit_target = rest
            else:
                _edit_target = None
                _edit_block_ord += 1
            in_edit_block = True
            _edit_section = "find"
            _edit_find_lines = []
            _edit_replace_lines = []

        # Inline code block via bare "Content:" keyword (no "> " prefix).
        # The "> content:" variant is already handled inside the "> " branch
        # above.  Here we catch the unindented form that the planner sometimes
        # emits when the step has a pre-written code body.
        elif line.lower().startswith("content:"):
            in_code_block = True
            in_markdown_fence = False
            code_lines = []
            rest = line[8:].strip()  # anything after "Content:" on the same line
            # Strip opening markdown fence (```jsx, ```, etc.)
            if rest.startswith("```"):
                rest = rest[3:].lstrip("abcdefghijklmnopqrstuvwxyz").strip()
            if rest:
                code_lines.append(rest)

        # Description line (anything else) — but check for mid-line content:
        elif not line.startswith("=="):
            _lower_line = line.lower()
            _cpos = _lower_line.find(" content:")
            if _cpos >= 0:
                _after_c = line[_cpos + 9:].strip()
                if not _after_c or _after_c.startswith("```"):
                    # Split: text before content: goes to description,
                    # everything after enters code block mode
                    before = line[:_cpos].strip()
                    if before:
                        desc_lines.append(before)
                    in_code_block = True
                    in_markdown_fence = False
                    code_lines = []
                    if _after_c.startswith("```"):
                        _after_c = _after_c[3:].lstrip(
                            "abcdefghijklmnopqrstuvwxyz").strip()
                    if _after_c:
                        code_lines.append(_after_c)
                    continue
            desc_lines.append(line)

    # Flush any open edit block (LLM omitted the final <<<END>>>).
    if in_edit_block and current is not None and (_edit_find_lines or _edit_replace_lines):
        _find_str = "\n".join(_edit_find_lines)
        _repl_str = "\n".join(_edit_replace_lines)
        # Skip if the find string is only file-content markers (e.g.
        # ---file-content-end--- accumulated after the last <<<END>>>).
        _find_meaningful = [
            l for l in _edit_find_lines
            if l.strip() and not l.strip().startswith("---")
        ]
        if _find_meaningful:
            _tgt = _norm_target_path(
                _edit_target or _default_edit_target())
            if _tgt:
                current.inline_edits.setdefault(_tgt, []).append((_find_str, _repl_str))

    # Flush any open inline code block.
    # If the code block was never closed (no ---file-content-end---, ==END==,
    # or next --STEP marker), the LLM output was truncated.  Assign what we
    # have but mark the step so validate_plan() can clear it.
    if in_code_block and current is not None and code_lines:
        _assign_inline_code(current, code_lines)
        current._inline_truncated = True  # type: ignore[attr-defined]

    # Flush last step
    if current is not None:
        current.description = " ".join(desc_lines).strip()
        _flush_echo_inline(current, cmd_lines)
        steps.append(current)

    # Assign 0-based indices
    for idx, step in enumerate(steps):
        step.index = idx

    # Multi-line CMD steps: drop `cd X` segments that re-enter the
    # directory the joined && chain is already in (each plan line was
    # written assuming the project root).
    for _step in steps:
        _step.command = dedupe_redundant_cd(_step.command)

    # Rescue CMD steps that only write file content via echo/redirect
    # chains — run them as CODE so the file is written directly instead of
    # through a fragile (and on Windows, broken) shell chain.
    _reclassify_file_creation_cmds(steps)

    # Derive imported_by from imports_from relationships across all steps.
    # This is free (no LLM cost) and ensures that when step B declares
    # imports_from step A's target file, step A gets imported_by = step B's target.
    _derive_imported_by(steps)

    # For steps that still have no imported_by after derivation, infer it from
    # the plan structure (entry-point files in later waves).  Works even when the
    # planner forgets to add a wiring step or an explicit imported_by: line.
    _infer_missing_imported_by(steps)

    return steps


def _build_file_to_step(steps: list[PlanStep]) -> dict[str, PlanStep]:
    """Map every step's target file (normalized path + basename) to that step."""
    import os as _os

    file_to_step: dict[str, PlanStep] = {}
    for step in steps:
        for tf in step.target_files:
            norm = tf.replace("\\", "/")
            file_to_step[norm] = step
            file_to_step[_os.path.basename(norm)] = step
    return file_to_step


def _resolve_producer(src_file: str,
                      file_to_step: dict[str, PlanStep]) -> PlanStep | None:
    """Find the step whose ``target_files`` provides *src_file*, if any.

    Matches on the normalized path first, then the basename, then Python
    dotted-module notation (``src.map`` → target ``src/map.py``).
    """
    import os as _os

    src_norm = src_file.replace("\\", "/")
    producer = (file_to_step.get(src_norm)
                or file_to_step.get(_os.path.basename(src_norm)))
    if producer is None and "/" not in src_norm:
        # Python module notation: 'src.snake' → target 'src/snake.py'
        dotted_path = src_norm.replace(".", "/")
        for key, st in file_to_step.items():
            if _os.path.splitext(key)[0] == dotted_path:
                return st
    return producer


def _derive_imported_by(steps: list[PlanStep]) -> None:
    """Populate ``imported_by`` on each step from other steps' ``imports_from``.

    For every consumer step that declares ``imports_from: {file: [symbols]}``,
    find the producer step whose ``target_files`` includes *file* and add the
    consumer's target files to the producer's ``imported_by`` list.

    This is zero-cost (no LLM) and fires automatically after parsing so that
    DepCheck can use ``plan_step.imported_by`` instead of guessing heuristically.
    Explicit ``imported_by:`` lines in the plan take precedence (they are set
    first during parsing; derivation only appends new entries, never clears them).
    """
    file_to_step = _build_file_to_step(steps)

    for consumer_step in steps:
        consumer_files = consumer_step.target_files
        if not consumer_files:
            continue
        for src_file in consumer_step.imports_from:
            producer = _resolve_producer(src_file, file_to_step)
            if producer is None:
                continue
            for cf in consumer_files:
                if cf not in producer.imported_by:
                    producer.imported_by.append(cf)


# Entry-point file basenames that commonly import/mount other components.
# Ordered from most-specific to least-specific.
_ENTRY_POINT_BASENAMES = (
    "main.tsx", "main.ts", "main.jsx", "main.js",
    "App.tsx", "App.ts", "App.jsx", "App.js",
    "index.tsx", "index.ts", "index.jsx", "index.js",
    "router.tsx", "router.ts", "router.jsx", "router.js",
    "__init__.py", "main.py", "app.py", "index.py",
)


def _infer_missing_imported_by(steps: list[PlanStep]) -> None:
    """For CODE steps with exports but no ``imported_by``, infer the consumer
    from other plan steps whose target files look like entry-points or whose
    wave number is later.

    Strategy (in order, first match wins):
    1. Another step in a later wave whose target is an entry-point file and
       that step's description mentions the orphaned step's exported symbol
       or file stem.
    2. Any step in a later wave whose target is an entry-point file.
    3. Any step in a later wave that has no ``imports_from`` declared
       (likely a wiring/mounting step with incomplete metadata).

    Only fires when ``imported_by`` is still empty after ``_derive_imported_by``
    — i.e. the planner neither added a wiring step nor wrote ``imported_by:``.
    Operates purely on plan metadata, zero LLM cost.
    """
    import os as _os

    def _wave(step: PlanStep) -> int:
        """Return the wave number from the step id (e.g. '3.1' → 3)."""
        try:
            return int(step.id.split(".")[0])
        except (ValueError, IndexError):
            return 0

    def _is_entry_point(path: str) -> bool:
        base = _os.path.basename(path).lower()
        return base in {ep.lower() for ep in _ENTRY_POINT_BASENAMES}

    for step in steps:
        # Only care about CODE steps with exports and no consumer yet
        if step.step_type not in ("CODE", "UNCLASSIFIED"):
            continue
        if not step.exports or step.imported_by:
            continue
        if not step.target_files:
            continue

        step_wave = _wave(step)
        exported_lower = {e.lower() for e in step.exports}
        stem_lower = {
            _os.path.splitext(_os.path.basename(tf))[0].lower()
            for tf in step.target_files
        }

        # Collect candidate steps: later wave, has target files
        candidates = [
            s for s in steps
            if _wave(s) > step_wave and s.target_files and s is not step
        ]

        # Strategy 1: entry-point target + description mentions our symbol/stem
        for cand in candidates:
            if not any(_is_entry_point(tf) for tf in cand.target_files):
                continue
            desc_lower = cand.description.lower()
            if exported_lower & set(desc_lower.split()) or stem_lower & set(desc_lower.split()):
                step.imported_by = list(cand.target_files[:1])
                break

        if step.imported_by:
            continue

        # Strategy 2: any entry-point target in a later wave
        for cand in candidates:
            ep_targets = [tf for tf in cand.target_files if _is_entry_point(tf)]
            if ep_targets:
                step.imported_by = [ep_targets[0]]
                break

        if step.imported_by:
            continue

        # Strategy 3: a later step with no imports_from (likely incomplete wiring step)
        for cand in candidates:
            if not cand.imports_from and cand.step_type in ("CODE", "UNCLASSIFIED"):
                step.imported_by = list(cand.target_files[:1])
                break


# File header comment pattern for splitting multi-file inline code blocks.
# Supports "//" (JS/C/Go/Rust/Java) and "#" (Python/Ruby/shell/YAML) and
# "--" (SQL/Lua) line-comment styles, since the planner picks the comment
# style that matches the target file's language. Path chars include "\"
# for Windows-style relative paths (e.g. "snake_game\logic\board.py").
_FILE_COMMENT_RE = re.compile(
    r"^(?://|#|--)\s*([\w./\\-]+\.\w{1,5})\s*$"
)

# Sentinel inserted into code_lines at each closing ``` fence boundary
# so _assign_inline_code can split multiple files without // header comments.
_FENCE_BOUNDARY = "\x00FENCE_BOUNDARY\x00"


def _assign_inline_code(step: PlanStep, code_lines: list[str]) -> None:
    """Assign captured inline code to a step's ``inline_code`` dict.

    For single-target steps, all code goes to that target file.
    For multi-target steps the parser tries three strategies in order:

    1. ``// FileName.ext`` comment headers between blocks.
    2. Fence boundaries (``_FENCE_BOUNDARY`` sentinels inserted at each
       closing `` ``` `` fence) — if the number of non-empty fence blocks
       matches the number of targets, assign block N to target N.
    3. Fallback: assign all code to the first target that has not yet
       been populated, so consecutive ``content:`` blocks (one per
       target file) land on different files instead of overwriting
       ``targets[0]`` repeatedly.
    """
    targets = step.target_files

    # ── Strategy 1: // filename.ext comment headers ──
    if len(targets) > 1:
        current_file: Optional[str] = None
        file_lines: dict[str, list[str]] = {}
        for line in code_lines:
            if line == _FENCE_BOUNDARY:
                continue
            m = _FILE_COMMENT_RE.match(line.strip())
            if m:
                fname = m.group(1)
                matched = _match_target(fname, targets)
                current_file = matched or fname
                file_lines.setdefault(current_file, [])
                continue
            if current_file is not None:
                file_lines[current_file] = file_lines.get(current_file, [])
                file_lines[current_file].append(line)
        if file_lines:
            for fpath, lines in file_lines.items():
                content = "\n".join(lines).strip()
                if content:
                    step.inline_code[fpath] = content
            return

    # ── Strategy 2: fence boundary splitting (no // headers) ──
    if len(targets) > 1:
        fence_blocks: list[list[str]] = []
        current_block: list[str] = []
        for line in code_lines:
            if line == _FENCE_BOUNDARY:
                if current_block:
                    fence_blocks.append(current_block)
                current_block = []
            else:
                current_block.append(line)
        if current_block:
            fence_blocks.append(current_block)

        non_empty = [b for b in fence_blocks if any(ln.strip() for ln in b)]
        if len(non_empty) == len(targets):
            for target, block_lines in zip(targets, non_empty):
                content = "\n".join(block_lines).strip()
                if content:
                    step.inline_code[target] = content
            return

    # ── Strategy 3: fallback — first UNASSIGNED target ──
    # Picking the first unassigned target (instead of always
    # ``targets[0]``) handles the common multi-target pattern where
    # the planner emits one ``content:`` block per file:
    #
    #     target: vite.config.js, vitest.setup.js
    #     content: ```js ...vite config... ``` ---file-content-end---
    #     content: ```js ...setup... ```      ---file-content-end---
    #
    # Each ``---file-content-end---`` triggers one call here, and
    # without this rule the second call would silently overwrite
    # ``vite.config.js`` with the setup-file content.
    clean_lines = [ln for ln in code_lines if ln != _FENCE_BOUNDARY]
    full_code = "\n".join(clean_lines).strip()
    if not full_code:
        return

    if not targets:
        return
    unassigned = [t for t in targets if t not in step.inline_code]
    if unassigned:
        step.inline_code[unassigned[0]] = full_code
    # else: every target already populated — drop the extra block
    # rather than clobbering an existing assignment.


def _match_target(name: str, targets: list[str]) -> Optional[str]:
    """Match a short filename against the full target paths."""
    import os
    for t in targets:
        if os.path.basename(t) == name or t.endswith("/" + name) or t == name:
            return t
    return None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_plan(steps: list[PlanStep], working_dir: Optional[str] = None) -> list[str]:
    """Validate a parsed plan for structural correctness.

    Returns a list of error messages (empty = valid).
    """
    errors: list[str] = []
    all_ids = {s.id for s in steps}

    # Track which files are produced by which steps
    produced_files: dict[str, str] = {}  # file -> step_id

    for step in steps:
        # Check depends_on references exist
        for dep in step.depends_on:
            if dep not in all_ids:
                errors.append(
                    f"Step {step.id} depends on unknown step '{dep}'"
                )

        # Track produced files
        for fpath in step.target_files:
            produced_files[fpath] = step.id

        # Check imports reference files that some step produces or will produce
        for file_path in step.imports_from:
            # It's OK if the file is produced by a later step — the plan
            # declares intent. But if NO step produces it, warn.
            if file_path not in produced_files:
                producers = [
                    s for s in steps
                    if file_path in s.target_files
                ]
                if not producers:
                    # Not an error — could be an existing project file
                    pass

    # ── Nested directory collision detection ─────────────────────────────
    # Detect when a CMD step creates a workspace directory with the same
    # name as a Python package created inside it (e.g. `mkdir snake_game`
    # followed by `cd snake_game && mkdir -p snake_game/tests`).  This
    # results in `snake_game/snake_game/` which causes import confusion.
    import re as _re_plan

    def _resolve_mkdir_paths(command: str) -> tuple[set[str], set[str]]:
        """Parse a compound shell command and return (workspace_dirs, all_abs_dirs).

        Tracks ``cd`` context so ``cd foo && mkdir bar`` yields ``foo/bar``.
        ``workspace_dirs`` are top-level dirs created by ``mkdir`` without
        a preceding ``cd`` (i.e. the project workspace root).
        """
        workspaces: set[str] = set()
        all_dirs: set[str] = set()
        cwd = ""
        for seg in command.split('&&'):
            seg = seg.strip()
            cd_m = _re_plan.match(r'^\s*cd\s+(\S+)', seg)
            if cd_m:
                _cd_target = cd_m.group(1).strip().rstrip('/')
                if cwd:
                    cwd = cwd + '/' + _cd_target
                else:
                    cwd = _cd_target
                continue
            mk_m = _re_plan.match(r'^\s*mkdir\s+(?:-p\s+)?(.+)', seg)
            if mk_m:
                for d in mk_m.group(1).split():
                    d = d.strip().rstrip('/')
                    if not d or d.startswith('-'):
                        continue
                    abs_d = (cwd + '/' + d) if cwd else d
                    all_dirs.add(abs_d)
                    if not cwd:
                        # Top-level mkdir = workspace dir
                        workspaces.add(d.split('/')[0])
        return workspaces, all_dirs

    _workspace_dirs: set[str] = set()
    _all_created_dirs: set[str] = set()
    for step in steps:
        if step.step_type == "CMD" and step.command:
            ws, dirs = _resolve_mkdir_paths(step.command)
            _workspace_dirs.update(ws)
            _all_created_dirs.update(dirs)

    if _workspace_dirs:
        _reported: set[str] = set()
        # Check created dirs for workspace/package name collision
        for d in _all_created_dirs:
            parts = d.replace('\\', '/').split('/')
            if (len(parts) >= 2
                    and parts[0] in _workspace_dirs
                    and parts[1] == parts[0]
                    and parts[0] not in _reported):
                _reported.add(parts[0])
                errors.append(
                    f"Plan creates nested '{parts[0]}/{parts[1]}/' inside "
                    f"workspace '{parts[0]}/' — Python will confuse the "
                    f"workspace dir with the package dir, causing import "
                    f"failures. Rename the workspace (e.g. '{parts[0]}-project') "
                    f"or write package files directly without a wrapper dir."
                )

    # Check inline code for truncation: if the parser never saw a closing
    # marker (---file-content-end---, ==END==, or next --STEP), the LLM
    # output was cut off.  Preserve the partial code as a hint for the coder
    # (so it can complete rather than regenerate from scratch), then clear
    # inline_code so the normal coder path handles full generation.
    import os as _os
    for step in steps:
        if not step.inline_code:
            continue
        if getattr(step, "_inline_truncated", False):
            # Preserve partial content as a completion hint before clearing
            step._partial_inline_code = dict(step.inline_code)  # type: ignore[attr-defined]
            step.inline_code.clear()
            errors.append(
                f"Step {step.id}: inline code was truncated (no closing marker) "
                f"— preserved as partial hint, coder will complete it"
            )
            continue

    # Check inline code imports: if inline code contains local imports
    # (e.g. import './Hero.css'), verify that some step produces the file.
    # When a dangling import is found, clear the entire step's inline_code
    # so the regular coder path handles it — the coder can generate both the
    # component AND the missing file with full KB/memory context.
    import re as _re

    # Language-specific local import patterns (group 1 = the import path)
    _IMPORT_PATTERNS: dict[str, list[_re.Pattern]] = {
        # JS/TS: import X from './Y'  |  import './Y'
        "js": [
            _re.compile(r"""(?:import\s+.*?from\s+|import\s+)['"](\.[^'"]+)['"]"""),
            _re.compile(r"""@import\s+['"](\.[^'"]+)['"]"""),   # CSS @import
            _re.compile(r"""require\s*\(\s*['"](\.[^'"]+)['"]\s*\)"""),  # CJS require
        ],
        # Python: from .module import X  |  from . import module
        "py": [
            _re.compile(r"""from\s+(\.[\w.]+)\s+import"""),
        ],
        # Go: import "./pkg"  |  in import block
        "go": [
            _re.compile(r"""(?:import\s+|")(\./[^"]+)"""),
        ],
        # Rust: mod submodule;  (local module declaration)
        "rs": [
            _re.compile(r"""mod\s+(\w+)\s*;"""),
        ],
    }

    # Map file extensions to pattern keys
    _EXT_TO_PATTERN_KEY = {
        ".js": "js", ".jsx": "js", ".mjs": "js", ".cjs": "js",
        ".ts": "js", ".tsx": "js",
        ".css": "js",   # CSS uses @import
        ".py": "py",
        ".go": "go",
        ".rs": "rs",
    }

    # Extension candidates per language when import has no extension
    _EXT_CANDIDATES = {
        "js": [".js", ".jsx", ".ts", ".tsx", ".css", ".mjs"],
        "py": [".py"],
        "go": [".go"],
        "rs": [".rs"],
    }

    for step in steps:
        if not step.inline_code:
            continue
        has_dangling = False
        for fpath, code in step.inline_code.items():
            if has_dangling:
                break
            ext = _os.path.splitext(fpath)[1].lower()
            pattern_key = _EXT_TO_PATTERN_KEY.get(ext)
            if not pattern_key:
                continue
            patterns = _IMPORT_PATTERNS.get(pattern_key, [])
            file_dir = _os.path.dirname(fpath)
            for pat in patterns:
                if has_dangling:
                    break
                for m in pat.finditer(code):
                    imp_path = m.group(1)
                    # Python relative imports use dots: from .models import X
                    if pattern_key == "py" and imp_path.startswith("."):
                        # Convert .models to ./models for path resolution
                        dotless = imp_path.lstrip(".")
                        depth = len(imp_path) - len(dotless)
                        base = file_dir
                        for _ in range(depth - 1):
                            base = _os.path.dirname(base)
                        imp_path = "./" + dotless.replace(".", "/")
                    # Rust mod X; → ./X.rs or ./X/mod.rs
                    if pattern_key == "rs":
                        imp_path = "./" + imp_path
                    resolved = _os.path.normpath(
                        _os.path.join(file_dir, imp_path)
                    ).replace("\\", "/")
                    candidates = [resolved]
                    if not _os.path.splitext(resolved)[1]:
                        for cand_ext in _EXT_CANDIDATES.get(pattern_key, []):
                            candidates.append(resolved + cand_ext)
                        # Rust: mod X → X/mod.rs
                        if pattern_key == "rs":
                            candidates.append(resolved + "/mod.rs")
                    # Scaffold CMD steps declare globs (`produces: app/src/*`)
                    # — match candidates against those patterns too, or every
                    # import of a scaffold-created file (./index.css) reads as
                    # dangling and nukes perfectly good inline code (observed:
                    # a correct main.jsx cleared, costing an 8-turn loop).
                    import fnmatch as _fnmatch
                    _wild_produced = [tf for tf in produced_files
                                      if "*" in tf or "?" in tf]
                    is_produced = any(
                        c in produced_files or any(
                            c == tf or tf.endswith("/" + _os.path.basename(c))
                            for tf in produced_files
                        ) or any(
                            _fnmatch.fnmatch(c, w) for w in _wild_produced
                        )
                        for c in candidates
                    )
                    if is_produced:
                        continue
                    if any(
                        _os.path.basename(c) in _os.path.basename(tf)
                        for c in candidates for tf in produced_files
                    ):
                        continue
                    # Also accept files that already exist on disk
                    _base = working_dir or _os.getcwd()
                    if any(_os.path.exists(_os.path.join(_base, c)) for c in candidates):
                        continue
                    has_dangling = True
                    break
        if has_dangling:
            step.inline_code.clear()
            errors.append(
                f"Step {step.id}: inline code imports a file no step produces "
                f"— cleared inline_code, will use coder LLM call instead"
            )

    # Check for circular dependencies
    if _has_cycle(steps):
        errors.append("Circular dependency detected in plan")

    # Check valid step types
    valid_types = {"CMD", "CODE", "TEST", "IGNORE", "SEARCH"}
    for step in steps:
        if step.step_type not in valid_types:
            errors.append(
                f"Step {step.id} has unknown type '{step.step_type}'"
            )

    return errors


# ---------------------------------------------------------------------------
# Auto-fix: nested workspace/package name collision
# ---------------------------------------------------------------------------

def fix_nested_workspace_collision(steps: list[PlanStep]) -> list[str]:
    """Detect and fix ``mkdir X && cd X && mkdir X/...`` patterns.

    When a CMD step creates a workspace directory with the same name as
    the Python package inside it (e.g. ``mkdir snake_game`` followed by
    ``cd snake_game && mkdir snake_game/``), imports break because Python
    confuses the outer workspace with the inner package.

    Fix strategy: rewrite CMD steps to drop the workspace ``mkdir`` and
    ``cd`` prefix, so the package is created directly at the repo root.
    All target_files, inline_code keys, and produces that reference the
    nested path are adjusted accordingly.

    Returns a list of human-readable descriptions of fixes applied.
    """
    import re as _re

    fixes: list[str] = []

    # Phase 1: detect workspace dirs and collisions (same logic as validate_plan)
    def _resolve_mkdir_paths(command: str) -> tuple[set[str], set[str]]:
        workspaces: set[str] = set()
        all_dirs: set[str] = set()
        cwd = ""
        for seg in command.split('&&'):
            seg = seg.strip()
            cd_m = _re.match(r'^\s*cd\s+(\S+)', seg)
            if cd_m:
                _cd_target = cd_m.group(1).strip().rstrip('/')
                cwd = (cwd + '/' + _cd_target) if cwd else _cd_target
                continue
            mk_m = _re.match(r'^\s*mkdir\s+(?:-p\s+)?(.+)', seg)
            if mk_m:
                for d in mk_m.group(1).split():
                    d = d.strip().rstrip('/')
                    if not d or d.startswith('-'):
                        continue
                    abs_d = (cwd + '/' + d) if cwd else d
                    all_dirs.add(abs_d)
                    if not cwd:
                        workspaces.add(d.split('/')[0])
        return workspaces, all_dirs

    workspace_dirs: set[str] = set()
    all_created: set[str] = set()
    for step in steps:
        if step.step_type == "CMD" and step.command:
            ws, dirs = _resolve_mkdir_paths(step.command)
            workspace_dirs.update(ws)
            all_created.update(dirs)

    # Find collision: workspace/workspace pattern
    colliding: set[str] = set()
    for d in all_created:
        parts = d.replace('\\', '/').split('/')
        if (len(parts) >= 2
                and parts[0] in workspace_dirs
                and parts[1] == parts[0]):
            colliding.add(parts[0])

    if not colliding:
        return fixes

    # Phase 2: rewrite steps to remove the workspace wrapper.
    # IMPORTANT: only strip the workspace prefix from DOUBLE-NESTED paths
    # (e.g. snake_game/snake_game/game.py → snake_game/game.py).
    # Single-nested paths (e.g. snake_game/game.py) are legitimate package
    # paths and must NOT be stripped — otherwise files end up at the repo
    # root instead of inside the package directory.
    for ws in colliding:
        ws_prefix = ws + '/'
        double_prefix = ws + '/' + ws + '/'

        def _strip_double(path: str) -> tuple[str, bool]:
            """Strip workspace prefix only from double-nested paths."""
            if path.startswith(double_prefix):
                return path[len(ws_prefix):], True
            return path, False

        for step in steps:
            # Rewrite CMD commands: strip "cd <ws> &&" and "mkdir <ws>"
            # IMPORTANT: only strip "cd <ws>" when the remaining segments
            # are ALL directory/file creation commands (mkdir, touch).
            # If any segment is a runtime command (pip, python, pytest,
            # npm, etc.), keep the cd — the command needs to run inside
            # the workspace even though the package prefix is stripped.
            if step.step_type == "CMD" and step.command:
                original = step.command
                segments = step.command.split('&&')
                cd_indices: list[int] = []
                mkdir_indices: list[int] = []
                other_indices: list[int] = []

                for i, seg in enumerate(segments):
                    seg_stripped = seg.strip()
                    if _re.match(rf'^\s*cd\s+{_re.escape(ws)}\s*$', seg_stripped):
                        cd_indices.append(i)
                    elif _re.match(
                        rf'^\s*mkdir\s+(?:-p\s+)?{_re.escape(ws)}\s*$',
                        seg_stripped,
                    ):
                        mkdir_indices.append(i)
                    else:
                        other_indices.append(i)

                # Always drop bare mkdir <workspace>
                drop = set(mkdir_indices)
                # Only drop cd <workspace> when all remaining segments
                # are directory/file scaffolding, not runtime commands
                _SCAFFOLD_RE = _re.compile(
                    r'^\s*(mkdir|touch|ln|cp|mv)\b', _re.IGNORECASE)
                _non_cd_non_mkdir = [
                    segments[i].strip() for i in other_indices]
                _all_scaffold = all(
                    _SCAFFOLD_RE.match(s) for s in _non_cd_non_mkdir
                ) if _non_cd_non_mkdir else True
                if _all_scaffold:
                    drop.update(cd_indices)

                new_segments = [
                    segments[i] for i in range(len(segments))
                    if i not in drop
                ]
                if new_segments:
                    step.command = ' && '.join(s.strip() for s in new_segments)
                else:
                    step.command = 'true'  # noop — all segments were workspace-related

                if step.command != original:
                    fixes.append(
                        f"Step {step.id}: stripped workspace '{ws}/' "
                        f"from CMD to avoid nested package collision"
                    )

            # Rewrite target_files: only strip double-nested prefix
            new_targets: list[str] = []
            changed_targets = False
            for tf in step.target_files:
                new_tf, changed = _strip_double(tf)
                new_targets.append(new_tf)
                if changed:
                    changed_targets = True
            if changed_targets:
                step.target_files = new_targets

            # Rewrite inline_code keys: only strip double-nested prefix
            if step.inline_code:
                new_inline: dict[str, str] = {}
                for k, v in step.inline_code.items():
                    new_k, _ = _strip_double(k)
                    new_inline[new_k] = v
                step.inline_code = new_inline

            # Rewrite inline_edits keys: only strip double-nested prefix
            if step.inline_edits:
                new_edits: dict[str, list[tuple[str, str]]] = {}
                for k, v in step.inline_edits.items():
                    new_k, _ = _strip_double(k)
                    new_edits[new_k] = v
                step.inline_edits = new_edits

    return fixes


# ---------------------------------------------------------------------------
# Acceptance-gate quality
# ---------------------------------------------------------------------------

# A gate that runs a real test suite is substantive by construction — the
# suite carries the assertions.
_TEST_RUNNER_RE = re.compile(
    r"\b(pytest|unittest|nose2|tox|jest|vitest|mocha|ava|karma|rspec|"
    r"phpunit|manage\.py\s+test|go\s+test|cargo\s+test|dotnet\s+test|"
    r"mvn\s+test|gradle\s+test|ctest)\b"
    r"|\b(npm|yarn|pnpm)\s+(run\s+)?test\b",
    re.IGNORECASE,
)

# `python -c "..."` / `node -e "..."` inline scripts — the form the planner
# reaches for, and the form that is easy to write vacuously. The body must
# tolerate escaped quotes (`assert hasattr(m, \"X\")`); a plain non-greedy
# `.+?` stops at the first `\"` and silently truncates the payload, which
# made every escaped gate look unparseable and slip through unjudged.
#
# The interpreter is rarely a bare name: plans reach for
# ``venv\Scripts\python.exe -c``, ``../venv/bin/python3 -c``, ``py -c``.
# An interpreter pattern that only matched ``python`` skipped those gates
# entirely — not "judged and passed", *skipped* — so a whole run's worth
# of gates went unchecked. Allow a leading path and an ``.exe`` suffix.
_INTERP = r"""(?:\S*[\\/])?(?:python[0-9.]*|py)(?:\.exe)?"""
_JS_INTERP = r"""(?:\S*[\\/])?(?:node|deno)(?:\.exe)?"""

_INLINE_SCRIPT_RE = re.compile(
    _INTERP + r"""\s+-c\s+(?P<pq>["'])"""
    r"""(?P<py>(?:\\.|(?!(?P=pq)).)*)(?P=pq)"""
    r"""|""" + _JS_INTERP + r"""\s+-e\s+(?P<jq>["'])"""
    r"""(?P<js>(?:\\.|(?!(?P=jq)).)*)(?P=jq)""",
    re.DOTALL,
)

# Anything that can fail on a wrong VALUE rather than a wrong import.
_JS_ASSERTION_RE = re.compile(
    r"\bassert\b|\bthrow\b|process\.exit\s*\(\s*[1-9]", re.IGNORECASE)

# The assertion is not always inside the payload. A gate can put the teeth
# in the SHELL — pipe the script's output through a matcher and turn a hit
# into a failing status:
#
#   python -c "import main" 2>&1 | findstr /i "error" && exit 1 || exit 0
#
# That command fails when main.py raises at import time, which is exactly
# the behavioural defect the step could have. Judging it on its Python
# payload alone ("imports and prints but never asserts") is wrong, and the
# cost of being wrong is not academic: it triggers a full re-plan, and the
# planner's second try replaced a real smoke gate with `assert True` —
# a vacuous gate that satisfies the checker. Recognise the shell form so
# the pressure lands only on gates that genuinely cannot fail.
_SHELL_TEETH_RE = re.compile(
    r"(?:\|\s*(?:findstr|grep|Select-String)\b[^|&]*(?:&&|\|\|)\s*exit\s+[1-9])"
    r"|(?:\|\|\s*exit\s+[1-9])"
    r"|(?:&&\s*exit\s+[1-9])",
    re.IGNORECASE,
)


def shell_level_assertion(cmd: str) -> bool:
    """True when *cmd* converts the script's output into a failing status."""
    return bool(cmd) and bool(_SHELL_TEETH_RE.search(cmd))


def shallow_gate_reason(cmd: str) -> Optional[str]:
    """Explain why *cmd* cannot detect a behavioural defect, or None.

    A step's ``verify:`` is the only thing standing between a broken
    implementation and a green run, yet planners habitually emit
    ``python -c "from game import Game; print(Game)"`` — which passes as
    long as the file parses. Observed consequence: a Pac-Man run shipped
    with three of four ghosts spawned inside wall tiles, unable to ever
    move. Every gate was green, the smoke test launched fine, and the
    pipeline reported success, because nothing in the plan asserted a
    single value.

    The rule is deliberately blunt and easy to satisfy: a gate must
    either run a test suite or assert something. Commands we cannot
    classify (``python manage.py check``, ``npm run build``) are left
    alone — they do real work and second-guessing them would be noise.
    """
    if not cmd or not cmd.strip():
        return None
    if _TEST_RUNNER_RE.search(cmd):
        return None
    if shell_level_assertion(cmd):
        return None

    match = _INLINE_SCRIPT_RE.search(cmd)
    if match is None:
        # Not an inline script — a build/check/lint command we can't judge.
        return None

    payload_py = match.group("py")
    if payload_py is not None:
        return _shallow_python_reason(payload_py)

    payload_js = match.group("js") or ""
    if _JS_ASSERTION_RE.search(payload_js):
        return None
    return ("runs an inline script that never asserts anything — it passes "
            "as long as the module loads")


def _same_expr(a, b) -> bool:
    """True when two AST nodes are the same expression, structurally."""
    import ast as _ast
    try:
        return _ast.dump(a) == _ast.dump(b)
    except Exception:
        return False


def _always_true(node) -> bool:
    """True when *node* is a test that NO runtime value can falsify.

    `assert True` was already rejected as punctuation, but the same
    intent survives in forms the constant check cannot see. Observed on a
    Pac-Man run whose plan gated its whole Game class on

        assert isinstance(g.player, type(g.player))
        assert isinstance(g.map,    type(g.map))

    — true for every object that has ever existed. Both gates went green
    against a game where nothing moved in 600 frames, and the pipeline
    reported success. `verified-early` sharpens the cost: the loop exits
    the moment the gate passes, so a gate that cannot fail ends the step
    on turn one.

    Recognised: truthy constants, `isinstance(x, type(x))`,
    `isinstance(x, object)`, and a comparison of an expression with
    itself. `and` is tautological only when every operand is; `or` when
    any is.

    Deliberately narrow. A false positive here costs a re-plan, so only
    forms that are true by construction are claimed — `x == x` is
    included even though NaN falsifies it, because a gate asserting it
    is not checking behaviour either way.
    """
    import ast as _ast

    if isinstance(node, _ast.Constant):
        return bool(node.value)

    if isinstance(node, _ast.BoolOp):
        if isinstance(node.op, _ast.And):
            return all(_always_true(v) for v in node.values)
        return any(_always_true(v) for v in node.values)

    if isinstance(node, _ast.Call):
        fn = node.func
        if (isinstance(fn, _ast.Name) and fn.id == "isinstance"
                and len(node.args) == 2):
            obj, cls = node.args
            # isinstance(x, type(x))
            if (isinstance(cls, _ast.Call)
                    and isinstance(cls.func, _ast.Name)
                    and cls.func.id == "type"
                    and len(cls.args) == 1
                    and _same_expr(cls.args[0], obj)):
                return True
            # isinstance(x, object)
            if isinstance(cls, _ast.Name) and cls.id == "object":
                return True
        return False

    if isinstance(node, _ast.Compare) and len(node.ops) == 1:
        if (isinstance(node.ops[0], (_ast.Eq, _ast.Is, _ast.LtE, _ast.GtE))
                and _same_expr(node.left, node.comparators[0])):
            return True

    return False


def _describe(node) -> str:
    import ast as _ast
    try:
        return _ast.unparse(node)
    except Exception:                       # pragma: no cover - <3.9 only
        return "the assertion"


def _shallow_python_reason(payload: str) -> Optional[str]:
    """Classify a ``python -c`` payload. None when it can fail on a value."""
    import ast as _ast

    # Planner payloads arrive with shell escaping still applied.
    source = payload.replace('\\"', '"').replace("\\'", "'")
    try:
        tree = _ast.parse(source)
    except SyntaxError:
        # Can't judge it — don't manufacture a complaint.
        return None

    body = tree.body
    if not body:
        return None

    if all(isinstance(node, (_ast.Import, _ast.ImportFrom)) for node in body):
        return ("only imports the module — it proves the file parses, not "
                "that it behaves correctly")

    # Assertions that cannot fail, kept for the message. `assert True` is
    # not an assertion, it is punctuation — and so is every other form in
    # `_always_true`. Only an assert that some runtime value could falsify
    # counts as a gate.
    tautologies: list[str] = []

    for node in _ast.walk(tree):
        if isinstance(node, _ast.Assert):
            if _always_true(node.test):
                tautologies.append(_describe(node.test))
                continue
            return None
        # `sys.exit(1)` / `raise` on a bad value are assertions in spirit.
        if isinstance(node, _ast.Raise):
            return None

    if tautologies:
        shown = "; ".join(f"`{t}`" for t in tautologies[:2])
        return (f"asserts only things that are true for every possible "
                f"value ({shown}) — the gate passes no matter what the "
                f"code does. Assert a concrete expected value instead")

    return ("imports and prints but never asserts — it passes whatever the "
            "values are, so it cannot detect wrong behaviour")


# `from a.b import x` / `import a.b` inside a verify payload.
_PY_IMPORT_RE = re.compile(
    r"\bfrom\s+([A-Za-z_][\w.]*)\s+import\b|\bimport\s+([A-Za-z_][\w.]*)")


def check_gate_consistency(steps: list[PlanStep]) -> list[tuple[str, str]]:
    """Find verify commands that assume a working directory they won't get.

    Gates run from the repo root. A plan that targets
    ``pacman_clone/src/config.py`` while its verify says
    ``from src.config import ...`` has written a gate that cannot pass
    there — the module only resolves from inside ``pacman_clone/``.

    That mismatch is not a harmless red gate: the agent loop treats the
    gate as ground truth and makes it pass. Observed on a Pygame run where
    every step wrote its declared target, watched the gate fail, and then
    wrote a *second copy* of the module at the repo root to satisfy it —
    shipping a fully duplicated source tree, 439k tokens, and a green
    pipeline. Catch it at plan time instead.

    Returns ``(step_id, reason)`` pairs, empty when the plan is coherent.
    """
    from .plan_graph import PlanGraph

    graph = PlanGraph(steps)
    issues: list[tuple[str, str]] = []
    for step in steps:
        cmd = step.verify_cmd
        if not cmd:
            continue
        match = _INLINE_SCRIPT_RE.search(cmd)
        if match is None:
            continue
        payload = match.group("py")
        if not payload:
            continue
        for imp in _PY_IMPORT_RE.finditer(payload):
            module = imp.group(1) or imp.group(2)
            if not module:
                continue
            key = module.replace(".", "/")
            if graph.has_module(key):
                continue          # resolves from the repo root — fine
            prefix = graph.prefix_for(key)
            if prefix:
                issues.append((step.id, (
                    f"imports `{module}`, but the plan targets that module "
                    f"at `{prefix}/{key}.py`. The gate runs from the project "
                    f"root, where this import fails")))
                break
    return issues


def unrunnable_gate_reason(cmd: str) -> Optional[str]:
    """Explain why *cmd* can never run at all, or None.

    A gate is executed on every monotonic recheck, so one that cannot even
    parse is a permanent, unsatisfiable blocker rather than a test of the
    code. Observed: a planner wrote a `python -c` payload containing a
    literal backslash-n to fake a multi-line loop —

        ...; changed=False; \nfor _ in range(40): g.update(0.05, ...)

    — which Python rejects with "unexpected character after line
    continuation character". No implementation of ghost.py could satisfy
    it. Three attempts across two models spent 24 turns and ~20 command
    runs on it before the run was abandoned.

    Structural defects are judged first, because they need no payload at
    all: an interpreter pointed at another executable, and a placeholder
    left in the command. Both are unsatisfiable for the same reason as a
    broken payload — no output of the step can change the verdict.

    Only the inline `python -c` payload is judged; a real newline inside
    it is perfectly legal and must not be flagged.
    """
    if not cmd:
        return None
    for structural in (_interpreter_target_error, _placeholder_error,
                       _posix_idiom_error, _node_jsx_error):
        reason = structural(cmd)
        if reason:
            return reason
    match = _INLINE_SCRIPT_RE.search(cmd)
    if not match:
        return None
    groups = match.groupdict()
    for key, check in (("py", _python_payload_error),
                       ("js", _js_payload_error)):
        payload = groups.get(key)
        if payload:
            return check(_unescape_shell_quotes(payload))
    return None


_NODE_JSX_RE = re.compile(
    r"""\bnode\b(?![^"']*\bvitest\b)[^\n]*?"""
    r"""(?:import|require)\s*\(\s*['"][^'"]+\.(?:jsx|tsx)['"]""",
    re.IGNORECASE)


def _node_jsx_error(cmd):
    """A gate asking bare Node to load a .jsx/.tsx module.

    Structural, and permanently so: JSX is not JavaScript, and no version
    of Node parses it — only a bundler or a transform-aware runner
    (vitest, jest, tsx, babel-node) does. So no output of the step can
    make this gate pass, which is exactly the bar the other structural
    branches meet.

    Measured 2026-08-19, gate::

        cd frontend && node -e "import('./src/context/AuthContext.jsx')
                                 .then(m=>{...})"

    The step ran three full loops, 30 turns, and the agent's eventual
    "fix" was to deform the toolchain rather than the code: it wrote
    `frontend/jsx-loader.mjs` (an esbuild transform hook) and
    `frontend/node.cmd` — a shim that SHADOWS the real `node` for
    anything run from that directory. The run failed anyway. A gate that
    can only be satisfied by replacing the interpreter is not measuring
    the artifact.

    Deliberately silent when the command already routes through a
    transform-aware runner, since `node ... vitest` handles JSX fine.
    """
    if not cmd:
        return None
    if _NODE_JSX_RE.search(cmd):
        return ("the gate asks Node to load a .jsx/.tsx module, which no "
                "version of Node can parse - JSX needs a bundler or a "
                "transform-aware runner. Assert against the file's text, "
                "or run the project's test runner (vitest/jest), which "
                "transforms JSX before importing it")
    return None


def _posix_idiom_error(cmd: str) -> Optional[str]:
    """A shell construct the running platform cannot execute.

    Structural like the two above it: no output of the step can change
    the verdict, because the shell rejects the command before the code
    is consulted. Lives in `gate_integrity`, which owns the question of
    what a given shell does with a given text, and is silent on POSIX
    where these idioms are correct.
    """
    from .gate_integrity import posix_only_idiom_reason
    return posix_only_idiom_reason(cmd)


def _unescape_shell_quotes(payload: str) -> str:
    """Undo the one escape the capture keeps but the interpreter never sees.

    The payload is captured from between the command's quotes, so a
    ``\\"`` written for the shell is still a backslash-quote here — while
    the interpreter is handed a plain ``"``. Checking the raw capture
    therefore fails on perfectly good gates: verified, both
    ``python -c "... open(\\"p.json\\") ..."`` and the JS equivalent were
    reported unrunnable before this, which sends the planner off to
    rewrite a command that was never broken.

    Only ``\\"`` is undone — every shell agrees on that one. ``\\\\`` is
    deliberately left alone: POSIX collapses it and cmd.exe does not (see
    :mod:`.gate_integrity`), and it cannot change a payload's syntactic
    validity either way.
    """
    return payload.replace('\\"', '"')


# An interpreter handed another executable as the script to run. Observed on
# a hello-world run whose planner wrote:
#
#     python venv\Scripts\python.exe hello_world.py | find /i "Hello World"
#
# Python then tries to PARSE python.exe as source and dies with
# "SyntaxError: Non-UTF-8 code starting with '\x90'". No edit to the target
# file can ever satisfy that, and three diagnosis attempts were spent on a
# program that already worked — `python hello_world.py` exited 0 midway
# through the same run. The flag alternation lets `python -X utf8 foo.exe`
# through to the script slot; `python -m pytest` lands on `pytest`, which is
# not an executable path and is correctly ignored.
_INTERPRETER_RUNS_BINARY_RE = re.compile(
    r'\b(?:python[23]?|node|ruby|perl)(?:\.exe)?\s+'
    r'(?:-[A-Za-z]\w*\s+)*'
    r'(?P<script>[^\s"\';|&]*'
    r'(?:\.exe\b|[\\/](?:python[23]?|node|ruby|perl)(?:\.exe)?\b)'
    r'[^\s"\';|&]*)',
    re.IGNORECASE)

# `<filename>`, `<path to file>` — a placeholder the planner was told never
# to emit, run verbatim. Observed: `python <filename>` executed as-is.
_PLACEHOLDER_RE = re.compile(r'<[A-Za-z_][\w\-]*(?:\s+[\w\-]+)*>')

_QUOTED_RE = re.compile(r'"[^"]*"|\'[^\']*\'')


def _strip_quoted(cmd: str) -> str:
    """Blank out quoted spans so payload contents cannot trip outer checks.

    `node -e "if (a<b) ..."` and an HTML assertion inside a python -c string
    both contain angle brackets that are not placeholders. The payloads are
    judged separately by the -c/-e checks; here they are noise.
    """
    return _QUOTED_RE.sub(" ", cmd)


def _interpreter_target_error(cmd: str) -> Optional[str]:
    match = _INTERPRETER_RUNS_BINARY_RE.search(_strip_quoted(cmd))
    if not match:
        return None
    script = match.group("script")
    return (f"the verify command runs an interpreter on another executable "
            f"({script}) — the interpreter parses that binary as source and "
            f"fails with a decoding/syntax error, so the gate can never pass "
            f"whatever the code does. Invoke the script directly, e.g. "
            f"`python your_script.py`, using the pipeline's own Python")


def _placeholder_error(cmd: str) -> Optional[str]:
    match = _PLACEHOLDER_RE.search(_strip_quoted(cmd))
    if not match:
        return None
    return (f"the verify command still contains the placeholder "
            f"`{match.group(0)}` — it is run verbatim, so the gate can never "
            f"pass. Write the real path or command")


def _python_payload_error(payload: str) -> Optional[str]:
    try:
        ast.parse(payload)
    except SyntaxError as exc:
        return (f"the verify command's python -c payload is not valid "
                f"Python ({exc.msg}) — it can never pass, whatever the "
                f"code does. Write it as a single line of statements "
                f"separated by ';', or move the logic into a test file")
    except ValueError as exc:          # e.g. embedded null bytes
        return (f"the verify command's python -c payload cannot be parsed "
                f"({exc}) — it can never pass")
    return None


def _js_payload_error(payload: str) -> Optional[str]:
    """Syntax-check a ``node -e`` payload, or stay silent.

    Observed: a plan put ``&& npm --prefix react-home run build`` INSIDE
    the ``node -e "..."`` string. As JavaScript that is a syntax error, so
    the gate could not pass whatever the code did. The loop diagnosed it,
    recovered the step via an equivalent command — and the monotonic
    ledger then rechecked the original, called it a regression and rolled
    a wave of correct work back. 153k tokens for a misplaced quote.

    Silence is the safe answer: a missing node, a timeout or any other
    surprise means "not judged", never "rejected". A false rejection
    costs a full replan of a plan that was fine.
    """
    if shutil.which("node") is None:
        return None
    try:
        proc = subprocess.run(
            ["node", "--check"], input=payload, capture_output=True,
            text=True, timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode == 0:
        return None
    detail = (proc.stderr or "").strip().splitlines()
    # Line 1 is "[stdin]:N"; the message itself is a little further down.
    reason = next((ln.strip() for ln in detail
                   if "Error" in ln or "error" in ln), "syntax error")
    return (f"the verify command's node -e payload is not valid JavaScript "
            f"({reason[:120]}) — it can never pass, whatever the code does. "
            f"A common cause is putting a shell chain such as "
            f"`&& npm run build` INSIDE the quoted script instead of after "
            f"the closing quote")


# Files a JS/Python test suite cannot execute. Editing one of these can
# no more fail `npm test` than leaving it untouched can.
_INERT_TARGET_SUFFIXES = (".css", ".scss", ".sass", ".less", ".styl")


def irrelevant_gate_reason(step: PlanStep) -> Optional[str]:
    """Explain why *step*'s gate cannot fail on *this* step's work, or None.

    ``shallow_gate_reason`` clears any command matching a test runner, on
    the reasonable ground that a suite asserts real behaviour. But a suite
    only asserts the behaviour it can reach, and a stylesheet is not
    reachable: no CSS edit can turn `npm test` red.

    Observed: a step whose brief was to add a full footer layout — brand
    area, navigation grid, legal row, responsive breakpoints — was gated
    on `cd react-home && npm test -- --run`. It deleted two words from a
    selector, wrote no footer styling whatsoever, and passed on turn 2.
    The markup shipped with eight classes and none of them styled, and
    every later check (suite, build, smoke test) was equally green,
    because none of them could see the difference.

    Deliberately narrow: only when EVERY target is a stylesheet AND the
    gate is nothing but a runner invocation. A gate that also asserts
    something about the file is fine, and any step touching executable
    code is left alone.
    """
    targets = list(getattr(step, "target_files", None) or [])
    if not targets:
        return None
    if not all(t.lower().endswith(_INERT_TARGET_SUFFIXES) for t in targets):
        return None

    cmd = (getattr(step, "verify_cmd", None) or "").strip()
    if not cmd or not _TEST_RUNNER_RE.search(cmd):
        return None
    # An assertion about the file itself makes the gate relevant again,
    # whatever else the command also runs.
    if shell_level_assertion(cmd) or _INLINE_SCRIPT_RE.search(cmd):
        return None

    return (f"the gate only runs a test suite, but this step's target(s) "
            f"({', '.join(targets[:3])}) are stylesheets that no test can "
            f"execute — it passes whether or not the styling was written, "
            f"and did exactly that on a step that produced none. Assert the "
            f"stylesheet's own content instead (that the selectors and "
            f"declarations this step promises are present), optionally "
            f"alongside the suite")


def check_gate_quality(steps: list[PlanStep]) -> list[tuple[str, str]]:
    """Find CODE steps whose ``verify:`` cannot detect a behavioural defect.

    Returns ``(step_id, reason)`` pairs, empty when every gate has teeth.

    Shallowness is judged for CODE steps only: a TEST step's gate is
    normally the suite itself, with the assertions in the test file rather
    than the command. Being UNRUNNABLE is checked for every step type —
    a gate that cannot parse is impossible to satisfy no matter what the
    step produces, and exempting TEST steps from that would leave the
    worst kind of gate in place.
    """
    gaps: list[tuple[str, str]] = []
    for step in steps:
        if not step.verify_cmd:
            continue  # missing verify entirely is a separate check
        # Unrunnable outranks shallow: a gate that cannot parse is not
        # weak, it is impossible, and reporting "too shallow" would send
        # the planner to fix the wrong thing.
        reason = unrunnable_gate_reason(step.verify_cmd)
        if reason is None and step.step_type == "CODE":
            reason = shallow_gate_reason(step.verify_cmd)
        if reason is None and step.step_type == "CODE":
            # Last: a gate can be runnable AND assert plenty, and still
            # be unable to observe the file this step was asked to write.
            reason = irrelevant_gate_reason(step)
        if reason:
            gaps.append((step.id, reason))
    return gaps


# One line per repaired gate: `<step id>: <command>`. The prompt asks for a
# bare id, but the model echoes the label it was given the gap under —
# `step 2.5: python -c ...` — so a parser that insists on a bare id throws
# away a correct answer and falls back to the re-plan it exists to avoid
# (measured: 506/140 tokens discarded, 8,214/3,219 spent instead).
_VERIFY_REPLY_RE = re.compile(
    r"^\s*(?:step\s+)?#?([0-9]+(?:\.[0-9]+)*)\s*[:.\-]\s*(\S.*)$",
    re.IGNORECASE)


def repair_verify_commands(steps: list[PlanStep],
                           gaps: list[tuple[str, str]],
                           llm_client,
                           task: str = "") -> list[str]:
    """Rewrite only the offending ``verify:`` lines, in place.

    A weak gate used to send the WHOLE plan back to the planner. That is
    the wrong shape of correction twice over. It is expensive — a re-plan
    costs a full generation (measured: 7.7k sent / 2.1k received) to fix
    one line — and it is destructive: each re-plan is a fresh generation,
    so the step list, the targets and the dependency graph all churn while
    the only thing wrong was a single command. Observed on a Pac-Man run:
    three plan attempts, three different step decompositions, ~20k tokens,
    to replace one smoke gate.

    Ask for the commands instead. Returns the ids actually repaired, so
    the caller can fall back to a re-plan when this yields nothing.
    """
    if not gaps or llm_client is None:
        return []
    by_id = {s.id: s for s in steps}
    wanted = [(sid, why) for sid, why in gaps if sid in by_id]
    if not wanted:
        return []

    lines = [
        "Some acceptance gates in the plan you wrote are unusable. "
        "Rewrite ONLY those gate commands — the rest of the plan stands.",
    ]
    if task:
        lines.append(f"\nOverall task: {task.strip()[:600]}")
    lines.append("\nGates to fix:")
    for sid, why in wanted:
        step = by_id[sid]
        lines.append(f"\nstep {sid} ({step.step_type})")
        lines.append(f"  does: {step.description.strip()[:300]}")
        if step.target_files:
            lines.append(f"  target: {', '.join(step.target_files[:4])}")
        if step.exports:
            lines.append(f"  exports: {', '.join(step.exports[:8])}")
        lines.append(f"  current verify: {step.verify_cmd}")
        lines.append(f"  problem: {why}")
    lines.append(
        "\nRules for each replacement:\n"
        "- It must FAIL (non-zero exit) if the step's promise is broken, "
        "and pass otherwise.\n"
        "- It runs from the PROJECT ROOT — imports must resolve from "
        "there, not from a subdirectory.\n"
        "- Assert a concrete value the step produces, or run the step's "
        "test suite. `assert True` is not an assertion.\n"
        "- One line, runnable from the project root, no shell heredocs.\n"
        "- Do not invent APIs the step does not export.\n"
        "\nReply with one line per gate and nothing else:\n"
        "<step id>: <command>")

    try:
        reply = llm_client.generate_response("\n".join(lines))
    except Exception:                       # network, provider, parse — all
        return []                           # fall back to the full re-plan
    if not reply:
        return []

    repaired: list[str] = []
    for raw in reply.splitlines():
        m = _VERIFY_REPLY_RE.match(raw.strip().lstrip("-* ").strip())
        if not m:
            continue
        sid, cmd = m.group(1), m.group(2).strip().strip("`").strip()
        step = by_id.get(sid)
        if step is None or not cmd:
            continue
        # Never accept a replacement that has the same defect — that would
        # burn the call and leave the gate toothless anyway.
        if unrunnable_gate_reason(cmd):
            continue
        if step.step_type == "CODE" and shallow_gate_reason(cmd):
            continue
        step.verify_cmd = cmd
        repaired.append(sid)
    return repaired


def _gate_key(step: PlanStep) -> Optional[str]:
    """What makes two steps across re-plans "the same work".

    The primary target file, because that is the one thing a re-plan
    keeps stable while ids, ordering and dependencies all churn — the
    churn is precisely why :func:`repair_verify_commands` exists. Falls
    back to the id for steps that declare no target (CMD steps mostly),
    where the id is the only handle there is.
    """
    for target in (step.target_files or []):
        norm = _norm_target_path(target)
        if norm:
            return f"file:{norm}"
    return f"id:{step.id}" if step.id else None


def carry_forward_strong_gates(previous: list[PlanStep],
                               current: list[PlanStep]) -> list[str]:
    """Keep gates a re-plan would otherwise silently weaken.

    A re-plan is triggered by ONE unusable gate, but it regenerates every
    step, so gates that were never in question are rewritten too — and a
    planner asked to strengthen step 4's gate has no reason to preserve
    the strength of step 3's.

    Measured. Plan attempt 1 of a Pac-Man run declared::

        step 3.1 verify: python -c "... g.run_frame(0.02);
                 assert g.player.pos[0] != g.player.prev_pos[0]"

    which fails against exactly the artifact that run went on to ship: a
    ``run_frame(dt)`` that ignores dt and a player that never moves. The
    re-plan was triggered by step *4.1*, and attempt 2's step 3.1 came
    back with ``assert len(g.ghosts)==4 and all(not g.map.is_wall(*pos)
    ...)`` — true of a stub. One weak gate on one step cost the strongest
    gate in the plan, and the run shipped a Pac-Man whose player could
    not move, with everything green.

    Note what that loss was NOT: both gates pass ``check_gate_quality``.
    The replacement asserts a concrete value, so no weakness check can
    see it — the plan simply traded a gate that would have caught the
    defect for one that would not. So "keep the new one unless it looks
    weak" is not enough, and the rule is:

    * new gate absent or judged weak  → restore the old one
    * new gate stands on its own      → keep BOTH, joined by ``&&``

    Conjoining rather than choosing is the honest resolution when two
    reasonable gates disagree: each was written to catch something, and
    a step that must satisfy both is strictly better checked.

    The bound on all of it is applicability. A carried gate that names a
    module the new plan no longer produces would fail a correct step
    forever, which is the failure mode ``gate_integrity`` exists for — a
    bad gate once cost a run 182k tokens and failed working code. So a
    gate is carried only when every *project* module it imports is still
    produced by some step of the new plan. Returns the ids whose gate was
    restored or strengthened.
    """
    if not previous or not current:
        return []

    weak_before = {sid for sid, _ in
                   check_gate_quality(previous) + check_gate_consistency(previous)}
    weak_now = {sid for sid, _ in
                check_gate_quality(current) + check_gate_consistency(current)}

    strong_old: dict[str, str] = {}
    for step in previous:
        key = _gate_key(step)
        cmd = (step.verify_cmd or "").strip()
        if not key or not cmd or step.id in weak_before:
            continue
        strong_old.setdefault(key, cmd)

    old_modules = _plan_modules(previous)
    new_modules = _plan_modules(current)

    restored: list[str] = []
    for step in current:
        key = _gate_key(step)
        if not key or key not in strong_old:
            continue
        carried = strong_old[key]
        cmd_now = (step.verify_cmd or "").strip()
        if cmd_now == carried or carried in cmd_now:
            continue                      # already carrying this assertion
        if not _gate_still_applies(carried, old_modules, new_modules):
            continue                      # would fail a correct step forever
        if cmd_now and step.id not in weak_now:
            if _gates_are_redundant(carried, cmd_now):
                continue
            # Both can fail on wrong behaviour. Keep both rather than
            # picking — this is the case that lost the measured gate.
            step.verify_cmd = f"{carried} && {cmd_now}"
        else:
            step.verify_cmd = carried
        restored.append(step.id)
    return restored


_SAME_SUITE_RUNNER_RE = re.compile(r"-m\s+(unittest|pytest|nose2)\b")


def _gates_are_redundant(old: str, new: str) -> bool:
    """Would running both add nothing over running the newer one?

    Two cases seen replaying the measured plans, where conjoining was
    technically correct and plainly wasteful:

    * the older gate's assertions are a prefix of the newer one's,
      differing only in spacing — ``assert TILE_SIZE == 20`` against
      ``assert TILE_SIZE==20; assert WALL==1``
    * both invoke the same test runner, so the second run re-executes the
      first's suite for nothing (``unittest -v <path>`` and
      ``unittest <path>``)
    """
    # Quotes go too, not just whitespace: the older command's closing `"`
    # is what stopped the containment from matching when it really was a
    # prefix. Over-matching here only declines to ADD a gate, which is the
    # behaviour before this function existed — never a new failure.
    def _squeeze(cmd: str) -> str:
        return re.sub(r"""[\s"']+""", "", cmd)

    if _squeeze(old) in _squeeze(new):
        return True
    old_runner = _SAME_SUITE_RUNNER_RE.search(old)
    new_runner = _SAME_SUITE_RUNNER_RE.search(new)
    return bool(old_runner and new_runner
                and old_runner.group(1) == new_runner.group(1))


_GATE_MODULE_RE = re.compile(r"\b(?:from|import)\s+([A-Za-z_][\w.]*)")


def _plan_modules(steps: list[PlanStep]) -> set[str]:
    """Importable module names the plan's own steps produce."""
    out: set[str] = set()
    for step in steps:
        for target in (step.target_files or []):
            norm = _norm_target_path(target)
            if not norm.endswith(".py"):
                continue
            dotted = norm[:-3].replace("/", ".").replace("\\", ".")
            out.add(dotted)
            out.add(dotted.rsplit(".", 1)[-1])   # importable from its own dir
    return out


def _gate_still_applies(cmd: str, old_modules: set[str],
                        new_modules: set[str]) -> bool:
    """Would this command still be runnable against the new plan?

    Only modules the OLD plan produced are judged. Anything else in the
    command is stdlib or a third-party package, whose availability has
    nothing to do with which decomposition the planner chose this time.
    """
    for name in _GATE_MODULE_RE.findall(cmd or ""):
        root = name.split(".")[0]
        for candidate in (name, root):
            if candidate in old_modules and candidate not in new_modules:
                return False
    return True


# ---------------------------------------------------------------------------
# Auto-fix: inject missing import-based dependencies
# ---------------------------------------------------------------------------

def project_file_reader(path: str) -> Optional[str]:
    """Read *path* relative to the project root, or None.

    Plan fixing runs BEFORE the run pre-loads sources into FileMemory, so
    memory is empty at that point and disk is the only place a step's
    existing target file can be read from.
    """
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except (OSError, ValueError):
        return None


def _source_derived_import_deps(steps: list[PlanStep], graph,
                                read_file) -> list[str]:
    """Edges the planner never declared, read from the files themselves.

    ``imports:`` is the planner's opinion, and it is optional. When a step
    that edits an EXISTING file declares ``imports: none``, the loop above
    has nothing to iterate and the edge is simply lost — producer and
    consumer then land in the same wave and run concurrently.

    Observed: `src/App.jsx` (whose first line is ``import './App.css'``)
    and `src/App.css` were both declared ``imports: none``, scheduled as
    ``[[0, 1]]``, and written in parallel. Neither could see the other, so
    the markup used ``site-footer__nav-title`` while the stylesheet
    defined ``site-footer__heading`` — 3 of 8 classes unstyled and 6 CSS
    rules matching nothing. Tests and the build both passed, because
    unmatched CSS classes are still valid CSS.

    Only files that already exist are consulted: for a file this run is
    about to create there is nothing to read, and the declared imports
    remain the only available signal.
    """
    if read_file is None:
        return []
    from .dependency_check import extract_file_deps

    fixes: list[str] = []
    for step in steps:
        for target in step.target_files:
            try:
                content = read_file(target)
            except Exception:
                content = None
            if not content:
                continue
            try:
                imports = extract_file_deps(target, content).imports
            except Exception:
                continue
            for spec in imports:
                if not spec.startswith("."):
                    continue          # package import, not a plan artifact
                base = posixpath.dirname(target.replace("\\", "/"))
                resolved = posixpath.normpath(posixpath.join(base, spec))
                for ext in ("", ".js", ".ts", ".tsx", ".jsx", ".css",
                            ".scss", ".py"):
                    producer_id = graph.producer_of(resolved + ext, None)
                    if producer_id is None:
                        continue
                    if producer_id == step.id or producer_id in step.depends_on:
                        break
                    step.depends_on.append(producer_id)
                    fixes.append(
                        f"Step {step.id}: {target} imports {spec} "
                        f"(produced by step {producer_id}) — added "
                        f"depends:{producer_id} [from source, undeclared]"
                    )
                    break
    return fixes


def fix_import_dependencies(steps: list[PlanStep], read_file=None) -> list[str]:
    """Auto-inject missing ``depends_on`` entries based on import relationships.

    If step B declares ``imports: src/Foo.jsx:Foo`` and step A has
    ``target: src/Foo.jsx``, then B must depend on A.  When the LLM
    forgets to declare this dependency the wave builder may schedule B
    before A, causing an import failure at runtime.

    This function detects such gaps and patches ``step.depends_on``
    in-place.  Returns a list of human-readable descriptions of fixes
    applied (empty list = nothing changed).

    Must be called **after** ``validate_plan()`` and **before**
    ``build_waves()``.

    Resolution goes through :class:`~.plan_graph.PlanGraph` rather than
    string comparison. Matching ``imports:`` text against ``target:`` text
    failed once per notation the planner invented — path, Windows path,
    dotted module, dotted-with-extension, bare filename — and every miss
    silently dropped an edge, putting producer and consumer in the same
    wave. Observed twice: a player step overwrote the map step's target
    mid-execution, and later a ghost step clobbered two sibling steps'
    files in a three-way race. The graph keys on several identities at
    once, including the exported symbol, which no spelling can disguise.
    """
    from .plan_graph import PlanGraph

    fixes: list[str] = []
    graph = PlanGraph(steps)

    for step in steps:
        for file_path, symbols in step.imports_from.items():
            producer_id = graph.producer_of(file_path, symbols)
            if producer_id is None:
                continue  # existing project file, not produced by plan
            if producer_id == step.id:
                continue  # self-reference (step modifies + imports same file)
            if producer_id in step.depends_on:
                continue  # already declared

            # Inject the missing dependency
            step.depends_on.append(producer_id)
            fixes.append(
                f"Step {step.id} imports from {file_path} "
                f"(produced by step {producer_id}) — added depends:{producer_id}"
            )

    fixes.extend(_source_derived_import_deps(steps, graph, read_file))
    fixes.extend(_infer_package_init_export_deps(steps))

    # Safety: verify we didn't introduce a cycle
    if fixes and _has_cycle(steps):
        # A package initializer that re-exports its siblings is the usual
        # cause, and the cycle is one-sided: `pacman/__init__.py` genuinely
        # needs pacman/map.py etc. to exist, but nothing needs the
        # initializer written before them — Python makes the package from
        # the directory. Drop the edges INTO the initializer and the order
        # resolves correctly, with __init__ last.
        #
        # Rolling everything back instead left the initializer scheduled
        # FIRST, which is the one order guaranteed to break. Observed: a
        # plan where 3.1 declared depends:2.1 and 2.1 imported Game from
        # 3.1; the rollback ran 2.1 in wave 2, its gate
        # `from pacman import Player, Ghost, Map, Game` passed against
        # placeholder classes the model wrote to satisfy it, the real
        # map.py landed in wave 3, the gate regressed, and the run rolled
        # back and reported failure having spent 133k tokens.
        freed = _break_cycle_at_package_inits(steps)
        if freed:
            fixes.extend(freed)
        if _has_cycle(steps):
            # Still cyclic — fall back to the blunt rollback.
            _rollback = {f.split(" — ")[0]: f.split("depends:")[1]
                         for f in fixes if "depends:" in f}
            for step in steps:
                for desc, dep_id in _rollback.items():
                    if dep_id in step.depends_on and f"Step {step.id}" in desc:
                        step.depends_on.remove(dep_id)
            fixes.append("WARNING: rolled back fixes — injecting deps "
                         "would create a cycle")

    return fixes


def _is_package_init(step: PlanStep) -> bool:
    """True when *step* only produces package ``__init__`` files."""
    targets = [t.replace("\\", "/") for t in (step.target_files or [])]
    return bool(targets) and all(
        t.rsplit("/", 1)[-1] in ("__init__.py", "index.js", "index.ts")
        for t in targets
    )


def _package_dir_of(step: PlanStep) -> Optional[str]:
    """Directory a package-initializer step writes into, ``""`` for root."""
    targets = [t.replace("\\", "/") for t in (step.target_files or [])]
    if not targets:
        return None
    head = targets[0].rsplit("/", 1)
    return head[0] if len(head) == 2 else ""


def _infer_package_init_export_deps(steps: list[PlanStep]) -> list[str]:
    """Make a package initializer wait for whoever implements its exports.

    ``fix_import_dependencies`` derives edges from the plan's ``imports:``
    line. An initializer that declares ``imports: none`` therefore gets no
    edge at all — even though re-exporting is the only thing it does, so
    some other step must define those symbols. Its ``exports:`` line still
    names them, and the step producing the same symbol under the same
    package is the producer no spelling can disguise.

    Observed live on a Pac-Man run: 2.1 wrote ``src/__init__.py``
    (``exports: Player, Ghost, Map, Game``, ``imports: none``) and 2.2 wrote
    ``src/pacman.py`` exporting the same four. Both declared only
    ``depends:1.1``, so they shared a wave. 2.1 ran with nothing to import
    and satisfied its gate — ``from src import Player, Ghost, Map, Game;
    assert all(x is not None ...)`` — by writing four ``class X: pass``
    stub modules. Every gate passed, the smoke test passed, 8/8 unittests
    passed (they import ``src.pacman`` directly), and the run was reported
    green while ``from src import Game`` returned an empty shell.

    Only ever adds edges INTO the initializer: Python builds the package
    from the directory, so nothing needs ``__init__`` written first. Any
    edge that would close a cycle is dropped rather than kept.
    """
    inits = [s for s in steps if _is_package_init(s) and s.exports]
    if not inits:
        return []

    added: list[str] = []
    for init in inits:
        pkg_dir = _package_dir_of(init)
        if pkg_dir is None:
            continue
        wanted = {e.strip() for e in init.exports if e and e.strip()}
        for other in steps:
            if other.id == init.id or _is_package_init(other):
                continue
            if other.id in init.depends_on:
                continue
            produced = {e.strip() for e in (other.exports or []) if e.strip()}
            shared = wanted & produced
            if not shared:
                continue
            # Only siblings under the initializer's own package: a step
            # elsewhere in the tree that happens to export "Game" is a
            # different Game.
            in_package = any(
                (t.replace("\\", "/").rsplit("/", 1)[0]
                 if "/" in t.replace("\\", "/") else "") == pkg_dir
                or t.replace("\\", "/").startswith(f"{pkg_dir}/")
                for t in (other.target_files or [])
            ) if pkg_dir else bool(other.target_files)
            if not in_package:
                continue

            init.depends_on.append(other.id)
            if _has_cycle(steps):
                init.depends_on.remove(other.id)
                continue
            added.append(
                f"Step {init.id} re-exports {', '.join(sorted(shared))} "
                f"(produced by step {other.id}) — added depends:{other.id}"
            )
    return added


def _break_cycle_at_package_inits(steps: list[PlanStep]) -> list[str]:
    """Remove dependencies *on* package initializers to break a cycle.

    Returns a description of each edge removed. Only touches steps that
    produce nothing but ``__init__``-style files, so an ordinary module
    keeps every edge it declared.
    """
    init_ids = {s.id for s in steps if _is_package_init(s)}
    if not init_ids:
        return []
    removed: list[str] = []
    for step in steps:
        if step.id in init_ids:
            continue        # the initializer's own deps are the real ones
        for dep in list(step.depends_on):
            if dep in init_ids:
                step.depends_on.remove(dep)
                removed.append(
                    f"Step {step.id} no longer waits on package initializer "
                    f"step {dep} — nothing needs __init__ written first, and "
                    f"the reverse edge is real (it re-exports this module)"
                )
    return removed


def _has_cycle(steps: list[PlanStep]) -> bool:
    """Detect circular dependencies via DFS."""
    all_ids = {s.id for s in steps}
    # Build adjacency: step depends on → dependency
    visited: set[str] = set()
    in_stack: set[str] = set()

    def dfs(sid: str) -> bool:
        if sid in in_stack:
            return True
        if sid in visited:
            return False
        visited.add(sid)
        in_stack.add(sid)
        step = next((s for s in steps if s.id == sid), None)
        if step:
            for dep in step.depends_on:
                if dep in all_ids and dfs(dep):
                    return True
        in_stack.discard(sid)
        return False

    for s in steps:
        if dfs(s.id):
            return True
    return False


# ---------------------------------------------------------------------------
# Wave builder
# ---------------------------------------------------------------------------

def _phase_of(step_id: str) -> int:
    """Primary step number, e.g. 2 for ``2.3``. Non-numeric IDs sort first."""
    head = (step_id or "").split(".")[0]
    return int(head) if head.isdigit() else 0


def _effective_phases(steps: list[PlanStep]) -> dict[str, int]:
    """Phase per step, raised so no step precedes something it depends on.

    A step keeps its declared phase unless a dependency sits in a later
    one, in which case it joins that phase — the intra-phase topological
    sort then orders the two correctly. Only ever moves steps later, so
    the "setup phases finish first" guarantee is untouched.
    """
    known = {s.id for s in steps}
    eff = {s.id: _phase_of(s.id) for s in steps}
    for _ in range(len(steps) + 1):        # fixed point, bounded
        changed = False
        for s in steps:
            for dep in s.depends_on:
                if dep in known and eff[dep] > eff[s.id]:
                    eff[s.id] = eff[dep]
                    changed = True
        if not changed:
            break
    return eff


def build_waves(steps: list[PlanStep]) -> list[list[PlanStep]]:
    """Topological sort into parallel execution waves, respecting phase order.

    Steps are grouped by their **primary step number** (the integer
    before the dot, e.g. ``1`` in ``1.1``, ``1.2``).  All waves within
    phase 1 complete before phase 2 starts.  Within each phase, steps
    are scheduled based on ``depends_on`` — independent sub-steps
    within the same phase can execute in parallel.

    This ensures that setup phases (CMD scaffolding) finish before
    code-generation phases begin, even when the planner omits explicit
    cross-phase dependencies.
    """
    step_map = {s.id: s for s in steps}

    # ── Group steps by primary number ──
    # A step that depends on a LATER phase is promoted to that phase.
    # Phases are otherwise walked in ID order, so such a dependency could
    # never be satisfied: the step fell into the "circular or missing
    # deps" escape hatch below and ran anyway, in the one order guaranteed
    # to fail. Observed with a package initializer — `pacman/__init__.py`
    # re-exports Game from a phase-3 step, so it belongs after it, but it
    # ran in phase 2 against classes that did not exist yet, passed its
    # gate on placeholders, then regressed when the real module landed and
    # took the whole run down with it.
    _eff = _effective_phases(steps)

    from collections import OrderedDict
    phase_groups: OrderedDict[int, list[PlanStep]] = OrderedDict()
    for s in sorted(steps, key=lambda s: (_eff[s.id], s.id)):
        phase_groups.setdefault(_eff[s.id], []).append(s)

    # ── Infer implicit deps: within a phase, steps with no declared deps
    #    should wait for any CMD steps that precede them (by step ID).
    #    This handles the common case where the LLM scaffolds a project in
    #    step 1.1 [CMD] and then writes files in 1.4–1.9 [CODE] with
    #    depends:none, causing them to run concurrently with the scaffold.
    for phase_steps in phase_groups.values():
        # Collect CMD step IDs in sorted order within the phase
        cmd_ids_in_phase = sorted(
            s.id for s in phase_steps if s.step_type == "CMD"
        )
        if not cmd_ids_in_phase:
            continue
        for s in phase_steps:
            if s.step_type == "CMD":
                # Chain CMD steps sequentially within the same phase to
                # prevent parallel package-manager operations (npm/pip/yarn)
                # targeting the same directory — causes ENOTEMPTY on Windows.
                # Always enforce this: even if a CMD step already has explicit
                # deps (e.g. depends:1.1), it must also wait for the immediately
                # preceding CMD step so installs don't run concurrently.
                preceding = [cid for cid in cmd_ids_in_phase if cid < s.id]
                if preceding:
                    prev_cmd = preceding[-1]
                    if prev_cmd not in s.depends_on:
                        s.depends_on = list(s.depends_on) + [prev_cmd]
                continue
            if s.depends_on:
                continue  # already has explicit deps — don't override
            # Add any CMD steps whose ID sorts before this step's ID
            implicit = [cid for cid in cmd_ids_in_phase if cid < s.id]
            if implicit:
                s.depends_on = list(implicit)

    waves: list[list[PlanStep]] = []
    completed: set[str] = set()

    for _phase_key, phase_steps in phase_groups.items():
        remaining = {s.id for s in phase_steps}

        while remaining:
            ready = [
                sid for sid in sorted(remaining)
                if all(d in completed for d in step_map[sid].depends_on)
            ]
            if not ready:
                # Circular or missing deps — pick the smallest ID to unblock
                ready = [min(remaining)]
            wave = [step_map[sid] for sid in ready]
            waves.append(wave)
            completed.update(ready)
            remaining -= set(ready)

    return waves


# ---------------------------------------------------------------------------
# Context builder (per-step)
# ---------------------------------------------------------------------------

def build_step_context(
    step: PlanStep,
    all_steps: list[PlanStep],
    memory,
    read_from_disk=None,
) -> dict[str, str]:
    """Build the file context for a step using plan-declared imports.

    Returns a dict of file_path -> content to inject into the LLM prompt.

    Parameters
    ----------
    step:
        The step about to execute.
    all_steps:
        All steps in the plan (for ghost contracts).
    memory:
        FileMemory instance with completed files.
    read_from_disk:
        Optional callable(file_path) -> str|None to read existing files.
    """
    files: dict[str, str] = {}

    # `imports:` entries keep the spelling the planner used, which for
    # Python is almost always the DOTTED module (`pacman_game.map`) — not
    # a path any lookup here can hit. Every `memory.get()` and every disk
    # read below therefore missed, this function returned {}, and the two
    # things that consume its output (the classic path's plan-context
    # block and the agent loop's preload) silently got nothing. Observed:
    # every CODE step opening with `read_file, read_file, read_file` to
    # fetch files the pipeline had already resolved and loaded. Resolve
    # through the plan graph, which knows all six spellings.
    from .plan_graph import PlanGraph
    _graph = PlanGraph(all_steps or [])

    def _declared_path(spec: str, symbols) -> str:
        node = _graph.resolve(spec, symbols)
        return node.path if node is not None else spec

    # 1. Plan-declared imports (real or ghost)
    for _spec, symbols in step.imports_from.items():
        file_path = _declared_path(_spec, symbols)
        # Compute the correct Python import path relative to each target file.
        # The first target is used as the reference; if there are no targets,
        # fall back to a simple path→module conversion.
        ref_target = step.target_files[0] if step.target_files else ""
        import_hint = ""
        _JS_EXTS = (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".mts", ".cts")
        if ref_target and file_path.endswith(".py"):
            # When the planner wrote a dotted module, that spelling IS the
            # answer — it is the same one its `verify:` gate imports, run
            # from the project root. _relative_import_path assumes a flat
            # script layout and would turn `pacman_game.map` into a bare
            # `from map import Map`, which fails inside a package.
            if _is_dotted_module_spec(_spec):
                import_module = _spec
            else:
                import_module = _relative_import_path(ref_target, file_path)
            symbol_str = ", ".join(symbols) if symbols else "..."
            import_hint = f"# Correct Python import: from {import_module} import {symbol_str}\n"
        elif ref_target and any(file_path.endswith(ext) for ext in _JS_EXTS):
            import_module = _relative_import_path(ref_target, file_path)
            # JS/TS relative imports need ./ prefix for same/sub-directory files
            if not import_module.startswith(".."):
                import_module = "./" + import_module
            symbol_str = ", ".join(symbols) if symbols else "..."
            import_hint = f"// Correct import: import {{ {symbol_str} }} from '{import_module}'\n"

        content = memory.get(file_path) if memory else None
        if not content and read_from_disk:
            content = read_from_disk(file_path)
        if content:
            files[file_path] = import_hint + content
        else:
            # Ghost contract: file not yet created, include planned info.
            # Reached whenever the file has no content yet — the `elif
            # read_from_disk:` this replaces made the ghost branch dead
            # code for every caller that supplies a reader (i.e. all of
            # them), so a step importing a not-yet-written file got no
            # contract at all.
            producer = _find_producer(file_path, all_steps)
            if producer and producer.status != "completed":
                ghost = (
                    f"# [PLANNED FILE — will be created by step {producer.id}]\n"
                    f"# Exports: {', '.join(producer.exports) if producer.exports else 'TBD'}\n"
                    + import_hint
                )
                files[file_path] = ghost

    # 2. Target files being modified — read current content + parse imports
    for target in step.target_files:
        if target in files:
            continue
        content = (memory.get(target) if memory else None)
        if content is None and read_from_disk:
            content = read_from_disk(target)
        if content:
            files[target] = content
            # Parse actual imports from the target file to catch undeclared deps
            try:
                from .dependency_check import extract_file_deps
                deps = extract_file_deps(target, content)
                for imp in deps.imports:
                    imp_file = _resolve_import_to_file(
                        imp, memory, read_from_disk, from_file=target)
                    if imp_file and imp_file not in files:
                        imp_content = (memory.get(imp_file) if memory else None)
                        if imp_content is None and read_from_disk:
                            imp_content = read_from_disk(imp_file)
                        if imp_content:
                            files[imp_file] = imp_content
            except Exception:
                pass  # best-effort dependency resolution

    return files


def _relative_import_path(target_file: str, dep_file: str) -> str:
    """
    Compute the correct Python import module name for *dep_file* as seen
    from *target_file*.

    Both paths use forward slashes and are relative to the project root.

    Examples
    --------
    >>> _relative_import_path("src/main.py", "src/cracker_anim.py")
    'cracker_anim'
    >>> _relative_import_path("main.py", "src/cracker_anim.py")
    'src.cracker_anim'
    >>> _relative_import_path("a/b/c.py", "a/b/utils.py")
    'utils'
    >>> _relative_import_path("a/b/c.py", "a/helpers.py")
    'a.helpers'
    """
    import posixpath

    target_dir = posixpath.dirname(target_file.replace("\\", "/"))
    dep_norm = dep_file.replace("\\", "/")
    # Strip .py extension to get module path
    dep_module = dep_norm[:-3] if dep_norm.endswith(".py") else dep_norm

    target_parts = [p for p in target_dir.split("/") if p]
    dep_parts = dep_module.split("/")

    # Find common prefix
    common = 0
    for t, d in zip(target_parts, dep_parts):
        if t == d:
            common += 1
        else:
            break

    # If dep is directly accessible from the same directory, use the remainder
    remaining = dep_parts[common:]
    return ".".join(remaining) if remaining else dep_module.replace("/", ".")


_SOURCE_EXTS = (".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
                ".mts", ".cts", ".go", ".rs", ".java", ".rb")


def _is_dotted_module_spec(spec: str) -> bool:
    """True when *spec* is a dotted module path (``pacman_game.map``).

    Distinguishes the planner's dotted spelling from a file path — a path
    names a directory or carries a source extension, a module does
    neither.
    """
    s = (spec or "").replace("\\", "/")
    if not s or "/" in s or "." not in s:
        return False
    return not s.endswith(_SOURCE_EXTS)


def _find_producer(file_path: str, steps: list[PlanStep]) -> Optional[PlanStep]:
    """Find the step that produces a given file."""
    for s in steps:
        if file_path in s.target_files:
            return s
    return None


def _resolve_import_to_file(
    import_source: str,
    memory,
    read_from_disk=None,
    from_file: Optional[str] = None,
) -> Optional[str]:
    """Best-effort resolution of an import string to a file path in memory.

    *from_file* is the file the import was written in. It is what makes a
    relative specifier resolvable at all: ``./App.css`` inside
    ``src/App.jsx`` means ``src/App.css``, and nothing below can know that
    without the importer's directory.

    Observed when it was missing: a plan whose two steps wrote
    ``src/App.jsx`` and ``src/App.css`` declared ``imports: none``, so the
    only thing that could have linked them was this source-derived
    fallback — and it returned None. The steps landed in the SAME wave and
    ran concurrently, neither seeing the other, and invented different
    class vocabularies: markup used ``site-footer__nav-title`` while the
    stylesheet defined ``site-footer__heading``. Tests and build both
    passed, because unmatched CSS classes are valid CSS.
    """
    if memory is None:
        return None
    all_files = memory.all_files()

    # Direct match
    if import_source in all_files:
        return import_source

    # Relative specifiers resolve against the IMPORTING file's directory.
    # Tried before the dotted-module branch below, which would otherwise
    # mangle "./App.css" into "//App/css" via its `.` -> `/` rewrite.
    if from_file and import_source.startswith("."):
        base = posixpath.dirname(from_file.replace("\\", "/"))
        joined = posixpath.normpath(posixpath.join(base, import_source))
        for ext in ("", ".js", ".ts", ".tsx", ".jsx", ".css", ".scss",
                    "/index.js", "/index.ts", "/index.jsx", "/index.tsx"):
            candidate = joined + ext
            if candidate in all_files:
                return candidate

    # Python: dots to path (e.g. "utils.helpers" -> "utils/helpers.py")
    as_path = import_source.replace(".", "/")
    for ext in (".py", ".js", ".ts", ".tsx", ".jsx", ""):
        candidate = as_path + ext
        if candidate in all_files:
            return candidate

    # JS relative, project-root spelling: "./utils" -> "utils.js"
    clean = import_source.lstrip("./")
    for ext in (".js", ".ts", ".tsx", ".jsx", ".css", ".scss",
                "/index.js", "/index.ts", ""):
        candidate = clean + ext
        if candidate in all_files:
            return candidate

    return None


# ---------------------------------------------------------------------------
# Post-step update
# ---------------------------------------------------------------------------

def update_step_after_execution(
    step: PlanStep,
    generated_files: dict[str, str],
) -> None:
    """Update a step's metadata after successful execution.

    Parses real exports from generated code (zero LLM cost).
    """
    step.status = "completed"
    actual_exports: list[str] = []

    for fpath, content in generated_files.items():
        try:
            from .dependency_check import extract_file_deps
            deps = extract_file_deps(fpath, content)
            actual_exports.extend(deps.exports)
        except Exception:
            pass

        # Add to target_files if not already tracked
        if fpath not in step.target_files:
            step.target_files.append(fpath)

    step.actual_exports = actual_exports


# ---------------------------------------------------------------------------
# Legacy compatibility helpers
# ---------------------------------------------------------------------------

def _synthesized_description(step: PlanStep) -> str:
    """A usable instruction for a step whose prose the planner omitted.

    ``description`` is not decoration — it is the text the coder, the
    classifier and the status line all work from, so an empty one leaves
    the step with no instruction at all. Weaker models in content mode
    routinely emit ``target:`` plus a ``content:`` body and no prose
    (observed: 7 of 9 steps from a 20B plan), which crashed the wave
    banner outright and would have handed the coder an empty task.

    Everything here comes from what the step already declares, so nothing
    is invented — it is the same facts, rendered as a sentence.
    """
    verb = "Run" if step.step_type == "CMD" else "Create"
    targets = ", ".join(step.target_files) if step.target_files else ""
    if step.step_type == "CMD" and step.command:
        return f"Run: {step.command}"
    if targets:
        text = f"{verb} {targets}"
        if step.exports:
            text += f" exporting {', '.join(step.exports)}"
        return text
    if step.command:
        return f"Run: {step.command}"
    return f"Step {step.id}"


def steps_as_text_list(steps: list[PlanStep]) -> list[str]:
    """Convert PlanStep list to legacy list[str] for backward compat.

    A step whose description the planner omitted gets one synthesized
    from its own declarations rather than an empty string.
    """
    return [s.description.strip() or _synthesized_description(s)
            for s in steps]


def steps_dependencies_dict(steps: list[PlanStep]) -> dict[int, set[int]]:
    """Convert PlanStep depends_on to legacy dict[int, set[int]] format."""
    id_to_idx = {s.id: s.index for s in steps}
    deps: dict[int, set[int]] = {}
    for s in steps:
        dep_indices = set()
        for dep_id in s.depends_on:
            if dep_id in id_to_idx:
                dep_indices.add(id_to_idx[dep_id])
        deps[s.index] = dep_indices
    return deps


# ---------------------------------------------------------------------------
# Fallback: convert old numbered-list plan to PlanStep objects
# ---------------------------------------------------------------------------

def from_legacy_steps(
    steps: list[str],
    dependencies: dict[int, set[int]],
) -> list[PlanStep]:
    """Convert old-format (list[str] + deps dict) to PlanStep objects.

    Used for checkpoint backward compatibility and gradual migration.
    Step type defaults to 'CODE' (will be classified at runtime).
    """
    result: list[PlanStep] = []
    idx_to_id = {i: str(i + 1) for i in range(len(steps))}

    for idx, text in enumerate(steps):
        dep_ids = [
            idx_to_id[d] for d in dependencies.get(idx, set())
            if d in idx_to_id
        ]
        result.append(PlanStep(
            id=idx_to_id[idx],
            step_type="UNCLASSIFIED",  # needs runtime classification
            description=text,
            depends_on=dep_ids,
            index=idx,
        ))

    return result


# ---------------------------------------------------------------------------
# Heuristic fallback parser for weaker LLMs
# ---------------------------------------------------------------------------

# Matches markdown section headers like: ### Step 1: description
#                                         ## Step 1.1 - description
_HEURISTIC_HEADER_RE = re.compile(
    r'^#{1,4}\s+(?:Step\s+)([\d.]+)[:.)\s-]*\s*(.*)?$',
    re.IGNORECASE,
)

# Matches bold key-value lines: **Type:** CODE  or  **Dependencies:** 1.1
# Handles both `**Key:** value` (markdown bold with colon inside closing **)
# and plain `Key: value` formats. The middle `\*{0,2}:` consumes `**:` or `:`.
_HEURISTIC_KV_RE = re.compile(
    r'^\*{0,2}(step\s*(?:id|#|number)?|type|dependenc(?:y|ies)|depends?(?:\s*on)?'
    r'|target|exports?|imports?|produces?|description)\*{0,2}:\s*(.*)$',
    re.IGNORECASE,
)

# Standalone **Step ID:** 1.1 — can appear as step boundary without a header
_HEURISTIC_STEPID_RE = re.compile(
    r'^\*{0,2}Step\s*(?:ID|#|Number)\*{0,2}:\s*\*{0,2}\s*([\d.]+)',
    re.IGNORECASE,
)

_VALID_STEP_TYPES = {"CMD", "CODE", "TEST", "IGNORE", "SEARCH"}


def parse_heuristic_plan(text: str) -> list[PlanStep]:
    """Fallback parser for non-standard plan formats produced by weaker LLMs.

    Handles markdown-heavy outputs like::

        ### Step 1: Create Pricelist Component
        **Step ID:** 1.1
        **Type:** CODE
        **Dependencies:** None
        **Target:** src/components/Pricelist.jsx
        **Exports:** Pricelist
        **Imports:** None

        ### Step 2: Integrate into App
        **Step ID:** 1.2
        **Type:** CODE
        **Dependencies:** 1.1
        **Target:** src/App.jsx

    Also handles ``> command`` lines and commands inside ```bash fences.

    Returns an empty list if no recognisable steps are found — safe to
    call unconditionally before the legacy numbered-list parser.
    """
    steps: list[PlanStep] = []
    current: Optional[PlanStep] = None
    desc_lines: list[str] = []
    cmd_lines: list[str] = []
    in_code_fence = False

    def _flush() -> None:
        nonlocal current, desc_lines, cmd_lines
        if current is None:
            return
        if not current.description and desc_lines:
            current.description = " ".join(desc_lines).strip()
        _flush_echo_inline(current, cmd_lines)
        steps.append(current)
        current = None
        desc_lines = []
        cmd_lines = []

    def _strip_value(raw: str) -> str:
        """Strip markdown bold markers, backticks, and trailing whitespace."""
        v = raw.strip()
        # Strip leading ** (residual closing bold from **key:** format)
        if v.startswith("**"):
            v = v[2:].strip()
        # Strip backtick wrapping (e.g. `value`)
        v = v.strip("`")
        # Strip trailing markdown bold/whitespace
        v = v.rstrip("* \t").strip()
        return v

    for raw_line in text.splitlines():
        line = raw_line.strip()

        # ── Code fence tracking ──
        if line.startswith("```"):
            in_code_fence = not in_code_fence
            continue

        # ── Markdown section header → new step ──
        m_hdr = _HEURISTIC_HEADER_RE.match(line)
        if m_hdr:
            _flush()
            step_id = m_hdr.group(1)
            description = (m_hdr.group(2) or "").strip()
            current = PlanStep(id=step_id, step_type="UNCLASSIFIED",
                               description=description)
            desc_lines = []
            cmd_lines = []
            continue

        # ── Standalone **Step ID:** as boundary when no header was used ──
        m_sid = _HEURISTIC_STEPID_RE.match(line)
        if m_sid:
            if current is None:
                # No header yet — start a new step
                _flush()
                current = PlanStep(id=m_sid.group(1), step_type="UNCLASSIFIED")
                desc_lines = []
                cmd_lines = []
            else:
                # Inside a step: refine the ID (header may have said "Step 1",
                # **Step ID:** may say "1.1" for more precision)
                current.id = m_sid.group(1)
            continue

        if current is None:
            continue  # skip preamble lines before any step is detected

        # ── Key-value metadata line ──
        m_kv = _HEURISTIC_KV_RE.match(line)
        if m_kv:
            key = m_kv.group(1).lower().replace(" ", "").rstrip("*")
            val = _strip_value(m_kv.group(2))

            if "type" in key:
                t = val.upper()
                if t in _VALID_STEP_TYPES:
                    current.step_type = t

            elif "depend" in key:
                if val.lower() not in ("none", "n/a", ""):
                    current.depends_on = [
                        d.strip() for d in val.split(",") if d.strip()
                    ]

            elif key == "target":
                if val.lower() not in ("none", "n/a", ""):
                    current.target_files = [
                        f.strip() for f in val.split(",") if f.strip()
                    ]

            elif "export" in key:
                if val.lower() not in ("none", "n/a", ""):
                    current.exports = [
                        e.strip() for e in val.split(",") if e.strip()
                    ]

            elif "import" in key:
                if val.lower() not in ("none", "n/a", ""):
                    for entry in val.split(","):
                        entry = entry.strip()
                        if ":" in entry:
                            fp, sym = entry.rsplit(":", 1)
                            current.imports_from.setdefault(
                                _norm_target_path(fp), []
                            ).append(sym.strip())

            elif "produce" in key:
                if val.lower() not in ("none", "n/a", ""):
                    for f in val.split(","):
                        f = f.strip()
                        if f and f not in current.target_files:
                            current.target_files.append(f)

            elif "description" in key:
                if val:
                    current.description = val

            elif "stepid" in key or "step#" in key or "stepnumber" in key:
                if re.match(r'^[\d.]+$', val):
                    current.id = val

            continue  # metadata line — don't fall through to description

        # ── Command line (> cmd) ── works both inside and outside fences
        if line.startswith("> "):
            cmd_text = line[2:].strip()
            # Strip backtick wrapping added by LLMs (e.g. `> `npm install`` -> `npm install`)
            if len(cmd_text) >= 2 and cmd_text[0] == "`" and cmd_text[-1] == "`":
                cmd_text = cmd_text[1:-1]
            _bare = cmd_text.lstrip("*_ \t")
            _meta_prefixes = ("produces:", "note:", "output:", "creates:",
                              "result:", "generates:", "returns:")
            if not cmd_text.startswith("**") and not any(
                _bare.lower().startswith(p) for p in _meta_prefixes
            ):
                if current.command:
                    current.command = current.command + " && " + cmd_text
                else:
                    current.command = cmd_text
                cmd_lines.append(cmd_text)
            continue

        # ── Ignore horizontal rules and plan boundary markers ──
        if line.startswith("---") or line.startswith("===") or line.startswith("***"):
            continue

        # ── Description continuation (plain text, not a metadata key) ──
        if line and not line.startswith("==") and not line.startswith("**"):
            desc_lines.append(line)

    _flush()

    # ── Post-process: infer type from content when still UNCLASSIFIED ──
    for idx, step in enumerate(steps):
        step.index = idx
        if step.step_type == "UNCLASSIFIED":
            if step.command and not step.target_files:
                step.step_type = "CMD"
            elif step.target_files:
                step.step_type = "CODE"

    for s in steps:
        s.command = dedupe_redundant_cd(s.command)

    # Only return steps that have at least a description or target/command
    return [s for s in steps if s.description or s.target_files or s.command]


def is_structured_plan(text: str) -> bool:
    """Check if LLM output is in the new structured format."""
    # Check each line independently (handles leading whitespace in raw text)
    for line in text.splitlines():
        if _STEP_RE.match(line.strip()):
            return True
    return False


# ---------------------------------------------------------------------------
# Manifest reclassification — CODE → CMD for protected manifest files
# ---------------------------------------------------------------------------

# Manifest basenames that are protected from direct LLM overwrite.
# Keep in sync with Executor._PROTECTED_FILENAMES.
_MANIFEST_BASENAMES: frozenset = frozenset({
    'package.json', 'requirements.txt', 'Cargo.toml', 'go.mod',
    'Gemfile', 'composer.json', 'Pipfile', 'pyproject.toml', 'setup.py',
})

# Manifest basename → package-manager install prefix
_MANIFEST_PM_PREFIX: dict = {
    'package.json':    'npm install',
    'requirements.txt': 'pip install',
    'Cargo.toml':      'cargo add',
    'go.mod':          'go get',
    'Gemfile':         'bundle add',
    'composer.json':   'composer require',
    'Pipfile':         'pipenv install',
    'pyproject.toml':  'pip install',
    'setup.py':        'pip install',
}

# package.json fields that are metadata, not dependency names
_PKG_JSON_NON_DEP_FIELDS = frozenset({
    'name', 'version', 'description', 'main', 'module', 'browser',
    'type', 'author', 'license', 'homepage', 'repository', 'bugs',
    'private', 'engines', 'os', 'cpu',
    # Script names and the "scripts" key itself are not packages
    'scripts', 'dev', 'build', 'start', 'test', 'lint', 'preview',
    'pretest', 'posttest', 'prebuild', 'postbuild', 'prepare',
    'prepublishOnly', 'preinstall', 'postinstall',
})


def _extract_packages_from_inline_edits(step: 'PlanStep') -> list:
    """Return package names added by the step's inline find/replace edits."""
    import os as _os
    packages: list = []
    for fpath, pairs in (step.inline_edits or {}).items():
        basename = _os.path.basename(fpath)
        for find_str, replace_str in pairs:
            find_lines = {l.strip().rstrip(',') for l in find_str.splitlines() if l.strip()}
            for raw_line in replace_str.splitlines():
                stripped = raw_line.strip().rstrip(',')
                if not stripped or stripped in find_lines:
                    continue
                if basename == 'package.json':
                    # Match: "animejs": "^4.2.0"
                    m = re.match(r'^"([^"]+)"\s*:\s*"([^"]*)"$', stripped)
                    if m and m.group(1) not in _PKG_JSON_NON_DEP_FIELDS:
                        # Reject entries whose value looks like a script
                        # command (e.g. "vite build") rather than a semver
                        # version (e.g. "^4.2.0").  Script values contain
                        # spaces or don't start with a version-range prefix.
                        val = m.group(2)
                        _is_version = bool(re.match(
                            r'^[\^~>=<*0-9]', val)) and ' ' not in val
                        if _is_version:
                            packages.append(m.group(1))
                elif basename in ('requirements.txt', 'Pipfile', 'pyproject.toml', 'setup.py'):
                    # Match: animejs==4.2.0  or  animejs>=3
                    m = re.match(r'^([A-Za-z0-9_.-]+)', stripped)
                    if m and not stripped.startswith('#') and not stripped.startswith('['):
                        packages.append(m.group(1))
                elif basename == 'Cargo.toml':
                    # Match: animejs = "4.2.0"  (skip section headers)
                    if not stripped.startswith('[') and '=' in stripped:
                        pkg_name = stripped.split('=')[0].strip()
                        if re.match(r'^[a-zA-Z0-9_-]+$', pkg_name):
                            packages.append(pkg_name)
                elif basename == 'go.mod':
                    # Match: require github.com/foo/bar v1.2.3
                    m = re.match(r'^(?:require\s+)?(\S+/\S+)\s+v', stripped)
                    if m:
                        packages.append(m.group(1))
    return packages


def reclassify_manifest_steps(plan_steps: list) -> list:
    """Convert CODE steps whose targets are *only* dependency manifest files into
    CMD package-manager install steps.

    Rationale: The executor's protected-file guard blocks direct overwrites of
    ``package.json``, ``requirements.txt``, etc.  Converting to ``npm install``
    (or the ecosystem equivalent) lets the package manager do the safe, atomic
    update and actually installs the package into ``node_modules`` /
    ``site-packages``.  The guard still exists as a safety net for any stray
    LLM-generated writes that slip through.

    Steps where no packages can be detected from the inline edits are left
    unchanged so they fall through to the normal CODE path (which will log a
    warning about the protected write).
    """
    import os as _os
    import logging as _logging
    _log = _logging.getLogger(__name__)

    for step in plan_steps:
        if step.step_type != "CODE":
            continue
        if not step.target_files:
            continue

        basenames = [_os.path.basename(f) for f in step.target_files]
        if not all(b in _MANIFEST_BASENAMES for b in basenames):
            continue  # at least one non-manifest target — leave as CODE

        packages = _extract_packages_from_inline_edits(step)

        if not packages:
            _log.debug(
                "[PlanStep] Step %s targets manifest(s) only but no packages "
                "detected from inline edits — leaving as CODE", step.id,
            )
            continue

        primary = basenames[0]
        prefix = _MANIFEST_PM_PREFIX.get(primary, 'pip install')
        install_cmd = f"{prefix} {' '.join(packages)}"

        _log.info(
            "[PlanStep] Step %s reclassified CODE→CMD "
            "(manifest-only target): %s", step.id, install_cmd,
        )
        step.step_type = "CMD"
        step.command = install_cmd
        step.inline_edits = {}
        step.inline_code = {}

    return plan_steps
