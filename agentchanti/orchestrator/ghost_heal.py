"""Deterministic repair of the gaps :mod:`ghost` finds — no LLM calls.

What this is for
----------------
The shadow proved it can see gaps the pipeline cannot: a dependency
declared in the manifest but absent from the interpreter that runs the
app, an import the plan promised and no file makes. Seeing them at wave 2
and doing nothing until the smoke test crashes at wave 8 wastes the
detection. This module closes the gaps it can close mechanically, so the
plan's declared goals are actually wired rather than merely reported as
unwired.

The rule that governs everything here
-------------------------------------
**Never invent content. Freely restore content the PLAN already
specified.**

Those are different things, and the difference is the whole design. A
healer that makes up an empty ``.site-header {}`` to satisfy "the
stylesheet is missing that class" converts a real styling bug into a
green gate and a clean report — a detectable defect made undetectable,
which is the exact failure mode `reachability.py` exists to catch. But a
healer that writes the ``.site-header`` rule **the planner itself put in
``PlanStep.inline_code``** invents nothing: it enforces a decision
already made, deterministically, with no model in the loop.

That second case is the one that matters in practice. A smaller model
plans the work correctly and then drifts while executing an individual
step — the file it writes is missing a class, an export, a whole
section the plan spelled out. The plan is right and the artifact is
wrong, and nothing in the pipeline reconciles them. `PLAN_ANCHORS`
detects that drift and the healers below repair it from the plan's own
body.

So:

  * installing a declared dependency        — state, healable
  * creating an empty ``__init__.py``       — empty IS the content
  * adding an import of a symbol that
    demonstrably exists                     — mechanical, healable
  * restoring a file, class, or export the
    plan's own body declares                — the planner's content,
                                              healable
  * content NO source specifies — not the
    plan, not the filesystem                — REPORTED, never written

Restoration is refused when the written file declares anything the
plan's body does not: the step may have added real work beyond the plan,
and overwriting that would be its own defect. Such a conflict is
reported for a human or the model to settle.

Every heal is verified
----------------------
A heal is only kept if re-resolving the expectation afterwards actually
turns it green. Source edits are snapshotted first and restored on
failure, so a heal that does not work leaves nothing behind. Everything
attempted is recorded and surfaced in the report — a run must never be
able to look clean *because* the healer touched it without saying so.
"""

from __future__ import annotations

import ast
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

from .ghost import (
    HOLDS, KIND_EXISTS, KIND_EXPORTS, KIND_IMPORT_EDGE, _is_unbound_use,
    KIND_PKG_PRESENT, KIND_PLAN_ANCHORS, VIOLATED,
    _pep503, _read, _requirements_names,
)

_logger = logging.getLogger(__name__)


@dataclass
class HealResult:
    """One repair attempt and what became of it."""

    exp_id: str
    kind: str
    action: str                  # what was tried, in human terms
    ok: bool = False             # did the expectation turn green?
    detail: str = ""
    reverted: bool = False

    def describe(self) -> str:
        state = "healed" if self.ok else (
            "reverted" if self.reverted else "failed")
        return f"{state}: {self.action}" + (f" — {self.detail}"
                                            if self.detail else "")


@dataclass
class _Snapshot:
    """Bytes of a file before a source edit, for exact restoration."""

    path: str
    existed: bool
    content: Optional[str] = None


class GhostHealer:
    """Closes the gaps the ghost finds, mechanically and verifiably."""

    def __init__(self, ghost, executor, allow_source_edits: bool = True,
                 timeout: int = 300) -> None:
        self.ghost = ghost
        self.executor = executor
        self.allow_source_edits = allow_source_edits
        self.timeout = timeout
        self.results: list[HealResult] = []
        self._attempted: set[str] = set()

    # -- plumbing ------------------------------------------------------

    def _abs(self, rel: str) -> str:
        return os.path.join(self.ghost.root, rel.replace("/", os.sep))

    def _snapshot(self, rel: str) -> _Snapshot:
        full = self._abs(rel)
        if not os.path.isfile(full):
            return _Snapshot(rel, existed=False)
        try:
            with open(full, "r", encoding="utf-8", errors="replace") as fh:
                return _Snapshot(rel, existed=True, content=fh.read())
        except OSError:
            return _Snapshot(rel, existed=False)

    def _restore(self, snap: _Snapshot) -> None:
        full = self._abs(snap.path)
        try:
            if not snap.existed:
                if os.path.isfile(full):
                    os.remove(full)
                return
            with open(full, "w", encoding="utf-8", newline="") as fh:
                fh.write(snap.content or "")
        except OSError as exc:
            _logger.warning("[GhostHeal] could not restore %s: %s",
                            snap.path, exc)

    def _write(self, rel: str, text: str) -> bool:
        full = self._abs(rel)
        try:
            os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
            with open(full, "w", encoding="utf-8", newline="") as fh:
                fh.write(text)
            return True
        except OSError as exc:
            _logger.warning("[GhostHeal] could not write %s: %s", rel, exc)
            return False

    def _reresolve(self, exp_id: str, language: str | None,
                   gate_cmds: Iterable[str]) -> str:
        """Re-check one expectation and return its fresh verdict."""
        exp = self.ghost.expectations.get(exp_id)
        if exp is None:
            return VIOLATED
        owners = [sid for sid, node in self.ghost.steps.items()
                  if exp_id in node["produces"] | node["requires"]]
        self.ghost.resolve(owners, language=language, gate_cmds=gate_cmds,
                           stage="post-heal")
        return exp.verdict

    # -- entry point ---------------------------------------------------

    def heal(self, step_ids: Iterable[str], *, language: str | None = None,
             gate_cmds: Iterable[str] = (), stage: str = "") -> list[HealResult]:
        """Attempt every registered repair for currently-violated nodes.

        Each expectation is attempted at most once per run: a repair that
        did not take is not going to take on the next wave either, and
        retrying it would spend a subprocess to learn nothing.
        """
        wanted: set[str] = set()
        for sid in step_ids:
            node = self.ghost.steps.get(sid)
            if node:
                wanted |= node["produces"] | node["requires"]

        out: list[HealResult] = []
        for exp_id in sorted(wanted):
            exp = self.ghost.expectations.get(exp_id)
            if exp is None or exp.verdict != VIOLATED:
                continue
            if exp_id in self._attempted:
                continue
            healer = _HEALERS.get(exp.kind)
            if healer is None:
                continue
            self._attempted.add(exp_id)
            try:
                result = healer(self, exp)
            except Exception as exc:          # never fail a run
                _logger.debug("[GhostHeal] %s raised: %s", exp_id, exc)
                continue
            if result is None:
                continue
            if result.ok:
                # "ok" from the healer means the action ran; the
                # expectation itself is the judge of whether it worked.
                verdict = self._reresolve(exp_id, language, gate_cmds)
                result.ok = verdict == HOLDS
                if not result.ok and result.kind != KIND_PKG_PRESENT:
                    snap = getattr(result, "_snapshot", None)
                    if snap is not None:
                        self._restore(snap)
                        result.reverted = True
                        self._reresolve(exp_id, language, gate_cmds)
            _logger.info("[GhostHeal] %s (%s) — %s",
                         exp.kind, exp.subject, result.describe())
            out.append(result)
            self.results.append(result)
        return out

    def heal_uncollected_tests(self) -> list[HealResult]:
        """Make a test directory reachable by the run's own test command.

        ``unittest`` discovery recurses only into importable packages, so
        a suite written to ``tests/`` with no ``__init__.py`` contributes
        nothing to ``python -m unittest`` — and the gate stays green on
        whatever else the root happens to hold. Measured: a step spent
        eight turns watching its own gate pass while the six tests it had
        just written were never collected.

        This is the ``EXISTS`` healer's rule applied to a file the plan
        did not name: empty *is* the correct content of an ``__init__.py``,
        so creating one invents nothing. It restores a decision the plan
        already made — the step declared ``tests/test_game.py`` as its
        target and the run declared ``python -m unittest`` as its gate;
        only the marker that connects them is missing.

        Refuses, as everywhere else, rather than guessing: nothing happens
        unless the file is genuinely unreachable, and a directory that
        already holds an ``__init__.py`` is left alone.
        """
        from .ghost import (declared_runner, is_python_test_file,
                            discovers_from_project_root,
                            unreachable_package_dir)

        out: list[HealResult] = []
        try:
            commands = list(self.ghost.declared_commands)
        except Exception:
            return out
        if declared_runner(commands) != "unittest":
            return out
        if not discovers_from_project_root(commands):
            return out

        seen: set[str] = set()
        for path in sorted(self.ghost.files):
            if not is_python_test_file(path):
                continue
            missing = unreachable_package_dir(self.ghost.root, path)
            if not missing or missing in seen:
                continue
            seen.add(missing)
            key = f"uncollected:{missing}"
            if key in self._attempted:
                continue
            self._attempted.add(key)
            init = os.path.join(self.ghost.root, *missing.split("/"),
                                "__init__.py")
            action = (f"create {missing}/__init__.py so `unittest` "
                      f"discovery can reach {path}")
            try:
                os.makedirs(os.path.dirname(init), exist_ok=True)
                with open(init, "w", encoding="utf-8"):
                    pass
            except OSError as exc:
                out.append(HealResult(key, "TESTS_COLLECTED", action,
                                      ok=False, detail=str(exc)))
                continue
            # Verified the same way every other heal is: by re-asking the
            # question, not by trusting that the write happened.
            ok = unreachable_package_dir(self.ghost.root, path) is None
            res = HealResult(key, "TESTS_COLLECTED", action, ok=ok)
            _logger.info("[GhostHeal] tests-never-collected (%s) — %s",
                         missing, res.describe())
            out.append(res)
            self.results.append(res)
        return out

    def summary(self) -> str:
        if not self.results:
            return ""
        healed = [r for r in self.results if r.ok]
        failed = [r for r in self.results if not r.ok]
        return (f"[GhostHeal] {len(healed)} gap(s) closed without an LLM, "
                f"{len(failed)} could not be repaired mechanically")


# ── Individual healers ───────────────────────────────────────────────
#
# Each returns a HealResult with ok=True when its ACTION ran; the caller
# re-resolves the expectation to decide whether it actually worked.


def _heal_packages(h: GhostHealer, exp) -> Optional[HealResult]:
    """Install declared dependencies missing from the app's environment.

    Pure environment state — no project file is touched. The install
    targets the interpreter the gates and the app actually use, which is
    the whole point: the defect being repaired is precisely a package
    that went to the wrong interpreter.
    """
    manifest = exp.subject
    text = _read(h.ghost.root, manifest)
    if text is None:
        return None
    base = os.path.basename(manifest).lower()

    if base == "package.json":
        # The install must run where the manifest lives, for the same
        # reason the check reads that directory's node_modules: `npm
        # install` at the repo root writes a root package.json and a root
        # node_modules, neither of which is the environment the app runs
        # in. Observed installing the backend's four dependencies and the
        # frontend's three at the top level while both sub-projects
        # already had them correctly installed.
        pkg_dir = os.path.dirname(manifest)
        missing = _missing_node_deps(
            os.path.join(h.ghost.root, pkg_dir), text)
        if not missing:
            return None
        cmd = "npm install " + " ".join(missing)
        if pkg_dir:
            cmd = f"npm --prefix {pkg_dir} install " + " ".join(missing)
    else:
        missing = _missing_python_deps(h.ghost.root, text)
        if not missing:
            return None
        from .agent_loop import _venv_python
        py = _venv_python(h.ghost.root) or "python"
        py_tok = f'"{py}"' if py != "python" else py
        cmd = f"{py_tok} -m pip install " + " ".join(missing)

    action = f"install {', '.join(missing)} into the project environment"
    _logger.info("[GhostHeal] %s", cmd)
    try:
        ok, out = h.executor.run_command(cmd, cwd=h.ghost.root,
                                         timeout=h.timeout)
    except Exception as exc:
        return HealResult(exp.id, exp.kind, action, ok=False, detail=str(exc))
    return HealResult(exp.id, exp.kind, action, ok=bool(ok),
                      detail="" if ok else (out or "")[-300:])


def _missing_python_deps(root: str, text: str) -> list[str]:
    from .ghost import _installed_names, _site_packages
    from .agent_loop import _venv_python

    py = _venv_python(root)
    if not py:
        return []
    site = _site_packages(os.path.dirname(py))
    if not site:
        return []
    installed = _installed_names(site)
    if installed is None:
        return []
    return [n for n in _requirements_names(text)
            if _pep503(n) not in installed]


def _missing_node_deps(root: str, text: str) -> list[str]:
    import json
    try:
        deps = list((json.loads(text).get("dependencies") or {}).keys())
    except (ValueError, AttributeError):
        return []
    nm = os.path.join(root, "node_modules")
    return [d for d in deps
            if not os.path.exists(os.path.join(nm, *d.split("/")))]


def _heal_missing_file(h: GhostHealer, exp) -> Optional[HealResult]:
    """Restore a file the plan promised, from the plan's own body.

    Two sources, neither of them invention:

      * ``PlanStep.inline_code`` — the planner wrote this file's contents
        itself, so writing them is enforcing the plan, not guessing at
        it. This is the small-model failure mode: the plan is right and
        the step that was supposed to execute it produced nothing (or
        the wrong path), and today nothing reconciles the two.
      * an ``__init__.py`` package marker, whose correct contents are
        empty and therefore need no source at all.

    A file the plan gave no body for is left missing and reported: there
    is nothing to restore it from, and a stub would only hide the gap.
    """
    path = exp.subject
    if not h.allow_source_edits:
        return None

    body = h.ghost.plan_content.get(path)
    if body:
        snap = h._snapshot(path)
        if not h._write(path, body):
            return None
        result = HealResult(
            exp.id, exp.kind,
            f"restore {path} from the plan's own body "
            f"({len(body)} chars)", ok=True,
            detail="content came from the planner, not from this healer")
        result._snapshot = snap                    # type: ignore[attr-defined]
        return result

    if os.path.basename(path) != "__init__.py":
        return None
    snap = h._snapshot(path)
    if not h._write(path, ""):
        return None
    result = HealResult(exp.id, exp.kind,
                        f"create empty package marker {path}", ok=True)
    result._snapshot = snap                        # type: ignore[attr-defined]
    return result


def _heal_plan_drift(h: GhostHealer, exp) -> Optional[HealResult]:
    """Reconcile a file that drifted from what the plan specified.

    The plan's body for this file declares names the written file does
    not have. Rather than overwrite wholesale — the step may have added
    real work beyond the plan, and discarding it would be its own defect
    — the plan's body is only restored when the written file is a strict
    *regression*: it declares nothing the plan's body does not already
    declare. When the file has diverged in both directions, the conflict
    is reported and left for a human or the model to resolve.
    """
    if not h.allow_source_edits:
        return None
    path = exp.subject
    body = h.ghost.plan_content.get(path)
    if not body:
        return None
    current = _read(h.ghost.root, path)
    if current is None:
        return None

    from .ghost import plan_anchors
    planned = plan_anchors(path, body)
    present = plan_anchors(path, current)
    extra = present - planned
    if extra:
        return HealResult(
            exp.id, exp.kind,
            f"NOT restoring {path} from the plan", ok=False,
            detail=(f"the written file also declares {', '.join(sorted(extra))}, "
                    f"which the plan's body does not — restoring would "
                    f"discard work, so this is reported instead"))

    snap = h._snapshot(path)
    if not h._write(path, body):
        return None
    result = HealResult(
        exp.id, exp.kind,
        f"restore {path} from the plan's own body — the step dropped "
        f"{', '.join(sorted(planned - present))}", ok=True,
        detail="content came from the planner, not from this healer")
    result._snapshot = snap                        # type: ignore[attr-defined]
    return result


_IMPORT_RE = re.compile(r"^\s*(?:import|from)\s", re.MULTILINE)


def _module_symbols(text: str) -> set[str]:
    """Module-level and class-level names a Python file defines."""
    names: set[str] = set()
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return names
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target,
                                                            ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets
                         if isinstance(t, ast.Name))
    return names


def _heal_import_edge(h: GhostHealer, exp) -> Optional[HealResult]:
    """Add an import the plan declared and no file makes.

    Deliberately narrow, because a wrong import is worse than a missing
    one — it can shadow a name or create a cycle. All of these must hold:

      * both files are Python and both are readable;
      * source and consumer sit in the SAME directory, so the module name
        is unambiguous without knowing what ends up on ``sys.path``;
      * exactly one consumer is a real module (a bare ``__init__.py``
        package marker is never the intended importer);
      * every declared symbol genuinely exists in the source module —
        an import of a name that is not there would turn a reported gap
        into an ImportError;
      * the consumer still parses afterwards.

    Anything else is left for the report. The caller re-resolves and
    restores the original bytes if the edit did not close the gap.
    """
    if not h.allow_source_edits:
        return None
    src = exp.subject
    if not src.endswith(".py"):
        return None
    src_text = _read(h.ghost.root, src)
    if src_text is None:
        return None

    candidates = [c for c in exp.consumers
                  if c.endswith(".py")
                  and os.path.basename(c) != "__init__.py"
                  and _read(h.ghost.root, c) is not None]
    if len(candidates) != 1:
        return None
    consumer = candidates[0]

    # Same directory keeps the module name unambiguous.
    if os.path.dirname(src) != os.path.dirname(consumer):
        return None

    symbols = [s.strip() for s in (exp.detail or "").split(",") if s.strip()]
    if not symbols:
        return None
    available = _module_symbols(src_text)
    if not available or any(s not in available for s in symbols):
        return None

    # The plan declaring an import is NOT sufficient reason to write one.
    # Observed: a plan declared `imports: player.py:Direction` for a step
    # targeting main.py; main.py routes input through `game.handle_event`
    # and never needs Direction. This healer added the import anyway and
    # turned the node green — dead code, written to satisfy the letter of
    # a declaration that was simply wrong. Enforcing a plan is only
    # legitimate where the artifact actually contradicts itself.
    #
    # So the bar is a real defect: the consumer USES the name and nothing
    # binds it, which is a NameError waiting to happen. Adding the import
    # then fixes a bug rather than decorating a mismatch.
    consumer_text_pre = _read(h.ghost.root, consumer) or ""
    needed = [s for s in symbols
              if _is_unbound_use(consumer_text_pre, s)]
    if not needed:
        return HealResult(
            exp.id, exp.kind,
            f"NOT adding an import to {consumer}", ok=False,
            detail=(f"{consumer} never uses {', '.join(symbols)} — the plan "
                    f"declared an import the file does not need, so writing "
                    f"one would add dead code to make a check pass"))
    symbols = needed

    module = os.path.basename(src)[:-3]
    if not module.isidentifier():
        return None

    consumer_text = _read(h.ghost.root, consumer) or ""
    line = f"from {module} import {', '.join(symbols)}"
    if line in consumer_text:
        return None

    # Place it after the existing import block, or at the top of the file
    # below any module docstring.
    lines = consumer_text.splitlines()
    insert_at = 0
    try:
        tree = ast.parse(consumer_text)
        if (tree.body and isinstance(tree.body[0], ast.Expr)
                and isinstance(tree.body[0].value, ast.Constant)
                and isinstance(tree.body[0].value.value, str)):
            insert_at = tree.body[0].end_lineno or 0
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                insert_at = max(insert_at, node.end_lineno or insert_at)
    except (SyntaxError, ValueError):
        return None

    snap = h._snapshot(consumer)
    new_lines = lines[:insert_at] + [line] + lines[insert_at:]
    new_text = "\n".join(new_lines) + ("\n" if consumer_text.endswith("\n")
                                       else "")
    try:
        ast.parse(new_text)
    except SyntaxError:
        return None
    if not h._write(consumer, new_text):
        return None

    result = HealResult(
        exp.id, exp.kind,
        f"add `{line}` to {consumer}", ok=True,
        detail=f"symbols verified present in {src}")
    result._snapshot = snap                        # type: ignore[attr-defined]
    return result


def _heal_exports_from_plan(h: GhostHealer, exp) -> Optional[HealResult]:
    """A declared export the file lost, restored from the plan's body.

    Only fires when the plan supplied this file's contents AND those
    contents genuinely declare the missing symbol — i.e. the plan was
    right and the step drifted. Where the plan gave no body, or its body
    lacks the symbol too, the export is the planner's own error and no
    amount of mechanical repair can invent it.
    """
    if not h.allow_source_edits:
        return None
    path = exp.subject
    body = h.ghost.plan_content.get(path)
    if not body:
        return None
    symbol = (exp.detail or "").strip()
    from .ghost import plan_anchors
    if symbol not in plan_anchors(path, body):
        return None
    return _heal_plan_drift(h, exp)


# Kinds absent from this map have no mechanical repair. PARSES, TOUCHED
# and GATE_PASSED would each need content that neither the plan nor the
# filesystem supplies — healing them would mean writing code to satisfy
# a check about whether that code was written, which turns a detectable
# defect into an undetectable one.
_HEALERS: dict[str, Callable[[GhostHealer, object], Optional[HealResult]]] = {
    KIND_PKG_PRESENT: _heal_packages,
    KIND_EXISTS: _heal_missing_file,
    KIND_IMPORT_EDGE: _heal_import_edge,
    KIND_PLAN_ANCHORS: _heal_plan_drift,
    KIND_EXPORTS: _heal_exports_from_plan,
}


# ── Module-level handle (mirrors get_ghost / get_gate_ledger) ────────

_healer: Optional[GhostHealer] = None


def start_healer(ghost, executor,
                 allow_source_edits: bool = True) -> Optional[GhostHealer]:
    global _healer
    try:
        _healer = GhostHealer(ghost, executor,
                              allow_source_edits=allow_source_edits)
    except Exception as exc:
        _healer = None
        _logger.debug("[GhostHeal] disabled — init failed: %s", exc)
    return _healer


def get_healer() -> Optional[GhostHealer]:
    return _healer


def reset_healer() -> None:
    global _healer
    _healer = None
