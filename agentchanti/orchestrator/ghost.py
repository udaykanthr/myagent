"""Shadow reconciliation of a plan's *declared effects* against reality.

What this is
------------
After the planner produces a structured plan, every step already declares
what it will do to the project: ``target:`` names files, ``exports:``
names symbols, ``imports:`` names cross-file edges, ``verify:`` names an
acceptance command. Those declarations are checkable **postconditions**,
and checking them costs no LLM call — only a stat, a hash, and an AST
parse.

This module builds that set of postconditions up front ("the ghost"),
records the project's real pre-state, and afterwards compares what the
plan promised against what the filesystem actually shows. It reports
where the pipeline's own verdict and the evidence disagree.

Read-only by construction
-------------------------
*This module* never writes to the project, never runs a command, and
never changes a verdict — so it cannot slow a run down or fail one.
``GATE_PASSED`` is resolved by asking the
:class:`~.wave_snapshots.GateLedger` what already passed, never by
re-running anything. Every entry point swallows its own exceptions.

Repair lives next door, in :mod:`ghost_heal`, which acts on what is
found here: installing a declared dependency into the interpreter that
actually runs the app, creating an absent package marker, adding a
declared-but-missing import. It is bounded by one rule — heal *state*,
never fabricate *content* — because a healer that invents a CSS rule or
a function body to satisfy a check turns a detectable defect into an
undetectable one. Detection stays honest here regardless: every heal is
verified by re-resolving the expectation it targeted, so a repair that
did not work leaves the verdict red.

Deliberately three-valued
-------------------------
Verdicts are ``HOLDS`` / ``VIOLATED`` / ``UNKNOWN`` / ``INAPPLICABLE``,
never a boolean. The rest of this package learned that the hard way:
``GateLedger._sample_gate`` separates crash and harness errors from real
failures, and ``verify_dt_invariance`` reserves exit code 2 for
"could not verify". Collapsing "no evidence" into "failed" manufactures
regressions out of silence, so an unreadable file, an unknown language
or a missing extractor all resolve to ``UNKNOWN`` and are counted
nowhere.

What it can see that nothing else does
--------------------------------------
* **Planned but untouched** — a step reports success and its target
  file's bytes never changed.
* **Touched but unplanned** — a recovery or agent loop rewrote a file no
  step ever claimed.
* **No checkable claim** — a step whose expectations are all
  tautologies (only "the file we ourselves wrote from the plan exists"),
  i.e. a step that certifies nothing. This mirrors ``gate_integrity``'s
  rule for acceptance commands, applied one layer up to the plan.
* **Failed but clean** — the run was marked failed while every declared
  postcondition holds, which historically has meant a harness defect
  rather than a model failure.
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from threading import Lock
from typing import Iterable, Optional

# `_export_satisfied` encodes hard-won knowledge about the many ways a
# planner spells an export ("default Footer", prose, "(none)"). Reusing it
# is the whole point — a second, naive comparison here would reproduce the
# false-warning history that function exists to end.
#
# `plan_graph.normalize_path` is deliberately NOT used: its `lstrip("./")`
# strips any leading dot, so `.agentchanti/log.txt` becomes
# `agentchanti/log.txt` and `.env` becomes `env`. That is harmless for a
# graph of planned modules, but this module stats real files and compares
# against FileMemory keys (which keep their dots), so it needs a
# normaliser that only collapses separators and a leading `./`.
from .plan_graph import _canonical_name, _export_satisfied, module_key

_logger = logging.getLogger(__name__)

# ── Verdict lattice ──────────────────────────────────────────────────
UNKNOWN = "UNKNOWN"
HOLDS = "HOLDS"
VIOLATED = "VIOLATED"
INAPPLICABLE = "INAPPLICABLE"

# ── Expectation kinds ────────────────────────────────────────────────
KIND_EXISTS = "EXISTS"
KIND_TOUCHED = "TOUCHED"
KIND_PARSES = "PARSES"
KIND_EXPORTS = "EXPORTS"
KIND_IMPORT_EDGE = "IMPORT_EDGE"
KIND_PKG_PRESENT = "PKG_PRESENT"
KIND_PLAN_ANCHORS = "PLAN_ANCHORS"
KIND_GATE_PASSED = "GATE_PASSED"

# Manifests whose declared runtime dependencies can be checked against the
# environment the app will actually run in.
_MANIFESTS = ("requirements.txt", "package.json")

# How much each kind counts toward a step having asserted anything real.
# EXISTS is scored at build time instead (0 for a file the plan itself
# supplies the bytes for — the pipeline writing its own inline content and
# then observing that it landed proves nothing about the task).
_WEIGHTS = {
    KIND_TOUCHED: 1,
    KIND_PARSES: 1,
    KIND_EXPORTS: 2,
    KIND_IMPORT_EDGE: 3,
    KIND_PKG_PRESENT: 4,
    KIND_PLAN_ANCHORS: 4,
    KIND_GATE_PASSED: 5,
}

# A step whose resolved evidence weighs less than this asserted nothing
# that could have failed. One EXISTS on a command-produced file clears it;
# one EXISTS on a file we pasted from the plan does not.
MIN_STEP_STRENGTH = 1

_TEXT_READ_LIMIT = 2_000_000     # don't hash a stray binary/asset blob


def _norm(path: str) -> str:
    """Collapse separators and strip a leading ``./`` — dots survive."""
    p = re.sub(r"[\\/]+", "/", (path or "").strip())
    while p.startswith("./"):
        p = p[2:]
    return p


def _looks_like_path(target: str) -> bool:
    """Is *target* a file path, or prose the planner wrote instead?

    ``produces:`` is where planners put whatever they consider the step's
    output, and a weaker model answers it in English. Observed on a
    20B-model run: ``produces: pygame package`` became a "planned target"
    that could never exist on disk, so the shadow reported a missing file
    and then scored the step at zero evidence for good measure — two
    fabricated findings from one prose line.

    Whitespace is the giveaway. Real paths in generated projects do not
    contain spaces, while a prose answer almost always does; ``venv`` and
    ``tests`` stay valid because a bare directory name is a legitimate
    target.
    """
    t = (target or "").strip()
    if not t or t.lower() in ("none", "n/a", "na", "-"):
        return False
    return not any(ch.isspace() for ch in t)


def _actual_spelling(full_path: str) -> Optional[str]:
    """The name as the directory really spells it, or ``None``."""
    directory, name = os.path.split(full_path)
    try:
        entries = os.listdir(directory or ".")
    except OSError:
        return None
    if name in entries:
        return name
    lowered = name.lower()
    for entry in entries:
        if entry.lower() == lowered:
            return entry
    return None


def _near_miss(full_path: str) -> Optional[str]:
    """A sibling whose name differs from *full_path*'s only trivially.

    Answers "the plan said `board.py` and it is not there — is something
    almost-that there?" Case differences and a changed extension are the
    two spellings a generated project actually gets wrong, and on Windows
    the case variant still imports, so the mismatch stays invisible until
    the project is checked out somewhere case-sensitive.
    """
    directory, name = os.path.split(full_path)
    if not name:
        return None
    try:
        entries = os.listdir(directory or ".")
    except OSError:
        return None
    stem, _ = os.path.splitext(name)
    lowered = name.lower()
    for entry in entries:
        if entry == name:
            continue
        if entry.lower() == lowered:
            return entry
    for entry in entries:
        e_stem, e_ext = os.path.splitext(entry)
        if e_ext and e_stem.lower() == stem.lower():
            return entry
    return None


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:16]


def _read(root: str, rel: str) -> Optional[str]:
    """File contents, or ``None`` when it cannot be read as text."""
    try:
        full = os.path.join(root, rel.replace("/", os.sep))
        if not os.path.isfile(full):
            return None
        if os.path.getsize(full) > _TEXT_READ_LIMIT:
            return None
        with open(full, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return None


# ── Data model ───────────────────────────────────────────────────────


@dataclass
class Expectation:
    """One checkable postcondition of the plan.

    The ``id`` is canonical and interned, so two steps declaring the same
    fact share a single node — which is what makes a cross-step
    contradiction (step 3 exports ``Board``, step 7 imports it) visible
    as one object rather than two independent opinions.
    """

    id: str
    kind: str
    subject: str                              # path, cmd, or "a.py->b.py"
    detail: str = ""                          # symbol name, etc.
    weight: int = 1
    claimed_by: list[str] = field(default_factory=list)   # producing steps
    required_by: list[str] = field(default_factory=list)  # consuming steps
    # IMPORT_EDGE only: every file the declaring step produces. `imports:`
    # is a step-level declaration, so any one of them satisfying it is
    # enough — see `_check_edge`.
    consumers: list[str] = field(default_factory=list)
    verdict: str = UNKNOWN
    evidence: str = ""
    # True once this postcondition has been observed broken, even if a
    # later pass found it repaired. Kept separate from `verdict`, which
    # always describes the CURRENT state — see `observe`.
    ever_violated: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id, "kind": self.kind, "subject": self.subject,
            "detail": self.detail, "weight": self.weight,
            "claimed_by": list(self.claimed_by),
            "required_by": list(self.required_by),
            "verdict": self.verdict, "evidence": self.evidence[:400],
            "ever_violated": self.ever_violated,
        }


@dataclass
class GhostFile:
    """A file the plan intends to change, and its real pre-state."""

    path: str
    pre_hash: Optional[str]                   # None = did not exist
    writers: list[str] = field(default_factory=list)
    inline: bool = False                      # bytes came from the plan
    post_hash: Optional[str] = None

    @property
    def touched(self) -> bool:
        return self.post_hash != self.pre_hash


@dataclass(frozen=True)
class Observation:
    """One append-only journal entry. Verdicts are a fold over these."""

    exp_id: str
    verdict: str
    evidence: str
    stage: str


@dataclass(frozen=True)
class Disagreement:
    """A place where the evidence and the pipeline's verdict differ."""

    kind: str                                 # kebab-case slug
    step_id: str
    detail: str


# ── The ghost ────────────────────────────────────────────────────────


class GhostPlan:
    """Declared postconditions of a plan, reconciled against the tree."""

    def __init__(self, project_root: str = ".") -> None:
        self.root = os.path.abspath(project_root)
        self.expectations: dict[str, Expectation] = {}
        self.files: dict[str, GhostFile] = {}
        self.steps: dict[str, dict] = {}      # step id -> {produces, requires}
        # Steps a checkpoint says were finished BEFORE this run started.
        # Their files are still registered — they are planned targets, and
        # dropping them would turn every later write to one into an
        # `unplanned-write` — but nothing they declare is a claim THIS run
        # makes, so no verdict about them is reported. See `_carried_note`.
        self.carried: set[str] = set()
        self.journal: list[Observation] = []
        # gate expectation id -> {path: digest} at the wave the gate first
        # went green. A gate's verdict expires when the files it was
        # exercising are rewritten under it; see `_gate_still_witnessed`.
        self._gate_witness: dict[str, dict[str, str]] = {}
        # path -> the spelling actually on disk, when it differs only by
        # case. Case-insensitive filesystems resolve the plan's spelling
        # happily, so this never fails EXISTS; it is reported separately.
        self.case_mismatches: dict[str, str] = {}
        # What the PLAN itself says each file should contain. This is not
        # a guess — the planner wrote it — so repairing a drifted file
        # from it invents nothing. Empty under intent mode, where the
        # plan deliberately supplies goals instead of bodies.
        self.plan_content: dict[str, str] = {}
        self.plan_edits: dict[str, list[tuple[str, str]]] = {}
        # Every command the plan names — verify gates and CMD bodies —
        # so the run's own test runner can be identified.
        self.declared_commands: list[str] = []
        self._lock = Lock()

    # -- construction --------------------------------------------------

    @classmethod
    def build(cls, steps: Iterable, project_root: str = ".",
              carried_step_ids: Iterable[str] = ()) -> "GhostPlan":
        """Derive expectations from a finalized plan and snapshot pre-state.

        Must be called after every plan-repair pass (blind-edit routing,
        verify repair, reclassification) and before the first step runs —
        the pre-state hashes are only meaningful if nothing has executed.

        ``carried_step_ids`` are steps a checkpoint completed in an
        EARLIER run. Their postconditions were satisfied against a tree
        this run never saw, and the pre-state below is captured after
        their work, so every one of them reads as "the step changed
        nothing". Measured 2026-08-18: a resume that executed one step
        reported ten `violated-touched` findings against four steps that
        had finished in the previous run — noise that buried the single
        real finding in the same list.
        """
        ghost = cls(project_root)
        for step in steps or ():
            ghost._add_step(step)
        ghost.carried = {sid for sid in carried_step_ids if sid in ghost.steps}
        ghost._mark_carried_inapplicable()
        ghost._capture_pre_state()
        return ghost

    def _mark_carried_inapplicable(self) -> None:
        """Retire expectations no pending step also declares.

        Expectations are interned and shared, so one declared by both a
        carried step and a pending one must still be resolved — the
        pending step is going to make that claim for real. Only the
        exclusively-carried ones are retired, which keeps the tally and
        the evidence weight honest about what this run actually checked.
        """
        if not self.carried:
            return
        live: set[str] = set()
        for sid, node in self.steps.items():
            if sid in self.carried:
                continue
            live |= node["produces"] | node["requires"]
        for sid in self.carried:
            node = self.steps.get(sid) or {}
            for exp_id in (node.get("produces", set())
                           | node.get("requires", set())):
                if exp_id in live:
                    continue
                exp = self.expectations.get(exp_id)
                if exp is not None:
                    exp.verdict = INAPPLICABLE
                    exp.evidence = "step completed in an earlier run"

    def _add_step(self, step) -> None:
        sid = getattr(step, "id", "?")
        for _cmd in (getattr(step, "verify_cmd", None),
                     getattr(step, "command", None)):
            if _cmd:
                self.declared_commands.append(_cmd)
        node = self.steps.setdefault(sid, {"produces": set(), "requires": set()})
        targets = [_norm(t) for t in
                   (getattr(step, "target_files", None) or []) if t]
        targets = [t for t in targets if _looks_like_path(t)]
        inline = {_norm(p) for p in
                  (getattr(step, "inline_code", None) or {})}
        inline |= {_norm(p) for p in
                   (getattr(step, "inline_edits", None) or {})}

        for path in targets:
            gf = self.files.setdefault(
                path, GhostFile(path=path, pre_hash=None))
            if sid not in gf.writers:
                gf.writers.append(sid)
            gf.inline = gf.inline or path in inline

            # A file whose bytes the plan supplies is one the pipeline
            # writes itself; observing that it then exists is circular.
            self._claim(node, Expectation(
                id=f"file:{path}#exists", kind=KIND_EXISTS, subject=path,
                weight=0 if path in inline else 1))
            self._claim(node, Expectation(
                id=f"file:{path}#touched", kind=KIND_TOUCHED, subject=path,
                weight=_WEIGHTS[KIND_TOUCHED]))
            if _parseable(path):
                self._claim(node, Expectation(
                    id=f"file:{path}#parses", kind=KIND_PARSES, subject=path,
                    weight=_WEIGHTS[KIND_PARSES]))
            # A manifest is a promise about the environment, not just a
            # file. The dependency list inside it does not exist yet at
            # plan time, so the node is created now and the list is read
            # when it is resolved.
            if os.path.basename(path).lower() in _MANIFESTS:
                self._claim(node, Expectation(
                    id=f"pkg:{path}#deps-installed", kind=KIND_PKG_PRESENT,
                    subject=path, weight=_WEIGHTS[KIND_PKG_PRESENT]))

        # Where the plan supplied the body itself, the written file must
        # still contain what that body declared. This is the step-drift
        # check: the planner got it right and the per-step model wrote
        # something else — common with smaller models, and invisible to
        # every other check because the file exists, parses, and may even
        # satisfy a weak gate.
        inline_code = getattr(step, "inline_code", None) or {}
        for raw_path, body in inline_code.items():
            path = _norm(raw_path)
            if not path or not (body or "").strip():
                continue
            self.plan_content[path] = body
            anchors = plan_anchors(path, body)
            if anchors:
                self._claim(node, Expectation(
                    id=f"plan:{path}#anchors", kind=KIND_PLAN_ANCHORS,
                    subject=path, detail=",".join(sorted(anchors)),
                    weight=_WEIGHTS[KIND_PLAN_ANCHORS]))
        for raw_path, pairs in (getattr(step, "inline_edits", None) or {}).items():
            self.plan_edits.setdefault(_norm(raw_path), []).extend(pairs)

        # Exports attach to the step's first target: that is the file the
        # plan format means them to describe.
        primary = targets[0] if targets else ""
        for sym in (getattr(step, "exports", None) or []):
            sym = (sym or "").strip()
            if not sym or not primary:
                continue
            self._claim(node, Expectation(
                id=f"file:{primary}#exports:{sym}", kind=KIND_EXPORTS,
                subject=primary, detail=sym,
                weight=_WEIGHTS[KIND_EXPORTS]))

        # One edge per (step, source) — NOT per target file. `imports:` is
        # declared once for the whole step, so fanning it out across every
        # target accuses the step's incidental files of failing to import
        # something they were never going to. Observed: a TEST step with
        # `target: tests/__init__.py, tests/test_game_invariants.py`
        # produced three "the import edge was never wired" findings
        # against the package marker, while the sibling test file next to
        # it imported all three symbols correctly.
        for src, syms in (getattr(step, "imports_from", None) or {}).items():
            src_n = _norm(src)
            if not src_n or not targets:
                continue
            exp = Expectation(
                id=f"edge:{src_n}->step:{sid}", kind=KIND_IMPORT_EDGE,
                subject=src_n,
                detail=",".join(s.strip() for s in (syms or []) if s),
                consumers=list(targets),
                weight=_WEIGHTS[KIND_IMPORT_EDGE])
            self._require(node, exp)

        verify = (getattr(step, "verify_cmd", None) or "").strip()
        if not verify:
            # A CMD step's command IS its acceptance criterion when the
            # command is a test suite. Observed: a step whose entire body
            # was `set SDL_VIDEODRIVER=dummy && python -m unittest -v`
            # declared no target and no verify, so it carried no
            # expectations at all and was reported as a step that
            # "asserted nothing that could have failed" — while running
            # the project's whole acceptance suite.
            command = (getattr(step, "command", None) or "").strip()
            if command:
                try:
                    from .wave_snapshots import is_suite_gate
                    if is_suite_gate(command):
                        verify = command
                except Exception:
                    pass
        if verify:
            self._claim(node, Expectation(
                id=f"gate:{sid}:{_digest(verify)}", kind=KIND_GATE_PASSED,
                subject=verify, weight=_WEIGHTS[KIND_GATE_PASSED]))

    def _intern(self, exp: Expectation) -> Expectation:
        existing = self.expectations.get(exp.id)
        if existing is not None:
            return existing
        self.expectations[exp.id] = exp
        return exp

    def _claim(self, node: dict, exp: Expectation) -> None:
        exp = self._intern(exp)
        node["produces"].add(exp.id)

    def _require(self, node: dict, exp: Expectation) -> None:
        exp = self._intern(exp)
        node["requires"].add(exp.id)

    def _capture_pre_state(self) -> None:
        """Hash every target file as it exists *before* the run."""
        for gf in self.files.values():
            content = _read(self.root, gf.path)
            gf.pre_hash = _digest(content) if content is not None else None

    # -- observation ---------------------------------------------------

    def observe(self, exp_id: str, verdict: str, evidence: str = "",
                stage: str = "") -> None:
        """Append one journal entry and fold it into the node's verdict.

        ``verdict`` always reflects the LATEST observation, because the
        question this module answers is what the run actually shipped.
        An earlier failure is preserved in ``ever_violated`` and in the
        journal rather than in the verdict.

        This was originally the other way round — ``VIOLATED`` was sticky
        — and it produced a confidently false report. Observed: a plan
        created a venv and installed into the wrong interpreter, so at
        wave 2 the declared dependency genuinely was absent and the
        shadow said so correctly; at wave 6 the agent loop's env
        self-heal reinstalled it into the project venv, and the run
        finished green with the package present. The stale ``VIOLATED``
        was still reported at the end, contradicting a run that was by
        then entirely correct. "Was broken once" and "is broken" are
        different claims, and only the second belongs in a verdict.
        """
        with self._lock:
            exp = self.expectations.get(exp_id)
            if exp is None:
                return
            self.journal.append(Observation(exp_id, verdict, evidence, stage))
            if verdict == VIOLATED:
                exp.ever_violated = True
            exp.verdict = verdict
            exp.evidence = evidence

    # -- resolution ----------------------------------------------------

    def resolve(self, step_ids: Iterable[str], *, language: str | None = None,
                gate_cmds: Iterable[str] = (), stage: str = "") -> None:
        """Check every expectation owned by *step_ids* against the tree.

        Safe to call repeatedly (per wave, then once at the end): each
        pass re-reads the files and appends fresh observations.
        """
        wanted: set[str] = set()
        for sid in step_ids:
            if sid in self.carried:
                continue      # not a claim this run made
            node = self.steps.get(sid)
            if node:
                wanted |= node["produces"] | node["requires"]
        gates = list(gate_cmds)
        cache: dict[str, Optional[str]] = {}

        def content(path: str) -> Optional[str]:
            if path not in cache:
                cache[path] = _read(self.root, path)
            return cache[path]

        for exp_id in sorted(wanted):
            exp = self.expectations.get(exp_id)
            if exp is None:
                continue
            try:
                verdict, evidence = self._check(exp, content, gates, language)
            except Exception as exc:            # never fail a run
                _logger.debug("[Ghost] check raised for %s: %s", exp_id, exc)
                continue
            self.observe(exp_id, verdict, evidence, stage)

    # The expectation kinds whose subject is a path in the tree — the
    # files a step's gate was exercising when it went green.
    _FILE_KINDS = frozenset({KIND_EXISTS, KIND_TOUCHED, KIND_PARSES,
                             KIND_EXPORTS, KIND_PLAN_ANCHORS})

    def _export_has_consumer(self, exp: Expectation) -> bool:
        """Does any step declare that it imports this symbol from here?

        A broken export contract only costs something when a step tries to
        import it — that is the `gate_integrity` shape, where one bad name
        rejected working code for a whole run. With no consumer the same
        disagreement is a naming preference the code won on its own.

        Convention-insensitive on the symbol, for the same reason
        `_export_satisfied` is: a plan shown only JavaScript examples
        writes `runHeadless` where the importer wrote `run_headless`, and
        that is one name, not two.
        """
        want = (exp.detail or "").strip()
        if not want:
            return False
        want_key = _canonical_name(want)
        for other in self.expectations.values():
            if other.kind != KIND_IMPORT_EDGE or other.subject != exp.subject:
                continue
            for sym in (other.detail or "").split(","):
                sym = sym.strip()
                if sym and (sym == want or _canonical_name(sym) == want_key):
                    return True
        return False

    def _gate_still_witnessed(self, exp: Expectation, content,
                              evidence: str) -> tuple[str, str]:
        """Does a gate that passed still describe the tree on disk?

        The ledger records that a command went green; it cannot record
        that the files it exercised were rewritten afterwards, and the
        verdict was folded as HOLDS forever regardless. Measured: gate 3.1
        of a Pac-Man run passed against a `game.py` that step 5's
        diagnosis then rewrote into a `TypeError` on every `advance()`.
        The run shipped that file while the ghost still reported the gate
        green, and `failed-but-clean` cited it as grounds to blame the
        harness.

        Resolution happens once per wave, so the first HOLDS records what
        the step's declared files hashed to at the moment it passed, and
        every later wave compares. A changed file does not make the gate
        VIOLATED — the command really did pass once, and it might well
        pass again — it makes it UNKNOWN, which is the whole point of the
        four-valued discipline: an answer about a tree that no longer
        exists is not evidence about the one that does.
        """
        sid = exp.id.split(":")[1] if exp.id.startswith("gate:") else ""
        node = self.steps.get(sid)
        if not node:
            return HOLDS, evidence
        now: dict[str, str] = {}
        for other_id in sorted(node.get("produces", set())):
            other = self.expectations.get(other_id)
            if other is None or other.kind not in self._FILE_KINDS:
                continue
            text = content(other.subject)
            if text is None:
                continue
            now[other.subject] = _digest(text)
        if not now:
            return HOLDS, evidence
        seen = self._gate_witness.get(exp.id)
        if seen is None:
            self._gate_witness[exp.id] = now
            return HOLDS, evidence
        changed = sorted(p for p, h in now.items()
                         if p in seen and seen[p] != h)
        if changed:
            listed = ", ".join(changed[:3])
            if len(changed) > 3:
                listed += f" (+{len(changed) - 3} more)"
            return UNKNOWN, (
                f"passed earlier, but {listed} changed afterwards — the "
                f"gate's verdict describes a state of the tree that no "
                f"longer exists")
        return HOLDS, evidence

    def _check(self, exp: Expectation, content, gates: list[str],
               language: str | None) -> tuple[str, str]:
        if exp.kind == KIND_GATE_PASSED:
            verdict, evidence = _check_gate(exp.subject, gates)
            if verdict == HOLDS:
                verdict, evidence = self._gate_still_witnessed(
                    exp, content, evidence)
            return verdict, evidence

        if exp.kind == KIND_IMPORT_EDGE:
            return _check_edge(exp.subject, exp.detail,
                               [(c, content(c)) for c in exp.consumers])

        if exp.kind == KIND_PKG_PRESENT:
            return _check_packages(self.root, exp.subject,
                                   content(exp.subject))

        if exp.kind == KIND_PLAN_ANCHORS:
            return _check_plan_anchors(exp.subject, exp.detail,
                                       content(exp.subject))

        path = exp.subject
        full = os.path.join(self.root, path.replace("/", os.sep))
        text = content(path)

        # A plan target is not always a file. CMD steps legitimately
        # declare `produces: venv` or `produces: src/assets`, and judging
        # a directory by `isfile` reports a target that is plainly there
        # as missing — which then drags the step's evidence weight to
        # zero and manufactures a second, equally false "asserted
        # nothing" finding on top of it.
        is_dir = os.path.isdir(full)

        if exp.kind == KIND_EXISTS:
            if text is not None or is_dir:
                # The file is usable here, so this is not a missing
                # target — but on a case-insensitive filesystem it can be
                # usable under a DIFFERENT spelling than the plan asked
                # for, and that difference is invisible until the project
                # is checked out somewhere case-sensitive. Record it as
                # its own finding rather than distorting this verdict.
                actual = _actual_spelling(full)
                if actual and actual != os.path.basename(path):
                    self.case_mismatches[path] = actual
                return HOLDS, "directory" if is_dir else ""
            if os.path.exists(full):
                return UNKNOWN, "exists but could not be read as text"
            near = _near_miss(full)
            if near:
                # A rename is deliberately NOT attempted: every other
                # file that references either spelling would have to move
                # with it, and on Windows the wrong case still imports
                # fine, so the mismatch only breaks on someone else's
                # machine. Naming the candidate makes it a one-line fix
                # for a human without risking a repo-wide edit here.
                return VIOLATED, (
                    f"planned target does not exist, but `{near}` does — "
                    f"filename mismatch (case or extension)")
            return VIOLATED, "planned target does not exist on disk"

        if is_dir:
            # Hashing or parsing a directory is not a question with an
            # answer; existence is the only claim it can settle.
            return INAPPLICABLE, "target is a directory"

        if text is None:
            return UNKNOWN, "file unreadable — no evidence either way"

        gf = self.files.get(path)
        if exp.kind == KIND_TOUCHED:
            if gf is None:
                return UNKNOWN, ""
            gf.post_hash = _digest(text)
            if gf.pre_hash is None:
                return HOLDS, "created"
            if gf.post_hash == gf.pre_hash:
                return VIOLATED, ("bytes identical to the pre-run state — "
                                  "the step changed nothing")
            return HOLDS, "modified"

        if exp.kind == KIND_PARSES:
            return _check_parses(path, text)

        if exp.kind == KIND_EXPORTS:
            return _check_exports(text, exp.detail, path, language)

        return UNKNOWN, ""

    # -- reporting -----------------------------------------------------

    def unplanned_writes(self, tracked: Iterable[str]) -> list[str]:
        """Files the run wrote that no step ever claimed as a target."""
        from .memory import _should_skip_for_context

        planned = set(self.files)
        out: list[str] = []
        for path in tracked or ():
            norm = _norm(path)
            if not norm or norm in planned:
                continue
            if _should_skip_for_context(norm):
                continue
            out.append(norm)
        return sorted(out)

    def _declares_a_file(self) -> bool:
        """Did any step name a target that is a file rather than a dir?

        A plan whose only target is ``produces: venv\\`` has declared
        nothing about its own source. Directory-ness is read from disk
        rather than from the spelling, so a target named ``Makefile``
        counts and one named ``tests`` does not.
        """
        for path in self.files:
            if not os.path.isdir(os.path.join(self.root, path)):
                return True
        return False

    def step_strength(self, step_id: str) -> int:
        """Weight of the evidence a step actually produced.

        Only ``HOLDS`` counts. A step can be strong on paper and weak in
        fact — an unreadable file resolves ``UNKNOWN`` and proves nothing,
        which is exactly the situation this number exists to expose.
        """
        node = self.steps.get(step_id)
        if not node:
            return 0
        total = 0
        for exp_id in node["produces"] | node["requires"]:
            exp = self.expectations.get(exp_id)
            if exp is not None and exp.verdict == HOLDS:
                total += exp.weight
        return total

    def declared_strength(self, step_id: str) -> int:
        """Weight of what the step CLAIMED, whatever came of checking it.

        Distinct from :meth:`step_strength`, which counts only confirmed
        evidence. "Did this step assert anything that could have failed?"
        is a question about the declaration, not about whether we managed
        to confirm it.

        Observed: a CMD step declared
        ``verify: python -c "import pygame; assert pygame.version.verstr
        .startswith('2.6')"`` — a genuinely falsifiable gate — but the
        gate never entered the ledger, so it resolved UNKNOWN, banked no
        evidence, and the step was reported as asserting nothing that
        could have failed. The step's claim was real; only our record of
        it was missing.
        """
        node = self.steps.get(step_id)
        if not node:
            return 0
        total = 0
        for exp_id in node["produces"] | node["requires"]:
            exp = self.expectations.get(exp_id)
            if exp is not None and exp.verdict != INAPPLICABLE:
                total += exp.weight
        return total

    def tally(self) -> dict[str, int]:
        counts = {HOLDS: 0, VIOLATED: 0, UNKNOWN: 0, INAPPLICABLE: 0}
        for exp in self.expectations.values():
            counts[exp.verdict] = counts.get(exp.verdict, 0) + 1
        return counts

    def disagreements(self, done_step_ids: Iterable[str], *,
                      tracked_files: Iterable[str] = (),
                      pipeline_success: bool = True) -> list[Disagreement]:
        """Places the evidence contradicts the pipeline's own verdict."""
        # A step this run did not execute cannot have disagreed with it.
        done = [sid for sid in done_step_ids if sid not in self.carried]
        out: list[Disagreement] = []

        drifted: list[str] = []
        for sid in done:
            node = self.steps.get(sid)
            if not node:
                continue
            for exp_id in sorted(node["produces"] | node["requires"]):
                exp = self.expectations.get(exp_id)
                if exp is None or exp.verdict != VIOLATED:
                    continue
                # A declared export nobody imports is a naming
                # disagreement, not a defect: the contract is broken and
                # nothing can break because of it. The pipeline already
                # draws this line — `_missing_declared_exports` fires
                # "only on an import/attribute error naming a declared
                # export" — and the noise is real: three findings in one
                # run whose artifact passed all nine external probes,
                # where the coder had written `Collectible` for `Pellet`
                # and `draw_*` for `render_game`. Enough of those bury the
                # one that matters. They collapse into a single run-level
                # note instead, the same way six `unplanned-write`
                # findings collapse into `plan-declares-no-targets`.
                if (exp.kind == KIND_EXPORTS
                        and not self._export_has_consumer(exp)):
                    drifted.append(f"{exp.subject} [{exp.detail}]")
                    continue
                out.append(Disagreement(
                    kind=f"violated-{exp.kind.lower().replace('_', '-')}",
                    step_id=sid,
                    detail=f"{exp.subject}"
                           + (f" [{exp.detail}]" if exp.detail else "")
                           + (f" — {exp.evidence}" if exp.evidence else "")))
            if self.declared_strength(sid) < MIN_STEP_STRENGTH:
                out.append(Disagreement(
                    kind="no-checkable-claim", step_id=sid,
                    detail=("step reported done but declared nothing that "
                            "could have failed — no target, no gate, and "
                            "any file it names is one the plan supplied "
                            "the contents of")))

        # "No step declared this file" says something quite different
        # depending on whether the plan declared any file at all. Against
        # a plan with targets it names the one file that slipped past
        # them; against a plan with none it is true of everything the run
        # wrote, and repeating it per file buries the fact that matters —
        # that the whole file layer of this check was never armed.
        _writes = self.unplanned_writes(tracked_files)
        if _writes and not self._declares_a_file():
            shown = ", ".join(_writes[:6])
            more = f", +{len(_writes) - 6} more" if len(_writes) > 6 else ""
            out.append(Disagreement(
                kind="plan-declares-no-targets", step_id="-",
                detail=(f"no step declared a file target, so nothing the "
                        f"run wrote could be reconciled against the plan — "
                        f"{len(_writes)} file(s) were written unchecked "
                        f"({shown}{more}). Exports, anchors and content "
                        f"regressions all went unexamined.")))
        else:
            for path in _writes:
                out.append(Disagreement(
                    kind="unplanned-write", step_id="-",
                    detail=f"{path} was written but no step declared it"))

        # Test files the run's own acceptance command will never collect.
        # Deliberately spans planned targets AND untracked writes: the
        # four modules that exposed this were written by the agent loop
        # and declared by no step, so a per-step check would have missed
        # every one of them.
        _runner = declared_runner(self.declared_commands)
        _candidates = set(self.files) | {
            _norm(p) for p in (tracked_files or ())}
        for path, reason in uncollected_test_files(
                self.root, _candidates, _runner, self.declared_commands):
            out.append(Disagreement(
                kind="tests-never-collected", step_id="-",
                detail=f"{path}: {reason}"))

        # Long assertion loops that stop simulating partway through. Same
        # candidate set as above, and for the same reason: the suites that
        # exposed this were as often unplanned writes as planned targets.
        for path, reason in degenerate_long_runs(self.root, _candidates):
            out.append(Disagreement(
                kind="degenerate-long-run", step_id="-",
                detail=f"{path}: {reason}"))

        # And the sibling defect: the loop kept running, but the input it
        # was varying never reached anything that reads it.
        for path, reason in ignored_varied_inputs(self.root, _candidates):
            out.append(Disagreement(
                kind="varied-input-ignored", step_id="-",
                detail=f"{path}: {reason}"))

        # The third in the family: the loop ran, the input was read, work
        # happened every iteration — and nothing ever asserted that any of
        # it accomplished anything.
        for path, reason in unprogressed_long_runs(self.root, _candidates):
            out.append(Disagreement(
                kind="unprogressed-long-run", step_id="-",
                detail=f"{path}: {reason}"))

        if drifted:
            shown = ", ".join(sorted(drifted)[:4])
            if len(drifted) > 4:
                shown += f" (+{len(drifted) - 4} more)"
            out.append(Disagreement(
                kind="export-drift", step_id="-",
                detail=(f"the plan declared {len(drifted)} export(s) the "
                        f"code renamed, and no step imports any of them, "
                        f"so nothing can break: {shown}")))

        for planned, actual in sorted(self.case_mismatches.items()):
            out.append(Disagreement(
                kind="filename-case-mismatch", step_id="-",
                detail=(f"the plan targets `{planned}` but the file on "
                        f"disk is `{actual}` — this resolves on a "
                        f"case-insensitive filesystem and breaks on a "
                        f"case-sensitive one")))

        if not pipeline_success:
            counts = self.tally()
            # Every other kind here proves SHAPE — the file exists, parses,
            # declares the right names, matches the plan's body. None of
            # them can see behaviour, so on their own they are no basis at
            # all for telling someone their failure is the harness's fault.
            #
            # Observed: a 20B run produced eight structurally perfect files
            # — 41 postconditions green, every plan-declared anchor present
            # — whose suite failed with "Ghost out of map bounds at (5, 7)".
            # A real logic bug, a correctly-failed run, and this check
            # confidently blamed the harness. A confirmed-green acceptance
            # gate is the only evidence that speaks to behaviour, so
            # without one the honest answer is silence.
            #
            # And a green gate is only evidence about the step it belongs
            # to. Asking "is there ANY green gate in this run" answers a
            # different question from "is there green evidence about the
            # thing that FAILED", and the two come apart the moment a plan
            # has more than one step. Measured twice, both blaming the
            # harness for the model: a run whose gates 2.1/3.1/4.1 were
            # green and whose step 5 failed having never recorded a gate
            # at all, and a run with four green gates whose step 6 failed
            # `verify` three times over. So when the run halted partway,
            # only a green gate on a step that did NOT complete counts.
            #
            # When every step DID complete and the run still failed, the
            # original question is the right one — that is the shape this
            # finding was written for.
            done_set = set(done)
            incomplete = [sid for sid in self.steps if sid not in done_set]
            if incomplete:
                behavioural = [
                    e for sid in incomplete
                    for exp_id in (self.steps.get(sid) or {}).get(
                        "produces", set())
                    | (self.steps.get(sid) or {}).get("requires", set())
                    for e in (self.expectations.get(exp_id),)
                    if e is not None and e.kind == KIND_GATE_PASSED
                    and e.verdict == HOLDS
                ]
            else:
                behavioural = [
                    e for e in self.expectations.values()
                    if e.kind == KIND_GATE_PASSED and e.verdict == HOLDS
                ]
            if counts[VIOLATED] == 0 and behavioural:
                out.append(Disagreement(
                    kind="failed-but-clean", step_id="-",
                    detail=(f"run marked FAILED while all {counts[HOLDS]} "
                            f"resolved postcondition(s) hold, including "
                            f"{len(behavioural)} acceptance gate(s) that "
                            f"went green — suspect the harness before the "
                            f"model")))
        return out

    def report(self, done_step_ids: Iterable[str], *,
               tracked_files: Iterable[str] = (),
               pipeline_success: bool = True) -> list[Disagreement]:
        """Log the shadow summary and any disagreements. Returns them."""
        counts = self.tally()
        gaps = self.disagreements(done_step_ids,
                                  tracked_files=tracked_files,
                                  pipeline_success=pipeline_success)
        strength = sum(self.step_strength(s) for s in self.steps)
        _logger.info(
            "[Ghost] shadow: %d expectation(s) over %d step(s) — "
            "%d hold, %d violated, %d unknown; evidence weight %d; "
            "%d disagreement(s)",
            len(self.expectations), len(self.steps), counts[HOLDS],
            counts[VIOLATED], counts[UNKNOWN], strength, len(gaps))
        # Repaired-in-flight is not a defect, but it is worth seeing: it
        # is the trace of a self-heal or fix round doing its job, and a
        # postcondition that keeps needing repair is a plan smell.
        repaired = [e for e in self.expectations.values()
                    if e.ever_violated and e.verdict == HOLDS]
        if repaired:
            _logger.info(
                "[Ghost] %d postcondition(s) were broken mid-run and are "
                "green now (repaired in flight): %s", len(repaired),
                ", ".join(f"{e.kind}:{e.subject}" for e in repaired[:5]))
        for gap in gaps:
            _logger.warning("[Ghost] %s (step %s): %s",
                            gap.kind, gap.step_id, gap.detail)
        return gaps

    def to_dict(self) -> dict:
        """Serialize for checkpoints, benchmarks, and offline comparison."""
        return {
            "root": self.root,
            "expectations": [e.to_dict()
                             for e in self.expectations.values()],
            "files": [{"path": f.path, "pre_hash": f.pre_hash,
                       "post_hash": f.post_hash, "inline": f.inline,
                       "writers": list(f.writers)}
                      for f in self.files.values()],
            "steps": {s: {"produces": sorted(n["produces"]),
                          "requires": sorted(n["requires"])}
                      for s, n in self.steps.items()},
            "tally": self.tally(),
        }


# ── Individual checks ────────────────────────────────────────────────


_PARSEABLE_EXTS = (".py", ".json")


def _parseable(path: str) -> bool:
    return path.lower().endswith(_PARSEABLE_EXTS)


def _check_parses(path: str, text: str) -> tuple[str, str]:
    """Syntax-only check for the formats we can judge without a toolchain."""
    low = path.lower()
    try:
        if low.endswith(".py"):
            ast.parse(text)
        elif low.endswith(".json"):
            json.loads(text)
        else:
            return INAPPLICABLE, ""
    except SyntaxError as exc:
        return VIOLATED, f"SyntaxError line {exc.lineno}: {exc.msg}"
    except (json.JSONDecodeError, ValueError) as exc:
        return VIOLATED, f"invalid JSON: {exc}"
    except (RecursionError, MemoryError):
        return UNKNOWN, "parser gave up"
    return HOLDS, ""


def _python_class_members(text: str) -> set[str]:
    """Names defined one level inside a class body.

    The language backend reports module-level names only, so a constant
    living on the class it belongs to reads as missing. Observed:
    ``class Map:`` declares ``TILE_SIZE = 32`` and `game.py` uses
    ``Map.TILE_SIZE`` — the plan's ``exports: TILE_SIZE`` was correct and
    was reported as a broken promise. Both the bare and qualified
    spellings are returned, since a plan may write either.
    """
    names: set[str] = set()
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return names
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        for item in node.body:
            targets: list[str] = []
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                targets = [item.name]
            elif isinstance(item, ast.AnnAssign) and isinstance(
                    item.target, ast.Name):
                targets = [item.target.id]
            elif isinstance(item, ast.Assign):
                targets = [t.id for t in item.targets
                           if isinstance(t, ast.Name)]
            for name in targets:
                names.add(name)
                names.add(f"{node.name}.{name}")
    return names


def _export_evidence(module_level: Iterable[str],
                     members: Iterable[str] = (),
                     limit: int = 12) -> str:
    """The 'file exports …' evidence for a violated export claim.

    Plain ``sorted(actual)[:8]`` over the merged set was actively
    misleading. ``_python_class_members`` contributes both ``member`` and
    ``Class.member`` spellings, and an alphabetical head could be spent
    entirely on those: for the ``entities.py`` that really declared
    ``Player``, ``GridMover``, ``add_direction`` and 11 other module-level
    names, the evidence showed eight entries, three of them
    ``Ghost.__init__``-style, and none of the names a reader would look
    for. A correct finding read like a false positive.

    So the file's own module-level names come first — a plan's ``exports:``
    almost always names one — class members fill only the room left over,
    and the elided count is always stated, because a silent truncation is
    what made the old line read as a complete list.
    """
    module_level = sorted(set(module_level))
    members = sorted(set(members) - set(module_level))
    shown = module_level[:limit]
    if len(shown) < limit:
        shown += members[:limit - len(shown)]
    omitted = (len(module_level) + len(members)) - len(shown)
    listing = ", ".join(shown) if shown else "nothing"
    if omitted > 0:
        listing += f" (+{omitted} more)"
    return listing


def _check_exports(text: str, symbol: str, path: str,
                   language: str | None) -> tuple[str, str]:
    """Is *symbol* actually exported by the file's real contents?

    An extractor that cannot run yields ``UNKNOWN``: per
    ``plan_graph._export_satisfied``'s history, a confident claim built on
    a missing extractor is wrong far more often than it is right.
    """
    try:
        from ..language_backend import get_backend
        backend = get_backend(_language_for(path, language))
        exported = set(backend.extract_exports(text) or [])
    except Exception:
        return UNKNOWN, "no export extractor for this file"
    members: set[str] = set()
    if path.lower().endswith(".py"):
        members = _python_class_members(text)
    # The verdict is decided on the union, exactly as before; only the
    # evidence distinguishes the two sources, so the reader sees the
    # file's own top-level names before one class's method list.
    actual = exported | members
    if not actual:
        return UNKNOWN, "extractor found no exports at all — inconclusive"
    if _export_satisfied(symbol, actual):
        return HOLDS, ""
    return VIOLATED, (f"declared export not found; file exports "
                      f"{_export_evidence(exported, members)}")


_EXT_LANG = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript",
    ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".go": "go", ".rs": "rust", ".java": "java", ".rb": "ruby",
}


def _language_for(path: str, fallback: str | None) -> str | None:
    return _EXT_LANG.get(os.path.splitext(path)[1].lower()) or fallback


def _is_unbound_use(text: str, symbol: str) -> bool:
    """Does *text* read *symbol* without anything binding it?

    That combination is a genuine defect — a NameError the moment the
    line runs — and it is the only condition under which writing an
    import is a repair rather than decoration.

    Conservative in the safe direction: any binding anywhere in the
    module (import, assignment, def, class, parameter, comprehension
    target) counts, even in an inner scope. A false "it is bound"
    declines a repair; a false "it is unbound" would write a duplicate
    or shadowing import, which is the more damaging error.
    """
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return False

    used = False
    bound = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load) and node.id == symbol:
                used = True
            elif node.id == symbol:
                bound = True
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if (alias.asname or alias.name.split(".")[0]) == symbol:
                    bound = True
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                               ast.ClassDef)):
            if node.name == symbol:
                bound = True
        elif isinstance(node, ast.arg) and node.arg == symbol:
            bound = True
        elif isinstance(node, ast.Attribute) and node.attr == symbol:
            # `mod.Symbol` is a use that an import of `mod` already
            # satisfies — not evidence that `Symbol` itself is needed.
            pass
    return used and not bound


def _check_edge(src: str, symbols: str,
                consumers: list[tuple[str, Optional[str]]]) -> tuple[str, str]:
    """Does ANY file the step produces reference the module it imports?

    Deliberately weak twice over. It looks for the producer's module stem
    *or* any declared symbol anywhere in the consumer — barrel
    re-exports, dynamic imports and aliasing all make a stricter test
    wrong. And one satisfying file settles it for the whole step, because
    `imports:` is a step-level declaration: a step that produces a
    package marker and a test module has wired the import if the test
    module imports it.

    Only when every one of the step's files is readable and none of them
    mentions the module is the edge called unwired.
    """
    readable = [(p, t) for p, t in consumers if t is not None]
    if not readable:
        return UNKNOWN, "no consumer file could be read"
    stem = os.path.basename(module_key(src))
    declared = [s.strip() for s in (symbols or "").split(",") if s.strip()]
    needles = [stem] + declared

    # Checked BEFORE the mention test, because it is strictly stronger.
    # A file that reads a declared symbol while nothing binds it is a
    # NameError waiting to run — yet it "mentions" the symbol, so the
    # mention test below would have called it correctly wired.
    for path, text in readable:
        if not path.endswith(".py"):
            continue
        unbound = [s for s in declared if _is_unbound_use(text, s)]
        if unbound:
            return VIOLATED, (
                f"{path} uses {', '.join(unbound)} but nothing imports or "
                f"defines it — the import edge is missing and the file "
                f"raises NameError when that line runs")

    for path, text in readable:
        for needle in needles:
            if needle and re.search(rf"\b{re.escape(needle)}\b", text):
                return HOLDS, f"wired in {path}"
    if len(readable) < len(consumers):
        return UNKNOWN, "some of the step's files could not be read"
    return VIOLATED, (
        f"none of the step's file(s) ({', '.join(p for p, _ in readable)}) "
        f"mentions `{stem}` or any declared symbol — the import edge was "
        f"never wired")


# ── Declared dependencies vs. the environment that will run the app ──
#
# WHY THIS EXISTS
# A plan step wrote `python -m venv venv && python -m pip install pygame`.
# `venv` was created but never activated, so the second `python` was still
# the pipeline's interpreter: pygame landed in the pipeline's environment
# and never in the project's. Every gate then passed — the game modules
# were cleanly headless and imported no pygame — and the suite went fully
# green. Only `main.py`, which imports pygame inside `main()`, ever needed
# it, and it ran under the project venv where it was absent. Both the
# classic and the agent-loop arm of a benchmark shipped an application
# that could not start, with every check green.
#
# The claim checked here is the one nothing else made: every dependency
# the plan's own manifest declares must be present in the environment the
# app will actually run in. Purely a filesystem question — no subprocess,
# no import — so the shadow stays read-only.


def _pep503(name: str) -> str:
    return re.sub(r"[-_.]+", "-", (name or "").strip()).lower()


def _requirements_names(text: str) -> list[str]:
    """Distribution names from a requirements.txt, specifiers stripped."""
    names: list[str] = []
    for raw in (text or "").splitlines():
        line = raw.split("#", 1)[0].strip()
        # Options (-r, -e, --index-url) and direct URLs name no
        # distribution we can look up by directory.
        if not line or line.startswith("-") or "://" in line:
            continue
        name = re.split(r"[\[<>=!~;\s]", line, maxsplit=1)[0].strip()
        if name:
            names.append(name)
    return names


def _site_packages(venv_bin: str) -> Optional[str]:
    """The site-packages dir belonging to *venv_bin*'s environment."""
    root = os.path.dirname(venv_bin)
    win = os.path.join(root, "Lib", "site-packages")
    if os.path.isdir(win):
        return win
    lib = os.path.join(root, "lib")
    if os.path.isdir(lib):
        try:
            for entry in sorted(os.listdir(lib)):
                cand = os.path.join(lib, entry, "site-packages")
                if os.path.isdir(cand):
                    return cand
        except OSError:
            return None
    return None


def _installed_names(site_dir: str) -> Optional[set[str]]:
    """Every distribution and top-level module name under *site_dir*.

    Both are collected because the two vocabularies differ: a dependency
    is declared by distribution name (``beautifulsoup4``) and imported by
    module name (``bs4``). Matching either is enough to say it is there —
    the question being asked is presence, not spelling.

    Returns ``None`` only when the directory cannot be read. An empty set
    is a real answer: a readable site-packages with nothing in it means
    the dependencies are genuinely absent, which is precisely the state
    this check exists to catch.
    """
    found: set[str] = set()
    try:
        entries = os.listdir(site_dir)
    except OSError:
        return None
    for entry in entries:
        if entry.endswith((".dist-info", ".egg-info")):
            stem = entry.rsplit(".", 1)[0]
            found.add(_pep503(stem.rsplit("-", 1)[0]))
            continue
        if entry.endswith(".py"):
            found.add(_pep503(entry[:-3]))
            continue
        if not entry.startswith("_") and "." not in entry:
            found.add(_pep503(entry))
    return found


def _check_packages(root: str, manifest: str,
                    text: Optional[str]) -> tuple[str, str]:
    """Are the manifest's declared dependencies in the app's environment?

    ``UNKNOWN`` whenever the environment cannot be identified — no
    project venv, no readable site-packages, an unsupported manifest.
    Absence of an environment is not absence of a dependency, and this
    check must never accuse a project that simply runs on the ambient
    interpreter.
    """
    if text is None:
        return UNKNOWN, "manifest unreadable"
    base = os.path.basename(manifest).lower()

    if base == "package.json":
        try:
            deps = list((json.loads(text).get("dependencies") or {}).keys())
        except (ValueError, AttributeError):
            return UNKNOWN, "manifest is not readable JSON"
        if not deps:
            return INAPPLICABLE, "no runtime dependencies declared"
        # Node resolves from the manifest's OWN directory, so that is the
        # environment this manifest describes. Looking in the repo root
        # instead is wrong for every multi-root layout: measured
        # 2026-08-19, `backend/node_modules` held all 101 packages and
        # `frontend/node_modules` all 93, yet both manifests were judged
        # against the root and reported VIOLATED — after which the healer
        # installed both dependency sets at the top level, creating a
        # `package.json`, a lockfile and a 107-package `node_modules` that
        # belong to no project in the repo.
        pkg_dir = os.path.join(root, os.path.dirname(manifest))
        node_modules = os.path.join(pkg_dir, "node_modules")
        if not os.path.isdir(node_modules):
            return UNKNOWN, "no node_modules — cannot tell what is installed"
        missing = [d for d in deps
                   if not os.path.exists(os.path.join(node_modules, *d.split("/")))]
        if missing:
            return VIOLATED, (
                f"declared but not installed in node_modules: "
                f"{', '.join(sorted(missing))}")
        return HOLDS, f"{len(deps)} dependency(ies) present"

    names = _requirements_names(text)
    if not names:
        return INAPPLICABLE, "no dependencies declared"

    try:
        from ..executor import Executor
        venv_bin = Executor._venv_bin_dir(root)
    except Exception:
        return UNKNOWN, "could not locate the project interpreter"
    if not venv_bin:
        # No project venv: the app runs on whatever interpreter is
        # ambient, which this check cannot inspect from disk.
        return UNKNOWN, "no project venv — app runs on the ambient interpreter"

    site_dir = _site_packages(venv_bin)
    if not site_dir:
        return UNKNOWN, f"no site-packages under {venv_bin}"
    installed = _installed_names(site_dir)
    if installed is None:
        return UNKNOWN, f"could not read {site_dir}"

    missing = [n for n in names if _pep503(n) not in installed]
    if missing:
        return VIOLATED, (
            f"declared in {manifest} but absent from the environment the "
            f"app runs in ({site_dir}): {', '.join(sorted(missing))} — "
            f"gates can still pass if no tested module imports them")
    return HOLDS, f"{len(names)} dependency(ies) present in {site_dir}"


def _canon_cmd(cmd: str) -> str:
    """Whitespace-free form of a command, for identity comparison only.

    The pipeline rewrites a gate's spacing before running it, so the
    plan's string and the ledger's string are routinely different bytes
    for the same command. Observed: the plan declared
    ``set SDL_VIDEODRIVER=dummy && python -m unittest -v`` while the
    ledger recorded ``set SDL_VIDEODRIVER=dummy&& python -m unittest -v``
    — the space is deliberately removed, because on Windows cmd.exe
    ``set VAR=dummy `` assigns the trailing space into the variable and
    breaks SDL. Comparing raw strings called a gate that had just passed
    "never passed". Two commands that differ only in whitespace are the
    same command for this purpose.
    """
    return re.sub(r"\s+", "", cmd or "")


# ── Plan-declared anchors: did the step build what the plan said? ────
#
# An "anchor" is a name the PLAN's own body for a file declares — a CSS
# class, a Python def/class, a JS export. Checking that the written file
# still contains them catches per-step drift: the planner specified the
# right thing and the model that executed the step produced something
# else. Nothing else in the pipeline notices, because the file exists,
# parses, and often satisfies a gate that never names the missing piece.
#
# Only structural names are used, never whole bodies. The claim being
# made is "the plan said this file declares `.site-header` and it does
# not", which is checkable and true or false — not "the styling is
# wrong", which is a judgement.

_CSS_SELECTOR_RE = re.compile(r"(?:^|[\s,{}>+~])([.#][A-Za-z_][\w-]*)")
_JS_EXPORT_RE = re.compile(
    r"export\s+(?:default\s+)?(?:async\s+)?"
    r"(?:function|class|const|let|var)\s+([A-Za-z_$][\w$]*)")
_CSS_SUFFIXES = (".css", ".scss", ".sass", ".less", ".styl")


def plan_anchors(path: str, body: str) -> set[str]:
    """Structural names the plan's own body for *path* declares."""
    low = path.lower()
    if low.endswith(_CSS_SUFFIXES):
        return {m.group(1) for m in _CSS_SELECTOR_RE.finditer(body)}
    if low.endswith(".py"):
        names: set[str] = set()
        try:
            tree = ast.parse(body)
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            return names
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.Assign):
                names.update(t.id for t in node.targets
                             if isinstance(t, ast.Name) and t.id.isupper())
        return names
    if low.endswith((".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs")):
        return {m.group(1) for m in _JS_EXPORT_RE.finditer(body)}
    return set()


def _check_plan_anchors(path: str, declared: str,
                        text: Optional[str]) -> tuple[str, str]:
    """Does the written file still declare what the plan's body did?"""
    if text is None:
        return UNKNOWN, "file unreadable"
    wanted = [a for a in (declared or "").split(",") if a]
    if not wanted:
        return INAPPLICABLE, "the plan body declared no structural names"
    missing = []
    for anchor in wanted:
        # A CSS selector carries its own sigil; identifiers get word
        # boundaries so `Board` does not match `Dashboard`.
        pattern = (re.escape(anchor) if anchor[0] in ".#"
                   else rf"\b{re.escape(anchor)}\b")
        if not re.search(pattern, text):
            missing.append(anchor)
    if not missing:
        return HOLDS, f"all {len(wanted)} plan-declared name(s) present"
    return VIOLATED, (
        f"the step drifted from the plan — the plan's own body for this "
        f"file declares {', '.join(sorted(missing))}, and the written "
        f"file does not")


# ── Test files the declared runner will never collect ────────────────
#
# WHY THIS EXISTS
# An agent loop wrote four test modules — 18KB across test_player.py,
# test_main.py, test_ghost.py and test_game_map.py — in pytest style:
# module-level `def test_x(Player, tmp_path)` with fixtures. The
# project's own acceptance command was `python -m unittest -v`, which
# collects only TestCase subclasses, so all four contributed exactly
# zero tests. `python -m unittest` reported 2 tests and passed; pytest
# on the same directory reported 22. Twenty tests were invisible to the
# command the task was graded on, the files imported cleanly so nothing
# errored, and every check in the pipeline stayed green.
#
# Collection is decided statically here rather than by running anything:
# the rules are simple enough to read off the AST, and this module does
# not execute commands.

_TEST_NAME_RE = re.compile(r"(^|[/_.])test[_s]?\d*\.py$|conftest\.py$",
                           re.IGNORECASE)

_RUNNERS = (
    ("unittest", re.compile(r"\bpython[0-9.]*\s+-m\s+unittest\b|\bunittest\b")),
    ("pytest", re.compile(r"\bpytest\b")),
)


def is_python_test_file(path: str) -> bool:
    base = os.path.basename(path.replace("\\", "/"))
    if not base.endswith(".py") or base == "conftest.py":
        return False
    return base.startswith("test_") or base.endswith("_test.py")


def declared_runner(commands: Iterable[str]) -> Optional[str]:
    """Which Python test runner the plan's own commands name.

    ``pytest`` wins a tie: it collects everything unittest does plus
    module-level functions, so a project running both is only in trouble
    when the unittest-only command is the acceptance gate.
    """
    found: set[str] = set()
    for cmd in commands:
        for name, pattern in _RUNNERS:
            if pattern.search(cmd or ""):
                found.add(name)
    if "pytest" in found:
        return "pytest"
    if "unittest" in found:
        return "unittest"
    return None


def _python_test_counts(text: str) -> tuple[int, int]:
    """``(unittest_visible, pytest_only)`` test counts for one module."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return -1, -1                      # unparseable: no opinion
    unittest_visible = 0
    pytest_only = 0
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            bases = []
            for b in node.bases:
                if isinstance(b, ast.Attribute):
                    bases.append(b.attr)
                elif isinstance(b, ast.Name):
                    bases.append(b.id)
            if any(b.endswith("TestCase") for b in bases):
                unittest_visible += sum(
                    1 for item in node.body
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and item.name.startswith("test"))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test"):
                pytest_only += 1
    return unittest_visible, pytest_only


# ── Long runs that stop running ──────────────────────────────────────
#
# The blind spot this closes: a suite can satisfy "run >= 2000 frames and
# assert the invariants hold" while simulating fifty. `Game.update` opens
# with `if self.state is not PLAYING: return`, so once the ghosts catch a
# parked player every later iteration is a no-op — and the invariant
# assertions keep passing, against a frozen state, all the way to 2000.
# Every gate goes green and no other check in the pipeline can see it.
#
# Detected structurally, from the two halves that have to be true at once:
# the advance method early-returns on a state guard, and a long assertion
# loop calls it without ever pinning that attribute to a working value.
# Both come
# from the project's own source, so nothing here is a hardcoded guess
# about what a "game" or a "frame" is.

_MIN_LONG_RUN = 200            # below this, a loop is not claiming endurance

# `unprogressed-long-run` uses a higher bar than `degenerate-long-run`,
# because it fires on far more loops: it does not require the object to be
# able to stop, only for the assertions to be blind to whether it did.
# At 200 it flagged a *state-vocabulary* test — 200 frames sampling
# `state` into a set, then asserting the set's members are legal — from a
# run whose artifact passed all nine external probes. That test is not
# claiming endurance and asserting progress is not its job. Every loop
# measured as genuinely unprotected was 600 or more.
_MIN_ENDURANCE_RUN = 500


@dataclass(frozen=True)
class _EarlyReturnGuard:
    """Which values of an attribute make a method return without working.

    A guard splits an attribute's values in two — those it bails on and
    those it works on — and it may name either side. ``if self.mode is
    not READY: return`` names the value it works on; ``if self.mode in
    (FAILED, DONE): return`` names the ones it bails on. Both spellings
    are ordinary, and reading only the first left a real method looking
    unguarded, which is how a 700-iteration loop that did work in 17 of
    them went unreported.

    So the names are kept on the side they were written, and callers ask
    :meth:`proceeds` / :meth:`halts` instead of testing set membership,
    which is only correct for one of the two spellings.
    """

    attr: str
    names: frozenset      # the values the guard condition actually names
    halting: bool         # True when `names` are the ones it returns on

    def proceeds(self, name: str) -> bool:
        """Does *name* let the method get past the guard and do work?"""
        return (name not in self.names) if self.halting else (
            name in self.names)

    def halts(self, name: str) -> bool:
        """Does *name* make the method return immediately?"""
        return (name in self.names) if self.halting else (
            name not in self.names)

    def all_proceed(self, names) -> bool:
        return bool(names) and all(self.proceeds(n) for n in names)

    def describe_exit(self) -> str:
        """How the finding says the attribute crossed into halting."""
        listed = "/".join(sorted(self.names))
        return (f"reaches {listed}" if self.halting else f"leaves {listed}")


def _state_guard(fn) -> Optional[_EarlyReturnGuard]:
    """The state guard *fn* opens with, if it opens with one.

    Only the unambiguous spellings are read — ``is not X``, ``!= X``,
    ``not in (X, Y)``, their negated forms, and the mirror-image
    spellings that name the terminal states instead (``== X``, ``in (X,
    Y)``). Anything cleverer yields ``None``, because a guard this check
    misreads would accuse a test that is doing nothing wrong.

    The whole guard prologue is scanned, not just the first statement: a
    real ``update()`` opened with ``if not math.isfinite(dt): raise`` and
    put the state guard second, which a ``body[0]`` reading missed
    entirely.
    """
    body = [s for s in fn.body
            if not (isinstance(s, ast.Expr)
                    and isinstance(s.value, ast.Constant)
                    and isinstance(s.value.value, str))]
    for stmt in body:
        # The prologue is guards only; the first real work ends it.
        if not isinstance(stmt, ast.If) or stmt.orelse or not stmt.body:
            break
        leave = stmt.body[-1]
        if isinstance(leave, ast.Raise):
            continue                     # validation guard, keep looking
        # The body may do a little work before leaving — one real guard
        # ran `self.assert_invariants()` first, which is exactly why its
        # frozen frames kept passing — but it must still leave.
        if not isinstance(leave, ast.Return) or leave.value is not None:
            break
        guard = _parse_state_guard(stmt.test)
        if guard is not None:
            return guard
    return None


def _state_value_name(node) -> Optional[str]:
    """The state value a node names, however the project spells it.

    A state vocabulary is as often a set of plain strings as a set of
    module-level constants, and reading only `Name`/`Attribute` left
    ``if self._state in ("win", "game_over"): return`` unparsed — which
    found no guard, which made the suite no candidate, which disarmed
    `degenerate-long-run` entirely for every project that spells its
    states as strings.

    Measured 2026-08-17: a glm-5.2 run shipped a game whose entities never
    moved at dt below ~0.2, under a 20000-iteration test loop that passed
    because a frozen board violates no invariant. The task prompt for it
    *mandates* string states (``state -> "start" | "playing" | "win" |
    "game_over"``), so the check could never have fired on that task.

    Only `str` constants count. A state is a value from a named
    vocabulary; numbers and booleans are not, and admitting them would
    read `if self.done is True: return` as a state guard.
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _parse_state_guard(test) -> Optional[_EarlyReturnGuard]:
    """The state fact a bare-return guard condition expresses.

    An ``or`` is enough on its own: ``if self.state != PLAYING or dt ==
    0.0: return`` leaves early whenever the state leaves PLAYING, whatever
    the other disjunct says. An ``and`` is not, since the guard may then
    decline to fire on a halting value, so it is left unread.
    """
    if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.Or):
        for value in test.values:
            guard = _parse_state_guard(value)
            if guard is not None:
                return guard
        return None
    negated = False
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        test, negated = test.operand, True
    if not isinstance(test, ast.Compare) or len(test.ops) != 1:
        return None
    if not isinstance(test.left, ast.Attribute):
        return None
    attr = test.left.attr
    op, right = test.ops[0], test.comparators[0]

    def _names(node) -> set[str]:
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            out: set[str] = set()
            for elt in node.elts:
                got = _state_value_name(elt)
                if got is None:
                    return set()
                out.add(got)
            return out
        one = _state_value_name(node)
        return {one} if one is not None else set()

    named = _names(right)
    if not named:
        return None
    # Only equality and membership say anything unambiguous about which
    # values a guard fires on; an ordering comparison does not.
    if not isinstance(op, (ast.Is, ast.IsNot, ast.Eq, ast.NotEq,
                           ast.In, ast.NotIn)):
        return None
    # The method returns when the condition is true. So `is not X` names
    # the value it works on, and `is X` names one it returns on.
    # Negation swaps which side that is.
    names_the_working_value = isinstance(op, (ast.IsNot, ast.NotEq,
                                              ast.NotIn))
    if negated:
        names_the_working_value = not names_the_working_value
    return _EarlyReturnGuard(attr=attr, names=frozenset(named),
                             halting=not names_the_working_value)


def guarded_advance_methods(text: str) -> dict[str, _EarlyReturnGuard]:
    """``"Class.method" -> _EarlyReturnGuard`` for every guarded advancer.

    Keyed by class, not by bare method name, because the name alone does
    not identify the code a call reaches. `Game.update` is state-guarded
    while `Player.update` and `Ghost.update` — same name, same project —
    are not, and a test driving the player and ghost directly skips no
    frames at all. Matching on `update` reported it as if it did.
    """
    out: dict[str, _EarlyReturnGuard] = {}
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return out
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for item in node.body:
            if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            guard = _state_guard(item)
            if guard is None:
                continue
            out[f"{node.name}.{item.name}"] = guard
    return out


def _class_bindings(tree) -> dict[str, str]:
    """``receiver -> class`` for every ``x = Class(...)`` in a module.

    Covers the two spellings tests actually use: a local ``game =
    Game(seed)`` and a fixture ``self.game = Game()`` set up in `setUp`.
    A receiver bound to two different classes is dropped — an unresolved
    receiver must not be guessed at.
    """
    out: dict[str, str] = {}
    conflicted: set[str] = set()

    def _key(target) -> Optional[str]:
        if isinstance(target, ast.Name):
            return target.id
        if (isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)):
            return f"{target.value.id}.{target.attr}"
        return None

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(
                node.value, ast.Call):
            continue
        func = node.value.func
        cls = (func.id if isinstance(func, ast.Name)
               else func.attr if isinstance(func, ast.Attribute) else None)
        if cls is None or not cls[:1].isupper():
            continue
        for target in node.targets:
            key = _key(target)
            if key is None:
                continue
            if out.get(key, cls) != cls:
                conflicted.add(key)
            out[key] = cls
    for key in conflicted:
        out.pop(key, None)
    return out


def _binding_key(node) -> Optional[str]:
    """The binding key an expression names, if it names one at all.

    The same two spellings :func:`_class_bindings` records: a bare
    ``game`` and a fixture's ``self.game``.
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return f"{node.value.id}.{node.attr}"
    return None


def _receiver_key(func: ast.Attribute) -> Optional[str]:
    """The binding key for the object a method call is made on."""
    return _binding_key(func.value)


# How far to follow a test's own helpers looking for the advance call.
# One level is what the artifacts need; the limit exists so a mutually
# recursive pair cannot walk forever.
_MAX_HELPER_DEPTH = 3


def _local_functions(tree) -> dict[str, ast.AST]:
    """``name -> def`` for the module's own functions and methods.

    A name defined twice is dropped rather than guessed at, for the same
    reason :func:`_class_bindings` drops a conflicted receiver.
    """
    out: dict[str, ast.AST] = {}
    seen_twice: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in out:
                seen_twice.add(node.name)
            out[node.name] = node
    for name in seen_twice:
        out.pop(name, None)
    return out


def _called_helper(func) -> Optional[str]:
    """The local function a call names — ``helper()`` or ``self.helper()``."""
    if isinstance(func, ast.Name):
        return func.id
    if (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
            and func.value.id == "self"):
        return func.attr
    return None


def _reached_advancers(node, bindings: dict[str, str], advancers: dict,
                       helpers: dict[str, ast.AST], *,
                       local: Optional[dict[str, str]] = None,
                       depth: int = 0,
                       seen: frozenset = frozenset()
                       ) -> tuple[set[str], list]:
    """``(advancer keys, helper bodies)`` reachable from *node*.

    A loop rarely calls the advance method itself. The natural way to
    write "assert the invariants every frame" is a helper that updates
    and then asserts, and the loop calls the helper — which is exactly
    what the artifact that motivated this did:

        def update_and_check(self, game, dt):
            game.update(dt, self.keys)
            self.assert_walkable(game)

        for frame in range(700):
            self.update_and_check(self.game, dt)

    Reading only direct calls, ``called`` came back empty and a loop
    running 17 of its 700 frames was never looked at. So a call into a
    local helper is followed, with the helper's parameters bound from
    the arguments at the call site — that is how ``game`` inside the
    helper is known to be the ``Game`` the fixture built.

    The bodies come back too, because every silence has to be judged
    against them as well: a helper that pins the state, or restarts a
    finished run, is doing the honest thing on behalf of its callers.
    """
    local = local or {}
    called: set[str] = set()
    bodies: list = []

    def _resolve(key: Optional[str]) -> Optional[str]:
        if not key:
            return None
        return local.get(key) or bindings.get(key)

    for n in ast.walk(node):
        if not isinstance(n, ast.Call):
            continue
        if isinstance(n.func, ast.Attribute):
            cls = _resolve(_receiver_key(n.func))
            if cls and f"{cls}.{n.func.attr}" in advancers:
                called.add(f"{cls}.{n.func.attr}")
        if depth >= _MAX_HELPER_DEPTH:
            continue
        name = _called_helper(n.func)
        helper = helpers.get(name) if name else None
        if helper is None or name in seen:
            continue
        params = [a.arg for a in helper.args.args]
        if params and params[0] == "self":
            params = params[1:]
        sub: dict[str, str] = {}
        for param, arg in zip(params, n.args):
            cls = _resolve(_binding_key(arg))
            if cls:
                sub[param] = cls
        deeper, deeper_bodies = _reached_advancers(
            helper, bindings, advancers, helpers,
            local=sub, depth=depth + 1, seen=seen | {name})
        if deeper:
            called |= deeper
            bodies.append(helper)
            bodies.extend(deeper_bodies)
    return called, bodies


def _loop_iterations(node: ast.For) -> int:
    """Literal ``range(...)`` length, or ``-1`` when it is not knowable."""
    call = node.iter
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
        return -1
    if call.func.id != "range" or call.keywords:
        return -1
    args = call.args
    if not all(isinstance(a, ast.Constant) and isinstance(a.value, int)
               for a in args):
        return -1
    if len(args) == 1:
        return args[0].value
    if len(args) == 2:
        return args[1].value - args[0].value
    return -1


def _pins_state_working(stmts, spec: _EarlyReturnGuard) -> bool:
    """Does any statement here constrain the attribute to a value the
    guarded method still does work on?

    A tautology is deliberately not a guard: ``assertIn(game.state,
    (PLAYING, WIN, GAME_OVER))`` admits the terminal states it was
    supposed to rule out, and was written verbatim by a real run.
    """
    _PINNING = {"assertEqual", "assertIs", "assertNotEqual", "assertIsNot"}
    attr = spec.attr

    def _reads_state(node) -> bool:
        return any(isinstance(n, ast.Attribute) and n.attr == attr
                   for n in ast.walk(node))

    def _compare_pins(cmp_node) -> bool:
        if not isinstance(cmp_node, ast.Compare) or len(cmp_node.ops) != 1:
            return False
        if not _reads_state(cmp_node.left):
            return False
        op = cmp_node.ops[0]
        right = cmp_node.comparators[0]
        # Read the same spellings the guard side reads, or a suite that
        # pins with `assertEqual(game.state, "playing")` would look
        # unpinned and get flagged for a run it genuinely protects.
        name = _state_value_name(right)
        if name is None:
            return False
        if isinstance(op, (ast.Eq, ast.Is)):
            return spec.proceeds(name)
        if isinstance(op, (ast.NotEq, ast.IsNot)):
            return spec.halts(name)   # ruling out a terminal state
        return False

    for stmt in stmts:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Assert) and _compare_pins(node.test):
                return True
            if not isinstance(node, ast.Call):
                continue
            fname = (node.func.attr if isinstance(node.func, ast.Attribute)
                     else node.func.id if isinstance(node.func, ast.Name)
                     else "")
            if fname in _PINNING and len(node.args) >= 2:
                target, expected = node.args[0], node.args[1]
                if not _reads_state(target):
                    continue
                name = _state_value_name(expected)
                if name is None:
                    continue
                if fname in ("assertEqual", "assertIs") and spec.proceeds(name):
                    return True
                if (fname in ("assertNotEqual", "assertIsNot")
                        and spec.halts(name)):
                    return True
            elif fname in ("assertTrue", "assertFalse") and node.args:
                if fname == "assertTrue" and _compare_pins(node.args[0]):
                    return True
            elif fname == "assertIn" and len(node.args) >= 2:
                # Only a guard when every admitted value is one it works on.
                if not _reads_state(node.args[0]):
                    continue
                allowed = node.args[1]
                if isinstance(allowed, (ast.Tuple, ast.List, ast.Set)):
                    names = {_state_value_name(e) for e in allowed.elts}
                    if None not in names and spec.all_proceed(names):
                        return True
    return False


def degenerate_long_runs(root: str, paths: Iterable[str]
                         ) -> list[tuple[str, str]]:
    """``(path, reason)`` for long assertion loops that can quietly stop.

    Silent unless the project itself supplies both halves of the proof:
    a method that early-returns on one of its own attributes, and a test
    looping over it enough times to be claiming endurance. A test that
    pins that attribute to a value the method works on — inside the loop
    or right after it — is doing the honest thing and is never reported,
    and neither is one that breaks out on its own.
    """
    paths = sorted(set(paths))
    advancers: dict[str, tuple[str, set[str]]] = {}
    for path in paths:
        if not path.endswith(".py") or is_python_test_file(path):
            continue
        text = _read(root, path)
        if text is None:
            continue
        for qualified, guard in guarded_advance_methods(text).items():
            advancers.setdefault(qualified, guard)
    if not advancers:
        return []

    out: list[tuple[str, str]] = []
    for path in paths:
        if not is_python_test_file(path):
            continue
        text = _read(root, path)
        if text is None:
            continue
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            continue
        bindings = _class_bindings(tree)
        helpers = _local_functions(tree)
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for loop in ast.walk(fn):
                if not isinstance(loop, ast.For):
                    continue
                iterations = _loop_iterations(loop)
                if iterations < _MIN_LONG_RUN:
                    continue
                # Resolve each call's receiver to the class it was built
                # from, following the test's own helpers; an unresolved
                # receiver is simply not evidence.
                called, bodies = _reached_advancers(
                    loop, bindings, advancers, helpers)
                if not called:
                    continue
                if any(isinstance(n, ast.Break) for n in ast.walk(loop)):
                    continue           # the test already handles stopping
                method = sorted(called)[0]
                spec = advancers[method]
                before, after = _siblings(fn, loop)
                reached = list(loop.body) + after + bodies
                if _pins_state_working(reached, spec):
                    continue
                if _handles_termination([loop] + bodies, spec):
                    continue
                if _extends_lifetime(before):
                    continue
                out.append((path, (
                    f"{fn.name} loops {iterations} times over `{method}()`, "
                    f"which returns immediately once `{spec.attr}` "
                    f"{spec.describe_exit()} — nothing in the loop or after it "
                    f"pins `{spec.attr}` to a value it works on, so the "
                    f"object can stop advancing while every remaining "
                    f"iteration asserts against unchanged state and "
                    f"still passes")))
                break                  # one finding per test function
    return out


def _asserts_progress(stmts) -> bool:
    """Does any assertion here require that something CHANGED?

    The distinction the check turns on. An endurance loop is only worth
    the frames it runs if at least one of its assertions would fail
    against a frozen world, and most do not:

      assertFalse(map.is_wall(*e.tile))   invariant — true of a still board
      assertLessEqual(cur, prev)          monotone — 172 <= 172 passes
      assertGreater(pellets, 0)           strict, but against a literal
      assertIn(state, VALID_STATES)       a vocabulary, not a movement

    Only a STRICT relation between two things that both vary can say
    "this moved": ``assertNotEqual(g.player.tile, before)``,
    ``assertLess(counts[-1], initial)``. Non-strict operators admit
    equality by construction, and a comparison against a literal is
    fixed at both ends of the run, so neither is evidence.
    """
    _STRICT = {"assertNotEqual", "assertIsNot", "assertLess", "assertGreater"}

    def _varies(node) -> bool:
        # A literal cannot have changed; anything computed might have.
        return not isinstance(node, ast.Constant)

    def _strict_compare(node) -> bool:
        if not isinstance(node, ast.Compare) or len(node.ops) != 1:
            return False
        if not isinstance(node.ops[0], (ast.NotEq, ast.IsNot, ast.Lt, ast.Gt)):
            return False
        return _varies(node.left) and _varies(node.comparators[0])

    for stmt in stmts:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Assert) and _strict_compare(node.test):
                return True
            if not isinstance(node, ast.Call):
                continue
            fname = (node.func.attr if isinstance(node.func, ast.Attribute)
                     else node.func.id if isinstance(node.func, ast.Name)
                     else "")
            if fname in _STRICT and len(node.args) >= 2:
                if _varies(node.args[0]) and _varies(node.args[1]):
                    return True
            elif fname == "assertTrue" and node.args:
                if _strict_compare(node.args[0]):
                    return True
            elif fname == "assertFalse" and node.args:
                # assertFalse(a == b) is assertNotEqual(a, b).
                inner = node.args[0]
                if (isinstance(inner, ast.Compare) and len(inner.ops) == 1
                        and isinstance(inner.ops[0], (ast.Eq, ast.Is))
                        and _varies(inner.left)
                        and _varies(inner.comparators[0])):
                    return True
    return False


def unprogressed_long_runs(root: str, paths: Iterable[str]
                           ) -> list[tuple[str, str]]:
    """``(path, reason)`` for long runs that never assert anything moved.

    The sibling `degenerate-long-run` cannot see, and the third defect
    class in this family. That check asks whether the loop stopped doing
    work; `varied-input-ignored` asks whether the work read the input at
    all. Neither covers a loop whose object reads its input, does work
    every single iteration, and still goes nowhere.

    Measured 2026-08-17, glm-5.2 on the Pac-Man task with agent_loop on.
    `Entity._move` stepped toward the current tile centre before stepping
    forward, so once past the centre the correction moved it *backward*,
    and at small dt that consumed the whole travel budget:

        advance(0.01) -> x=1.04 -> x=1.0 -> x=1.04 ...

    Net displacement zero, forever, at any dt below roughly 0.2. The suite
    drove 20000 frames at 1/60 and passed, because its only assertion was
    `assertLessEqual(cur, prev)` on a pellet count that never moved. Every
    external probe of dt-scaling and of `press()` failed on the shipped
    artifact while all 19 of its own tests were green.

    What this reports is that the endurance claim is **unprotected** — the
    loop's assertions would all hold if the object had frozen on iteration
    one — not that the run is measurably degenerate. That discriminator is
    dynamic and not in the AST, exactly as for `varied-input-ignored`. The
    fix for both is the same one line: assert that something changed.

    Deliberately NOT silenced by a `break`. Breaking on a terminal state is
    how a careful test ends a run; it says nothing about whether the run
    accomplished anything, which is the entire question here.
    """
    paths = sorted(set(paths))
    advancers: dict[str, tuple[str, set[str]]] = {}
    for path in paths:
        if not path.endswith(".py") or is_python_test_file(path):
            continue
        text = _read(root, path)
        if text is None:
            continue
        for qualified, guard in guarded_advance_methods(text).items():
            advancers.setdefault(qualified, guard)
    if not advancers:
        return []

    out: list[tuple[str, str]] = []
    for path in paths:
        if not is_python_test_file(path):
            continue
        text = _read(root, path)
        if text is None:
            continue
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            continue
        bindings = _class_bindings(tree)
        helpers = _local_functions(tree)
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for loop in ast.walk(fn):
                if not isinstance(loop, ast.For):
                    continue
                iterations = _loop_iterations(loop)
                if iterations < _MIN_ENDURANCE_RUN:
                    continue
                called, bodies = _reached_advancers(
                    loop, bindings, advancers, helpers)
                if not called:
                    continue
                before, after = _siblings(fn, loop)
                # The whole test function gets the benefit of the doubt:
                # a progress assertion after the loop protects the run
                # just as well as one inside it.
                reached = list(loop.body) + after + bodies
                if _asserts_progress(reached):
                    continue
                method = sorted(called)[0]
                out.append((path, (
                    f"{fn.name} loops {iterations} times over `{method}()` "
                    f"but never asserts that anything changed — every "
                    f"assertion it makes would still hold if the object "
                    f"had frozen on the first iteration, so the endurance "
                    f"claim is unprotected")))
                break                  # one finding per test function
    return out


def _siblings(fn, loop: ast.For) -> tuple[list, list]:
    """``(before, after)`` statements around *loop* in its own block.

    Both halves matter. A post-loop ``assertEqual(game.state, PLAYING)``
    is a real guard — it fails when the run ended early — and a pre-loop
    statement is where a test disables the thing that would end the run.
    """
    for node in ast.walk(fn):
        for field in ("body", "orelse", "finalbody"):
            block = getattr(node, field, None)
            if isinstance(block, list) and loop in block:
                i = block.index(loop)
                return block[:i], block[i + 1:]
    return [], []


# A timer set to this or beyond is nobody's real gameplay value; it is a
# test deliberately holding the simulation open.
_KEEPALIVE_MIN = 1000


def _handles_termination(nodes, spec: _EarlyReturnGuard) -> bool:
    """Does the loop branch on the run having *ended*?

    Takes every node the loop reaches, helper methods included: a suite
    that restarts a finished run does it wherever the per-frame work
    lives, and that is as often in a shared helper as in the loop body.

    The third honest pattern, after pinning and breaking: notice the run
    ended and start another one. A real suite wrote

        if game.state is GameState.GAME_OVER:
            game.restart()

    and commented that it restarts "so this test continues to exercise the
    actual PLAYING update loop" — 2000 of 2000 frames live across six
    restarts.

    Branching on a value the method still works on is a different thing
    and is not
    accepted: ``if game.state == PLAYING and frame % 11 == 0: send_input()``
    only gates input, and the loop it appeared in degenerated anyway. The
    condition has to be one that a halted object satisfies.
    """
    def _tests_terminal(test) -> bool:
        if isinstance(test, ast.BoolOp):
            return any(_tests_terminal(v) for v in test.values)
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            return False        # `not <working>` is read below, not here
        if not isinstance(test, ast.Compare) or len(test.ops) != 1:
            return False
        if not any(isinstance(n, ast.Attribute) and n.attr == spec.attr
                   for n in ast.walk(test.left)):
            return False
        op, right = test.ops[0], test.comparators[0]
        names = {right.id} if isinstance(right, ast.Name) else (
            {right.attr} if isinstance(right, ast.Attribute) else set())
        if isinstance(right, (ast.Tuple, ast.List, ast.Set)):
            names = {e.id if isinstance(e, ast.Name) else
                     e.attr if isinstance(e, ast.Attribute) else ""
                     for e in right.elts}
        if not names or "" in names:
            return False
        if isinstance(op, (ast.Eq, ast.Is, ast.In)):
            # compares against a value the method halts on
            return all(spec.halts(n) for n in names)
        if isinstance(op, (ast.NotEq, ast.IsNot, ast.NotIn)):
            # "not still working"
            return any(spec.proceeds(n) for n in names)
        return False

    for root in nodes:
        for node in ast.walk(root):
            if (isinstance(node, (ast.If, ast.While))
                    and _tests_terminal(node.test)):
                return True
    return False


def _extends_lifetime(stmts) -> bool:
    """Did the test deliberately stop the run from ending?

    Two independent runs wrote exactly this before a long loop —
    ``game.frightened_timer = 1_000_000.0`` and
    ``game.spawn_protection_timer = 1_000_000.0`` — to keep ghosts from
    ending a simulation they wanted to observe for 2000 frames. Both were
    live for every frame. A test that did this thought about termination,
    and accusing it of ignoring termination is how a check gets ignored.
    """
    for stmt in stmts:
        for node in ast.walk(stmt):
            if not isinstance(node, (ast.Assign, ast.AugAssign)):
                continue
            targets = (node.targets if isinstance(node, ast.Assign)
                       else [node.target])
            if not any(isinstance(t, ast.Attribute) for t in targets):
                continue
            value = node.value
            if (isinstance(value, ast.Constant)
                    and isinstance(value.value, (int, float))
                    and not isinstance(value.value, bool)
                    and value.value >= _KEEPALIVE_MIN):
                return True
    return False


# ── inputs a test varies that the code never reads ───────────────────
#
# The sibling of `degenerate-long-run`, and the one it cannot see. That
# check asks whether the loop stopped doing work; this one asks whether
# the work ever depended on what the loop was varying.
#
# Measured: a task demanded "run >= 600 frames with dt drawn randomly
# from 0.008..0.05, not a fixed 1/60". The suite did exactly that, over
# 600 and then 2000 iterations, and passed. `run_frame(self, dt)` never
# mentions `dt` in its body — its own docstring says the parameter is
# "accepted to mirror the interface the tests expect". Replaying the
# suite with dt at 1e-9, 1/60 and 1e9 gives byte-identical results, so
# the adversarial condition the whole task was written around could not
# have failed. Every gate passed and no other check had anything to say.
#
# Nothing here is about time, frames or games: the rule is that a value
# a loop deliberately varies must reach code that reads it.


def _varying_names(loop: ast.For) -> set[str]:
    """Names this loop rebinds to something fresh each iteration.

    The loop variable itself, and anything assigned from a `random.*`
    draw inside the body — the two ways a test says "this input is not
    the same twice".
    """
    out: set[str] = set()
    target = loop.target
    for node in ast.walk(target):
        if isinstance(node, ast.Name):
            out.add(node.id)
    for stmt in loop.body:
        for node in ast.walk(stmt):
            if not isinstance(node, ast.Assign) or not _draws_randomly(
                    node.value):
                continue
            for t in node.targets:
                if isinstance(t, ast.Name):
                    out.add(t.id)
    return out


def _draws_randomly(node) -> bool:
    """Does this expression pull a fresh value from `random`?"""
    for n in ast.walk(node):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and isinstance(n.func.value, ast.Name)
                and n.func.value.id == "random"):
            return True
    return False


def _is_varied(arg, varying: set[str]) -> bool:
    if _draws_randomly(arg):
        return True
    return any(isinstance(n, ast.Name) and n.id in varying
               for n in ast.walk(arg))


def _reads_parameter(fn, param: str) -> bool:
    """Does *fn* mention *param* anywhere in its body?

    Any mention counts, load or store. A parameter that is reassigned
    before use is doing something odd, but it is not proof the input was
    discarded, and this check must only fire when that proof is total.
    """
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and node.id == param:
            return True
    return False


def _project_methods(root: str, paths: Iterable[str]) -> dict[str, ast.AST]:
    """``"Class.method" -> def`` across the project's non-test sources."""
    out: dict[str, ast.AST] = {}
    seen_twice: set[str] = set()
    for path in paths:
        if not path.endswith(".py") or is_python_test_file(path):
            continue
        text = _read(root, path)
        if text is None:
            continue
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    key = f"{node.name}.{item.name}"
                    if key in out:
                        seen_twice.add(key)
                    out[key] = item
    for key in seen_twice:
        out.pop(key, None)          # never guess at an ambiguous target
    return out


def ignored_varied_inputs(root: str, paths: Iterable[str]
                          ) -> list[tuple[str, str]]:
    """``(path, reason)`` where a loop varies an input the callee drops.

    Silent unless the proof is complete: the argument demonstrably
    changes per iteration, the receiver resolves to a class built in
    this project, that class's method is unambiguous, and the parameter
    it lands on appears nowhere in the method body. A method taking
    ``*args``/``**kwargs`` is never accused — the value may reach it by
    a route this cannot see.
    """
    paths = sorted(set(paths))
    methods = _project_methods(root, paths)
    if not methods:
        return []

    out: list[tuple[str, str]] = []
    for path in paths:
        if not is_python_test_file(path):
            continue
        text = _read(root, path)
        if text is None:
            continue
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            continue
        bindings = _class_bindings(tree)
        reported: set[tuple[str, str, str]] = set()
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for loop in ast.walk(fn):
                if not isinstance(loop, ast.For):
                    continue
                varying = _varying_names(loop)
                if not varying and not any(
                        _draws_randomly(n) for n in ast.walk(loop)):
                    continue
                for call in ast.walk(loop):
                    if not isinstance(call, ast.Call) or not isinstance(
                            call.func, ast.Attribute):
                        continue
                    cls = bindings.get(_receiver_key(call.func) or "")
                    if not cls:
                        continue
                    target = methods.get(f"{cls}.{call.func.attr}")
                    if target is None:
                        continue
                    if target.args.vararg or target.args.kwarg:
                        continue
                    params = [a.arg for a in target.args.args]
                    if params and params[0] == "self":
                        params = params[1:]
                    for index, arg in enumerate(call.args):
                        if index >= len(params):
                            break
                        param = params[index]
                        if not _is_varied(arg, varying):
                            continue
                        if _reads_parameter(target, param):
                            continue
                        key = (fn.name, f"{cls}.{call.func.attr}", param)
                        if key in reported:
                            continue
                        reported.add(key)
                        out.append((path, (
                            f"{fn.name} varies the argument passed as "
                            f"`{param}` on every iteration and hands it to "
                            f"`{cls}.{call.func.attr}()`, whose `{param}` "
                            f"parameter is never read in its body — the "
                            f"variation cannot change the outcome, so the "
                            f"condition this loop exists to test is not "
                            f"actually being applied")))
    return out


# `-s`/`-t` move discovery's start or top level, and a start directory
# is importable as itself — `python -m unittest discover -s tests` runs a
# suite that bare discovery from the root cannot see. Naming any explicit
# test target is not discovery at all.
_UNITTEST_DIR_FLAG_RE = re.compile(
    r'(?:^|\s)(?:-s|-t|--start-directory|--top-level-directory)(?:[=\s]|$)')
_UNITTEST_SEG_RE = re.compile(r'(?:^|\s)-m\s+unittest(?:\s+(.*))?$')


def discovers_from_project_root(commands: Iterable[str]) -> bool:
    """Does any declared command run bare ``unittest`` discovery at the root?

    Only that spelling makes a directory's importability decide whether
    its tests exist, so only that spelling licenses the finding below.
    """
    for cmd in commands or ():
        for seg in re.split(r'&&|\|\||;', cmd or ""):
            m = _UNITTEST_SEG_RE.search(seg.strip())
            if not m:
                continue
            rest = (m.group(1) or "").strip()
            if _UNITTEST_DIR_FLAG_RE.search(" " + rest):
                continue
            args = [t for t in rest.split()
                    if t and t != "discover" and not t.startswith("-")]
            if args:
                continue              # names explicit tests, not discovery
            return True
    return False


def unreachable_package_dir(root: str, path: str) -> Optional[str]:
    """First directory above *path* that root discovery cannot enter.

    ``unittest`` recurses only into importable packages: a directory with
    no ``__init__.py`` is skipped whole, and every file under it is
    invisible to the run's own acceptance command. Verified against the
    interpreter rather than assumed — a ``tests/`` holding six tests
    contributed nothing to ``python -m unittest`` until the file existed.
    """
    parts = [p for p in path.replace("\\", "/").split("/")[:-1] if p]
    walked: list[str] = []
    for part in parts:
        walked.append(part)
        rel = "/".join(walked)
        if not os.path.isfile(os.path.join(root, *walked, "__init__.py")):
            return rel
    return None


def uncollected_test_files(root: str, paths: Iterable[str],
                           runner: Optional[str],
                           commands: Iterable[str] = ()) -> list[tuple[str, str]]:
    """``(path, reason)`` for test files the *runner* will never collect.

    Two ways a file goes uncollected, and they are independent: what the
    file *contains* (pytest-style functions under a unittest runner), and
    where it *sits* (a directory discovery cannot enter). The second was
    missed until a run wrote six tests into a ``tests/`` with no
    ``__init__.py``: the step's gate — ``python -m unittest`` — stayed
    green throughout on a *different* file's tests, the seeded acceptance
    contract at the root, so the step spent eight turns getting no signal
    from its own gate about its own work.

    Silent when the runner cannot be identified, when a file will not
    parse, or when the runner is pytest (which collects both styles) —
    absence of a clear rule is not evidence of a broken test file.
    """
    if runner != "unittest":
        return []
    rooted = discovers_from_project_root(commands)
    out: list[tuple[str, str]] = []
    for path in sorted(set(paths)):
        if not is_python_test_file(path):
            continue
        if rooted:
            missing = unreachable_package_dir(root, path)
            if missing:
                out.append((path, (
                    f"`{missing}/` has no __init__.py, and `unittest` "
                    f"discovery recurses only into importable packages — "
                    f"nothing under it runs under the project's own "
                    f"acceptance command")))
                continue
        text = _read(root, path)
        if text is None:
            continue
        visible, pytest_style = _python_test_counts(text)
        if visible < 0:
            continue                        # unparseable
        if visible > 0:
            continue
        if pytest_style > 0:
            out.append((path, (
                f"{pytest_style} test(s) are written pytest-style "
                f"(module-level functions), and `unittest` collects only "
                f"TestCase subclasses — none of them run under the "
                f"project's own acceptance command")))
        else:
            out.append((path, "defines no tests the declared runner collects"))
    return out


def _check_gate(cmd: str, gates: list[str]) -> tuple[str, str]:
    """Did this step's acceptance command ever go green?

    Answered purely from the ledger of gates that already passed — this
    module never runs a command. Matching is whitespace-insensitive and
    allows containment in either direction, because the pipeline may
    respell a gate or prefix it with a ``cd`` before recording it.
    """
    if not gates:
        return UNKNOWN, "no gate ledger available"
    want = _canon_cmd(cmd)
    if not want:
        return UNKNOWN, "empty verify command"
    for recorded in gates:
        got = _canon_cmd(recorded)
        if got == want or want in got or got in want:
            return HOLDS, ""
    # UNKNOWN, not VIOLATED. The ledger records gates that passed through
    # the normal step path; it is NOT a complete log of every gate that
    # ever ran. Observed: a CMD step's `python -m unittest -v` passed
    # inside the agent loop's recovery path and again in BulkTest, and
    # never entered the ledger — reporting "never passed" about a suite
    # that had just gone green twice is exactly the confident falsehood
    # this module's three-valued discipline exists to prevent. Absence
    # from an incomplete record is absence of evidence.
    return UNKNOWN, ("not found in the gate ledger — it may have passed "
                     "outside the ledger's recording path (agent-loop "
                     "recovery, BulkTest), so this is inconclusive")


# ── Module-level handle (mirrors get_gate_ledger) ────────────────────

_ghost: Optional[GhostPlan] = None


def start_ghost(steps, project_root: str = ".",
                carried_step_ids: Iterable[str] = ()) -> Optional[GhostPlan]:
    """Build and install the run's ghost. Returns ``None`` on any failure."""
    global _ghost
    try:
        _ghost = GhostPlan.build(steps, project_root, carried_step_ids)
        _logger.info(
            "[Ghost] tracking %d expectation(s) across %d step(s) and "
            "%d file(s)", len(_ghost.expectations), len(_ghost.steps),
            len(_ghost.files))
        if _ghost.carried:
            _logger.info(
                "[Ghost] %d step(s) completed in an earlier run are not "
                "judged by this one: %s",
                len(_ghost.carried), ", ".join(sorted(_ghost.carried)))
    except Exception as exc:
        _ghost = None
        _logger.debug("[Ghost] disabled — build failed: %s", exc)
    return _ghost


def get_ghost() -> Optional[GhostPlan]:
    return _ghost


def reset_ghost() -> None:
    global _ghost
    _ghost = None
