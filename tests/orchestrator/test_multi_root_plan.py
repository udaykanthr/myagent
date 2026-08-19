"""A plan that declares two roots must not have one folded into the other.

Replays the measured 2026-08-19 run: task "create a react responsive
project as FE and express js API as BE service". Step 1.1 scaffolded
``frontend/`` via ``npm create vite@latest frontend``; step 1.2's
``mkdir backend && cd backend && npm init -y`` was then re-rooted into
it and built the entire Express backend at ``frontend/backend/``.

The misroute was *self-consistent*, which is why nothing caught it: the
backend gates were prefixed ``cd frontend &&`` too, so they read the same
wrong copy and all four passed green. The run finished with the backend
on disk twice — ``frontend/backend/`` holding the installed node_modules
and the plan-declared ``backend/`` holding none.
"""

import pytest

from agentchanti.orchestrator.memory import FileMemory
from agentchanti.orchestrator.plan_step import PlanStep
from agentchanti.orchestrator.step_handlers import (
    _plan_sibling_roots,
    _prefix_subproject_paths,
    _step_belongs_to_subproject,
)


def _memory(roots=None, files=None):
    mem = FileMemory(embedding_store=None)
    if files:
        mem.update(files)
    if roots is not None:
        mem._plan_declared_roots = set(roots)
    return mem


def _step(sid, targets, verify=None, command=None):
    return PlanStep(id=sid, step_type="CODE", target_files=list(targets),
                    verify_cmd=verify, command=command)


# ─── the run's own plan ──────────────────────────────────────────────

# Every step of the measured plan that declares a target, verbatim.
MEASURED_ROOTS = {"frontend", "backend"}

BACKEND_STEPS = [
    _step("1.2", ["backend/package.json"],
          command="mkdir backend && cd backend && npm init -y"),
    _step("2.2", ["backend/package-lock.json"]),
    _step("3.2", ["backend/package.json", "backend/.env.example"]),
    _step("4.1", ["backend/data/userStore.js"]),
    _step("4.2", ["backend/services/authValidation.js"]),
    _step("5.1", ["backend/controllers/authController.js"]),
    _step("5.2", ["backend/routes/authRoutes.js"]),
    _step("6.1", ["backend/app.js"]),
    _step("6.2", ["backend/server.js"]),
]

FRONTEND_STEPS = [
    _step("1.1", ["frontend/package.json", "frontend/src/main.jsx",
                  "frontend/src/App.jsx"]),
    _step("2.1", ["frontend/package-lock.json"]),
    _step("3.1", ["frontend/package.json", "frontend/vitest.config.js",
                  "frontend/src/vitest.setup.js", "frontend/.env.example"]),
    _step("7.1", ["frontend/src/services/api.js"]),
    _step("7.2", ["frontend/src/context/AuthContext.jsx"]),
    _step("8.3", ["frontend/src/main.jsx", "frontend/src/index.css"]),
    _step("10.1", ["frontend/src/App.test.jsx",
                   "frontend/src/pages/AuthPages.test.jsx"]),
]


@pytest.mark.parametrize("step", BACKEND_STEPS, ids=lambda s: s.id)
def test_backend_steps_do_not_belong_to_the_frontend(step):
    assert _step_belongs_to_subproject(step, "frontend",
                                       _memory(MEASURED_ROOTS)) is False


@pytest.mark.parametrize("step", FRONTEND_STEPS, ids=lambda s: s.id)
def test_frontend_steps_do_belong_to_the_frontend(step):
    assert _step_belongs_to_subproject(step, "frontend",
                                       _memory(MEASURED_ROOTS)) is True


def test_readme_at_the_repo_root_is_not_claimed_by_either():
    """Step 9.1 targets `README.md`, whose gate reads it from the root.

    A `cd frontend &&` prefix would have made that gate unpassable — it
    survived only because the step declares no directory at all, so the
    root never entered `_plan_declared_roots`.
    """
    mem = _memory(MEASURED_ROOTS)
    step = _step("9.1", ["README.md"])
    assert _step_belongs_to_subproject(step, "frontend", mem) is False


# ─── the guard that keeps single-scaffold runs unchanged ─────────────

def test_unknown_when_the_plan_never_names_the_subproject():
    """The ordinary `npx create-next-app my-app` case.

    The planner writes `src/App.jsx`, meaning `my-app/src/App.jsx`, and
    the prefix machinery is right to apply itself. Nothing here may
    second-guess that, so the answer is None — never a guess.
    """
    mem = _memory({"src", "components"})
    step = _step("2.1", ["src/App.jsx"])
    assert _step_belongs_to_subproject(step, "my-app", mem) is None
    assert _plan_sibling_roots(mem, "my-app") is None


def test_unknown_without_a_recorded_plan():
    """A resume or a plan-less path records no roots; behaviour is unchanged."""
    mem = FileMemory(embedding_store=None)
    step = _step("1.2", ["backend/package.json"])
    assert _step_belongs_to_subproject(step, "frontend", mem) is None


def test_unknown_when_the_step_declares_no_target():
    mem = _memory(MEASURED_ROOTS)
    assert _step_belongs_to_subproject(_step("x", []), "frontend", mem) is None
    assert _step_belongs_to_subproject(None, "frontend", mem) is None


def test_no_subproject_is_unknown():
    assert _step_belongs_to_subproject(BACKEND_STEPS[0], "",
                                       _memory(MEASURED_ROOTS)) is None


# ─── the classic path's twin of the same defect ──────────────────────

def test_sibling_root_paths_are_not_prefixed():
    """`backend/app.js` is the repo's backend, not `frontend/backend/app.js`."""
    mem = _memory(MEASURED_ROOTS, {"frontend/package.json": "{}"})
    out = _prefix_subproject_paths(
        {"backend/app.js": "const express = require('express');"},
        "frontend", mem)
    assert "backend/app.js" in out
    assert "frontend/backend/app.js" not in out


def test_non_root_paths_are_still_prefixed_in_a_multi_root_plan():
    """The guard is scoped to declared siblings — it is not a blanket opt-out."""
    mem = _memory(MEASURED_ROOTS, {"frontend/package.json": "{}"})
    out = _prefix_subproject_paths({"src/Widget.jsx": "x"}, "frontend", mem)
    assert "frontend/src/Widget.jsx" in out
    assert "src/Widget.jsx" not in out


def test_single_root_plan_prefixing_is_untouched():
    mem = _memory({"src"}, {"my-app/package.json": "{}"})
    out = _prefix_subproject_paths({"components/Header.tsx": "h"},
                                   "my-app", mem)
    assert "my-app/components/Header.tsx" in out


# ─── the gate prefix: how the misroute stayed invisible ──────────────

def _scaffolded(tmp_path, monkeypatch, roots):
    """A repo with `frontend/` scaffolded and `backend/` beside it."""
    (tmp_path / "frontend").mkdir()
    (tmp_path / "frontend" / "package.json").write_text('{"type":"module"}')
    (tmp_path / "backend").mkdir()
    (tmp_path / "backend" / "package.json").write_text('{"type":"commonjs"}')
    monkeypatch.chdir(tmp_path)
    mem = _memory(roots, {"frontend/package.json": "{}"})
    mem._scaffolded_subproject = "frontend"
    return mem


def test_backend_gate_is_not_redirected_into_the_frontend(tmp_path, monkeypatch):
    """Step 3.2's gate, verbatim from the run.

    Prefixed with `cd frontend &&` it resolved `frontend/backend/
    package.json` — which existed, because the CMD step had been
    misrouted the same way. The gate passed, agreed with the misroute,
    and the two wrongs left nothing to disagree.
    """
    from agentchanti.orchestrator.step_handlers import _declared_verify_cmd
    mem = _scaffolded(tmp_path, monkeypatch, MEASURED_ROOTS)
    gate = ("node -e \"const p=require('./backend/package.json'); "
            "if(!p.scripts.dev)process.exit(1)\"")
    out = _declared_verify_cmd(
        _step("3.2", ["backend/package.json", "backend/.env.example"], gate),
        mem)
    assert not out.startswith("cd frontend")
    assert out == gate


def test_frontend_gate_still_gets_its_prefix(tmp_path, monkeypatch):
    from agentchanti.orchestrator.step_handlers import _declared_verify_cmd
    mem = _scaffolded(tmp_path, monkeypatch, MEASURED_ROOTS)
    out = _declared_verify_cmd(_step("7.1", ["frontend/src/services/api.js"],
                                     "npm run build"), mem)
    assert out == "cd frontend && npm run build"


def test_single_root_gate_prefixing_is_untouched(tmp_path, monkeypatch):
    from agentchanti.orchestrator.step_handlers import _declared_verify_cmd
    mem = _scaffolded(tmp_path, monkeypatch, {"src"})
    out = _declared_verify_cmd(_step("2.1", ["src/App.jsx"],
                                     "npm run build"), mem)
    assert out == "cd frontend && npm run build"


# ─── the gate must not be sent into a directory it already names ─────

def test_gate_naming_the_subproject_in_a_path_is_not_prefixed(
        tmp_path, monkeypatch):
    """Step 3.1's gate from run 2, verbatim.

    The planner wrote root-relative paths — `require('./frontend/
    package.json')` — which is correct. `_references_sub` did not see the
    reference because the character before `frontend` is `/`, not a quote,
    so `cd frontend &&` was prepended and the gate looked for
    `frontend/frontend/package.json`. GateIntegrity called it STALLED
    after three identical failures over three artifacts, and the recovery
    loop then satisfied it by CREATING `frontend/frontend/package.json`
    and `frontend/frontend/vitest.setup.js` — the duplicate nested package
    this guard exists to prevent.
    """
    from agentchanti.orchestrator.step_handlers import _declared_verify_cmd
    mem = _scaffolded(tmp_path, monkeypatch, MEASURED_ROOTS)
    gate = ("node -e \"const p=require('./frontend/package.json'); "
            "if(!p.scripts.test) process.exit(1); const fs=require('fs'); "
            "if(!fs.readFileSync('./frontend/vitest.setup.js','utf8')"
            ".includes('jest-dom')) process.exit(1)\"")
    out = _declared_verify_cmd(
        _step("3.1", ["frontend/package.json", "frontend/vitest.setup.js"],
              gate), mem)
    assert out == gate
    assert "cd frontend" not in out


@pytest.mark.parametrize("gate", [
    "node -e \"require('./frontend/package.json')\"",      # ./ prefix
    "node -e \"require('src/frontend/x.js')\"",            # mid-path
    'node -e "require(\'.\\frontend\\x.js\')"',        # windows sep
    "npm --prefix frontend run build",                     # bare token
    "cat frontend/package.json",                           # plain path
])
def test_every_spelling_of_a_self_reference_suppresses_the_prefix(
        gate, tmp_path, monkeypatch):
    from agentchanti.orchestrator.step_handlers import _declared_verify_cmd
    mem = _scaffolded(tmp_path, monkeypatch, MEASURED_ROOTS)
    out = _declared_verify_cmd(_step("3.1", ["frontend/package.json"], gate),
                               mem)
    assert not out.startswith("cd frontend"), out


def test_a_gate_naming_nothing_still_gets_the_prefix(tmp_path, monkeypatch):
    """The guard must stay narrow — this is the case the prefix is FOR."""
    from agentchanti.orchestrator.step_handlers import _declared_verify_cmd
    mem = _scaffolded(tmp_path, monkeypatch, MEASURED_ROOTS)
    out = _declared_verify_cmd(
        _step("7.1", ["frontend/src/services/api.js"], "npm run build"), mem)
    assert out == "cd frontend && npm run build"


# ─── the idempotency check must look where the command will run ──────

def test_idempotency_uses_the_same_root_the_command_will_run_in(
        tmp_path, monkeypatch):
    """`npm install express` for a BACKEND step, with express in frontend/.

    `_make_cmd_idempotent` answers "already satisfied?" by looking for
    `node_modules/<pkg>`. Asking that about `frontend/` while the command
    runs at the repo root skips a backend install because the frontend
    happened to have the package — and unlike the other members of this
    family, a wrong answer here does not mis-report, it SKIPS the work.
    """
    from agentchanti.orchestrator.step_handlers import _make_cmd_idempotent

    (tmp_path / "frontend" / "node_modules" / "express").mkdir(parents=True)
    (tmp_path / "backend").mkdir()
    monkeypatch.chdir(tmp_path)

    # Asked about frontend/, the install looks redundant...
    _cmd, why = _make_cmd_idempotent("npm install express", None,
                                     cwd="frontend")
    assert _cmd is None and "express" in why

    # ...but the backend step runs at the repo root, where it is not.
    cmd_root, why_root = _make_cmd_idempotent("npm install express", None,
                                              cwd=None)
    assert cmd_root is not None, why_root


def test_a_frontend_step_still_gets_the_subproject_root(tmp_path, monkeypatch):
    """The narrowing is per-step, not a blanket disable."""
    mem = _memory(MEASURED_ROOTS)
    step = _step("7.1", ["frontend/src/services/api.js"])
    assert _step_belongs_to_subproject(step, "frontend", mem) is True


# ─── a step spanning both roots must not be re-rooted ────────────────

MIXED_STEP = _step(
    "8.1", ["backend/.env.example", "frontend/.env.example", "README.md"],
    "node -e \"const fs=require('fs');"
    "const env=fs.readFileSync('backend/.env.example','utf8');"
    "const readme=fs.readFileSync('README.md','utf8');"
    "if(!env.includes('PORT')||!readme.includes('in-memory'))process.exit(1)\"")


def test_a_step_spanning_both_roots_does_not_belong_to_either():
    """Run 13 step 8.1, verbatim.

    Under `any()` this read as "belongs to frontend", so its gate ran as
    `cd frontend && ... readFileSync('backend/.env.example') ...
    readFileSync('README.md')` — two paths that do not exist there. The
    agent satisfied it by CREATING `frontend/backend/.env.example` and
    `frontend/README.md`, both flagged as unplanned writes. Re-rooting is
    only correct when the WHOLE step is under the sub-project.
    """
    assert _step_belongs_to_subproject(
        MIXED_STEP, "frontend", _memory(MEASURED_ROOTS)) is False


def test_the_mixed_steps_gate_keeps_its_root_relative_form(
        tmp_path, monkeypatch):
    from agentchanti.orchestrator.step_handlers import _declared_verify_cmd
    mem = _scaffolded(tmp_path, monkeypatch, MEASURED_ROOTS)
    out = _declared_verify_cmd(MIXED_STEP, mem)
    assert not out.startswith("cd frontend"), out
    assert out == MIXED_STEP.verify_cmd


def test_a_wholly_contained_step_still_belongs():
    """The narrowing must not disarm the ordinary case."""
    step = _step("3.1", ["frontend/package.json", "frontend/vitest.config.js",
                         "frontend/src/vitest.setup.js"])
    assert _step_belongs_to_subproject(
        step, "frontend", _memory(MEASURED_ROOTS)) is True
