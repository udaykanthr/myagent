"""An env self-heal installs beside the manifest that owns the step.

Measured 2026-08-19 run 6: a backend step failed on missing `jsonwebtoken`
and `supertest`, and both were installed into a repo-root `node_modules`,
leaving a root `package.json` belonging to no project in the repo. The
app worked only because Node walks up from `backend/`; shipping `backend/`
on its own would break.

`_cd_prefix` was the only directory signal, and it reads a `cd` the GATE
happens to carry. A correct root-relative backend gate — `node -e
"require('./backend/x')"` — carries none, so the heal fell to the root.
"""

import os

import pytest

BS = chr(92)  # a literal backslash, kept out of string escapes

from agentchanti.orchestrator.agent_loop import _npm_package_dir


def _mk(root, *dirs):
    for d in dirs:
        os.makedirs(os.path.join(root, d), exist_ok=True)
        with open(os.path.join(root, d, "package.json"), "w") as fh:
            fh.write('{"name":"x"}')


def test_backend_step_resolves_to_backend(tmp_path):
    root = str(tmp_path)
    _mk(root, "backend", "frontend")
    assert _npm_package_dir(root, ["backend/routes/auth.js"]) == "backend"
    assert _npm_package_dir(root, ["backend/test/api.test.js"]) == "backend"


def test_frontend_step_resolves_to_frontend(tmp_path):
    root = str(tmp_path)
    _mk(root, "backend", "frontend")
    assert _npm_package_dir(root, ["frontend/src/pages/HomePage.jsx"]) == "frontend"


def test_several_files_in_one_package_agree(tmp_path):
    root = str(tmp_path)
    _mk(root, "backend")
    got = _npm_package_dir(root, ["backend/app.js", "backend/routes/auth.js"])
    assert got == "backend"


def test_a_step_spanning_two_packages_declines(tmp_path):
    """No single right target — fall back to the caller's old behaviour."""
    root = str(tmp_path)
    _mk(root, "backend", "frontend")
    assert _npm_package_dir(
        root, ["backend/app.js", "frontend/src/main.jsx"]) == ""


def test_a_root_manifest_reads_as_empty(tmp_path):
    """Root and unknown are the same instruction: add no --prefix."""
    root = str(tmp_path)
    _mk(root, ".")
    assert _npm_package_dir(root, ["src/index.js"]) == ""


def test_no_manifest_anywhere_declines(tmp_path):
    assert _npm_package_dir(str(tmp_path), ["backend/app.js"]) == ""


def test_no_planned_files_declines(tmp_path):
    root = str(tmp_path)
    _mk(root, "backend")
    assert _npm_package_dir(root, None) == ""
    assert _npm_package_dir(root, []) == ""
    assert _npm_package_dir(root, ["", None]) == ""


def test_nearest_manifest_wins_over_an_outer_one(tmp_path):
    root = str(tmp_path)
    _mk(root, ".", "backend")
    assert _npm_package_dir(root, ["backend/src/app.js"]) == "backend"


def test_windows_separators_resolve(tmp_path):
    root = str(tmp_path)
    _mk(root, "backend")
    win = "backend" + BS + "routes" + BS + "auth.js"
    assert _npm_package_dir(root, [win]) == "backend"


# ─── the input must be the step's TARGETS, not its reading list ──────

def test_targets_decide_even_when_the_reading_list_spans_packages(tmp_path):
    """Run 13: the same loop healed to `backend/` then to the repo root.

    `_loop_preload_paths` hands the loop `target_files` PLUS every file
    the step declares an import from, so a backend step that reads a
    frontend module spans two packages. The unanimity rule then declines
    and the install falls back to the root — which is the bug the whole
    fix exists to prevent. The step's declared targets are what "the
    manifest owning this step" means.
    """
    root = str(tmp_path)
    _mk(root, "backend", "frontend")
    targets = ["backend/routes/authRoutes.js"]
    reading_list = targets + ["frontend/src/services/api.js",
                              "backend/middleware/auth.js"]

    # The reading list is genuinely ambiguous...
    assert _npm_package_dir(root, reading_list) == ""
    # ...but the targets are not.
    assert _npm_package_dir(root, targets) == "backend"


def test_the_heal_prefers_targets_over_planned_files(tmp_path, monkeypatch):
    """End-to-end: the command carries --prefix backend, not a bare install."""
    from agentchanti.orchestrator import agent_loop as al

    root = str(tmp_path)
    _mk(root, "backend", "frontend")
    monkeypatch.chdir(root)

    seen = {}

    class _Tools:
        project_root = root
        def execute(self, call):
            seen["cmd"] = call.arguments["command"]
            return "exit: success"

    from agentchanti.orchestrator import pipeline as pl
    monkeypatch.setattr(pl, "_missing_js_packages", lambda out: ["jsonwebtoken"])
    ok = al.attempt_env_self_heal(
        _Tools(), "Cannot find package 'jsonwebtoken'", "javascript", set(),
        verify_cmd=None,
        planned_files=["backend/app.js", "frontend/src/api.js"],
        target_files={"backend/app.js"})
    assert ok
    assert seen["cmd"] == "npm --prefix backend install -D jsonwebtoken"


def test_falls_back_to_planned_files_when_no_targets(tmp_path, monkeypatch):
    """A step with no declared target keeps the previous behaviour."""
    from agentchanti.orchestrator import agent_loop as al
    root = str(tmp_path)
    _mk(root, "backend")
    monkeypatch.chdir(root)
    seen = {}

    class _Tools:
        project_root = root
        def execute(self, call):
            seen["cmd"] = call.arguments["command"]
            return "exit: success"

    from agentchanti.orchestrator import pipeline as pl
    monkeypatch.setattr(pl, "_missing_js_packages", lambda out: ["supertest"])
    al.attempt_env_self_heal(_Tools(), "x", "javascript", set(),
                             planned_files=["backend/app.js"],
                             target_files=None)
    assert seen["cmd"] == "npm --prefix backend install -D supertest"
