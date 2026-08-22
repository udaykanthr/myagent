r"""A root that owns its own manifest is not a sub-project scaffold.

`_detect_subproject_root` infers "the project really lives in
subdirectory X" from finding a manifest in X. That inference is only
sound when the repo root is NOT itself a package -- which is the case
the function was written for (`npx create-next-app my-app` leaves the
root bare). When the root owns a manifest, the app is AT the root and a
sibling package is just a second package.

Measured 2026-08-22 run 31. The plan scaffolded Vite at the root
(`npm create vite@latest .`), so `package.json` carrying the real
`"test": "vitest run"` sat at the top and the suite lived in
`src/test/`. A planned `server/package.json` made this function answer
`server/`, and every downstream step then behaved correctly given a
wrong premise:

    [BulkTest] bulk test execution on: src/test/auth-flow.test.jsx, ...
    [SubProject] Detected sub-project root via disk manifest: server/
    [Executor] npm test --silent (cwd=server)          -> exit 1
    [Executor] npx vitest run (cwd=server)             -> No test files found
    [VitestEnv] Installing missing test deps (cwd=server)
    [BulkTest] Agent-loop fix attempt before per-file loop

The recovery loop "fixed" the empty suite by writing `server/app.test.js`
and rewriting `server/middleware/auth.js` from CommonJS into ESM. That
rewrite broke gates 4.2 and 7.1, which `require()` it, and the run ended
on a GATE CONFLICT. One wrong directory answer cost the whole run.
"""

import os

import pytest

from agentchanti.orchestrator.memory import FileMemory
from agentchanti.orchestrator.step_handlers import _detect_subproject_root

MANIFESTS = ["package.json", "requirements.txt", "go.mod", "Cargo.toml",
             "Gemfile", "pyproject.toml", "composer.json"]


@pytest.fixture
def in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _memory(paths):
    mem = FileMemory()
    mem.update({p: "x" for p in paths})
    return mem


def _write(root, rel, body="{}"):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body)


def test_the_incident_layout_has_no_subproject(in_tmp):
    """Vite at the root + a server/ package: the root is the project."""
    _write(in_tmp, "package.json", '{"scripts":{"test":"vitest run"}}')
    _write(in_tmp, "server/package.json", '{"name":"server"}')
    mem = _memory(["src/test/auth-flow.test.jsx", "src/App.jsx",
                   "server/app.js", "server/routes/auth.js"])
    assert _detect_subproject_root(mem) is None


@pytest.mark.parametrize("manifest", MANIFESTS)
def test_any_root_manifest_settles_it(in_tmp, manifest):
    """Whatever ecosystem the root belongs to, owning a manifest is enough.

    Files span two top-level directories, as in the incident — that is
    the shape where the manifest fallbacks operate. When every tracked
    file shares ONE prefix, an earlier and stronger branch answers first
    and is deliberately left alone: all the work really is in there.
    """
    _write(in_tmp, manifest)
    _write(in_tmp, "server/package.json")
    mem = _memory(["src/App.jsx", "server/app.js"])
    assert _detect_subproject_root(mem) is None


def test_a_bare_root_still_detects_the_subproject(in_tmp):
    """The case the function exists for must keep working."""
    _write(in_tmp, "frontend/package.json")
    mem = _memory(["frontend/src/App.jsx", "frontend/src/main.jsx"])
    assert _detect_subproject_root(mem) == "frontend"


def test_a_bare_root_with_two_packages_is_still_ambiguous(in_tmp):
    """Unchanged: two sibling manifests and no root one resolve to nothing."""
    _write(in_tmp, "frontend/package.json")
    _write(in_tmp, "backend/package.json")
    mem = _memory(["frontend/src/App.jsx", "backend/app.js"])
    assert _detect_subproject_root(mem) in (None, "frontend", "backend")


def test_an_explicit_scaffold_still_outranks_this(in_tmp):
    """A create-* command naming a directory is direct evidence.

    It is recorded on memory by the scaffold step and must survive a root
    manifest appearing later (a root README-driven `npm init`, say).
    """
    _write(in_tmp, "package.json")
    (in_tmp / "my-app").mkdir()
    mem = _memory(["my-app/src/App.jsx"])
    mem._scaffolded_subproject = "my-app"
    assert _detect_subproject_root(mem) == "my-app"


def test_no_manifest_anywhere_is_unchanged(in_tmp):
    mem = _memory(["src/App.jsx", "src/main.jsx"])
    assert _detect_subproject_root(mem) in (None, "src")
