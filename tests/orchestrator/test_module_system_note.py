"""The module system is a per-file fact, decided by the nearest manifest.

Replays the 2026-08-19 "React frontend + Express backend" run. The
dependency-fix prompt asked one run-wide question — does ANY file in
memory contain `import ... from` — so the React components made the
answer "ES Modules" for the whole repo, and the CommonJS branch sat in
an `elif` that could never be reached.

The fix prompt for a CommonJS Express module was told to use ESM. It
rewrote `module.exports = {...}` into `export function`, dropping every
declared export; the step's previously green gate went red twice, and
the monotonic check reported it as a GATE CONFLICT — telling the reader
to fix the plan's verify line for a gate that was correct all along.
"""

import pytest

from agentchanti.orchestrator.dependency_check import (
    _module_system_note,
    _module_system_of,
)

# The two manifests the run actually produced.
VITE_MANIFEST = '{"name":"frontend","type":"module","scripts":{}}'
EXPRESS_MANIFEST = '{"name":"backend","type":"commonjs","scripts":{}}'

MONOREPO = {
    "frontend/package.json": VITE_MANIFEST,
    "frontend/src/App.jsx": "import React from 'react';\nexport default App;",
    "backend/package.json": EXPRESS_MANIFEST,
    "backend/services/authValidation.js":
        "const x = require('node:crypto');\nmodule.exports = { validateSignupInput };",
    "backend/app.js": "const express = require('express');\nmodule.exports = { app };",
}


@pytest.mark.parametrize("fpath,expected", [
    ("backend/services/authValidation.js", "cjs"),
    ("backend/app.js", "cjs"),
    ("frontend/src/App.jsx", "esm"),
])
def test_nearest_manifest_decides(fpath, expected):
    assert _module_system_of(fpath, MONOREPO) == expected


def test_the_incident_file_is_not_called_esm():
    """The single assertion the broken code failed.

    `backend/services/authValidation.js` sits beside a `"type":
    "commonjs"` manifest. Nothing about a React component two
    directories away may change that.
    """
    note = _module_system_note({"backend/services/authValidation.js"}, MONOREPO)
    assert "CommonJS" in note
    assert "ES Modules" not in note


def test_mixed_repo_names_each_file_rather_than_averaging():
    note = _module_system_note(
        {"backend/app.js", "frontend/src/App.jsx"}, MONOREPO)
    assert "backend/app.js: CommonJS" in note
    assert "frontend/src/App.jsx: ES Modules" in note
    assert "Do NOT convert" in note


def test_single_root_esm_project_reads_as_before():
    files = {"package.json": VITE_MANIFEST,
             "src/App.jsx": "import React from 'react';"}
    note = _module_system_note({"src/App.jsx"}, files)
    assert note == "Project uses ES Modules (import/export) — use ESM syntax."


def test_single_root_cjs_project_reads_as_before():
    files = {"package.json": '{"name":"api"}',
             "server.js": "const express = require('express');"}
    note = _module_system_note({"server.js"}, files)
    assert note == ("Project uses CommonJS (require/module.exports) — "
                    "use CJS syntax.")


def test_explicit_extensions_win_over_the_manifest():
    files = {"package.json": VITE_MANIFEST}
    assert _module_system_of("scripts/build.cjs", files) == "cjs"
    assert _module_system_of("scripts/build.mjs", files) == "esm"


def test_manifest_without_a_type_is_commonjs():
    """Node's own default, and the boundary still stops the upward walk."""
    files = {"package.json": '{"type":"module"}',
             "tools/package.json": '{"name":"tools"}',
             "tools/run.js": "doSomething();"}
    assert _module_system_of("tools/run.js", files) == "cjs"


def test_falls_back_to_the_file_when_no_manifest_exists(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    files = {"a.js": "const x = require('y');"}
    assert _module_system_of("a.js", files) == "cjs"
    assert _module_system_of("b.js", {"b.js": "import y from 'y';"}) == "esm"


def test_undecidable_stays_unknown(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert _module_system_of("a.js", {"a.js": "let x = 1;"}) is None
    assert _module_system_note({"a.js"}, {"a.js": "let x = 1;"}) == \
        "Unknown module system."
