"""`PKG_PRESENT` describes the manifest's own environment, not the root's.

Node resolves dependencies from the directory the `package.json` lives
in. Measured 2026-08-19 on a two-root repo: `backend/node_modules` held
all 101 packages and `frontend/node_modules` all 93, yet both manifests
were checked against the repo root, both reported VIOLATED, and the
healer then ran `npm install` at the top level — leaving a `package.json`,
a lockfile and a 107-package `node_modules` belonging to no project.
"""

import json
import os

import pytest

from agentchanti.orchestrator.ghost import (
    HOLDS,
    INAPPLICABLE,
    UNKNOWN,
    VIOLATED,
    _check_packages,
)
from agentchanti.orchestrator.ghost_heal import _missing_node_deps

BACKEND = json.dumps({"dependencies": {"express": "^5", "cors": "^2"}})
FRONTEND = json.dumps({"dependencies": {"react": "^19", "react-dom": "^19"}})


def _tree(root, spec):
    """spec: {dir: [installed package names]}"""
    for d, pkgs in spec.items():
        nm = os.path.join(root, d, "node_modules") if d else os.path.join(root, "node_modules")
        os.makedirs(nm, exist_ok=True)
        for p in pkgs:
            os.makedirs(os.path.join(nm, *p.split("/")), exist_ok=True)


def test_installed_beside_its_manifest_holds(tmp_path):
    root = str(tmp_path)
    _tree(root, {"backend": ["express", "cors"], "frontend": ["react", "react-dom"]})
    assert _check_packages(root, "backend/package.json", BACKEND)[0] == HOLDS
    assert _check_packages(root, "frontend/package.json", FRONTEND)[0] == HOLDS


def test_the_incident_is_not_reported_violated(tmp_path):
    """Both sub-projects correct, repo root empty — the measured state."""
    root = str(tmp_path)
    _tree(root, {"backend": ["express", "cors"], "frontend": ["react", "react-dom"]})
    for manifest, text in (("backend/package.json", BACKEND),
                           ("frontend/package.json", FRONTEND)):
        verdict, why = _check_packages(root, manifest, text)
        assert verdict == HOLDS, f"{manifest}: {verdict} — {why}"


def test_a_genuinely_missing_dep_is_still_caught(tmp_path):
    root = str(tmp_path)
    _tree(root, {"backend": ["express"]})
    verdict, why = _check_packages(root, "backend/package.json", BACKEND)
    assert verdict == VIOLATED
    assert "cors" in why


def test_root_node_modules_does_not_satisfy_a_subproject(tmp_path):
    """The root having it is not the app having it — Node looks in backend/."""
    root = str(tmp_path)
    _tree(root, {"backend": [], "": ["express", "cors"]})
    assert _check_packages(root, "backend/package.json", BACKEND)[0] == VIOLATED


def test_no_node_modules_beside_the_manifest_is_unknown(tmp_path):
    root = str(tmp_path)
    os.makedirs(os.path.join(root, "backend"))
    assert _check_packages(root, "backend/package.json", BACKEND)[0] == UNKNOWN


def test_single_root_project_is_unchanged(tmp_path):
    root = str(tmp_path)
    _tree(root, {"": ["express", "cors"]})
    assert _check_packages(root, "package.json", BACKEND)[0] == HOLDS


def test_no_dependencies_is_inapplicable(tmp_path):
    root = str(tmp_path)
    assert _check_packages(root, "package.json", '{"name":"x"}')[0] == INAPPLICABLE


# ─── the healer reads the same directory ─────────────────────────────

def test_healer_sees_nothing_missing_when_installed_beside_manifest(tmp_path):
    root = str(tmp_path)
    _tree(root, {"backend": ["express", "cors"]})
    assert _missing_node_deps(os.path.join(root, "backend"), BACKEND) == []


def test_healer_reports_only_the_truly_missing(tmp_path):
    root = str(tmp_path)
    _tree(root, {"backend": ["express"]})
    assert _missing_node_deps(os.path.join(root, "backend"), BACKEND) == ["cors"]


def test_scoped_package_names_resolve(tmp_path):
    root = str(tmp_path)
    text = json.dumps({"dependencies": {"@scope/pkg": "^1"}})
    _tree(root, {"backend": ["@scope/pkg"]})
    assert _check_packages(root, "backend/package.json", text)[0] == HOLDS
    assert _missing_node_deps(os.path.join(root, "backend"), text) == []
