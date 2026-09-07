r"""A guard whose precondition the agent can write is not a guard.

`rootless_npm_install_reason` refuses `npm install <pkg>` at a root that
owns no manifest, with an escape hatch: if the root really IS a package,
the install is fine. That hatch was agent-writable.

Measured 2026-08-19 run 28. Step 5.1's gate did `require('jsonwebtoken')`
from the repo root, where it cannot resolve -- the package lives in
`backend/node_modules`. Everything upstream got it right:

    gate STALLED - the gate is not measuring the artifact
    NOT escalating - the gate is the defect, not the code
    refused rootless npm install: npm install jsonwebtoken --no-save

On the very next turn the agent wrote a root `package.json` with
`write_file`, and the recovery loop then ran `npm install jsonwebtoken
--save`, which the npm guard permitted because a root manifest now
existed.

The harm was real and nothing reported it: the stray manifest shadowed
the frontend's, turning `[SmokeTest] JS build check: npm run build
(cwd=frontend)` into `No build script in package.json - skipping`. Both
acceptance instruments still passed -- neither looks at the repo root.
"""

import json
import os

import pytest

from agentchanti.agent_tools import (
    phantom_root_manifest_reason,
    rootless_npm_install_reason,
)

# Verbatim from the incident. Hand-authored, not npm-generated: `npm init
# -y` names the directory and emits version/main/scripts.
INCIDENT_BODY = json.dumps({
    "name": "fullstack-auth-project",
    "private": True,
    "dependencies": {"jsonwebtoken": "^9.0.3"},
})


def _multiroot(tmp_path):
    """backend/ and frontend/ own manifests; the root owns none."""
    for sub in ("backend", "frontend"):
        d = tmp_path / sub
        d.mkdir()
        (d / "package.json").write_text('{"name": "%s"}' % sub)
    return str(tmp_path)


def test_the_incident_write_is_refused(tmp_path):
    why = phantom_root_manifest_reason("package.json", _multiroot(tmp_path))
    assert why is not None
    assert "backend" in why and "frontend" in why
    # It must name the real defect, not just say no.
    assert "defect in the GATE" in why
    assert "npm --prefix backend" in why


def test_the_escape_hatch_chain_is_closed(tmp_path):
    """The write is what made the refused install legal one turn later."""
    root = _multiroot(tmp_path)
    cmd = "npm install jsonwebtoken --save"

    assert rootless_npm_install_reason(cmd, root) is not None   # turn 8
    assert phantom_root_manifest_reason("package.json", root) is not None

    # Had the write gone through, the install would have been permitted.
    (tmp_path / "package.json").write_text(INCIDENT_BODY)
    assert rootless_npm_install_reason(cmd, root) is None


@pytest.mark.parametrize("spelling", [
    "package.json", "./package.json", "/package.json", ".\\package.json",
])
def test_path_spellings(spelling, tmp_path):
    assert phantom_root_manifest_reason(spelling, _multiroot(tmp_path))


# --- the guard must stay narrow -------------------------------------

def test_a_declared_root_manifest_is_allowed(tmp_path):
    """A workspaces root someone PLANNED is a decision, not a workaround."""
    assert phantom_root_manifest_reason(
        "package.json", _multiroot(tmp_path),
        declared={"package.json", "backend/src/app.js"}) is None


def test_a_sub_project_manifest_is_untouched(tmp_path):
    root = _multiroot(tmp_path)
    for p in ("backend/package.json", "frontend/package.json",
              "packages/ui/package.json"):
        assert phantom_root_manifest_reason(p, root) is None, p


def test_an_existing_root_package_may_be_rewritten(tmp_path):
    root = _multiroot(tmp_path)
    (tmp_path / "package.json").write_text('{"name": "real"}')
    assert phantom_root_manifest_reason("package.json", root) is None


def test_greenfield_root_is_left_alone(tmp_path):
    """No sibling manifests: an ordinary single-package project."""
    assert phantom_root_manifest_reason("package.json", str(tmp_path)) is None


def test_other_root_files_are_not_this_guards_business(tmp_path):
    root = _multiroot(tmp_path)
    for p in ("README.md", "package-lock.json", "tsconfig.json", "app.js"):
        assert phantom_root_manifest_reason(p, root) is None, p


def test_node_modules_is_not_a_sub_project(tmp_path):
    nm = tmp_path / "node_modules" / "left-pad"
    nm.mkdir(parents=True)
    (nm / "package.json").write_text("{}")
    assert phantom_root_manifest_reason("package.json", str(tmp_path)) is None


def test_a_missing_root_is_not_an_error(tmp_path):
    assert phantom_root_manifest_reason(
        "package.json", str(tmp_path / "nope")) is None


# --- the write path actually refuses --------------------------------

def test_write_file_refuses_the_incident(tmp_path):
    from agentchanti.agent_tools import AgentTools

    root = _multiroot(tmp_path)
    tools = AgentTools(project_root=root)

    out = tools._tool_write_file(path="package.json", content=INCIDENT_BODY)
    assert out.startswith("ERROR"), out
    assert not os.path.exists(os.path.join(root, "package.json"))

    # A NEW sub-project manifest through the same call still writes.
    # (An existing one is refused by a different, older guard -- overwriting
    # a manifest the project already has is its own hazard.)
    ok = tools._tool_write_file(path="services/api/package.json",
                                content='{"name": "api"}')
    assert not ok.startswith("ERROR"), ok
    assert os.path.exists(os.path.join(root, "services", "api", "package.json"))
