r"""Aliasing a dependency tree fakes module resolution.

A dependency tree is created by a package manager in the directory that
owns the manifest. Linking one into another location makes resolution
depend on a filesystem alias that survives no clone, copy, archive or
deploy — the project appears to work in place and nowhere else.

Measured 2026-08-19 run 23. A gate ran from the repo root and did a bare
`require('jsonwebtoken')`, which cannot resolve there in a multi-root
layout: the package lives in `backend/node_modules`. Rather than report
the gate as wrongly scoped, the agent ran `mklink /J node_modules
backend\node_modules` and the gate went green. Removing the junction
afterwards left the application working exactly as before — it carried
no weight, existing only to satisfy the measurement.

Fifth recorded instance of manufacture-the-path, after
frontend/frontend/package.json, frontend/node.cmd,
frontend/backend/.env.example and frontend/frontend/src/pages/*.jsx.
"""

import pytest

from agentchanti.agent_tools import AgentTools, fabricated_dependency_link
from agentchanti.llm.chat_types import ToolCall

BS = chr(92)
INCIDENT = "mklink /J node_modules backend" + BS + "node_modules"


def test_the_incident_is_named():
    assert fabricated_dependency_link(INCIDENT) == "node_modules"


@pytest.mark.parametrize("cmd", [
    INCIDENT,
    "mklink /D node_modules .." + BS + "shared" + BS + "node_modules",
    "ln -s backend/node_modules node_modules",
    "ln -sf backend/node_modules ./node_modules",
    "New-Item -ItemType SymbolicLink -Path node_modules -Target backend/node_modules",
    "New-Item -ItemType Junction -Path ./node_modules -Target backend/node_modules",
])
def test_every_link_syntax_is_caught(cmd):
    assert fabricated_dependency_link(cmd) == "node_modules"


@pytest.mark.parametrize("cmd,name", [
    ("ln -s ../shared/site-packages venv/lib/site-packages", "site-packages"),
    ("mklink /J vendor .." + BS + "vendor", "vendor"),
])
def test_other_dependency_dirs_too(cmd, name):
    assert fabricated_dependency_link(cmd) == name


@pytest.mark.parametrize("cmd", [
    "mklink /J shared .." + BS + "common",      # ordinary directory link
    "ln -s ../assets public/assets",
    "npm --prefix backend install express",
    "ln -s ../backend/node_modules/.bin/vitest bin/vitest",  # a file, not the tree
    "",
])
def test_ordinary_commands_are_untouched(cmd):
    assert fabricated_dependency_link(cmd) is None, cmd


def test_run_command_refuses_it_and_says_what_to_do(tmp_path):
    t = AgentTools(project_root=str(tmp_path))
    out = t.execute(ToolCall(name="run_command", id="1",
                             arguments={"command": INCIDENT}))
    assert out.startswith("ERROR")
    assert "node_modules" in out
    assert "--prefix" in out
    assert "the gate is running" in out
