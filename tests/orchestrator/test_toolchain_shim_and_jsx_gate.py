"""Run 11: a gate no code could satisfy, and the agent replacing the tool.

The plan's gate was `cd frontend && node -e "import('./src/context/
AuthContext.jsx').then(...)"`. JSX is not JavaScript and no version of
Node parses it, so no correct React component could ever pass. The step
ran three loops and 30 turns; the agent's eventual fix was to deform the
toolchain rather than the code, writing `frontend/jsx-loader.mjs` and
`frontend/node.cmd` — a shim that shadows the real `node` for anything
run from that directory. The run failed anyway.

Two seams: the gate is refused at plan time, and the shim is refused at
write time.
"""

import pytest

from agentchanti.agent_tools import AgentTools, toolchain_shim
from agentchanti.llm.chat_types import ToolCall
from agentchanti.orchestrator.plan_step import unrunnable_gate_reason

JSX_GATE = ("cd frontend && node -e \"import('./src/context/AuthContext.jsx')"
            ".then(m=>{if(typeof m.useAuth!=='function')process.exit(1)})\"")


# ─── the gate is unrunnable, structurally ────────────────────────────

def test_the_incident_gate_is_refused():
    why = unrunnable_gate_reason(JSX_GATE)
    assert why and "jsx" in why.lower()


@pytest.mark.parametrize("gate", [
    "node -e \"import('./src/App.jsx')\"",
    "node -e \"require('./src/App.jsx')\"",
    "node -e \"import('./ui/Widget.tsx').then(m=>m.default)\"",
    "cd frontend && node --experimental-vm-modules -e \"import('./a.jsx')\"",
])
def test_every_shape_of_node_loading_jsx_is_refused(gate):
    assert unrunnable_gate_reason(gate), gate


@pytest.mark.parametrize("gate", [
    "cd frontend && npx vitest run src/App.test.jsx",
    "node ./node_modules/vitest/vitest.mjs run App.test.jsx",
    "node -e \"require('./backend/app.js')\"",
    "node -e \"const fs=require('fs'); fs.readFileSync('./src/App.jsx','utf8')\"",
    "npm --prefix frontend run build",
])
def test_legitimate_gates_are_untouched(gate):
    """Reading a .jsx as TEXT is fine; so is a transform-aware runner."""
    assert unrunnable_gate_reason(gate) is None, gate


# ─── the shim is refused at write time ───────────────────────────────

@pytest.mark.parametrize("path,tool", [
    ("frontend/node.cmd", "node"),
    ("node.cmd", "node"),
    ("tools/npm.bat", "npm"),
    ("python.exe", "python"),
    ("bin/git", "git"),
    ("pnpm.ps1", "pnpm"),
])
def test_toolchain_names_are_detected(path, tool):
    assert toolchain_shim(path) == tool


@pytest.mark.parametrize("path", [
    "src/node.js",            # a module about node, not an executable
    "components/Node.jsx",    # a React component
    "nodemon.cmd",            # a different tool entirely
    "backend/app.js",
    "docs/python.md",
])
def test_ordinary_files_are_not_shims(path):
    assert toolchain_shim(path) is None


def test_write_file_refuses_the_incident_shim(tmp_path):
    t = AgentTools(project_root=str(tmp_path))
    out = t.execute(ToolCall(name="write_file", id="1", arguments={
        "path": "frontend/node.cmd",
        "content": '@echo off\n"%ProgramFiles%\nodejs\node.exe" %*'}))
    assert out.startswith("ERROR")
    assert "node" in out
    assert not (tmp_path / "frontend" / "node.cmd").exists()


def test_the_refusal_names_the_gate_as_the_defect(tmp_path):
    t = AgentTools(project_root=str(tmp_path))
    out = t.execute(ToolCall(name="write_file", id="1",
                             arguments={"path": "node.cmd", "content": "x"}))
    assert "defect in the GATE" in out
    assert "do not" in out.lower()


def test_ordinary_writes_still_work(tmp_path):
    t = AgentTools(project_root=str(tmp_path))
    out = t.execute(ToolCall(name="write_file", id="1", arguments={
        "path": "frontend/src/components/Node.jsx",
        "content": "export const Node = () => null"}))
    assert not out.startswith("ERROR"), out


def test_edit_file_also_refuses_a_shim(tmp_path):
    """Creation is blocked in write_file; this covers the other direction.

    A shim the project legitimately ships, or one an earlier run left
    behind, is not the agent's to rewrite either.
    """
    (tmp_path / "frontend").mkdir()
    (tmp_path / "frontend" / "node.cmd").write_text("@echo off\nnode.exe %*")
    t = AgentTools(project_root=str(tmp_path))
    out = t.execute(ToolCall(name="edit_file", id="1", arguments={
        "path": "frontend/node.cmd",
        "old_text": "node.exe %*", "new_text": "other.exe %*"}))
    assert out.startswith("ERROR")
    assert "node" in out
    assert "node.exe %*" in (tmp_path / "frontend" / "node.cmd").read_text()


def test_edit_file_still_works_on_ordinary_files(tmp_path):
    (tmp_path / "app.js").write_text("const a = 1;")
    t = AgentTools(project_root=str(tmp_path))
    out = t.execute(ToolCall(name="edit_file", id="1", arguments={
        "path": "app.js", "old_text": "const a = 1;",
        "new_text": "const a = 2;"}))
    assert not out.startswith("ERROR"), out
    assert (tmp_path / "app.js").read_text() == "const a = 2;"
