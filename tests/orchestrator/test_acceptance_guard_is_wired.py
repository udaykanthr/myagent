r"""The acceptance guard has to be reachable the way production builds it.

`test_acceptance_instrument_protection.py` calls
`tools.protect_acceptance_files(...)` directly, so it tested the unit
while the wiring was missing. In production nothing ever called it:
`cli.py` set `memory._acceptance_files`, `_acceptance_refusal` read
`self._acceptance_files`, and no code connected the two. The entire
mechanism was inert, and the existing test could not see it.

Demonstrated 2026-08-20 by driving `build_step_tools` and writing the
check: the write was accepted and the frozen contract was overwritten.

Every test here goes through `build_step_tools`, never through
`protect_acceptance_files`, because the seam is the wiring and a test
that reaches past it proves nothing about a real run.

It matters most now that a failing acceptance command is retried. Until
that existed the check ran once, at the very end, with nothing after it
-- so nothing in the run had any incentive to touch it. Retrying until
green makes rewriting the contract the cheapest path to green.
"""

import os

import pytest

from agentchanti.orchestrator.agent_loop import build_step_tools
from agentchanti.orchestrator.memory import FileMemory

BODY = "// the frozen contract\n"


@pytest.fixture
def project(tmp_path):
    (tmp_path / "acceptance_check.cjs").write_text(BODY)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.js").write_text("// app\n")
    mem = FileMemory()
    mem._acceptance_files = {"acceptance_check.cjs"}
    return str(tmp_path), mem


def _tools(project):
    root, mem = project
    return build_step_tools(None, mem, project_root=root), root


def test_the_guard_survives_the_real_construction_path(project):
    tools, _ = _tools(project)
    assert tools._acceptance_files == {"acceptance_check.cjs"}


def test_write_is_refused_and_the_bytes_are_untouched(project):
    tools, root = _tools(project)
    out = tools._tool_write_file(path="acceptance_check.cjs",
                                 content="// REWRITTEN\n")
    assert out.startswith("ERROR")
    assert open(os.path.join(root, "acceptance_check.cjs")).read() == BODY


def test_edit_is_refused_too(project):
    tools, root = _tools(project)
    out = tools._tool_edit_file(path="acceptance_check.cjs",
                                old_text="frozen", new_text="flexible")
    assert out.startswith("ERROR")
    assert open(os.path.join(root, "acceptance_check.cjs")).read() == BODY


def test_reading_stays_allowed(project):
    """The model must be able to see what it has to satisfy."""
    tools, _ = _tools(project)
    assert "frozen contract" in tools._tool_read_file(
        path="acceptance_check.cjs")


@pytest.mark.parametrize("spelling", [
    "acceptance_check.cjs", "./acceptance_check.cjs", ".\\acceptance_check.cjs",
])
def test_path_spellings_are_all_refused(project, spelling):
    tools, _ = _tools(project)
    assert tools._tool_write_file(path=spelling, content="x").startswith("ERROR")


def test_the_refusal_says_what_to_do_instead(project):
    tools, _ = _tools(project)
    out = tools._tool_write_file(path="acceptance_check.cjs", content="x")
    assert "change the PROJECT" in out or "change the PROJECT until it" in out
    assert "do not edit it" in out


def test_ordinary_files_are_unaffected(project):
    tools, root = _tools(project)
    out = tools._tool_write_file(path="src/app.js", content="// new\n")
    assert not out.startswith("ERROR"), out
    assert open(os.path.join(root, "src", "app.js")).read() == "// new\n"


def test_no_acceptance_files_means_no_protection(tmp_path):
    """A run without acceptance_cmds must behave exactly as before."""
    (tmp_path / "whatever.cjs").write_text("x\n")
    tools = build_step_tools(None, FileMemory(), project_root=str(tmp_path))
    assert tools._acceptance_files == set()
    assert not tools._tool_write_file(
        path="whatever.cjs", content="y\n").startswith("ERROR")


def test_a_memory_without_the_attribute_does_not_crash(tmp_path):
    class Bare:
        def update(self, *a, **k): pass

    tools = build_step_tools(None, Bare(), project_root=str(tmp_path))
    assert tools._acceptance_files == set()
