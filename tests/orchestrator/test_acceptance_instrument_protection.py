"""The acceptance instrument is read-only to the agent.

`acceptance_cmds` is described throughout this project as the one check
the model neither wrote nor CAN edit, and so the only one allowed to fail
a run on its own. The first half was always true — the command string
lives in config. The second half was never enforced for a command that
invokes a FILE: the script sat in the project root like any other source.

Observed 2026-08-19 run 8: a planner emitted `target: acceptance_check.cjs`
on a TEST step. It behaved — the step said "run the supplied unchanged
acceptance checker" and the bytes were identical afterwards — but nothing
made that the only possible outcome, and a run that rewrites its own
acceptance check reports independent evidence for a contract it authored.
"""

import os

import pytest

from agentchanti.agent_tools import AgentTools
from agentchanti.llm.chat_types import ToolCall
from agentchanti.orchestrator.evidence import acceptance_instrument_files


# ─── which files are instruments ─────────────────────────────────────

def test_names_the_script_and_ignores_the_rest(tmp_path):
    (tmp_path / "acceptance_check.cjs").write_text("// check")
    (tmp_path / "frontend").mkdir()
    got = acceptance_instrument_files(
        ["node acceptance_check.cjs", "npm --prefix frontend run build"],
        str(tmp_path))
    assert got == {"acceptance_check.cjs"}


def test_a_path_that_does_not_exist_is_not_protected(tmp_path):
    """Protecting a phantom would block the run creating an ordinary file."""
    assert acceptance_instrument_files(
        ["node acceptance_check.cjs"], str(tmp_path)) == set()


def test_leading_dot_slash_normalises(tmp_path):
    (tmp_path / "check.js").write_text("x")
    assert acceptance_instrument_files(
        ["node ./check.js"], str(tmp_path)) == {"check.js"}


def test_a_dotfile_is_not_mangled_by_normalisation(tmp_path):
    (tmp_path / ".env").write_text("X=1")
    assert acceptance_instrument_files(
        ["cat ./.env"], str(tmp_path)) == {".env"}


def test_nested_and_quoted_paths(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "contract.py").write_text("x")
    assert acceptance_instrument_files(
        ['pytest "tests/contract.py"'], str(tmp_path)) == {"tests/contract.py"}


def test_flags_are_never_instruments(tmp_path):
    (tmp_path / "-v").write_text("x")  # pathological but decidable
    got = acceptance_instrument_files(["pytest -v"], str(tmp_path))
    assert "-v" not in got


def test_no_commands_is_empty(tmp_path):
    assert acceptance_instrument_files(None, str(tmp_path)) == set()
    assert acceptance_instrument_files([], str(tmp_path)) == set()


# ─── the tools refuse to write them ──────────────────────────────────

def _tools(tmp_path, protected=()):
    t = AgentTools(project_root=str(tmp_path))
    t.protect_acceptance_files(protected)
    return t


def test_write_file_is_refused(tmp_path):
    (tmp_path / "acceptance_check.cjs").write_text("ORIGINAL")
    t = _tools(tmp_path, {"acceptance_check.cjs"})
    out = t.execute(ToolCall(name="write_file", id="1", arguments={
        "path": "acceptance_check.cjs", "content": "assert(true)"}))
    assert out.startswith("ERROR")
    assert "acceptance" in out.lower()
    assert (tmp_path / "acceptance_check.cjs").read_text() == "ORIGINAL"


def test_edit_file_is_refused(tmp_path):
    (tmp_path / "acceptance_check.cjs").write_text("ORIGINAL")
    t = _tools(tmp_path, {"acceptance_check.cjs"})
    out = t.execute(ToolCall(name="edit_file", id="1", arguments={
        "path": "acceptance_check.cjs",
        "old_text": "ORIGINAL", "new_text": "WEAKENED"}))
    assert out.startswith("ERROR")
    assert (tmp_path / "acceptance_check.cjs").read_text() == "ORIGINAL"


def test_the_refusal_says_what_to_do_instead(tmp_path):
    (tmp_path / "c.js").write_text("x")
    t = _tools(tmp_path, {"c.js"})
    out = t.execute(ToolCall(name="write_file", id="1", arguments={
        "path": "c.js", "content": "y"}))
    assert "change the PROJECT" in out
    assert "do not edit it" in out


@pytest.mark.parametrize("spelling", [
    "acceptance_check.cjs", "./acceptance_check.cjs",
])
def test_every_spelling_of_the_path_is_refused(tmp_path, spelling):
    (tmp_path / "acceptance_check.cjs").write_text("ORIGINAL")
    t = _tools(tmp_path, {"acceptance_check.cjs"})
    out = t.execute(ToolCall(name="write_file", id="1", arguments={
        "path": spelling, "content": "y"}))
    assert out.startswith("ERROR"), spelling


def test_reading_it_is_still_allowed(tmp_path):
    """The model must be able to see what it has to satisfy."""
    (tmp_path / "acceptance_check.cjs").write_text("// the contract")
    t = _tools(tmp_path, {"acceptance_check.cjs"})
    out = t.execute(ToolCall(name="read_file", id="1",
                             arguments={"path": "acceptance_check.cjs"}))
    assert "the contract" in out
    assert not out.startswith("ERROR")


def test_ordinary_files_are_untouched(tmp_path):
    t = _tools(tmp_path, {"acceptance_check.cjs"})
    out = t.execute(ToolCall(name="write_file", id="1", arguments={
        "path": "backend/app.js", "content": "const app = 1;"}))
    assert not out.startswith("ERROR"), out
    assert (tmp_path / "backend" / "app.js").exists()


def test_no_protection_configured_blocks_nothing(tmp_path):
    t = _tools(tmp_path)
    out = t.execute(ToolCall(name="write_file", id="1", arguments={
        "path": "acceptance_check.cjs", "content": "y"}))
    assert not out.startswith("ERROR"), out
