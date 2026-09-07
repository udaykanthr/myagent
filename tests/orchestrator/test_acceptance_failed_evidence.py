"""A failing acceptance command is evidence, not the absence of it.

`acceptance_cmds` is described throughout the architecture notes as the
one instrument the model neither wrote nor can edit, and therefore the
only check allowed to fail a run on its own. `classify` handled it
passing, and let it FAILING fall through to the self-authored branch --
whose advice is "Supply `acceptance_cmds` in .agentchanti.yaml",
addressed to a user who had already supplied them, about a run those
very commands had just failed.

Measured 2026-08-19 run 29. The acceptance check correctly failed an
artifact whose login issued a `randomUUID()` token that no route ever
verified (`sessionsByToken` was written and never read, there was no
middleware, and the dashboard page made no API call at all). The run
exited non-zero, which was right, and then printed:

    Evidence: self-authored (self-authored) - every test that passed was
    written during this run - the run marked its own homework. Supply
    `acceptance_cmds` in .agentchanti.yaml ...

Both halves of that explanation were wrong: an independent instrument
had run, and it had disagreed. This mirrors PRE_EXISTING_FAILED, which
already carried the same reasoning one layer down.
"""

import pytest

from agentchanti.orchestrator.evidence import (
    ACCEPTANCE_FAILED,
    INDEPENDENT_ACCEPTANCE,
    NO_TESTS,
    SELF_AUTHORED,
    classify,
)

CMDS = ["node ../test1-acceptance/acceptance_check.cjs",
        "node ../test1-acceptance/frontend_build_check.cjs"]


def _classify(tmp_path, **kw):
    kw.setdefault("snapshot", {})
    kw.setdefault("tests_ran", True)
    return classify(root=str(tmp_path), **kw)


def test_failed_acceptance_gets_its_own_kind(tmp_path):
    e = _classify(tmp_path, acceptance_passed=False, acceptance_cmds=CMDS)
    assert e.kind == ACCEPTANCE_FAILED
    assert e.independent is False


def test_it_does_not_tell_the_user_to_supply_what_they_supplied(tmp_path):
    """The whole point: the old message advised a fix already in place."""
    e = _classify(tmp_path, acceptance_passed=False, acceptance_cmds=CMDS)
    assert "Supply `acceptance_cmds`" not in e.detail
    assert "marked its own homework" not in e.detail


def test_it_says_the_instrument_disagreed(tmp_path):
    e = _classify(tmp_path, acceptance_passed=False, acceptance_cmds=CMDS)
    assert "FAILED" in e.detail
    assert "disagrees" in e.detail
    assert "neither wrote nor can edit" in e.detail
    assert CMDS[0] in e.detail


def test_the_command_count_is_reported(tmp_path):
    e = _classify(tmp_path, acceptance_passed=False, acceptance_cmds=CMDS)
    assert "2 user-supplied" in e.detail


def test_many_commands_are_elided_not_dumped(tmp_path):
    e = _classify(tmp_path, acceptance_passed=False,
                  acceptance_cmds=[f"cmd{i}" for i in range(5)])
    assert "(+3)" in e.detail


# --- the other two states must be untouched -------------------------

def test_passing_acceptance_is_unchanged(tmp_path):
    e = _classify(tmp_path, acceptance_passed=True, acceptance_cmds=CMDS)
    assert e.kind == INDEPENDENT_ACCEPTANCE
    assert e.independent is True


@pytest.mark.parametrize("tests_ran,expected", [
    (True, SELF_AUTHORED),
    (False, NO_TESTS),
])
def test_no_acceptance_commands_is_not_a_failure(tmp_path, tests_ran, expected):
    """None means "none supplied" and must never read as a failure."""
    e = _classify(tmp_path, tests_ran=tests_ran,
                  acceptance_passed=None, acceptance_cmds=[])
    assert e.kind == expected
    assert e.kind != ACCEPTANCE_FAILED


def test_the_supply_advice_still_appears_when_none_were_supplied(tmp_path):
    """The advice is correct in its own case and must survive there."""
    e = _classify(tmp_path, acceptance_passed=None, acceptance_cmds=[])
    assert "Supply `acceptance_cmds`" in e.detail


def test_failure_wins_over_surviving_pre_existing_tests(tmp_path):
    """The stronger instrument decides; it is checked first."""
    (tmp_path / "test_seeded.py").write_text("def test_x():\n    assert True\n")
    e = _classify(tmp_path, acceptance_passed=False, acceptance_cmds=CMDS,
                  survivors_passed=True)
    assert e.kind == ACCEPTANCE_FAILED
