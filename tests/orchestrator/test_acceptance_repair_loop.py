r"""Retrying a failed acceptance check, and staying honest about it.

`acceptance_cmds` used to run once, at the very end, with nothing after
it: a plan that never built what the contract requires failed the run
outright, after every one of its own gates had passed. Measured
2026-08-19 run 29 -- 765k tokens and 13 minutes to discover on the final
line that the backend had no protected endpoint, a fact already settled
at 23:42 when the plan chose `randomUUID()` sessions and never installed
a JWT library.

"Until green" is bounded twice on purpose, because an unbounded loop
against a contract nothing can satisfy would spend the whole budget
proving it: a round cap, and a progress requirement that stops when a
round leaves the failure byte-identical.
"""

import pytest

from agentchanti.orchestrator import agent_loop as _al
from agentchanti.orchestrator.evidence import (
    INDEPENDENT_ACCEPTANCE,
    Evidence,
    repair_failed_acceptance,
)

CMDS = ["node acceptance_check.cjs"]


class _Recorder:
    """Stands in for run_recovery_loop; counts how often it is asked."""

    def __init__(self):
        self.calls = []

    def __call__(self, client, tools, **kw):
        self.calls.append({"client": client, **kw})
        return True, "did something"


@pytest.fixture
def recovery(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr(_al, "run_recovery_loop", rec)
    monkeypatch.setattr(_al, "build_step_tools",
                        lambda *a, **k: object())
    return rec


def _acceptance(sequence):
    """A fake acceptance runner returning each (passed, failures) in turn."""
    seq = list(sequence)

    def run(executor, cmds):
        return seq.pop(0) if seq else (False, ["still failing"])
    return run


def test_a_repair_that_works_reports_the_round_count(recovery):
    passed, rounds, failures = repair_failed_acceptance(
        executor=None, cmds=CMDS, failures=["no endpoint"],
        llm_client=object(), memory=None, task="t",
        run_acceptance=_acceptance([(True, [])]))
    assert (passed, rounds, failures) == (True, 1, [])
    assert len(recovery.calls) == 1


def test_it_keeps_going_until_green(recovery):
    """Two failed rounds then a pass — the point of the feature."""
    passed, rounds, _ = repair_failed_acceptance(
        executor=None, cmds=CMDS, failures=["fail A"],
        llm_client=object(), memory=None, task="t", max_rounds=5,
        run_acceptance=_acceptance([(False, ["fail B"]),
                                    (False, ["fail C"]),
                                    (True, [])]))
    assert passed is True
    assert rounds == 3
    assert len(recovery.calls) == 3


def test_the_round_cap_is_honoured(recovery):
    passed, rounds, failures = repair_failed_acceptance(
        executor=None, cmds=CMDS, failures=["f0"],
        llm_client=object(), memory=None, task="t", max_rounds=2,
        run_acceptance=_acceptance([(False, ["f1"]), (False, ["f2"])]))
    assert passed is False
    assert rounds == 2
    assert len(recovery.calls) == 2
    assert failures == ["f2"]


def test_an_unchanging_failure_stops_early(recovery):
    """The progress requirement: a check that never moves is not measuring.

    Without this, "until green" against an unsatisfiable contract burns
    every remaining round proving it.
    """
    passed, rounds, _ = repair_failed_acceptance(
        executor=None, cmds=CMDS, failures=["identical"],
        llm_client=object(), memory=None, task="t", max_rounds=5,
        run_acceptance=_acceptance([(False, ["identical"]),
                                    (False, ["identical"])]))
    assert passed is False
    assert len(recovery.calls) < 5, "should not have used the whole budget"
    assert rounds < 5


def test_zero_rounds_disables_the_feature(recovery):
    passed, rounds, failures = repair_failed_acceptance(
        executor=None, cmds=CMDS, failures=["f"], llm_client=object(),
        memory=None, task="t", max_rounds=0,
        run_acceptance=_acceptance([(True, [])]))
    assert (passed, rounds, failures) == (False, 0, ["f"])
    assert recovery.calls == []


def test_no_llm_client_is_not_a_crash(recovery):
    passed, rounds, _ = repair_failed_acceptance(
        executor=None, cmds=CMDS, failures=["f"], llm_client=None,
        memory=None, task="t")
    assert (passed, rounds) == (False, 0)
    assert recovery.calls == []


def test_the_final_round_gets_the_stronger_model(recovery):
    weak, strong = object(), object()
    repair_failed_acceptance(
        executor=None, cmds=CMDS, failures=["f0"], llm_client=weak,
        memory=None, task="t", max_rounds=2, escalation_client=strong,
        run_acceptance=_acceptance([(False, ["f1"]), (False, ["f2"])]))
    assert recovery.calls[0]["client"] is weak
    assert recovery.calls[-1]["client"] is strong


def test_the_model_is_told_it_may_not_edit_the_check(recovery):
    """The refusal is what makes this loop safe; the prompt must say so."""
    repair_failed_acceptance(
        executor=None, cmds=CMDS, failures=["f"], llm_client=object(),
        memory=None, task="t", run_acceptance=_acceptance([(True, [])]))
    step_text = recovery.calls[0]["step_text"]
    assert "not edit it" in step_text
    assert "read it" in step_text.lower()
    assert "Change the PROJECT" in step_text
    assert CMDS[0] in step_text


def test_the_failure_output_reaches_the_model(recovery):
    repair_failed_acceptance(
        executor=None, cmds=CMDS, failures=["no authenticated endpoint found"],
        llm_client=object(), memory=None, task="t",
        run_acceptance=_acceptance([(True, [])]))
    assert "no authenticated endpoint" in recovery.calls[0]["error_info"]


# --- the banner must not overstate a repaired pass -------------------

def test_a_clean_pass_and_a_repaired_pass_read_differently():
    clean = Evidence(True, INDEPENDENT_ACCEPTANCE, "2 passed")
    fixed = Evidence(True, INDEPENDENT_ACCEPTANCE, "2 passed", repaired=2)
    assert clean.headline != fixed.headline
    assert "2 repair round(s)" in fixed.headline
    assert "2 repair round(s)" in fixed.log_line()
    assert "repair" not in clean.log_line()


def test_a_repaired_pass_is_still_independent():
    """The check was never edited and really did run, so it still counts."""
    fixed = Evidence(True, INDEPENDENT_ACCEPTANCE, "2 passed", repaired=3)
    assert fixed.independent is True
