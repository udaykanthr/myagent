"""A step still writing its targets is expected to fail its own gate.

`observe_gate_verdict` trips on identical failing verdicts across changed
artifact digests, reading that as "the gate is not measuring the
artifact". A multi-file step produces exactly that signature honestly:
until every file the gate reads exists, the failure is a constant ENOENT
naming the missing FILE, while the digest moves with each write.

Measured 2026-08-19 run 19: a step declaring five components was cut
short at `turns=5, write_file: 3`. The verdict was reached three times in
that run; a recovery loop then finished the work and passed the very same
gate, which is proof the gate was fine. That is the false positive the
check's own bias statement warns about — the kind that suppresses real
work — so the observation is withheld until the step has produced
everything it promised.
"""

import os

import pytest

from agentchanti.agent_tools import AgentTools
from agentchanti.orchestrator.agent_loop import _missing_required
from agentchanti.orchestrator.gate_integrity import (
    observe_gate_verdict,
    reset_gate_verdicts,
)


@pytest.fixture(autouse=True)
def _clean():
    try:
        reset_gate_verdicts()
    except Exception:
        pass
    yield


def _tools(tmp_path):
    return AgentTools(project_root=str(tmp_path))


def test_missing_targets_are_detected(tmp_path):
    (tmp_path / "a.jsx").write_text("x")
    t = _tools(tmp_path)
    required = {"a.jsx", "b.jsx", "c.jsx"}
    assert _missing_required(t, required) == ["b.jsx", "c.jsx"]


def test_complete_targets_report_nothing(tmp_path):
    for n in ("a.jsx", "b.jsx"):
        (tmp_path / n).write_text("x")
    assert _missing_required(_tools(tmp_path), {"a.jsx", "b.jsx"}) == []


def test_no_declared_targets_is_not_incomplete(tmp_path):
    """A step declaring nothing must not hold the check off forever."""
    assert _missing_required(_tools(tmp_path), None) == []
    assert _missing_required(_tools(tmp_path), set()) == []


def test_a_path_escaping_the_root_does_not_hold_the_step_open(tmp_path):
    assert _missing_required(_tools(tmp_path), {"../outside.js"}) == []


def test_the_incident_shape_is_gated_off(tmp_path):
    """Three identical ENOENT failures over changing digests.

    Left ungated this is a stall verdict; the loop must not reach it
    while a declared target is still absent.
    """
    (tmp_path / "Navigation.jsx").write_text("Login / Sign Up")
    t = _tools(tmp_path)
    required = {"Navigation.jsx", "ProtectedRoute.jsx"}

    enoent = ("Error: ENOENT: no such file or directory, open "
              "'frontend/src/components/ProtectedRoute.jsx'")
    gate = "node -e \"...readFileSync('ProtectedRoute.jsx')...\""

    # The raw observation WOULD trip — that is the false positive.
    tripped = None
    for digest in ("d1", "d2", "d3"):
        tripped = tripped or observe_gate_verdict(gate, enoent, digest)
    assert tripped is not None, "precondition: this shape does stall"

    # But the step is demonstrably incomplete, which is the guard.
    assert _missing_required(t, required) == ["ProtectedRoute.jsx"]


def test_once_every_target_exists_the_check_is_armed_again(tmp_path):
    for n in ("Navigation.jsx", "ProtectedRoute.jsx"):
        (tmp_path / n).write_text("x")
    assert _missing_required(_tools(tmp_path), {"Navigation.jsx",
                                                "ProtectedRoute.jsx"}) == []
