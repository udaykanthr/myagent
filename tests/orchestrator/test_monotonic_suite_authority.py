"""A suite outranks a gate only if it ran, and only over what it covers.

`_green_suites_contradicting` makes a strong claim on the suite's behalf
— "the suite encodes the task's stated invariants, so the gate is the
suspect, not the code" — and on 2026-08-19 it made that claim for
`cd frontend && npm run test -- --run`, whose script the agent had
written as `vitest --passWithNoTests --run` and which found no test files
at all. It overruled a real gate on `backend/services/authValidation.js`,
a file outside anything it could have run, and told the reader to go fix
a `verify:` line that was correct.
"""

import pytest

from agentchanti.orchestrator.cli import (
    _gate_scope,
    _green_suites_contradicting,
    _suite_covers,
)
from agentchanti.orchestrator.wave_snapshots import get_gate_ledger

# The run's own two gates, verbatim.
SUITE = "cd frontend && npm run test -- --run"
BACKEND_GATE = ("node -e \"const v=require('./backend/services/"
                "authValidation'); if(!v.validateSignupInput)process.exit(1)\"")
FRONTEND_GATE = "cd frontend && npm run build"

VITEST_EMPTY = ("RUN  v4.1.11\n\nNo test files found, exiting with code 0\n"
                "include: **/*.{test,spec}.?(c|m)[jt]s?(x)\n")
VITEST_REAL = "Test Files  2 passed (2)\n      Tests  9 passed (9)\n"


@pytest.fixture(autouse=True)
def _clean_ledger():
    get_gate_ledger().reset()
    yield
    get_gate_ledger().reset()


def _seed(cmd, label, output):
    led = get_gate_ledger()
    led.record(cmd, label)
    led._last_output[cmd] = output


# ─── the incident ────────────────────────────────────────────────────

def test_empty_suite_cannot_overrule_a_gate():
    """The exact verdict that shipped. Both new conditions fail here."""
    _seed(SUITE, "3.1", VITEST_EMPTY)
    _seed(BACKEND_GATE, "4.2", "AssertionError")
    assert _green_suites_contradicting(
        [(BACKEND_GATE, "4.2", "exports missing")]) == []


def test_a_real_suite_still_cannot_overrule_a_gate_it_cannot_reach():
    """Even with 9 passing tests, a frontend suite says nothing about backend."""
    _seed(SUITE, "3.1", VITEST_REAL)
    _seed(BACKEND_GATE, "4.2", "AssertionError")
    assert _green_suites_contradicting(
        [(BACKEND_GATE, "4.2", "exports missing")]) == []


def test_a_real_suite_does_overrule_a_gate_inside_its_scope():
    _seed(SUITE, "3.1", VITEST_REAL)
    _seed(FRONTEND_GATE, "7.1", "ok")
    assert _green_suites_contradicting(
        [(FRONTEND_GATE, "7.1", "build asserted")]) == [(SUITE, "3.1")]


# ─── the protection this must not lose ───────────────────────────────

def test_the_pacman_case_is_preserved():
    """The incident this rule was written for: a root suite vs an inline gate.

    Both run at the repo root, and the suite ran 12 tests, so it keeps
    every bit of its authority.
    """
    suite = "python -m unittest discover -v"
    gate = 'python -c "from game import Player; assert Player().can_move()"'
    _seed(suite, "5.1", "Ran 12 tests in 0.3s\n\nOK\n")
    _seed(gate, "3.1", "AssertionError")
    assert _green_suites_contradicting(
        [(gate, "3.1", "AssertionError")]) == [(suite, "5.1")]


def test_a_red_suite_never_contradicts():
    suite = "python -m unittest discover -v"
    _seed(suite, "5.1", "FAILED (failures=2)")
    assert _green_suites_contradicting([(suite, "5.1", "boom")]) == []


def test_no_suite_recorded_is_empty():
    _seed(BACKEND_GATE, "4.2", "AssertionError")
    assert _green_suites_contradicting(
        [(BACKEND_GATE, "4.2", "boom")]) == []


def test_a_suite_that_never_ran_is_not_trusted_blindly():
    """No recorded output means no evidence it collected anything.

    An absent output reads as empty rather than as a pass, which keeps
    the safe answer safe.
    """
    led = get_gate_ledger()
    led.record("python -m unittest discover", "5.1")
    gate = 'python -c "assert False"'
    led.record(gate, "3.1")
    # unittest at root covers the gate, and no output means no empty-marker
    # match — the rule falls back to the original behaviour.
    assert _green_suites_contradicting([(gate, "3.1", "x")]) == [
        ("python -m unittest discover", "5.1")]


# ─── the scope primitives ────────────────────────────────────────────

@pytest.mark.parametrize("cmd,expected", [
    ("cd frontend && npm run test -- --run", "frontend"),
    ("cd ./frontend && npm test", "frontend"),
    ("cd frontend/app && npm test", "frontend/app"),
    ("npm test", ""),
    ("python -m unittest discover", ""),
    ("", ""),
])
def test_gate_scope(cmd, expected):
    assert _gate_scope(cmd) == expected


@pytest.mark.parametrize("suite,gate,covers", [
    ("python -m unittest discover", "python -c 'assert x'", True),
    ("cd frontend && npm test", "cd frontend && npm run build", True),
    ("cd frontend && npm test", "cd frontend/sub && node x.js", True),
    ("cd frontend && npm test", "node -e \"require('./backend/x')\"", False),
    ("cd frontend && npm test", "cd backend && npm test", False),
])
def test_suite_covers(suite, gate, covers):
    assert _suite_covers(suite, gate) is covers


@pytest.mark.parametrize("out", [
    "No test files found, exiting with code 0",
    "no tests ran in 0.01s",
    "collected 0 items",
    "No tests found, exiting with code 0",
    "Ran 0 tests in 0.000s",
    "ok  example.com/pkg [no test files]",
])
def test_empty_suite_outputs_are_recognised(out):
    _seed(SUITE, "3.1", out)
    _seed(BACKEND_GATE, "4.2", "x")
    assert _green_suites_contradicting([(BACKEND_GATE, "4.2", "x")]) == []
