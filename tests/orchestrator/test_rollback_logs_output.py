"""A rollback must say WHY, not just which gate went red.

`_enforce_monotonic_gates` performs the most destructive action in the
pipeline — it discards a whole wave of work and fails the run — and until
now logged only the failing gate's command. `GateLedger.recheck` has
always captured the failing output (`(out or "")[-1500:]`); `_names`
dropped it, so a reader saw `output=810 chars` in the executor line and
nothing else.

Measured 2026-08-19: two separate investigations of a rolled-back run
(runs 18 and 20) could not determine the cause from the log, and the
rollback had already deleted the files needed to reproduce it by hand.
"""

import logging

import pytest

from agentchanti.orchestrator import cli


class _Snapshots:
    managed = True

    def __init__(self):
        self.rolled_back = False

    def commit_wave(self, stage): pass
    def mark_green(self): pass

    def rollback_to_last(self):
        self.rolled_back = True
        return True, "ok"


@pytest.fixture
def ledger_with_regression(monkeypatch):
    FAILURE = ("AssertionError [ERR_ASSERTION]: Expected /health to return "
               "200, got 404\n    at Object.<anonymous>")

    class _Ledger:
        def gates(self): return {"node -e \"...\"": "5.1"}
        def recheck(self, executor, timeout=300):
            return [("node -e \"...\"", "5.1", FAILURE)]

    monkeypatch.setattr(cli, "get_gate_ledger", lambda: _Ledger(),
                        raising=False)
    monkeypatch.setattr("agentchanti.orchestrator.wave_snapshots"
                        ".get_gate_ledger", lambda: _Ledger())
    return FAILURE


def test_the_failing_output_reaches_the_log(caplog, ledger_with_regression):
    snaps = _Snapshots()
    with caplog.at_level(logging.WARNING):
        ok = cli._enforce_monotonic_gates(snaps, executor=None, stage="wave 8")
    assert ok is False
    assert snaps.rolled_back
    text = caplog.text
    assert "Expected /health to return 200, got 404" in text, text
    assert "step 5.1 output" in text


def test_the_gate_name_is_still_logged(caplog, ledger_with_regression):
    with caplog.at_level(logging.WARNING):
        cli._enforce_monotonic_gates(_Snapshots(), executor=None,
                                     stage="wave 8")
    assert "left 1 gate(s) red" in caplog.text


def test_an_empty_output_says_so_rather_than_logging_nothing(caplog,
                                                             monkeypatch):
    class _Ledger:
        def gates(self): return {"cmd": "3.1"}
        def recheck(self, executor, timeout=300):
            return [("cmd", "3.1", "")]

    monkeypatch.setattr(cli, "get_gate_ledger", lambda: _Ledger(),
                        raising=False)
    monkeypatch.setattr("agentchanti.orchestrator.wave_snapshots"
                        ".get_gate_ledger", lambda: _Ledger())
    with caplog.at_level(logging.WARNING):
        cli._enforce_monotonic_gates(_Snapshots(), executor=None,
                                     stage="wave 2")
    assert "(no output captured)" in caplog.text
