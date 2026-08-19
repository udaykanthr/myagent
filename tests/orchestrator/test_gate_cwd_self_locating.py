r"""Two callers ask "does this command locate itself?" — they must agree.

`_declared_verify_cmd` decides whether to ADD a `cd {sub}` prefix;
`declared_gate_cwd` decides whether to hand the gate `cwd={sub}`. Both
are the same question, and `declared_gate_cwd`'s docstring has always
claimed they cannot disagree.

They did. `declared_gate_cwd` recognised only a leading `cd `, so a gate
that names the sub-project without cd-ing to it --

    npm --prefix frontend test -- --run && npm --prefix backend test -- --run

-- was launched with cwd=frontend, where `--prefix frontend` resolves to
`frontend/frontend`. Measured 2026-08-19: the gate died four times
(0xFFFFF026) and BulkTest logged "Plan-declared gate did not pass",
demoting a correct gate to the framework default — the exact
substitution that preflight exists to prevent. The identical command
passed from the repo root immediately before and after.

They now share `references_subproject`, so the divergence cannot recur.
"""

import pytest

from agentchanti.orchestrator.pipeline import declared_gate_cwd
from agentchanti.orchestrator.step_handlers import references_subproject

INCIDENT = ("npm --prefix frontend test -- --run "
            "&& npm --prefix backend test -- --run")


def test_the_incident_gate_runs_at_the_repo_root():
    assert declared_gate_cwd(INCIDENT, "frontend") is None


@pytest.mark.parametrize("cmd", [
    INCIDENT,
    "cd frontend && npm test",
    "cd ./frontend && npm test",
    "node -e \"require('./frontend/package.json')\"",
    "python -m unittest discover -s frontend",
    "npm --prefix frontend run build",
])
def test_self_locating_commands_get_no_cwd(cmd):
    assert declared_gate_cwd(cmd, "frontend") is None, cmd


@pytest.mark.parametrize("cmd", [
    "npm test -- --run",
    "npx vitest run",
    "npm run build",
])
def test_commands_needing_the_subproject_still_get_it(cmd):
    """The guard must stay narrow — this is what the cwd is FOR."""
    assert declared_gate_cwd(cmd, "frontend") == "frontend", cmd


def test_a_similarly_named_directory_is_not_a_reference():
    assert declared_gate_cwd("npm --prefix myfrontend test",
                             "frontend") == "frontend"


@pytest.mark.parametrize("sub", [None, ""])
def test_no_subproject_means_repo_root(sub):
    assert declared_gate_cwd("npm test", sub) in (None, "")


def test_empty_command_keeps_previous_behaviour():
    assert declared_gate_cwd("", "frontend") == "frontend"


# ─── the shared predicate itself ─────────────────────────────────────

@pytest.mark.parametrize("cmd,sub,expected", [
    (INCIDENT, "frontend", True),
    ("cd frontend && npm test", "frontend", True),
    ("node -e \"require('./frontend/package.json')\"", "frontend", True),
    ("python -m unittest discover -s game", "game", True),
    ("npm test -- --run", "frontend", False),
    ("npm --prefix myfrontend test", "frontend", False),
    ("npx vitest run", "frontend", False),
    ("", "frontend", False),
    ("npm test", "", False),
    ("npm test", None, False),
])
def test_references_subproject(cmd, sub, expected):
    assert references_subproject(cmd, sub) is expected


def test_both_callers_share_one_definition():
    """A regression here would let the two drift apart again."""
    for cmd in (INCIDENT, "cd frontend && npm test", "npm test -- --run"):
        shares = references_subproject(cmd, "frontend")
        cd_own = cmd.lstrip().lower().startswith("cd ")
        assert (declared_gate_cwd(cmd, "frontend") is None) == (shares or cd_own)
