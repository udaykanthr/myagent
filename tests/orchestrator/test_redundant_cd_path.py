r"""A gate that enters a directory and then repeats it in its paths.

Measured 2026-08-19 run 15::

    cd frontend && npm run build
      && findstr /c:"aria-label" frontend\src\components\Navigation.jsx >nul
      && ...

From inside `frontend/` those resolve to `frontend/frontend/src/...`, so
no output of the step could satisfy the gate. It failed six times over
five rewrites. `observe_gate_verdict` correctly called it stalled and
suppressed the escalation — but by then the agent had done what an
unsatisfiable gate always invites and CREATED the paths it named:
`frontend/frontend/src/pages/{Dashboard,ForgotPassword,Signup}Page.jsx`,
all reported as unplanned writes. That is the fourth recorded instance
of manufacture-the-path, after frontend/frontend/package.json,
frontend/node.cmd and frontend/backend/.env.example.
"""

import pytest

from agentchanti.orchestrator.gate_integrity import (
    platform_equivalent_variants,
    redundant_cd_path_variant,
)

BS = chr(92)
INCIDENT = ('cd frontend && npm run build && findstr /c:"aria-label" '
            'frontend' + BS + 'src' + BS + 'components' + BS
            + 'Navigation.jsx >nul')


def test_the_incident_is_corrected():
    out = redundant_cd_path_variant(INCIDENT)
    assert out is not None
    assert 'frontend' + BS + 'src' not in out
    assert 'src' + BS + 'components' + BS + 'Navigation.jsx' in out
    # the planner's own cd is kept — `npm run build` needs it
    assert out.startswith("cd frontend && npm run build")


@pytest.mark.parametrize("gate,expected_fragment", [
    ("cd frontend && cat frontend/src/App.jsx", "cat src/App.jsx"),
    ("cd frontend && cat ./frontend/src/App.jsx", "cat src/App.jsx"),
    ("cd app && node app/index.js", "node index.js"),
])
def test_every_spelling_of_the_repetition(gate, expected_fragment):
    assert expected_fragment in redundant_cd_path_variant(gate)


@pytest.mark.parametrize("gate", [
    "cd frontend && npm run build",              # nothing repeated
    "cd frontend && cat myfrontend/src/App.jsx",  # different directory
    "cd frontend && echo frontend",               # a bare word, not a path
    "npm --prefix frontend run build",            # no cd at all
    "",
])
def test_silent_when_there_is_no_repetition(gate):
    assert redundant_cd_path_variant(gate) is None, gate


def test_it_is_offered_as_a_variant():
    """So the loop tries it on the FIRST failure, not after six."""
    reasons = dict((r, v) for r, v in platform_equivalent_variants(INCIDENT))
    assert "redundant-cd-path" in reasons
    assert 'frontend' + BS + 'src' not in reasons["redundant-cd-path"]


def test_a_correct_gate_offers_no_such_variant():
    reasons = [r for r, _ in
               platform_equivalent_variants("cd frontend && npm run build")]
    assert "redundant-cd-path" not in reasons
