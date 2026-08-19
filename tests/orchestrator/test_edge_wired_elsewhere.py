"""A declared import edge satisfied by a different file is still satisfied.

The plan names the step that SHOULD consume a module; the code is free
to keep that contract from somewhere else. `export-drift` already
reasons this way -- "a contract with no consumer cannot break anything"
-- and the mirror is that a contract WITH a consumer is kept, wherever
the consumer turned out to live.

Measured 2026-08-19. The plan declared five page components importing
`AppLayout`. The code instead used React Router's layout-route form --
`<Route element={<AppLayout />}>` in App.jsx, with an `<Outlet />`
inside the layout -- which is idiomatic and strictly better than each
page wrapping itself. Every page rendered inside the layout, the build
passed, the acceptance suite passed, and the ghost reported "the import
edge was never wired" over working, well-factored code.

A finding that fires on the better of two correct designs trains the
reader to skip the whole category, which is the real cost.
"""

from agentchanti.orchestrator.ghost import HOLDS, UNKNOWN, VIOLATED, _check_edge

LAYOUT = """
import { Outlet } from 'react-router-dom';
export function AppLayout() { return <div><Outlet /></div>; }
"""

PAGE = """
import { useState } from 'react';
export function LoginPage() { return <form />; }
"""

APP = """
import { AppLayout } from './components/AppLayout.jsx';
export function App() {
  return <Route element={<AppLayout />}><Route index element={<HomePage />} /></Route>;
}
"""

SRC = "frontend/src/components/AppLayout.jsx"


def test_the_incident_edge_holds_when_wired_from_another_file():
    verdict, why = _check_edge(
        SRC, "AppLayout",
        consumers=[("frontend/src/pages/LoginPage.jsx", PAGE)],
        elsewhere=[("frontend/src/App.jsx", APP)],
    )
    assert verdict == HOLDS
    assert "frontend/src/App.jsx" in why


def test_the_step_s_own_file_still_wins_when_it_does_wire_it():
    verdict, why = _check_edge(
        SRC, "AppLayout",
        consumers=[("frontend/src/pages/LoginPage.jsx",
                    "import { AppLayout } from '../components/AppLayout.jsx';")],
        elsewhere=[("frontend/src/App.jsx", APP)],
    )
    assert verdict == HOLDS
    assert "rather than" not in why, "should not claim it moved when it did not"


def test_a_genuinely_unwired_edge_is_still_violated():
    """The check must stay able to fail — this is the whole point of it."""
    verdict, why = _check_edge(
        SRC, "AppLayout",
        consumers=[("frontend/src/pages/LoginPage.jsx", PAGE)],
        elsewhere=[("frontend/src/pages/HomePage.jsx", PAGE),
                   ("frontend/src/main.jsx", "createRoot(el).render(<App />);")],
    )
    assert verdict == VIOLATED
    assert "no other planned file does either" in why


def test_no_elsewhere_keeps_the_old_verdict():
    """Callers that pass nothing must behave exactly as before."""
    assert _check_edge(SRC, "AppLayout",
                       consumers=[("p.jsx", PAGE)])[0] == VIOLATED
    assert _check_edge(SRC, "AppLayout",
                       consumers=[("p.jsx", PAGE)], elsewhere=[])[0] == VIOLATED


def test_unreadable_elsewhere_files_are_skipped_not_matched():
    verdict, _ = _check_edge(
        SRC, "AppLayout",
        consumers=[("frontend/src/pages/LoginPage.jsx", PAGE)],
        elsewhere=[("frontend/src/App.jsx", None)],
    )
    assert verdict == VIOLATED


def test_unknown_still_wins_over_searching_elsewhere():
    """An unreadable consumer is not evidence either way."""
    verdict, _ = _check_edge(
        SRC, "AppLayout",
        consumers=[("a.jsx", PAGE), ("b.jsx", None)],
        elsewhere=[("frontend/src/App.jsx", APP)],
    )
    assert verdict == UNKNOWN


def test_python_module_stem_matches_elsewhere_too():
    verdict, why = _check_edge(
        "game/board.py", "Board",
        consumers=[("game/play.py", "x = 1\n")],
        elsewhere=[("game/main.py", "from game.board import Board\n")],
    )
    assert verdict == HOLDS
    assert "game/main.py" in why


def test_a_substring_is_not_a_match():
    """Word boundaries must survive — this regex was mangled once already."""
    verdict, _ = _check_edge(
        SRC, "AppLayout",
        consumers=[("frontend/src/pages/LoginPage.jsx", PAGE)],
        elsewhere=[("frontend/src/x.jsx", "const MyAppLayoutThing = 1;\n")],
    )
    assert verdict == VIOLATED


def test_the_source_module_never_vouches_for_itself():
    """A module declares the symbols being searched for and owns the stem.

    Counting it as a consumer turns every unwired edge into a pass. The
    first cut of this fix did exactly that, and the pre-existing
    test_unwired_import_edge_is_violated caught it: `board.py` vouched
    for a `main.py` that ignored it entirely.
    """
    verdict, _ = _check_edge(
        "game/board.py", "Board",
        consumers=[("game/main.py", "print('hi')\n")],
        elsewhere=[("game/board.py", "class Board:\n    pass\n")],
    )
    assert verdict == VIOLATED
