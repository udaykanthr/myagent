r"""A stylesheet nothing imports styles nothing.

`find_style_drift` asked "is this class defined in some .css on disk?".
The question that matters is "is it defined in a stylesheet the app
actually LOADS?", and the gap between them is not academic: it let the
smoke test's own repair loop drive itself green by writing rules into
dead code.

Measured 2026-08-20 run 30. Step 12.1 was to "update the React bootstrap
to import the global responsive stylesheet"; the agent edited `App.jsx`
instead of its declared target `main.jsx`, and the step's gate --
`npm run build --silent` -- passes whether or not any CSS is imported.
`client/src/styles/global.css` ended up 7023 bytes that nothing imports,
while `main.jsx` kept Vite's stock `import './index.css'`. The built
bundle contained no `home-hero`, proving it never reached the output.

This check then found the classes BROKEN, handed the list to a repair
loop, and the loop satisfied it by adding the missing rules **to
global.css** -- the unreachable file. It reported "Style coupling
repaired" having changed nothing a user would see. Every gate, both
acceptance instruments and the smoke test passed over an entirely
unstyled application; only the ghost's `violated-import-edge` noticed.
"""

import pytest

from agentchanti.orchestrator.style_coupling import (
    find_style_drift,
    reachable_stylesheets,
)

MARKUP = """
export function HomePage() {
  return <section className="home-hero"><p className="eyebrow">hi</p></section>;
}
"""
RULES = ".home-hero { display: flex; }\n.eyebrow { color: red; }\n"


def _project(tmp_path, entry_imports):
    src = tmp_path / "src"
    (src / "pages").mkdir(parents=True)
    (src / "styles").mkdir()
    (src / "pages" / "HomePage.jsx").write_text(MARKUP)
    (src / "styles" / "global.css").write_text(RULES)
    (src / "index.css").write_text("body { margin: 0; }\n")
    (src / "main.jsx").write_text(entry_imports)
    return str(tmp_path)


def test_the_incident_is_now_caught(tmp_path):
    """global.css defines the classes, but main.jsx imports index.css."""
    root = _project(tmp_path, "import './index.css'\n")
    drift = find_style_drift(root)
    assert drift is not None and drift.broken
    assert "home-hero" in drift.unstyled


def test_the_finding_names_the_real_defect(tmp_path):
    """Telling the reader to write rules that already exist is wrong."""
    root = _project(tmp_path, "import './index.css'\n")
    drift = find_style_drift(root)
    assert drift.defines_them == ["src/styles/global.css"]
    text = drift.describe()
    assert "ARE defined" in text
    assert "nothing imports that stylesheet" in text
    assert "Fix the IMPORT, not the rules" in text


def test_repairing_the_unreachable_file_does_not_go_green(tmp_path):
    """The exact move the repair loop made must no longer satisfy it."""
    root = _project(tmp_path, "import './index.css'\n")
    (tmp_path / "src" / "styles" / "global.css").write_text(
        RULES + ".extra { color: blue; }\n")
    assert find_style_drift(root).broken, "writing into dead CSS is not a fix"


def test_importing_it_is_the_fix(tmp_path):
    root = _project(tmp_path, "import './index.css'\nimport './styles/global.css'\n")
    assert not find_style_drift(root).broken


def test_moving_the_rules_into_the_loaded_file_is_also_a_fix(tmp_path):
    root = _project(tmp_path, "import './index.css'\n")
    (tmp_path / "src" / "index.css").write_text(RULES)
    assert not find_style_drift(root).broken


# --- reachability itself --------------------------------------------

def test_html_link_counts_as_reachable(tmp_path):
    root = _project(tmp_path, "console.log('no css import')\n")
    (tmp_path / "index.html").write_text(
        '<link rel="stylesheet" href="/src/styles/global.css">')
    assert not find_style_drift(root).broken


def test_at_import_chains_are_followed(tmp_path):
    root = _project(tmp_path, "import './index.css'\n")
    (tmp_path / "src" / "index.css").write_text(
        "@import './styles/global.css';\nbody { margin: 0; }\n")
    assert not find_style_drift(root).broken


def test_require_is_recognised_too(tmp_path):
    root = _project(tmp_path, "require('./styles/global.css')\n")
    assert not find_style_drift(root).broken


# --- the guard must not accuse what it cannot see --------------------

def test_no_reachability_signal_means_not_judged_that_way(tmp_path):
    """Some setups inject CSS by means this cannot see.

    Reporting every class unstyled there would be far worse than not
    judging, so with no signal at all the old behaviour stands.
    """
    root = _project(tmp_path, "console.log('nothing imports any css')\n")
    styles = [str(tmp_path / "src" / "styles" / "global.css")]
    markup = [str(tmp_path / "src" / "main.jsx")]
    assert reachable_stylesheets(root, styles, markup) is None
    # ...and the caller therefore judges against every stylesheet, as before.
    assert not find_style_drift(root).broken


def test_no_stylesheets_at_all_is_none(tmp_path):
    assert reachable_stylesheets(str(tmp_path), [], []) is None


def test_a_healthy_project_stays_silent(tmp_path):
    root = _project(tmp_path, "import './styles/global.css'\n")
    drift = find_style_drift(root)
    assert not drift.broken
    assert drift.defines_them == []
