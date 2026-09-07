r"""A missing FILE is not a missing package.

Node says `Cannot find module '<absolute path>'` when a file is missing,
not only when a package is. The old regex matched a package-shaped
PREFIX of the quoted specifier, so `C:\Users\...\run-tests.js` was
truncated to the drive letter and the loop ran `npm install -D C`
(measured 2026-08-19 run 14).

That is the supply-chain hazard `_missing_third_party_module` already
refuses on the Python side — installing a name that happens to exist on
the registry — and single-letter npm packages do exist.
"""

import pytest

from agentchanti.orchestrator.pipeline import (
    _missing_js_packages,
    _npm_package_of,
)

BS = chr(92)
WIN_PATH = "C:" + BS + "Users" + BS + "u" + BS + "backend" + BS + "run-tests.js"


def test_the_incident_installs_nothing():
    assert _missing_js_packages(f"Error: Cannot find module '{WIN_PATH}'") == []


@pytest.mark.parametrize("spec", [
    WIN_PATH,
    "D:" + BS + "proj" + BS + "x.js",
    "./Header",
    "../utils/helpers.js",
    "/abs/path/file.js",
    "~/thing.js",
    "",
])
def test_paths_are_never_packages(spec):
    assert _npm_package_of(spec) is None, spec


@pytest.mark.parametrize("spec,pkg", [
    ("supertest", "supertest"),
    ("@testing-library/react", "@testing-library/react"),
    ("@testing-library/jest-dom/vitest", "@testing-library/jest-dom"),
    ("lodash/debounce", "lodash"),
    ("react-router-dom", "react-router-dom"),
    ("vite-plugin-x.y", "vite-plugin-x.y"),
])
def test_real_specifiers_resolve_to_their_package(spec, pkg):
    assert _npm_package_of(spec) == pkg


def test_several_missing_packages_are_all_reported():
    out = ("Cannot find package 'vitest' imported from x\n"
           "Cannot find package 'jsdom' imported from y")
    assert _missing_js_packages(out) == ["vitest", "jsdom"]


def test_a_real_package_beside_a_path_still_resolves():
    out = (f"Error: Cannot find module '{WIN_PATH}'\n"
           "Cannot find package 'supertest' imported from z")
    assert _missing_js_packages(out) == ["supertest"]


def test_duplicates_collapse():
    out = "Cannot find package 'vitest'\nCannot find package 'vitest'"
    assert _missing_js_packages(out) == ["vitest"]
