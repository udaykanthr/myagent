"""Every omission in the JS export extractor reads as a MISSING export.

The ghost turns that into `violated-exports`, a finding that sends a
reader after code which is already correct. Measured 2026-08-19 run 16:
`export async function apiRequest(...)` was reported missing from a file
that plainly exports it, because the pattern had no `async` branch.

`ghost.plan_anchors` has carried the `async` case for some time — the two
parsers simply disagreed, and the verdict was decided by the weaker one.
The list form and the `module.exports = {...}` object form were missing
for the same reason, so every CommonJS backend module in this project's
own benchmark exported nothing as far as this extractor could tell.
"""

import pytest

from agentchanti.language_backend import JavaScriptBackend, TypeScriptBackend

JS = JavaScriptBackend()

# Verbatim shape of the run-16 file that triggered the false positive.
INCIDENT = """
export const TOKEN_KEY = 'app_auth_token'
export function getToken() { return null }
export async function apiRequest(path, opts = {}) { return fetch(path) }
export function login({ email, password }) { return apiRequest('/login') }
"""


def test_the_incident_export_is_found():
    got = JS.extract_exports(INCIDENT)
    assert "apiRequest" in got
    assert {"TOKEN_KEY", "getToken", "login"} <= set(got)


@pytest.mark.parametrize("src,expected", [
    ("export async function apiRequest(p){}", {"apiRequest"}),
    ("export function getToken(){}", {"getToken"}),
    ("export const API = 1", {"API"}),
    ("export class Thing {}", {"Thing"}),
    ("export let x = 1", {"x"}),
    ("export function* gen(){}", {"gen"}),
    ("export default async function main(){}", {"main", "default"}),
])
def test_declaration_forms(src, expected):
    assert expected <= set(JS.extract_exports(src))


@pytest.mark.parametrize("src,expected", [
    ("export { a, b }", {"a", "b"}),
    ("export { a as b }", {"b"}),
    ("export { one, two as three, four }", {"one", "three", "four"}),
])
def test_export_lists(src, expected):
    assert expected <= set(JS.extract_exports(src))


@pytest.mark.parametrize("src,expected", [
    ("module.exports = app;", {"app"}),
    ("module.exports = { createUser, findUserByEmail }",
     {"createUser", "findUserByEmail"}),
    ("module.exports = { signup: doSignup, login }", {"signup", "login"}),
    ("exports.authenticate = fn", {"authenticate"}),
    ("module.exports.app = app", {"app"}),
])
def test_commonjs_forms(src, expected):
    """The backend modules these runs actually produce."""
    assert expected <= set(JS.extract_exports(src))


def test_the_real_backend_shape():
    src = """
'use strict';
const bcrypt = require('bcryptjs');
async function signup(req, res) {}
function logout(req, res) {}
module.exports = {
  signup,
  logout,
};
"""
    assert {"signup", "logout"} <= set(JS.extract_exports(src))


def test_no_duplicates():
    src = "export const a = 1\nmodule.exports = { a }"
    got = JS.extract_exports(src)
    assert got.count("a") == 1


def test_a_file_with_no_exports_yields_none():
    assert JS.extract_exports("const x = 1\nfunction y(){}") == []


def test_typescript_inherits_the_fix():
    assert "apiRequest" in TypeScriptBackend().extract_exports(
        "export async function apiRequest(p: string): Promise<void> {}")


# ─── what survives a pattern match must actually be a name ───────────

def test_css_modules_export_block_yields_nothing():
    """`:export` is CSS Modules, not JavaScript.

    Measured 2026-08-19 run 22: the export-list branch added here for
    defect 15 matched `:export { globalStyles: globalStyles; }` in a
    .css file and produced the "export" `globalStyles: globalStyles;`,
    which the ghost then reported verbatim as the file's exports.
    """
    css = ":root { --c: red }\n:export {\n  globalStyles: globalStyles;\n}\n"
    assert JS.extract_exports(css) == []


@pytest.mark.parametrize("src", [
    ":export { a: a; }",
    "myexport { a }",          # not the keyword
    "reexport { a }",
])
def test_only_the_real_export_keyword_counts(src):
    assert JS.extract_exports(src) == []


@pytest.mark.parametrize("junk", [
    "a: b;", "1abc", "a-b", "", "   ", "a b",
])
def test_non_identifiers_are_never_exports(junk):
    assert junk not in JS.extract_exports("export { %s }" % junk)


def test_the_real_forms_still_survive_the_validation():
    """The guard must not cost anything the earlier fix gained."""
    src = ("export async function apiRequest(p){}\n"
           "export { a, b as c }\n"
           "module.exports = { createUser, findUserByEmail }\n"
           "export default function main(){}\n")
    got = set(JS.extract_exports(src))
    assert {"apiRequest", "a", "c", "createUser", "findUserByEmail",
            "main", "default"} <= got
