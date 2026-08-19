"""A bare `npm install <pkg>` where there is no package.json CREATES one.

It does not fail. It writes a `package.json`, a `node_modules` and a
lockfile — so in a repo whose real packages live in subdirectories it
manufactures a top-level package belonging to no project, and the app
keeps working only because Node resolution walks upward. Shipping the
sub-project alone then breaks.

Measured across three runs of one benchmark. Two came from healers and
were fixed at their source; the third was the agent itself running
`npm install jsonwebtoken && npm install jsonwebtoken --save --prefix
backend` — the second half correct, the first half leaving a root
package.json holding one stray dependency. This is the `run_command` gap
the architecture notes already name: tool sandboxing, not a gate check.
"""

import os

import pytest

from agentchanti.agent_tools import AgentTools, rootless_npm_install_reason
from agentchanti.llm.chat_types import ToolCall


@pytest.fixture
def multiroot(tmp_path):
    for sub in ("backend", "frontend"):
        (tmp_path / sub).mkdir()
        (tmp_path / sub / "package.json").write_text('{"name":"%s"}' % sub)
    return str(tmp_path)


def test_the_incident_is_refused(multiroot):
    why = rootless_npm_install_reason("npm install jsonwebtoken", multiroot)
    assert why is not None
    assert "backend" in why and "frontend" in why
    assert "npm --prefix backend install jsonwebtoken" in why


def test_the_shorthand_is_refused_too(multiroot):
    assert rootless_npm_install_reason("npm i express", multiroot)
    assert rootless_npm_install_reason("npm add express", multiroot)


@pytest.mark.parametrize("cmd", [
    "npm install jsonwebtoken --prefix backend",   # already directed
    "npm --prefix backend install jsonwebtoken",
    "npm install",                                 # restoring a tree
    "npm install --production",                    # flags only
    "npm run build",
    "npm test",
])
def test_legitimate_commands_are_allowed(cmd, multiroot):
    assert rootless_npm_install_reason(cmd, multiroot) is None, cmd


def test_a_real_root_package_is_allowed(tmp_path):
    """The root owning a manifest means installing there is correct."""
    (tmp_path / "package.json").write_text('{"name":"app"}')
    assert rootless_npm_install_reason(
        "npm install express", str(tmp_path)) is None


def test_greenfield_is_allowed(tmp_path):
    """No manifests anywhere — there is nothing better to suggest."""
    assert rootless_npm_install_reason(
        "npm install express", str(tmp_path)) is None


def test_node_modules_is_not_mistaken_for_a_package(tmp_path):
    (tmp_path / "node_modules" / "left-pad").mkdir(parents=True)
    (tmp_path / "node_modules" / "left-pad" / "package.json").write_text("{}")
    assert rootless_npm_install_reason(
        "npm install express", str(tmp_path)) is None


def test_run_command_refuses_it_end_to_end(multiroot):
    t = AgentTools(project_root=multiroot)
    out = t.execute(ToolCall(name="run_command", id="1", arguments={
        "command": "npm install jsonwebtoken"}))
    assert out.startswith("ERROR")
    assert not os.path.exists(os.path.join(multiroot, "package.json"))


# ─── a chain holds several independent invocations ──────────────────

INCIDENT_CHAIN = ("npm install --prefix backend jsonwebtoken --save-prod "
                  "&& npm install jsonwebtoken")


def test_a_directed_first_half_does_not_excuse_the_second(multiroot):
    """Run 25, verbatim.

    The anchored match saw only the correctly-prefixed first half, found
    `--prefix`, and allowed the whole line — so the bare root install
    after the `&&` was never examined, and created exactly the phantom
    root package this guard exists to prevent. `parse_failed_install_
    targets` in this same module already splits on these separators for
    the same reason.
    """
    why = rootless_npm_install_reason(INCIDENT_CHAIN, multiroot)
    assert why is not None
    assert "jsonwebtoken" in why


@pytest.mark.parametrize("sep", ["&&", "||", ";", "|"])
def test_every_separator_is_split(sep, multiroot):
    cmd = f"npm --prefix backend install express {sep} npm install lodash"
    assert rootless_npm_install_reason(cmd, multiroot) is not None


@pytest.mark.parametrize("cmd", [
    "npm --prefix backend install jsonwebtoken && npm --prefix frontend install react",
    "npm install --prefix backend jsonwebtoken",
    "npm run build && npm test",
    "npm install && npm run build",
])
def test_fully_directed_chains_stay_allowed(cmd, multiroot):
    assert rootless_npm_install_reason(cmd, multiroot) is None, cmd


def test_the_reported_package_is_the_undirected_one(multiroot):
    why = rootless_npm_install_reason(
        "npm --prefix backend install express && npm install lodash", multiroot)
    assert "lodash" in why
    assert "npm --prefix backend install lodash" in why


# ─── segments share sequential state ────────────────────────────────

SCAFFOLD = ("mkdir backend && cd backend && npm init -y "
            "&& npm install express cors dotenv jsonwebtoken bcryptjs")


def test_the_ordinary_scaffold_chain_is_allowed(multiroot):
    """Run 26: the segment-wise check refused this, and it is correct.

    It creates the directory, enters it, gives it a manifest, and installs
    THERE. Scanning segments independently loses the `cd` that makes the
    install right — the fix for "only the first segment is inspected"
    over-corrected into "no segment knows where it runs".
    """
    assert rootless_npm_install_reason(SCAFFOLD, multiroot) is None


@pytest.mark.parametrize("cmd", [
    "cd backend && npm install express",
    "cd ./backend && npm install express",
    "npm init -y && npm install express",
    "mkdir svc && cd svc && npm init -y && npm install fastify",
])
def test_a_cd_or_init_before_the_install_disarms_the_check(cmd, multiroot):
    assert rootless_npm_install_reason(cmd, multiroot) is None, cmd


def test_an_undirected_install_before_any_cd_is_still_refused(multiroot):
    """Order matters: the cd must come FIRST to excuse what follows."""
    assert rootless_npm_install_reason(
        "npm install lodash && cd backend && npm install express",
        multiroot) is not None
