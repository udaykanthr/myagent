"""Do the classes the markup renders actually exist in the stylesheet?

Two files can each be individually correct and jointly wrong. A component
step writes `site-footer__content`; a stylesheet step, running in the same
wave and unable to see it, writes `.site-footer__inner`. Both steps pass
their own gates, the suite passes, the production build passes — an
unmatched CSS class is still valid CSS — and the page renders unstyled.

WHY THIS EXISTS
---------------
Four of six consecutive runs on one Vite/React project drifted this way,
once completely::

    run    classes used    styled
    13:12       7             3
    13:39       8             5
    21:28       7             0      <- every element unstyled
    21:51       8             4

The 21:51 run is the one that settles the argument. Its acceptance gate
had already been strengthened, and asserted eight separate structural
properties of the stylesheet — full-bleed override, background AND
colour, a max-width container, a grid, a hover treatment, a divider, a
flex utility row, a responsive stacking rule. Every one was true. All
eight described selectors the markup never renders, so the gate passed on
a visibly broken footer.

No amount of single-file assertion can catch this: neither file is wrong
on its own. Only the join is.

SCOPE AND REFUSALS
------------------
Only "used but never defined" is treated as a defect — that is the part a
visitor sees. Orphaned rules are reported as context, never as failure:
dead CSS is untidy, not broken, and a project may legitimately keep
styles for markup rendered elsewhere.

Everything ambiguous is refused rather than guessed, because a false
accusation here sends a correct run into a fix loop:

* a utility or component framework in the dependencies (Tailwind,
  Bootstrap, MUI, …) puts class names in the markup that the project's
  own stylesheets are not expected to define;
* Sass/SCSS nesting composes selectors (`.a { &__b {} }`) that no static
  scan of the text can reconstruct;
* CSS Modules rename classes at build time, so the literal text never
  matches;
* a dynamic `className={...}` cannot be resolved at all — those
  expressions are skipped, though string literals in the same file are
  still checked.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field

_logger = logging.getLogger(__name__)

_MARKUP_SUFFIXES = (".jsx", ".tsx")
_STYLE_SUFFIXES = (".css",)
_NESTED_STYLE_SUFFIXES = (".scss", ".sass", ".less", ".styl")
_SKIP_DIRS = {"node_modules", "dist", "build", ".git", "__pycache__",
              ".next", "coverage", ".agentchanti", "venv", ".venv"}

# Dependencies that ship their own class vocabulary. Their presence makes
# "this class is not in our CSS" meaningless.
_FRAMEWORK_MARKERS = (
    "tailwindcss", "bootstrap", "bulma", "foundation-sites", "@mui/",
    "antd", "@chakra-ui/", "semantic-ui", "materialize-css", "primereact",
    "@radix-ui/themes", "daisyui",
)

_CLASSNAME_RE = re.compile(r'className\s*=\s*"([^"]*)"')
_DYNAMIC_CLASSNAME_RE = re.compile(r'className\s*=\s*\{')
_SELECTOR_RE = re.compile(r'\.(-?[_a-zA-Z][\w-]*)')
_CSS_COMMENT_RE = re.compile(r'/\*.*?\*/', re.DOTALL)
_TAILWIND_DIRECTIVE_RE = re.compile(r'@tailwind\b|@apply\b')


@dataclass
class StyleDrift:
    """Classes rendered by markup that no project stylesheet defines."""

    unstyled: dict[str, list[str]] = field(default_factory=dict)
    orphans: list[str] = field(default_factory=list)
    markup_files: int = 0
    style_files: int = 0
    # Stylesheets that exist but nothing imports. When the missing rules
    # are IN one of these, that is the whole defect and the fix is one
    # import line — not the rules, which are already written.
    unreachable: list[str] = field(default_factory=list)
    defines_them: list[str] = field(default_factory=list)

    @property
    def broken(self) -> bool:
        return bool(self.unstyled)

    def describe(self) -> str:
        lines = []
        for cls in sorted(self.unstyled):
            where = ", ".join(sorted(self.unstyled[cls])[:3])
            lines.append(f"  {cls}  (used in {where})")
        if self.defines_them:
            return (
                "These classes ARE defined — in "
                + ", ".join(sorted(self.defines_them))
                + " — but nothing imports that stylesheet, so none of it "
                  "reaches the browser and the elements render unstyled:\n"
                + "\n".join(lines)
                + "\n\nFix the IMPORT, not the rules: import the stylesheet "
                  "from the entry point (or from a module the entry point "
                  "already loads). Adding more rules to a file nothing "
                  "imports changes nothing a user sees.")
        text = ("These classes are rendered by the markup but no project "
                "stylesheet defines them, so those elements render "
                "unstyled:\n" + "\n".join(lines))
        if self.orphans:
            text += ("\n\nLikely counterpart — rules defined but never "
                     "rendered (the two files disagree on naming):\n  "
                     + ", ".join(sorted(self.orphans)[:12]))
        return text


def _walk(root: str, suffixes: tuple[str, ...]) -> list[str]:
    found = []
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        for name in files:
            if name.endswith(suffixes):
                found.append(os.path.join(base, name))
    return found


def _uses_a_class_framework(root: str) -> bool:
    path = os.path.join(root, "package.json")
    try:
        with open(path, encoding="utf-8") as fh:
            pkg = json.load(fh)
    except (OSError, ValueError):
        return False
    names = " ".join(list(pkg.get("dependencies") or {})
                     + list(pkg.get("devDependencies") or {}))
    return any(marker in names for marker in _FRAMEWORK_MARKERS)


_CSS_IMPORT_RE = re.compile(
    r"""(?:import\s+['"]|require\(\s*['"])([^'"]+\.css)['"]""")
_CSS_AT_IMPORT_RE = re.compile(r"""@import\s+(?:url\()?\s*['"]([^'"]+)['"]""")
_HTML_LINK_RE = re.compile(
    r"""<link[^>]+href\s*=\s*['"]([^'"]+\.css)['"]""", re.IGNORECASE)


def reachable_stylesheets(root: str, styles: list[str],
                          markup: list[str]) -> list[str] | None:
    """The stylesheets actually loaded, or None when that cannot be told.

    A rule only styles anything if the file holding it is REACHED from the
    entry point. Judging "is this class defined?" against every `.css` on
    disk answers a different question, and the difference is not academic:
    it lets a repair loop drive itself green by writing into dead code.

    Measured 2026-08-20 run 30. Step 12.1 was to "update the React
    bootstrap to import the global responsive stylesheet"; the agent
    edited `App.jsx` instead of its declared target `main.jsx`, and the
    step's gate -- `npm run build --silent` -- passes whether or not any
    CSS is imported. `client/src/styles/global.css` ended up 7023 bytes
    that nothing imports, while `main.jsx` kept Vite's stock
    `import './index.css'`. The built bundle contained no `home-hero`,
    proving it never reached the output.

    Then this very check found the classes "BROKEN", handed the list to a
    repair loop, and the loop satisfied it by adding the missing rules
    **to global.css** -- the unreachable file. It went green having
    changed nothing a user would see. Every gate, both acceptance
    instruments and the smoke test all passed over an entirely unstyled
    application; only the ghost's `violated-import-edge` noticed.

    Reachability is read from three places, because a project loads CSS
    in three ways: a JS/JSX `import`/`require` of a `.css`, an HTML
    `<link href>`, and `@import` chains from anything already reachable.

    Returns None when NO reachability signal exists anywhere -- some
    setups inject CSS by means this cannot see, and silently reporting
    every class as unstyled would be far worse than not judging. The
    caller keeps its existing "None means not judged" contract.
    """
    if not styles:
        return None

    by_norm = {os.path.normcase(os.path.abspath(p)): p for p in styles}
    seen: set[str] = set()

    def _claim(base_dir: str, spec: str) -> None:
        spec = spec.split("?")[0].split("#")[0]
        cand = os.path.normcase(os.path.abspath(
            os.path.join(base_dir, spec.lstrip("/"))))
        if cand in by_norm:
            seen.add(cand)
            return
        # A bare or aliased specifier: fall back to matching the basename,
        # which is what a human would do reading the import.
        tail = os.path.basename(spec)
        for norm, path in by_norm.items():
            if os.path.basename(path) == tail:
                seen.add(norm)

    found_any_signal = False
    for path in markup:
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError:
            continue
        hits = _CSS_IMPORT_RE.findall(text) + _HTML_LINK_RE.findall(text)
        if hits:
            found_any_signal = True
        for spec in hits:
            _claim(os.path.dirname(path), spec)

    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        for name in files:
            if not name.endswith(".html"):
                continue
            try:
                with open(os.path.join(base, name), encoding="utf-8",
                          errors="replace") as fh:
                    text = fh.read()
            except OSError:
                continue
            hits = _HTML_LINK_RE.findall(text)
            if hits:
                found_any_signal = True
            for spec in hits:
                _claim(base, spec)

    if not found_any_signal:
        return None                      # cannot tell; do not accuse

    # Follow @import chains out of everything already reached.
    changed = True
    while changed:
        changed = False
        for norm in list(seen):
            try:
                with open(by_norm[norm], encoding="utf-8",
                          errors="replace") as fh:
                    text = fh.read()
            except OSError:
                continue
            before = len(seen)
            for spec in _CSS_AT_IMPORT_RE.findall(text):
                _claim(os.path.dirname(by_norm[norm]), spec)
            if len(seen) != before:
                changed = True

    return [by_norm[n] for n in seen]


def find_style_drift(root: str = ".") -> StyleDrift | None:
    """Classes the markup renders that no stylesheet defines, or None.

    None means "not judged" — an unrecognised or unsupported layout, not
    a clean bill of health.
    """
    if _uses_a_class_framework(root):
        return None
    if _walk(root, _NESTED_STYLE_SUFFIXES):
        return None                      # Sass nesting composes selectors

    markup = _walk(root, _MARKUP_SUFFIXES)
    styles = [p for p in _walk(root, _STYLE_SUFFIXES)
              if not p.endswith(".module.css")]
    if not markup or not styles:
        return None

    # Only stylesheets the app actually LOADS can style anything. See
    # reachable_stylesheets: judging against every .css on disk let a
    # repair loop go green by writing rules into a file nothing imports.
    reached = reachable_stylesheets(root, styles, markup)
    unreachable = []
    if reached is not None:
        unreachable = [p for p in styles if p not in reached]
        styles = reached
        if not styles:
            return None

    defined: set[str] = set()
    for path in styles:
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                text = _CSS_COMMENT_RE.sub("", fh.read())
        except OSError:
            return None
        if _TAILWIND_DIRECTIVE_RE.search(text):
            return None                  # utility pipeline, not our names
        defined.update(_SELECTOR_RE.findall(text))

    used: dict[str, set[str]] = {}
    for path in markup:
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError:
            continue
        rel = os.path.relpath(path, root).replace("\\", "/")
        for literal in _CLASSNAME_RE.findall(text):
            for cls in literal.split():
                used.setdefault(cls, set()).add(rel)

    if not used:
        return None

    drift = StyleDrift(
        markup_files=len(markup), style_files=len(styles),
        unreachable=[os.path.relpath(p, root).replace(os.sep, "/")
                     for p in unreachable])
    for cls, where in used.items():
        if cls not in defined:
            drift.unstyled[cls] = sorted(where)
    # Orphans are reported only as the counterpart of a real break — on
    # their own they are dead CSS, which is untidy rather than wrong.
    if drift.unstyled and unreachable:
        # Are the missing names already sitting in a stylesheet nothing
        # loads? Then the rules are not missing at all and telling the
        # reader to write them is the wrong instruction entirely.
        missing = set(drift.unstyled)
        for path in unreachable:
            try:
                with open(path, encoding="utf-8", errors="replace") as fh:
                    orphaned = set(_SELECTOR_RE.findall(
                        _CSS_COMMENT_RE.sub("", fh.read())))
            except OSError:
                continue
            if orphaned & missing:
                drift.defines_them.append(
                    os.path.relpath(path, root).replace(os.sep, "/"))
    if drift.unstyled:
        rendered = set(used)
        prefixes = {c.split("__")[0] for c in drift.unstyled}
        drift.orphans = sorted(
            d for d in defined
            if d not in rendered and d.split("__")[0] in prefixes)
    return drift


def main(argv: list[str] | None = None) -> int:
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    root = argv[0] if argv else "."
    drift = find_style_drift(root)
    if drift is None:
        print("style-coupling: not judged (framework, nesting or no markup)")
        return 0
    if not drift.broken:
        print(f"style-coupling: OK — every class in {drift.markup_files} "
              f"markup file(s) is defined across {drift.style_files} "
              f"stylesheet(s)")
        return 0
    print(drift.describe())
    return 1


if __name__ == "__main__":       # usable as a gate: exit 1 == drift
    raise SystemExit(main())
