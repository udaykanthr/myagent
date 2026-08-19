"""
Unit tests for agentchanti.kb.context_builder

Tests the ContextBuilder class: intent detection, build_context(),
format_context_for_prompt(), and token budget management.
All KB dependencies (searcher, graph, global store) are mocked.
"""

from __future__ import annotations

import os
import tempfile
import unittest
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Import under test
# ---------------------------------------------------------------------------

from agentchanti.kb.context_builder import (
    ContextBuilder, KBContext,
    _ERROR_KEYWORDS, _REVIEW_KEYWORDS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def builder(tmp_path):
    """Return a ContextBuilder pointed at a tmp directory."""
    return ContextBuilder(project_root=str(tmp_path))


# ---------------------------------------------------------------------------
# Intent detection tests
# ---------------------------------------------------------------------------

class TestIntentDetection:

    def test_error_intent_positive(self):
        assert ContextBuilder._detect_error_intent("fix the login error")
        assert ContextBuilder._detect_error_intent("There is an exception in auth")
        assert ContextBuilder._detect_error_intent("Debug the crash")
        assert ContextBuilder._detect_error_intent("not working properly")

    def test_error_intent_negative(self):
        assert not ContextBuilder._detect_error_intent("add a new feature")
        assert not ContextBuilder._detect_error_intent("refactor auth module")

    def test_review_intent_positive(self):
        assert ContextBuilder._detect_review_intent("review the auth module")
        assert ContextBuilder._detect_review_intent("refactor the database layer")
        assert ContextBuilder._detect_review_intent("optimize the query")

    def test_review_intent_negative(self):
        assert not ContextBuilder._detect_review_intent("fix the login error")
        assert not ContextBuilder._detect_review_intent("create a new API endpoint")

    def test_language_detection(self):
        assert ContextBuilder._detect_language("src/auth.py") == "python"
        assert ContextBuilder._detect_language("app.js") == "javascript"
        assert ContextBuilder._detect_language("main.go") == "go"
        assert ContextBuilder._detect_language("Makefile") is None
        assert ContextBuilder._detect_language(None) is None


# ---------------------------------------------------------------------------
# Token estimation tests
# ---------------------------------------------------------------------------

class TestTokenEstimation:

    def test_estimate_tokens(self):
        assert ContextBuilder._estimate_tokens("") == 0
        assert ContextBuilder._estimate_tokens("a" * 100) == 25
        assert ContextBuilder._estimate_tokens("hello world") == 2


# ---------------------------------------------------------------------------
# KBContext dataclass tests
# ---------------------------------------------------------------------------

class TestKBContext:

    def test_defaults(self):
        ctx = KBContext()
        assert ctx.local_symbols == []
        assert ctx.related_symbols == []
        assert ctx.error_fixes == []
        assert ctx.global_patterns == []
        assert ctx.behavioral_instructions == []
        assert ctx.token_count == 0
        assert ctx.kb_available is False
        assert ctx.sources_used == []


# ---------------------------------------------------------------------------
# build_context tests
# ---------------------------------------------------------------------------

class TestBuildContext:

    def test_no_index_returns_unavailable(self, builder):
        """When no index exists, kb_available should be False."""
        ctx = builder.build_context("add login feature")
        assert ctx.kb_available is False
        assert ctx.local_symbols == []

    @patch("agentchanti.kb.context_builder.ContextBuilder._ensure_local")
    @patch("agentchanti.kb.context_builder.ContextBuilder._ensure_global")
    def test_build_context_with_mocked_local(self, mock_global, mock_local, builder):
        """Test build_context when local KB is available."""
        # Mock local to return True and set up searcher
        mock_local.return_value = True

        # Create a fake SearchResult
        fake_result = MagicMock()
        fake_result.symbol_name = "login"
        fake_result.symbol_type = "function"
        fake_result.file = "src/auth.py"
        fake_result.line_start = 10
        fake_result.line_end = 25
        fake_result.code_snippet = "def login(user, pwd): pass"
        fake_result.score = 0.95
        fake_result.related_symbols = []

        builder._searcher = MagicMock()
        builder._searcher.search.return_value = [fake_result]
        builder._graph = MagicMock()
        builder._graph.get_related_symbols.return_value = []

        ctx = builder.build_context("fix the login error")
        # local search should have been called
        builder._searcher.search.assert_called_once()
        assert len(ctx.local_symbols) == 1
        assert ctx.local_symbols[0].symbol_name == "login"

    @patch("agentchanti.kb.context_builder.ContextBuilder._ensure_local")
    @patch("agentchanti.kb.context_builder.ContextBuilder._ensure_global")
    def test_error_intent_triggers_error_lookup(self, mock_global, mock_local, builder):
        """Error-related tasks should trigger error_dict lookup."""
        mock_local.return_value = False

        fake_fix = MagicMock()
        fake_fix.error_type = "AttributeError"
        fake_fix.cause = "None attribute access"
        fake_fix.fix_template = "Check for None"
        fake_fix.tags = ""

        builder._global_store = MagicMock()
        builder._global_store.search_errors.return_value = [fake_fix]
        builder._global_store.batch_search.return_value = {
            "doc": [], "behavioral": [],
        }

        ctx = builder.build_context("fix the AttributeError exception")
        builder._global_store.search_errors.assert_called_once()
        assert len(ctx.error_fixes) == 1
        assert ctx.error_fixes[0].error_type == "AttributeError"

    @patch("agentchanti.kb.context_builder.ContextBuilder._ensure_local")
    @patch("agentchanti.kb.context_builder.ContextBuilder._ensure_global")
    def test_error_output_used_for_lookup(self, mock_global, mock_local, builder):
        """When error_output is provided, it should be used for error matching."""
        mock_local.return_value = False

        fake_fix = MagicMock()
        fake_fix.error_type = "NullPointerException"
        fake_fix.cause = "Null dereference"
        fake_fix.fix_template = "Add null check"
        fake_fix.tags = ""

        builder._global_store = MagicMock()
        builder._global_store.search_errors.return_value = [fake_fix]
        builder._global_store.batch_search.return_value = {
            "doc": [], "behavioral": [],
        }

        # Step description has no error keywords, but error_output forces error intent
        ctx = builder.build_context(
            "compile the project",
            error_output="java.lang.NullPointerException at Main.java:42",
        )
        builder._global_store.search_errors.assert_called_once()
        # Should have used the error_output, not the task description
        call_args = builder._global_store.search_errors.call_args
        assert "NullPointerException" in call_args[0][0]
        assert len(ctx.error_fixes) == 1

    @patch("agentchanti.kb.context_builder.ContextBuilder._ensure_local")
    @patch("agentchanti.kb.context_builder.ContextBuilder._ensure_global")
    def test_review_intent_triggers_pattern_search(self, mock_global, mock_local, builder):
        """Review-related tasks should trigger global pattern search."""
        mock_local.return_value = False

        fake_pattern = MagicMock()
        fake_pattern.title = "SOLID Principles"
        fake_pattern.content = "Use dependency injection"
        fake_pattern.category = "pattern"

        builder._global_store = MagicMock()
        builder._global_store.batch_search.return_value = {
            "pattern": [fake_pattern],
            "adr": [],
            "doc": [],
            "behavioral": [],
        }

        ctx = builder.build_context("review the auth module for patterns")
        builder._global_store.batch_search.assert_called_once()
        assert len(ctx.global_patterns) >= 1
        assert ctx.global_patterns[0].title == "SOLID Principles"

    @patch("agentchanti.kb.context_builder.ContextBuilder._ensure_local")
    @patch("agentchanti.kb.context_builder.ContextBuilder._ensure_global")
    def test_behavioral_always_included(self, mock_global, mock_local, builder):
        """Behavioral instructions should always be fetched via batch_search."""
        mock_local.return_value = False

        fake_behavioral = MagicMock()
        fake_behavioral.title = "Always use type hints"
        fake_behavioral.content = "Add type hints to all functions"

        builder._global_store = MagicMock()
        builder._global_store.batch_search.return_value = {
            "doc": [],
            "behavioral": [fake_behavioral],
        }

        ctx = builder.build_context("add a new feature")
        builder._global_store.batch_search.assert_called_once()
        assert len(ctx.behavioral_instructions) == 1

    @patch("agentchanti.kb.context_builder.ContextBuilder._ensure_local")
    @patch("agentchanti.kb.context_builder.ContextBuilder._ensure_global")
    def test_exception_does_not_crash(self, mock_global, mock_local, builder):
        """KB exceptions should be caught, not crash the build."""
        mock_local.side_effect = Exception("boom")

        # Should not raise
        ctx = builder.build_context("any task")
        assert isinstance(ctx, KBContext)


# ---------------------------------------------------------------------------
# Token budget tests
# ---------------------------------------------------------------------------

class TestTokenBudget:

    def test_trim_low_priority_first(self, builder):
        """When over budget, related_symbols and extra local_symbols are trimmed first."""
        ctx = KBContext()

        # Create fake items with enough "tokens"
        for i in range(10):
            r = MagicMock()
            r.code_snippet = "x" * 400  # ~100 tokens each
            r.symbol_name = f"sym_{i}"
            ctx.local_symbols.append(r)

        for i in range(5):
            ctx.related_symbols.append({"name": f"rel_{i}", "x": "y" * 400})

        # Low budget, requires flushing all related_symbols to fit (top 3 local cost ~315)
        result = builder._apply_token_budget(ctx, max_tokens=350)
        # local_symbols should be trimmed to 3
        assert len(result.local_symbols) <= 3
        # related_symbols should be empty
        assert len(result.related_symbols) == 0


# ---------------------------------------------------------------------------
# format_context_for_prompt tests
# ---------------------------------------------------------------------------

class TestFormatContext:

    def test_empty_context_returns_empty(self, builder):
        ctx = KBContext()
        result = builder.format_context_for_prompt(ctx)
        assert result == ""

    def test_format_with_local_symbols(self, builder):
        ctx = KBContext(kb_available=True)

        result_obj = MagicMock()
        result_obj.file = "src/auth.py"
        result_obj.line_start = 10
        result_obj.line_end = 25
        result_obj.code_snippet = "def login(): pass"
        result_obj.related_symbols = [{"name": "validate"}]
        ctx.local_symbols = [result_obj]

        output = builder.format_context_for_prompt(ctx)
        assert "KNOWLEDGE BASE CONTEXT" in output
        assert "RELEVANT CODE FROM THIS PROJECT" in output
        assert "src/auth.py" in output
        assert "def login(): pass" in output

    def test_format_with_error_fixes(self, builder):
        ctx = KBContext(kb_available=True)

        fix = MagicMock()
        fix.error_type = "AttributeError"
        fix.cause = "None access"
        fix.fix_template = "Check for None first"
        ctx.error_fixes = [fix]

        output = builder.format_context_for_prompt(ctx)
        assert "ERROR FIX PATTERNS" in output
        assert "AttributeError" in output
        assert "Check for None first" in output

    def test_format_with_behavioral(self, builder):
        ctx = KBContext()  # kb_available=False but has behavioral

        bi = MagicMock()
        bi.content = "Always validate inputs"
        bi.title = "Input Validation"
        ctx.behavioral_instructions = [bi]

        output = builder.format_context_for_prompt(ctx)
        assert "BEHAVIORAL INSTRUCTIONS" in output
        assert "Always validate inputs" in output

    def test_format_with_patterns(self, builder):
        ctx = KBContext(kb_available=True)

        pattern = MagicMock()
        pattern.title = "Repository Pattern"
        pattern.content = "Use repository pattern for data access"
        ctx.global_patterns = [pattern]

        output = builder.format_context_for_prompt(ctx)
        assert "CODING PATTERNS" in output
        assert "Repository Pattern" in output


class TestRawTaskTopicFallback(unittest.TestCase):
    """An unarmed topic filter admits every doc in the registry.

    `_passes_topic_filter` returns True when `_intent_topics` is empty.
    The vector-search branch was fixed to arm it even on a 0-doc search;
    the title fast-path still assigned `_fp_per_topic_kws or []` and the
    no-docs path assigned nothing. Live result on a Pygame run: React,
    Django, Vitest and Three.js docs injected into every step.

    These assert the *shape* the planner's fallback must produce, and the
    discrimination it has to achieve on the real shipped titles.
    """

    TASK = (
        "Build a Pac-Man clone using Python and Pygame. Tile-based maze, "
        "ghosts, pellets and power pellets. Organize code into classes. "
        "Test with randomised delta-time and assert no entity enters a "
        "wall. python -m unittest must pass."
    )

    def _topics(self):
        import re as _re
        from agentchanti.agents.planner import _TOPIC_STOPWORDS
        kws = {w.lower() for w in _re.findall(r'[a-zA-Z]{4,}', self.TASK)}
        kws -= _TOPIC_STOPWORDS
        return [{k} for k in sorted(kws)]

    @staticmethod
    def _passes(title, topics):
        def stem(a, b):
            n = min(len(a), len(b))
            if n < 3:
                return a == b
            return a[:min(n, 5)] == b[:min(n, 5)]
        words = {w.lower() for w in title.replace("-", " ").split()
                 if len(w) >= 3}
        for topic in topics:
            threshold = max(1, min(2, len(topic)))
            hits = sum(1 for w in words
                       if any(stem(w, t) for t in topic))
            if hits >= threshold:
                return True
        return False

    def test_each_keyword_is_its_own_topic(self):
        """One combined set raises the threshold to 2 and drops good docs."""
        topics = self._topics()
        self.assertTrue(all(len(t) == 1 for t in topics))
        self.assertTrue(
            self._passes("Pygame Setup Guide", topics),
            "a single strong match must survive; combining every keyword "
            "into one set makes the threshold 2 and drops this")

    def test_relevant_docs_survive(self):
        topics = self._topics()
        for title in ("Pygame Setup Guide", "Python Setup Guide",
                      "Python Test Generation Instructions"):
            self.assertTrue(self._passes(title, topics), title)

    def test_foreign_stack_docs_are_dropped(self):
        topics = self._topics()
        for title in ("React Component Export Instructions",
                      "React Router Setup Instructions",
                      "NPM Scripts Instructions",
                      "Three.js + Vitest: Fixing WebGL Context Errors",
                      "ADR-001: Use SQLite for Vector Store"):
            self.assertFalse(self._passes(title, topics), title)

    def test_short_keywords_cannot_stem_match_junk(self):
        """"pac" would otherwise match "packages" and readmit Vitest docs."""
        topics = self._topics()
        self.assertNotIn({"pac"}, topics)


def _shipped_registry_docs():
    """Paths of the shipped KB registry docs, or [] when absent.

    `agentchanti/kb/global_kb/registry/` is gitignored — its own .gitignore
    says "Registry content is pulled at runtime; do not commit" — so it is
    present on a working machine and ABSENT in CI. A test that reads it must
    therefore skip rather than fail, and must not pass vacuously on an empty
    directory either (an empty scan makes "no offenders found" trivially
    true, which looks green while checking nothing).
    """
    import glob
    import os

    import agentchanti.kb.global_kb.store as _store
    registry = os.path.join(
        os.path.dirname(os.path.abspath(_store.__file__)), "registry")
    return sorted(glob.glob(os.path.join(registry, "**", "*.md"),
                            recursive=True))


class TestFrameworkScopedDocs(unittest.TestCase):
    """A doc written FOR a framework needs that framework in the project.

    The language filter cannot catch this case: the Django docs are
    correctly tagged `language: "python"` and a Pygame game IS a Python
    project. Live evidence -- "Django Page Creation Pattern" and "Django
    Test Generation Instructions" (11.2KB, ~2.8k tokens) injected into
    steps of a Pac-Man clone.

    The rule is asymmetric on purpose: it only recognises the framework
    the DOC is about, so it never needs a vocabulary of every stack a
    project might use.
    """

    TASK_WORDS = {"pacman", "clone", "python", "pygame", "maze", "ghosts",
                  "pellets", "tile", "unittest", "test", "wall", "sprite"}

    def _passes(self, title, tags=()):
        # Calls the shipped rule rather than restating it: a test that
        # reimplements what it checks passes just as happily when the
        # real filter is deleted.
        from agentchanti.kb.context_builder import doc_survives_framework_scope
        return doc_survives_framework_scope(title, tags, self.TASK_WORDS)

    def test_django_docs_are_dropped_from_a_pygame_project(self):
        self.assertFalse(self._passes(
            "Django Test Generation Instructions",
            ("django", "testing", "pytest", "python")))
        self.assertFalse(self._passes("Django Page Creation Pattern",
                                      ("django", "python")))

    def test_framework_agnostic_python_docs_survive(self):
        self.assertTrue(self._passes(
            "Python Test Generation Instructions",
            ("python", "testing", "pytest", "mock")))
        self.assertTrue(self._passes("Clean Code Naming Conventions"))
        self.assertTrue(self._passes("Async Programming Patterns"))

    def test_the_projects_own_framework_survives(self):
        self.assertTrue(self._passes("Pygame Setup Guide", ("pygame",)))

    def test_foreign_js_frameworks_are_dropped(self):
        for title, tags in (
            ("React Component Export Instructions", ("react", "jsx")),
            ("Vitest React Testing Library Setup", ("vitest", "react")),
            ("Three.js + Vitest: Fixing WebGL Context Errors",
             ("threejs", "webgl", "vitest")),
        ):
            self.assertFalse(self._passes(title, tags), title)

    def test_a_bare_language_is_not_treated_as_a_framework(self):
        """Otherwise generic Python docs would be dropped from Python."""
        from agentchanti.kb.context_builder import _DOC_FRAMEWORK_TOKENS
        for lang in ("python", "javascript", "typescript", "java", "go"):
            self.assertNotIn(lang, _DOC_FRAMEWORK_TOKENS)

    def test_the_shipped_registry_splits_the_way_we_expect(self):
        """Guards the real frontmatter, not just the token set.

        Skipped where the registry is not on disk — see
        _shipped_registry_docs(). This is a data check on runtime-pulled
        content, not a CI-enforceable invariant; the filter logic itself is
        covered by the literal-title tests above.
        """
        import re as _re

        paths = _shipped_registry_docs()
        if not paths:
            self.skipTest("KB registry is gitignored/pulled at runtime — "
                          "not present here")
        verdicts = {}
        for path in paths:
            with open(path, encoding="utf-8") as fh:
                head = fh.read(800)
            title = _re.search(r'^title:\s*"([^"]*)"', head, _re.M)
            tags = _re.search(r'^tags:\s*"([^"]*)"', head, _re.M)
            if not title:
                continue
            verdicts[title.group(1)] = self._passes(
                title.group(1),
                tuple(t.strip() for t in (tags.group(1) if tags
                                          else "").split(",")))
        dropped = {t for t, ok in verdicts.items() if not ok}
        self.assertTrue(
            any("Django" in t for t in dropped),
            f"Django docs must not reach a Pygame project; dropped={dropped}")
        self.assertTrue(
            any("React" in t for t in dropped),
            f"React docs must not reach a Pygame project; dropped={dropped}")
        kept = {t for t, ok in verdicts.items() if ok}
        self.assertTrue(
            any("Python Test Generation" in t for t in kept),
            f"generic Python docs must survive; kept={kept}")


class TestTaskVocabulary(unittest.TestCase):
    """Both sides of the comparison must tokenise the same way.

    `.`, `+` and `#` belong in the character class so `three.js`, `c++`
    and `c#` survive as one token — which also means prose punctuation
    sticks to a word. "with Pygame." tokenised as `pygame.`, matched no
    framework name, and made the project's OWN framework look foreign:
    the fix that scoped the pre-analysis list would have dropped
    "Pygame Setup Guide" from a Pygame task.
    """

    def test_trailing_punctuation_does_not_hide_a_framework(self):
        from agentchanti.kb.context_builder import task_vocabulary
        vocab = task_vocabulary("Build a Pac-Man clone in Python with Pygame.")
        self.assertIn("pygame", vocab)

    def test_dotted_framework_names_survive(self):
        from agentchanti.kb.context_builder import task_vocabulary
        self.assertIn("three.js", task_vocabulary("render it with three.js"))

    def test_the_projects_own_framework_is_kept(self):
        from agentchanti.kb.context_builder import (
            doc_survives_framework_scope, task_vocabulary)
        vocab = task_vocabulary("Build a Pac-Man clone in Python with Pygame.")
        self.assertTrue(doc_survives_framework_scope(
            "Pygame Setup Guide", ("pygame",), vocab))


class _Doc:
    def __init__(self, title, tags=()):
        self.title = title
        self.tags = tags


class TestScopeDocsToProject(unittest.TestCase):
    """The pre-analysis title list is scoped, not just the per-step build.

    `language=` cannot separate these — the Django docs are correctly
    tagged `language: "python"` and a Pygame game IS a Python project —
    so every run of the Pac-Man benchmark surfaced "Django Page Creation
    Pattern" to the IntentAgent. The per-step builder dropped it a layer
    later, but this list is what the agent picks `KB docs:` from and what
    the force-include net scans, which is how "Vitest React Testing
    Library Setup" reached a Pygame plan.
    """

    TASK = ("Build a Pac-Man clone in Python with Pygame. maze ghosts "
            "pellets unittest")

    def _kept(self, *docs):
        from agentchanti.kb.context_builder import (
            scope_docs_to_project, task_vocabulary)
        return [d.title for d in scope_docs_to_project(
            list(docs), task_vocabulary(self.TASK))]

    def test_drops_foreign_frameworks_keeps_the_rest(self):
        kept = self._kept(
            _Doc("Clean Code Naming Conventions"),
            _Doc("Django Page Creation Pattern", ("django", "python")),
            _Doc("Vitest React Testing Library Setup", ("vitest", "react")),
            _Doc("Pygame Setup Guide", ("pygame",)),
            _Doc("Python Test Generation Instructions", ("python", "pytest")),
        )
        self.assertEqual(kept, ["Clean Code Naming Conventions",
                                "Pygame Setup Guide",
                                "Python Test Generation Instructions"])

    def test_no_task_text_keeps_everything(self):
        # No task words is no evidence that a doc is foreign; dropping on
        # a guess would lose good docs.
        from agentchanti.kb.context_builder import scope_docs_to_project
        docs = [_Doc("Django Page Creation Pattern", ("django",))]
        self.assertEqual(len(scope_docs_to_project(docs, set())), 1)

    def test_tolerates_empty_and_missing_input(self):
        from agentchanti.kb.context_builder import scope_docs_to_project
        self.assertEqual(scope_docs_to_project(None, {"pygame"}), [])
        self.assertEqual(scope_docs_to_project([], {"pygame"}), [])
