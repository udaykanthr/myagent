"""
Tests for the Phase 3 — Global Knowledge Base.

Covers:
- ErrorDict: CRUD, lookup with exact/regex/fuzzy matching
- Seeder: seed errors.db and markdown files (no embed)
- GlobalKBStore: search_errors, fallback file search
- Updater: version parsing, manifest loading
- Markdown chunking
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import tempfile
import unittest

# ---------------------------------------------------------------------------
# ErrorDict tests
# ---------------------------------------------------------------------------


class TestErrorDict(unittest.TestCase):
    """Tests for ``kb.global.error_dict.ErrorDict``."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test_errors.db")
        from multi_agent_coder.kb.global_kb.error_dict import ErrorDict
        self.edict = ErrorDict(self.db_path)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_add_and_count(self):
        """add() inserts a record; count() reflects it."""
        from multi_agent_coder.kb.global_kb.error_dict import ErrorFix
        ef = ErrorFix(
            error_type="TestError",
            language="python",
            pattern=r"TestError:.*",
            cause="test cause",
            fix_template="fix it",
            severity="error",
            tags="test",
        )
        self.edict.add(ef)
        self.assertEqual(self.edict.count(), 1)
        self.assertEqual(self.edict.count(language="python"), 1)
        self.assertEqual(self.edict.count(language="java"), 0)

    def test_bulk_insert(self):
        """bulk_insert() inserts multiple records."""
        from multi_agent_coder.kb.global_kb.error_dict import ErrorFix
        errors = [
            ErrorFix(error_type=f"Err{i}", language="python",
                     fix_template=f"fix {i}")
            for i in range(10)
        ]
        self.edict.bulk_insert(errors)
        self.assertEqual(self.edict.count(), 10)

    def test_lookup_exact(self):
        """lookup() matches by error_type substring."""
        from multi_agent_coder.kb.global_kb.error_dict import ErrorFix
        ef = ErrorFix(
            error_type="NullPointerException",
            language="java",
            pattern=r"NullPointerException",
            cause="null ref",
            fix_template="check for null",
        )
        self.edict.add(ef)
        results = self.edict.lookup("java.lang.NullPointerException at line 42")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].error_type, "NullPointerException")

    def test_lookup_regex(self):
        """lookup() falls back to regex pattern matching."""
        from multi_agent_coder.kb.global_kb.error_dict import ErrorFix
        ef = ErrorFix(
            error_type="ImportError",
            language="python",
            pattern=r"No module named '\w+'",
            cause="missing module",
            fix_template="pip install the module",
        )
        self.edict.add(ef)
        results = self.edict.lookup("No module named 'requests'", language="python")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].error_type, "ImportError")

    def test_lookup_fuzzy_tags(self):
        """lookup() falls back to tag-based fuzzy matching."""
        from multi_agent_coder.kb.global_kb.error_dict import ErrorFix
        ef = ErrorFix(
            error_type="MemoryLeak",
            language="all",
            pattern=r"^$",  # Won't match anything via regex
            cause="memory not freed",
            fix_template="free the memory",
            tags="memory,leak,heap",
        )
        self.edict.add(ef)
        # "memory" is in the error message and in the tags
        results = self.edict.lookup("possible memory issue detected")
        self.assertEqual(len(results), 1)

    def test_lookup_language_filter(self):
        """lookup() respects language filter, includes 'all'."""
        from multi_agent_coder.kb.global_kb.error_dict import ErrorFix
        self.edict.bulk_insert([
            ErrorFix(error_type="Err1", language="python", fix_template="fix1"),
            ErrorFix(error_type="Err2", language="java", fix_template="fix2"),
            ErrorFix(error_type="Err3", language="all", fix_template="fix3"),
        ])
        # Searching for python should get Err1 + Err3 (language=all)
        results = self.edict.lookup("Err1 occurred", language="python")
        languages = {r.language for r in results}
        self.assertIn("python", languages)
        # java should NOT be included
        self.assertNotIn("java", languages)

    def test_clear(self):
        """clear() removes all records."""
        from multi_agent_coder.kb.global_kb.error_dict import ErrorFix
        self.edict.bulk_insert([
            ErrorFix(error_type=f"E{i}", language="go", fix_template="f")
            for i in range(5)
        ])
        self.assertEqual(self.edict.count(), 5)
        self.edict.clear()
        self.assertEqual(self.edict.count(), 0)

    def test_count_by_language(self):
        """count_by_language() groups correctly."""
        from multi_agent_coder.kb.global_kb.error_dict import ErrorFix
        self.edict.bulk_insert([
            ErrorFix(error_type="E1", language="python", fix_template="f"),
            ErrorFix(error_type="E2", language="python", fix_template="f"),
            ErrorFix(error_type="E3", language="java", fix_template="f"),
        ])
        counts = self.edict.count_by_language()
        self.assertEqual(counts["python"], 2)
        self.assertEqual(counts["java"], 1)

    def test_errorfix_tag_list(self):
        """ErrorFix.tag_list() splits comma-separated tags."""
        from multi_agent_coder.kb.global_kb.error_dict import ErrorFix
        ef = ErrorFix(error_type="E", language="py", fix_template="f",
                      tags="a, b, c")
        self.assertEqual(ef.tag_list(), ["a", "b", "c"])

    def test_errorfix_empty_tags(self):
        """ErrorFix.tag_list() returns empty list for no tags."""
        from multi_agent_coder.kb.global_kb.error_dict import ErrorFix
        ef = ErrorFix(error_type="E", language="py", fix_template="f", tags="")
        self.assertEqual(ef.tag_list(), [])


# ---------------------------------------------------------------------------
# Seeder tests
# ---------------------------------------------------------------------------


class TestSeeder(unittest.TestCase):
    """Tests for ``kb.global.seeder``."""

    def test_seed_no_embed(self):
        """seed(embed=False) populates errors.db and writes .md files."""
        from multi_agent_coder.kb.global_kb.seeder import seed, _GLOBAL_DIR, _REGISTRY_DIR
        from multi_agent_coder.kb.global_kb.error_dict import ErrorDict

        summary = seed(embed=False)

        # Check errors.db was populated (35 errors: 5 * 7 languages)
        self.assertEqual(summary["errors_seeded"], 35)
        self.assertEqual(summary["docs_seeded"], 9)  # 3+2+2+2
        self.assertEqual(summary["chunks_embedded"], 0)

        # Verify error counts per language
        db_path = os.path.join(_GLOBAL_DIR, "core", "errors.db")
        edict = ErrorDict(db_path)
        counts = edict.count_by_language()
        for lang in ("python", "javascript", "typescript", "java",
                      "go", "rust", "csharp"):
            self.assertEqual(counts.get(lang, 0), 5,
                             f"Expected 5 errors for {lang}")

    def test_md_files_have_frontmatter(self):
        """All seeded .md files contain valid frontmatter."""
        from multi_agent_coder.kb.global_kb.seeder import seed, _REGISTRY_DIR

        seed(embed=False)

        for dirpath, _, filenames in os.walk(_REGISTRY_DIR):
            for fname in filenames:
                if not fname.endswith(".md"):
                    continue
                filepath = os.path.join(dirpath, fname)
                with open(filepath, encoding="utf-8") as fh:
                    content = fh.read()
                self.assertTrue(
                    content.startswith("---"),
                    f"{filepath} missing frontmatter",
                )
                parts = content.split("---", 2)
                self.assertGreaterEqual(len(parts), 3,
                    f"{filepath} has incomplete frontmatter")
                # Check required fields
                fm = parts[1]
                for field in ("title:", "category:", "tags:", "version:"):
                    self.assertIn(field, fm,
                        f"{filepath} missing frontmatter field: {field}")

    def test_chunk_markdown(self):
        """_chunk_markdown splits correctly."""
        from multi_agent_coder.kb.global_kb.seeder import _chunk_markdown

        text = "## Section 1\nShort.\n\n## Section 2\nAlso short."
        # Both sections are < 100 chars each, so they get merged
        chunks = _chunk_markdown(text, "Test Title", min_size=100)
        self.assertGreaterEqual(len(chunks), 1)
        # Every chunk should contain the title
        for chunk in chunks:
            self.assertIn("Test Title", chunk)

    def test_chunk_markdown_splits_large(self):
        """_chunk_markdown splits sections exceeding max_size."""
        from multi_agent_coder.kb.global_kb.seeder import _chunk_markdown

        # Create a large section with paragraph breaks so it can be split
        paragraphs = "\n\n".join(["word " * 30 for _ in range(10)])
        large_text = "## Big Section\n\n" + paragraphs
        chunks = _chunk_markdown(large_text, "Big", min_size=50, max_size=300)
        self.assertGreater(len(chunks), 1)


# ---------------------------------------------------------------------------
# Store tests (offline mode — no Qdrant)
# ---------------------------------------------------------------------------


class TestGlobalKBStore(unittest.TestCase):
    """Tests for ``kb.global.store.GlobalKBStore``."""

    @classmethod
    def setUpClass(cls):
        """Seed the DB once for all store tests."""
        from multi_agent_coder.kb.global_kb.seeder import seed
        seed(embed=False)

    def test_search_errors_exact(self):
        """search_errors finds NullPointerException for Java."""
        from multi_agent_coder.kb.global_kb.store import GlobalKBStore
        store = GlobalKBStore()
        results = store.search_errors("NullPointerException", language="java")
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].error_type, "NullPointerException")
        self.assertEqual(results[0].language, "java")

    def test_search_errors_regex(self):
        """search_errors finds Python AttributeError via regex."""
        from multi_agent_coder.kb.global_kb.store import GlobalKBStore
        store = GlobalKBStore()
        results = store.search_errors(
            "AttributeError: 'NoneType' object has no attribute 'foo'",
            language="python",
        )
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].error_type, "AttributeError")

    def test_search_errors_no_match(self):
        """search_errors returns empty for unknown error."""
        from multi_agent_coder.kb.global_kb.store import GlobalKBStore
        store = GlobalKBStore()
        results = store.search_errors("SomeCompletelyUnknownError12345")
        self.assertEqual(len(results), 0)

    def test_fallback_file_search(self):
        """search() falls back to file search when Qdrant is unavailable."""
        from multi_agent_coder.kb.global_kb.store import GlobalKBStore
        store = GlobalKBStore()
        # This will fail Qdrant and use fallback
        results = store.search("error handling best practices")
        # Should find the error-handling-best-practices.md doc
        self.assertGreater(len(results), 0)
        titles = [r.title for r in results]
        self.assertTrue(
            any("Error Handling" in t for t in titles),
            f"Expected 'Error Handling' in results, got: {titles}",
        )

    def test_fallback_search_category_filter(self):
        """Fallback search respects category filter."""
        from multi_agent_coder.kb.global_kb.store import GlobalKBStore
        store = GlobalKBStore()
        results = store.search("qdrant", categories=["adr"])
        for r in results:
            self.assertEqual(r.category, "adr")

    def test_get_behavioral_instructions(self):
        """get_behavioral_instructions returns behavioral docs."""
        from multi_agent_coder.kb.global_kb.store import GlobalKBStore
        store = GlobalKBStore()
        results = store.get_behavioral_instructions("reviewing code for quality")
        # Should find code-review-instructions
        self.assertGreater(len(results), 0)
        for r in results:
            self.assertEqual(r.category, "behavioral")


# ---------------------------------------------------------------------------
# Updater tests
# ---------------------------------------------------------------------------


class TestUpdater(unittest.TestCase):
    """Tests for ``kb.global.updater`` utility functions."""

    def test_parse_semver(self):
        """_parse_semver parses correctly."""
        from multi_agent_coder.kb.global_kb.updater import _parse_semver
        self.assertEqual(_parse_semver("1.2.3"), (1, 2, 3))
        self.assertEqual(_parse_semver("v2.0.0"), (2, 0, 0))
        self.assertGreater(_parse_semver("1.1.0"), _parse_semver("1.0.9"))

    def test_load_local_manifest(self):
        """_load_local_manifest reads the core manifest."""
        from multi_agent_coder.kb.global_kb.updater import _load_local_manifest
        manifest = _load_local_manifest()
        self.assertIn("version", manifest)
        self.assertIn("categories", manifest)

    def test_get_version(self):
        """get_version returns a version string."""
        from multi_agent_coder.kb.global_kb.updater import get_version
        version = get_version()
        self.assertIsInstance(version, str)
        self.assertRegex(version, r"\d+\.\d+\.\d+")

    def test_get_manifest_info(self):
        """get_manifest_info returns full manifest dict."""
        from multi_agent_coder.kb.global_kb.updater import get_manifest_info
        info = get_manifest_info()
        self.assertIn("version", info)
        self.assertIn("categories", info)
        self.assertIsInstance(info["categories"], list)

    def test_check_for_updates_no_owner(self):
        """check_for_updates gracefully handles nonexistent repo."""
        from multi_agent_coder.kb.global_kb.updater import check_for_updates
        status = check_for_updates("nonexistent-owner-xyz", "nonexistent-repo-xyz")
        # Should not crash; update_available should be False
        self.assertFalse(status.update_available)
        self.assertIsInstance(status.current_version, str)


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------


class TestCLIParsing(unittest.TestCase):
    """Tests that Phase 3 CLI subcommands parse correctly."""

    def _parse(self, argv: list[str]):
        from multi_agent_coder.kb.cli import _build_parser
        parser = _build_parser()
        return parser.parse_args(argv)

    def test_seed_command(self):
        args = self._parse(["seed"])
        self.assertEqual(args.kb_cmd, "seed")
        self.assertFalse(args.no_embed)

    def test_seed_no_embed(self):
        args = self._parse(["seed", "--no-embed"])
        self.assertTrue(args.no_embed)

    def test_version_command(self):
        args = self._parse(["version"])
        self.assertEqual(args.kb_cmd, "version")

    def test_error_lookup(self):
        args = self._parse(["error-lookup", "NullPointerException"])
        self.assertEqual(args.kb_cmd, "error-lookup")
        self.assertEqual(args.message, "NullPointerException")
        self.assertIsNone(args.language)

    def test_error_lookup_with_language(self):
        args = self._parse(["error-lookup", "TypeError", "--language", "python"])
        self.assertEqual(args.message, "TypeError")
        self.assertEqual(args.language, "python")

    def test_global_search(self):
        args = self._parse(["global-search", "error handling"])
        self.assertEqual(args.kb_cmd, "global-search")
        self.assertEqual(args.query, "error handling")

    def test_global_search_with_category(self):
        args = self._parse(["global-search", "qdrant", "--category", "adr"])
        self.assertEqual(args.category, "adr")

    def test_update_check(self):
        args = self._parse(["update", "--check"])
        self.assertEqual(args.kb_cmd, "update")
        self.assertTrue(args.check)

    def test_update_category(self):
        args = self._parse(["update", "--category", "errors"])
        self.assertEqual(args.category, "errors")


if __name__ == "__main__":
    unittest.main()
