"""
Tests for the Search Agent integration:
- SearchAgent query building
- SearchAgent error extraction
- SearchAgent result formatting
- SearchAgent graceful failure
- search_provider HTML text extraction
- Diagnosis integration with search context
- Config search settings
"""
import pytest
from unittest.mock import patch, MagicMock

from multi_agent_coder.agents.search import SearchAgent
from multi_agent_coder.search_provider import (
    _html_to_text, SearchResult, web_search, fetch_page_text,
)


# ── HTML text extraction tests ───────────────────────────────


def test_html_to_text_basic():
    """Basic HTML tags should be stripped."""
    html = "<p>Hello <b>world</b></p>"
    assert "Hello" in _html_to_text(html)
    assert "world" in _html_to_text(html)
    assert "<" not in _html_to_text(html)


def test_html_to_text_script_removal():
    """Script and style tags should be removed."""
    html = "<p>Visible</p><script>alert('x')</script><style>.x{}</style>"
    text = _html_to_text(html)
    assert "Visible" in text
    assert "alert" not in text
    assert ".x" not in text


def test_html_to_text_empty():
    """Empty HTML should return empty string."""
    assert _html_to_text("") == ""


# ── SearchAgent query building tests ─────────────────────────


class TestBuildSearchQuery:
    """Tests for SearchAgent._build_search_query()."""

    def setup_method(self):
        self.agent = SearchAgent()

    def test_python_error(self):
        """Python-style error should produce a focused query."""
        error = (
            "Traceback (most recent call last):\n"
            "  File \"/app/main.py\", line 42\n"
            "ModuleNotFoundError: No module named 'flask'"
        )
        query = self.agent._build_search_query(error, language="python")
        assert "ModuleNotFoundError" in query
        assert "flask" in query
        assert "python" in query
        # File paths should be stripped
        assert "/app/main.py" not in query

    def test_node_error(self):
        """Node.js error should produce a focused query."""
        error = "TypeError: Cannot read properties of undefined (reading 'map')"
        query = self.agent._build_search_query(error, language="javascript")
        assert "TypeError" in query
        assert "undefined" in query
        assert "javascript" in query or "node.js" in query

    def test_empty_error(self):
        """Empty error should return empty query."""
        assert self.agent._build_search_query("") == ""
        assert self.agent._build_search_query("   ") == ""

    def test_query_length_cap(self):
        """Very long errors should be capped."""
        error = "SomeError: " + "x" * 200
        query = self.agent._build_search_query(error)
        assert len(query) <= 150

    def test_no_language(self):
        """Query without language should still work."""
        error = "ImportError: No module named 'foo'"
        query = self.agent._build_search_query(error)
        assert "ImportError" in query
        assert "foo" in query

    def test_ansi_codes_stripped(self):
        """ANSI escape codes should be stripped from the query."""
        error = "\x1b[31mError: something failed\x1b[0m"
        query = self.agent._build_search_query(error)
        assert "\x1b" not in query
        assert "Error" in query


# ── SearchAgent key error line extraction ────────────────────


class TestExtractKeyErrorLine:
    """Tests for SearchAgent._extract_key_error_line()."""

    def test_python_traceback(self):
        """Should extract the last ErrorClass: message line."""
        error = (
            "Traceback (most recent call last):\n"
            "  File \"test.py\", line 3\n"
            "    import nonexistent\n"
            "ModuleNotFoundError: No module named 'nonexistent'"
        )
        line = SearchAgent._extract_key_error_line(error)
        assert line.startswith("ModuleNotFoundError")

    def test_node_error(self):
        """Should extract TypeError/ReferenceError etc."""
        error = (
            "at Object.<anonymous> (index.js:3:5)\n"
            "TypeError: foo is not a function\n"
            "    at Module._compile"
        )
        line = SearchAgent._extract_key_error_line(error)
        assert "TypeError" in line

    def test_generic_error(self):
        """Should extract lines containing 'error' (case-insensitive)."""
        error = "Command failed: npm install\nerror code ERESOLVE\nsome other line"
        line = SearchAgent._extract_key_error_line(error)
        assert "error" in line.lower()

    def test_empty_input(self):
        """Empty input should return empty string."""
        assert SearchAgent._extract_key_error_line("") == ""
        assert SearchAgent._extract_key_error_line("   \n  \n") == ""


# ── SearchAgent result formatting ────────────────────────────


class TestFormatResults:
    """Tests for SearchAgent._format_results()."""

    def setup_method(self):
        self.agent = SearchAgent()

    @patch("multi_agent_coder.agents.search.fetch_page_text", return_value="")
    def test_basic_formatting(self, mock_fetch):
        """Results should have numbered headers and URLs."""
        results = [
            SearchResult("Fix Flask Import", "https://example.com", "Install flask using pip"),
        ]
        formatted = self.agent._format_results(results)
        assert "[1]" in formatted
        assert "Fix Flask Import" in formatted
        assert "https://example.com" in formatted
        assert "Install flask using pip" in formatted

    @patch("multi_agent_coder.agents.search.fetch_page_text", return_value="Page content here")
    def test_includes_page_excerpt(self, mock_fetch):
        """Should include fetched page content."""
        results = [
            SearchResult("Title", "https://example.com", "Snippet"),
        ]
        formatted = self.agent._format_results(results)
        assert "Page excerpt:" in formatted
        assert "Page content here" in formatted

    def test_empty_results(self):
        """Empty results should return empty string."""
        assert self.agent._format_results([]) == ""

    @patch("multi_agent_coder.agents.search.fetch_page_text", return_value="")
    def test_header(self, mock_fetch):
        """Should contain the Web Search Results header."""
        results = [SearchResult("T", "http://x.com", "S")]
        formatted = self.agent._format_results(results)
        assert "Web Search Results" in formatted


# ── SearchAgent search_for_error (integration) ──────────────


class TestSearchForError:
    """Tests for the main search_for_error() method."""

    @patch("multi_agent_coder.agents.search.web_search")
    @patch("multi_agent_coder.agents.search.fetch_page_text", return_value="")
    def test_returns_context(self, mock_fetch, mock_search):
        """Should return formatted context when search finds results."""
        mock_search.return_value = [
            SearchResult("Fix Guide", "https://fix.com", "How to fix the error"),
        ]
        agent = SearchAgent()
        result = agent.search_for_error(
            "ModuleNotFoundError: No module named 'flask'",
            language="python",
        )
        assert "Fix Guide" in result
        assert "Web Search Results" in result
        mock_search.assert_called_once()

    @patch("multi_agent_coder.agents.search.web_search")
    def test_returns_empty_on_no_results(self, mock_search):
        """Should return empty string when no results found."""
        mock_search.return_value = []
        agent = SearchAgent()
        result = agent.search_for_error("SomeError: unknown")
        assert result == ""

    @patch("multi_agent_coder.agents.search.web_search",
           side_effect=Exception("Network error"))
    def test_graceful_failure(self, mock_search):
        """Should return empty string on exception, never raise."""
        agent = SearchAgent()
        result = agent.search_for_error("SomeError: failure")
        assert result == ""

    def test_empty_error_skips_search(self):
        """Empty error should return empty without searching."""
        agent = SearchAgent()
        result = agent.search_for_error("")
        assert result == ""


# ── web_search provider dispatch tests ───────────────────────


class TestWebSearch:
    """Tests for the web_search() dispatcher."""

    @patch("multi_agent_coder.search_provider._search_duckduckgo")
    def test_default_uses_duckduckgo(self, mock_ddg):
        """Default provider should be DuckDuckGo."""
        mock_ddg.return_value = []
        web_search("test query")
        mock_ddg.assert_called_once_with("test query", 3)

    @patch("multi_agent_coder.search_provider._search_google")
    def test_google_dispatch(self, mock_google):
        """Google provider should dispatch correctly."""
        mock_google.return_value = []
        web_search("test", provider="google", api_key="key:cx")
        mock_google.assert_called_once()

    @patch("multi_agent_coder.search_provider._search_serpapi")
    def test_serpapi_dispatch(self, mock_serp):
        """SerpAPI provider should dispatch correctly."""
        mock_serp.return_value = []
        web_search("test", provider="serpapi", api_key="key123")
        mock_serp.assert_called_once()

    def test_google_without_key_returns_empty(self):
        """Google without API key should return empty list."""
        results = web_search("test", provider="google", api_key="")
        assert results == []

    def test_serpapi_without_key_returns_empty(self):
        """SerpAPI without API key should return empty list."""
        results = web_search("test", provider="serpapi", api_key="")
        assert results == []

    @patch("multi_agent_coder.search_provider._search_duckduckgo")
    def test_unknown_provider_falls_back_to_ddg(self, mock_ddg):
        """Unknown provider should fall back to DuckDuckGo."""
        mock_ddg.return_value = []
        web_search("test", provider="bing")
        mock_ddg.assert_called_once()


# ── fetch_page_text tests ────────────────────────────────────


class TestFetchPageText:
    """Tests for fetch_page_text()."""

    @patch("multi_agent_coder.search_provider.requests.get")
    def test_fetches_html(self, mock_get):
        """Should extract text from HTML response."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body><p>Hello world</p></body></html>"
        mock_resp.headers = {"Content-Type": "text/html"}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp
        text = fetch_page_text("https://example.com")
        assert "Hello world" in text

    @patch("multi_agent_coder.search_provider.requests.get",
           side_effect=Exception("timeout"))
    def test_returns_empty_on_error(self, mock_get):
        """Should return empty string on any error."""
        text = fetch_page_text("https://example.com")
        assert text == ""

    @patch("multi_agent_coder.search_provider.requests.get")
    def test_respects_max_chars(self, mock_get):
        """Should truncate to max_chars."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<p>" + "x" * 5000 + "</p>"
        mock_resp.headers = {"Content-Type": "text/html"}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp
        text = fetch_page_text("https://example.com", max_chars=100)
        assert len(text) <= 100


# ── Config integration tests ────────────────────────────────


class TestSearchConfig:
    """Tests for search-related config settings."""

    def test_default_search_enabled(self):
        """Search should be enabled by default."""
        from multi_agent_coder.config import Config
        cfg = Config()
        assert cfg.SEARCH_ENABLED is True
        assert cfg.SEARCH_PROVIDER == "duckduckgo"
        assert cfg.SEARCH_MAX_RESULTS == 3
        assert cfg.SEARCH_MAX_PAGE_CHARS == 3000

    def test_yaml_override(self):
        """YAML settings should override defaults."""
        from multi_agent_coder.config import Config
        cfg = Config(yaml_data={
            "search_enabled": False,
            "search_provider": "google",
            "search_api_key": "mykey:mycx",
            "search_max_results": 5,
        })
        assert cfg.SEARCH_ENABLED is False
        assert cfg.SEARCH_PROVIDER == "google"
        assert cfg.SEARCH_API_KEY == "mykey:mycx"
        assert cfg.SEARCH_MAX_RESULTS == 5

    def test_env_override(self, monkeypatch):
        """Env vars should override YAML and defaults."""
        from multi_agent_coder.config import Config
        monkeypatch.setenv("SEARCH_ENABLED", "false")
        monkeypatch.setenv("SEARCH_PROVIDER", "serpapi")
        cfg = Config()
        assert cfg.SEARCH_ENABLED is False
        assert cfg.SEARCH_PROVIDER == "serpapi"

    def test_to_dict_includes_search(self):
        """to_dict() should include search settings."""
        from multi_agent_coder.config import Config
        cfg = Config()
        d = cfg.to_dict()
        assert "search_enabled" in d
        assert "search_provider" in d
        assert "search_api_key" in d
        assert "search_api_url" in d
        assert "search_max_results" in d
        assert "search_max_page_chars" in d


# ── Diagnosis integration test ───────────────────────────────


class TestDiagnosisWithSearch:
    """Test that _diagnose_failure uses search agent context."""

    @patch("multi_agent_coder.orchestrator.diagnosis.token_tracker")
    def test_diagnosis_includes_search_context(self, mock_tracker):
        """When search_agent is provided, its results should appear in the LLM prompt."""
        mock_tracker.total_prompt_tokens = 0
        mock_tracker.total_completion_tokens = 0

        mock_llm = MagicMock()
        mock_llm.generate_response.return_value = "ROOT CAUSE: test\nFIX: none"

        mock_display = MagicMock()
        mock_display.steps = [{"type": "CODE"}]

        mock_memory = MagicMock()
        mock_memory.related_context.return_value = ""
        mock_memory.all_files.return_value = {}
        mock_memory.summary.return_value = "1 file"

        mock_search = MagicMock()
        mock_search.search_for_error.return_value = (
            "=== Web Search Results ===\n[1] Fix Guide\nUse pip install flask"
        )

        from multi_agent_coder.orchestrator.diagnosis import _diagnose_failure
        _diagnose_failure(
            "Install dependencies",
            "CODE",
            "ModuleNotFoundError: No module named 'flask'",
            mock_memory, mock_llm, mock_display, 0,
            search_agent=mock_search,
            language="python",
        )

        # Verify the LLM was called with the search context
        call_args = mock_llm.generate_response.call_args[0][0]
        assert "Web Search Results" in call_args
        assert "Fix Guide" in call_args

    @patch("multi_agent_coder.orchestrator.diagnosis.token_tracker")
    def test_diagnosis_works_without_search(self, mock_tracker):
        """Without search_agent, diagnosis should work normally (backward compat)."""
        mock_tracker.total_prompt_tokens = 0
        mock_tracker.total_completion_tokens = 0

        mock_llm = MagicMock()
        mock_llm.generate_response.return_value = "ROOT CAUSE: test\nFIX: none"

        mock_display = MagicMock()
        mock_display.steps = [{"type": "CODE"}]

        mock_memory = MagicMock()
        mock_memory.related_context.return_value = ""
        mock_memory.all_files.return_value = {}
        mock_memory.summary.return_value = "1 file"

        from multi_agent_coder.orchestrator.diagnosis import _diagnose_failure
        result = _diagnose_failure(
            "Install dependencies",
            "CODE",
            "ModuleNotFoundError: No module named 'flask'",
            mock_memory, mock_llm, mock_display, 0,
        )

        assert result == "ROOT CAUSE: test\nFIX: none"
        # Search context should NOT be mentioned
        call_args = mock_llm.generate_response.call_args[0][0]
        assert "Web Search Results" not in call_args

    @patch("multi_agent_coder.orchestrator.diagnosis.token_tracker")
    def test_diagnosis_handles_search_exception(self, mock_tracker):
        """If search_agent.search_for_error raises, diagnosis should still work."""
        mock_tracker.total_prompt_tokens = 0
        mock_tracker.total_completion_tokens = 0

        mock_llm = MagicMock()
        mock_llm.generate_response.return_value = "ROOT CAUSE: test\nFIX: none"

        mock_display = MagicMock()
        mock_display.steps = [{"type": "CODE"}]

        mock_memory = MagicMock()
        mock_memory.related_context.return_value = ""
        mock_memory.all_files.return_value = {}
        mock_memory.summary.return_value = "1 file"

        mock_search = MagicMock()
        mock_search.search_for_error.side_effect = Exception("Network down")

        from multi_agent_coder.orchestrator.diagnosis import _diagnose_failure
        result = _diagnose_failure(
            "Install deps",
            "CODE",
            "Error: something failed",
            mock_memory, mock_llm, mock_display, 0,
            search_agent=mock_search,
        )

        # Should still return diagnosis, no crash
        assert result == "ROOT CAUSE: test\nFIX: none"
