"""
Tests for the [SEARCH] step type integration:
- Classifier recognizes SEARCH
- _handle_search_step calls search_agent and stores results
- _handle_search_step graceful skip when no search_agent
- _handle_search_step graceful failure on exception
"""
import pytest
from unittest.mock import MagicMock, patch


# ── Classifier tests ────────────────────────────────────────────


class TestClassifySearchStep:
    """Verify _classify_step recognizes SEARCH."""

    @patch("multi_agent_coder.orchestrator.classification.token_tracker")
    def test_classify_returns_search(self, mock_tracker):
        """When the LLM responds with SEARCH, _classify_step should return it."""
        mock_tracker.total_prompt_tokens = 0
        mock_tracker.total_completion_tokens = 0

        mock_llm = MagicMock()
        mock_llm.generate_response.return_value = "SEARCH"

        mock_display = MagicMock()

        from multi_agent_coder.orchestrator.classification import _classify_step
        result = _classify_step(
            "Search for the latest Next.js 15 migration guide",
            mock_llm, mock_display, 0,
        )
        assert result == "SEARCH"

    @patch("multi_agent_coder.orchestrator.classification.token_tracker")
    def test_classify_prompt_mentions_search(self, mock_tracker):
        """The classification prompt should mention SEARCH as a valid option."""
        mock_tracker.total_prompt_tokens = 0
        mock_tracker.total_completion_tokens = 0

        mock_llm = MagicMock()
        mock_llm.generate_response.return_value = "CODE"

        mock_display = MagicMock()

        from multi_agent_coder.orchestrator.classification import _classify_step
        _classify_step("dummy", mock_llm, mock_display, 0)

        # Inspect the prompt sent to the LLM
        prompt = mock_llm.generate_response.call_args[0][0]
        assert "SEARCH" in prompt
        assert "search the web" in prompt.lower()


# ── Handler tests ────────────────────────────────────────────────


class TestHandleSearchStep:
    """Tests for _handle_search_step."""

    def test_calls_search_agent(self):
        """Should call search_for_task and store results in memory."""
        from multi_agent_coder.orchestrator.step_handlers import _handle_search_step

        mock_search = MagicMock()
        mock_search.search_for_task.return_value = (
            "=== Web Search Context ===\n[1] Next.js 15 Guide\nMigration steps..."
        )

        mock_memory = MagicMock()
        mock_display = MagicMock()

        success, error = _handle_search_step(
            "Search for the latest Next.js 15 migration guide",
            mock_search, mock_memory, mock_display, step_idx=2,
            language="javascript",
        )

        assert success is True
        assert error == ""
        mock_search.search_for_task.assert_called_once_with(
            "Search for the latest Next.js 15 migration guide",
            language="javascript",
        )
        # Results should be stored in memory under _search_context/
        mock_memory.update.assert_called_once()
        stored = mock_memory.update.call_args[0][0]
        assert "_search_context/step_3.txt" in stored
        assert "Next.js 15 Guide" in stored["_search_context/step_3.txt"]

    def test_no_search_agent_skips(self):
        """When search_agent is None, should skip gracefully."""
        from multi_agent_coder.orchestrator.step_handlers import _handle_search_step

        mock_memory = MagicMock()
        mock_display = MagicMock()

        success, error = _handle_search_step(
            "Search for docs", None, mock_memory, mock_display, step_idx=0,
        )

        assert success is True
        assert error == ""
        # Memory should not be updated
        mock_memory.update.assert_not_called()

    def test_search_exception_handled(self):
        """If search_for_task raises, should return success (best-effort)."""
        from multi_agent_coder.orchestrator.step_handlers import _handle_search_step

        mock_search = MagicMock()
        mock_search.search_for_task.side_effect = Exception("Network down")

        mock_memory = MagicMock()
        mock_display = MagicMock()

        success, error = _handle_search_step(
            "Search for API docs", mock_search, mock_memory, mock_display,
            step_idx=1,
        )

        assert success is True
        assert error == ""
        mock_memory.update.assert_not_called()

    def test_empty_results(self):
        """When search returns empty string, should still succeed."""
        from multi_agent_coder.orchestrator.step_handlers import _handle_search_step

        mock_search = MagicMock()
        mock_search.search_for_task.return_value = ""

        mock_memory = MagicMock()
        mock_display = MagicMock()

        success, error = _handle_search_step(
            "Search for obscure thing", mock_search, mock_memory, mock_display,
            step_idx=0,
        )

        assert success is True
        assert error == ""
        # Should NOT store empty results
        mock_memory.update.assert_not_called()
