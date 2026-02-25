"""
Search Agent — searches the web for error documentation when the LLM
encounters an unrecognized error during diagnosis.
"""

import re

from ..cli_display import log
from ..search_provider import web_search, fetch_page_text, SearchResult


class SearchAgent:
    """Optional agent that enriches error diagnosis with web search context.

    Unlike other agents, SearchAgent does **not** use an LLM — it performs
    web searches and fetches documentation pages directly.

    All methods are best-effort: exceptions are caught and logged, never
    propagated.  An empty string return means "no useful context found".
    """

    def __init__(self, provider: str = "duckduckgo",
                 api_key: str = "", api_url: str = "",
                 max_results: int = 3, max_page_chars: int = 3000):
        self.provider = provider
        self.api_key = api_key
        self.api_url = api_url
        self.max_results = max_results
        self.max_page_chars = max_page_chars

    # ── Public API ───────────────────────────────────────────

    def search_for_error(self, error_info: str,
                         step_text: str = "",
                         language: str | None = None) -> str:
        """Search the web for information about an error.

        Args:
            error_info: The error output / traceback from the failed step.
            step_text: The step description (for additional context).
            language: Programming language (adds focused keywords).

        Returns:
            Formatted string with search results and excerpts, or empty
            string if nothing useful was found or on any error.
        """
        try:
            query = self._build_search_query(error_info, language)
            if not query:
                return ""

            log.info(f"[SearchAgent] Searching: {query}")

            results = web_search(
                query,
                provider=self.provider,
                api_key=self.api_key,
                api_url=self.api_url,
                max_results=self.max_results,
            )

            if not results:
                log.info("[SearchAgent] No search results found")
                return ""

            return self._format_results(results)

        except Exception as exc:
            log.warning(f"[SearchAgent] Search failed: {exc}")
            return ""

    def search_for_task(self, task: str,
                        language: str | None = None) -> str:
        """Search the web for latest documentation relevant to a task.

        Called **before** the planner generates steps so that local LLMs
        receive up-to-date, command-level guidance (framework CLI flags,
        install commands, API changes, etc.).

        Args:
            task: The user's task description / prompt.
            language: Programming language (adds focused keywords).

        Returns:
            Formatted string with search results and page excerpts, or
            empty string if nothing useful was found or on any error.
        """
        try:
            query = self._build_task_query(task, language)
            if not query:
                return ""

            log.info(f"[SearchAgent] Planning search: {query}")

            results = web_search(
                query,
                provider=self.provider,
                api_key=self.api_key,
                api_url=self.api_url,
                max_results=self.max_results,
            )

            if not results:
                log.info("[SearchAgent] No planning search results found")
                return ""

            header = ("=== Web Search Context (latest documentation) ===\n"
                      "Use the information below for accurate, up-to-date "
                      "commands and flags in your plan.")
            return self._format_results(results, header=header)

        except Exception as exc:
            log.warning(f"[SearchAgent] Planning search failed: {exc}")
            return ""

    # ── Internals ────────────────────────────────────────────

    def _build_search_query(self, error_info: str,
                            language: str | None = None) -> str:
        """Extract a focused search query from the error output.

        Strategy:
        1. Find the first "key error line" — the line that contains the
           actual error message (e.g., ``ModuleNotFoundError: No module
           named 'foo'``).
        2. Strip file paths, line numbers, and timestamp noise.
        3. Append language keyword for relevance.
        4. Cap length so the search engine doesn't choke.
        """
        if not error_info or not error_info.strip():
            return ""

        key_line = self._extract_key_error_line(error_info)
        if not key_line:
            # Fallback: use the last non-empty line
            lines = [l.strip() for l in error_info.strip().splitlines() if l.strip()]
            key_line = lines[-1] if lines else ""

        if not key_line:
            return ""

        # Strip file paths and line numbers
        query = re.sub(r'(?:File\s+)?["\']?[\w/\\.:]+\.(py|js|jsx|ts|tsx|go|rb|java|rs)\b["\']?',
                       '', key_line)
        query = re.sub(r',?\s*line\s+\d+', '', query, flags=re.IGNORECASE)
        # Strip ANSI escape codes
        query = re.sub(r'\x1b\[[0-9;]*m', '', query)
        # Collapse whitespace
        query = re.sub(r'\s+', ' ', query).strip()

        # Add language keyword
        lang_keywords = {
            "python": "python",
            "javascript": "javascript node.js",
            "typescript": "typescript",
            "go": "golang",
            "ruby": "ruby",
            "java": "java",
            "rust": "rust",
        }
        if language and language in lang_keywords:
            query = f"{query} {lang_keywords[language]}"

        # Cap length (search engines typically handle ~150 chars well)
        if len(query) > 150:
            query = query[:150].rsplit(' ', 1)[0]

        return query

    def _build_task_query(self, task: str,
                          language: str | None = None) -> str:
        """Build a documentation-focused search query from a task description.

        Unlike ``_build_search_query`` (which targets error messages), this
        method extracts **technology keywords** from the user prompt and
        appends terms like "latest docs guide" so the search engine returns
        setup guides, CLI references, and recent release notes.
        """
        if not task or not task.strip():
            return ""

        # Strip ANSI codes (just in case)
        clean = re.sub(r'\x1b\[[0-9;]*m', '', task)
        # Collapse whitespace
        clean = re.sub(r'\s+', ' ', clean).strip()

        # Build query: use the task itself (trimmed) + documentation focus
        query = clean

        # Add language keyword for relevance
        lang_keywords = {
            "python": "python",
            "javascript": "javascript node.js",
            "typescript": "typescript",
            "go": "golang",
            "ruby": "ruby",
            "java": "java",
            "rust": "rust",
        }
        if language and language in lang_keywords:
            query = f"{query} {lang_keywords[language]}"

        # Append documentation focus
        query = f"{query} latest docs guide"

        # Cap length
        if len(query) > 150:
            query = query[:150].rsplit(' ', 1)[0]

        return query

    @staticmethod
    def _extract_key_error_line(error_info: str) -> str:
        """Find the most informative error line in a traceback/output.

        Looks for common error patterns:
        - Python: ``SomeError: message``
        - Node.js: ``Error: message`` or ``TypeError: message``
        - Jest/Node: ``Cannot find module``, ``Module not found``
        - Generic: lines containing "error", "failed"
        """
        lines = error_info.strip().splitlines()

        # Python-style: "ErrorClassName: message"
        for line in reversed(lines):
            line = line.strip()
            if re.match(r'^[A-Z]\w*(Error|Exception|Warning):\s', line):
                return line

        # Node.js style: "Error: ..." or "TypeError: ..."
        for line in lines:
            line = line.strip()
            if re.match(r'^(Error|TypeError|ReferenceError|SyntaxError|'
                        r'RangeError|URIError|EvalError):', line):
                return line

        # Jest/Node descriptive errors (no "Error:" prefix but highly
        # informative — e.g. "Cannot find module '@testing-library/react'")
        _DESCRIPTIVE_PATTERNS = (
            r'Cannot find module\b',
            r'Module not found\b',
            r'Module build failed\b',
            r'Failed to compile\b',
            r'is not defined\b',
            r'is not a function\b',
            r'Unexpected token\b',
            r'Cannot read propert',       # Cannot read property / properties
            r'command not found\b',
            r'No such file or directory\b',
            r'Permission denied\b',
        )
        for line in lines:
            stripped = line.strip()
            if stripped and any(re.search(p, stripped, re.IGNORECASE)
                               for p in _DESCRIPTIVE_PATTERNS):
                return stripped

        # Generic: first line containing "error" or "failed" (but skip
        # bare Jest markers like "FAIL src/App.test.tsx")
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            # Skip bare FAIL markers (just "FAIL <path>")
            if re.match(r'^FAIL\s+\S+$', stripped):
                continue
            if re.search(r'\b(error|failed)\b', stripped, re.IGNORECASE):
                return stripped

        # Fallback: first non-empty line that isn't a bare marker
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("Traceback"):
                continue
            # Skip bare FAIL/PASS markers
            if re.match(r'^(FAIL|PASS)\s+\S+$', stripped):
                continue
            # Skip Jest section markers (● ...)
            if stripped.startswith('●'):
                continue
            return stripped

        return ""

    def _format_results(self, results: list[SearchResult], *,
                        header: str | None = None) -> str:
        """Format search results with fetched page excerpts."""
        sections: list[str] = []

        for i, result in enumerate(results, 1):
            section = f"[{i}] {result.title}\n    URL: {result.url}"
            if result.snippet:
                section += f"\n    Snippet: {result.snippet}"

            # Fetch page content for richer context
            page_text = fetch_page_text(result.url,
                                        max_chars=self.max_page_chars)
            if page_text:
                # Truncate to a reasonable excerpt
                excerpt = page_text[:1500]
                if len(page_text) > 1500:
                    excerpt = excerpt.rsplit(' ', 1)[0] + "..."
                section += f"\n    Page excerpt: {excerpt}"

            sections.append(section)

        if not sections:
            return ""

        if header is None:
            header = "=== Web Search Results (documentation/solutions found) ==="
        return header + "\n\n" + "\n\n".join(sections)
