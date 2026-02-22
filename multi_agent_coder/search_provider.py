"""
Search provider — web search abstraction supporting multiple backends.

Supports DuckDuckGo (free, no key), Google Custom Search, and SerpAPI.
"""

import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Optional
from urllib.parse import quote_plus, urlencode

import requests

from .cli_display import log


@dataclass
class SearchResult:
    """A single web search result."""
    title: str
    url: str
    snippet: str


# ── HTML text extraction ─────────────────────────────────────


class _HTMLTextExtractor(HTMLParser):
    """Strip HTML tags and extract readable text."""

    _SKIP_TAGS = frozenset({
        "script", "style", "nav", "footer", "header", "aside", "noscript",
    })

    def __init__(self):
        super().__init__()
        self._pieces: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data):
        if self._skip_depth == 0:
            text = data.strip()
            if text:
                self._pieces.append(text)

    def get_text(self) -> str:
        return " ".join(self._pieces)


def _html_to_text(html: str) -> str:
    """Convert HTML to plain text, stripping tags and scripts."""
    extractor = _HTMLTextExtractor()
    try:
        extractor.feed(html)
    except Exception:
        # Fallback: crude regex strip
        text = re.sub(r"<[^>]+>", " ", html)
        return re.sub(r"\s+", " ", text).strip()
    return extractor.get_text()


# ── Page fetching ────────────────────────────────────────────


def fetch_page_text(url: str, max_chars: int = 3000,
                    timeout: float = 5.0) -> str:
    """Fetch a URL and return extracted plain text (up to *max_chars*).

    Returns empty string on any failure.
    """
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (compatible; AgentChanti/1.0; "
                "+https://github.com/agentchanti)"
            ),
            "Accept": "text/html,application/xhtml+xml",
        }
        resp = requests.get(url, headers=headers, timeout=timeout,
                            allow_redirects=True)
        resp.raise_for_status()

        content_type = resp.headers.get("Content-Type", "")
        if "text/html" not in content_type and "text/plain" not in content_type:
            return ""

        text = _html_to_text(resp.text)
        return text[:max_chars] if len(text) > max_chars else text
    except Exception as exc:
        log.debug(f"[Search] Failed to fetch {url}: {exc}")
        return ""


# ── DuckDuckGo provider ─────────────────────────────────────


def _search_duckduckgo(query: str, max_results: int = 3) -> list[SearchResult]:
    """Search DuckDuckGo HTML and parse results."""
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
    }
    try:
        resp = requests.get(url, headers=headers, timeout=8)
        resp.raise_for_status()
    except Exception as exc:
        log.warning(f"[Search] DuckDuckGo request failed: {exc}")
        return []

    results: list[SearchResult] = []
    html = resp.text

    # Parse DuckDuckGo HTML results
    # Results are in <a class="result__a"> with snippets in
    # <a class="result__snippet">
    link_pattern = re.compile(
        r'<a[^>]+class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
        re.DOTALL,
    )
    snippet_pattern = re.compile(
        r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
        re.DOTALL,
    )

    links = link_pattern.findall(html)
    snippets = snippet_pattern.findall(html)

    for i, (href, title_html) in enumerate(links):
        if i >= max_results:
            break
        title = _html_to_text(title_html).strip()
        snippet = _html_to_text(snippets[i]).strip() if i < len(snippets) else ""
        # DuckDuckGo wraps URLs in a redirect — extract the real URL
        real_url = href
        if "uddg=" in href:
            match = re.search(r"uddg=([^&]+)", href)
            if match:
                from urllib.parse import unquote
                real_url = unquote(match.group(1))
        if title and real_url:
            results.append(SearchResult(title=title, url=real_url, snippet=snippet))

    return results


# ── Google Custom Search provider ────────────────────────────


def _search_google(query: str, api_key: str, api_url: str = "",
                   max_results: int = 3) -> list[SearchResult]:
    """Search using Google Custom Search JSON API.

    *api_key* should be in the format ``"API_KEY:CX_ID"`` where CX_ID is the
    Custom Search Engine ID.
    """
    parts = api_key.split(":", 1)
    if len(parts) != 2:
        log.warning("[Search] Google search_api_key must be 'API_KEY:CX_ID'")
        return []

    key, cx = parts
    base = api_url or "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": key,
        "cx": cx,
        "q": query,
        "num": min(max_results, 10),
    }

    try:
        resp = requests.get(base, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        log.warning(f"[Search] Google API request failed: {exc}")
        return []

    results: list[SearchResult] = []
    for item in data.get("items", [])[:max_results]:
        results.append(SearchResult(
            title=item.get("title", ""),
            url=item.get("link", ""),
            snippet=item.get("snippet", ""),
        ))
    return results


# ── SerpAPI provider ─────────────────────────────────────────


def _search_serpapi(query: str, api_key: str, api_url: str = "",
                    max_results: int = 3) -> list[SearchResult]:
    """Search using SerpAPI."""
    base = api_url or "https://serpapi.com/search"
    params = {
        "api_key": api_key,
        "q": query,
        "engine": "google",
        "num": min(max_results, 10),
    }

    try:
        resp = requests.get(base, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        log.warning(f"[Search] SerpAPI request failed: {exc}")
        return []

    results: list[SearchResult] = []
    for item in data.get("organic_results", [])[:max_results]:
        results.append(SearchResult(
            title=item.get("title", ""),
            url=item.get("link", ""),
            snippet=item.get("snippet", ""),
        ))
    return results


# ── Public API ───────────────────────────────────────────────


def web_search(query: str, provider: str = "duckduckgo",
               api_key: str = "", api_url: str = "",
               max_results: int = 3) -> list[SearchResult]:
    """Run a web search using the configured provider.

    Args:
        query: Search query string.
        provider: One of ``"duckduckgo"``, ``"google"``, ``"serpapi"``.
        api_key: API key (required for google/serpapi).
        api_url: Optional base URL override.
        max_results: Maximum number of results to return.

    Returns:
        List of :class:`SearchResult`. Empty list on failure.
    """
    provider = provider.lower().strip()

    if provider == "google":
        if not api_key:
            log.warning("[Search] Google provider requires search_api_key")
            return []
        return _search_google(query, api_key, api_url, max_results)

    elif provider == "serpapi":
        if not api_key:
            log.warning("[Search] SerpAPI provider requires search_api_key")
            return []
        return _search_serpapi(query, api_key, api_url, max_results)

    else:
        # Default: DuckDuckGo (free, no key needed)
        if provider != "duckduckgo":
            log.warning(f"[Search] Unknown provider '{provider}', "
                        f"falling back to DuckDuckGo")
        return _search_duckduckgo(query, max_results)
