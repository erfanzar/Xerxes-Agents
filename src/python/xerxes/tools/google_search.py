# Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Google search integration supporting both API and web scraping backends.

This module provides Google search functionality through the Custom Search API
(preferred when credentials are available) or HTML scraping fallback.

Example:
    >>> from xerxes.tools.google_search import GoogleSearch, configure_google_search
    >>> configure_google_search(api_key="...", cse_id="...")
    >>> results = GoogleSearch.static_call(query="machine learning")
"""

from __future__ import annotations

import logging
import os
import re
import typing as tp
import urllib.parse
from dataclasses import dataclass

from ..types import AgentBaseFn

logger = logging.getLogger(__name__)
SCRAPE_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15"
)


@dataclass
class GoogleSearchConfig:
    """Configuration for Google search operations.

    Attributes:
        api_key: Google API key for Custom Search API.
        cse_id: Custom Search Engine ID.
        api_base: Base URL for the API endpoint.
        scrape_base: Base URL for scraping.
        safe: Safe search setting ('active' or 'off').
        user_agent: User agent string for scraping.
    """

    api_key: str = ""
    cse_id: str = ""
    api_base: str = "https://www.googleapis.com/customsearch/v1"
    scrape_base: str = "https://www.google.com/search"
    safe: str = "off"
    user_agent: str = SCRAPE_USER_AGENT


_config = GoogleSearchConfig(
    api_key=os.environ.get("GOOGLE_API_KEY", ""),
    cse_id=os.environ.get("GOOGLE_CSE_ID", ""),
)
_http_client: tp.Any | None = None


def configure_google_search(
    *,
    api_key: str | None = None,
    cse_id: str | None = None,
    safe: str | None = None,
    user_agent: str | None = None,
) -> GoogleSearchConfig:
    """Configure Google search settings.

    Args:
        api_key: Google API key. Overrides environment variable.
        cse_id: Custom Search Engine ID. Overrides environment variable.
        safe: Safe search setting ('active' or 'off').
        user_agent: Custom user agent string.

    Returns:
        Updated GoogleSearchConfig.

    Example:
        >>> configure_google_search(
        ...     api_key="your-api-key",
        ...     cse_id="your-cse-id"
        ... )
    """
    global _config
    if api_key is not None:
        _config.api_key = api_key
    if cse_id is not None:
        _config.cse_id = cse_id
    if safe is not None:
        _config.safe = safe
    if user_agent is not None:
        _config.user_agent = user_agent
    return _config


def get_google_search_config() -> GoogleSearchConfig:
    """Get the current Google search configuration.

    Returns:
        The current GoogleSearchConfig instance.
    """
    return _config


def set_google_search_client(client: tp.Any | None) -> None:
    """Set a custom HTTP client for Google searches.

    Args:
        client: Custom HTTP client instance, or None to use default httpx.
    """
    global _http_client
    _http_client = client


def _http_get(url: str, *, headers: dict[str, str] | None = None, params: dict[str, str] | None = None) -> tp.Any:
    """Perform HTTP GET request using configured client.

    Args:
        url: URL to fetch.
        headers: Optional HTTP headers.
        params: Optional query parameters.

    Returns:
        HTTP response object.
    """
    if _http_client is not None:
        return _http_client.get(url, headers=headers, params=params)
    try:
        import httpx
    except ImportError as exc:
        raise RuntimeError("httpx required for GoogleSearch HTTP fallback") from exc
    return httpx.get(url, headers=headers or {}, params=params or {}, timeout=20.0, follow_redirects=True)


def _resp_text(resp: tp.Any) -> str:
    """Extract text content from HTTP response.

    Args:
        resp: HTTP response object.

    Returns:
        Response body as string.
    """
    text = getattr(resp, "text", None)
    if isinstance(text, str):
        return text
    body = getattr(resp, "body", "") or ""
    if isinstance(body, bytes):
        body = body.decode(errors="replace")
    return body


def _resp_json(resp: tp.Any) -> dict[str, tp.Any]:
    """Parse JSON from HTTP response.

    Args:
        resp: HTTP response object.

    Returns:
        Parsed JSON as dictionary.
    """
    if hasattr(resp, "json") and callable(resp.json):
        try:
            return resp.json()
        except Exception:
            pass
    text = _resp_text(resp)
    try:
        import json

        return json.loads(text)
    except Exception:
        return {}


def _resp_status(resp: tp.Any) -> int:
    """Get HTTP status code from response.

    Args:
        resp: HTTP response object.

    Returns:
        Status code as integer.
    """
    return int(getattr(resp, "status_code", 0) or 0)


def _search_via_api(
    query: str,
    *,
    n_results: int,
    cfg: GoogleSearchConfig,
    site: str | None,
    time_range: str | None,
) -> dict[str, tp.Any]:
    """Search using Google Custom Search API.

    Args:
        query: Search query string.
        n_results: Number of results to return.
        cfg: Search configuration.
        site: Optional site restriction.
        time_range: Optional time range filter.

    Returns:
        Dictionary with search results and metadata.
    """
    params: dict[str, str] = {
        "key": cfg.api_key,
        "cx": cfg.cse_id,
        "q": (f"site:{site} " if site else "") + query,
        "num": str(min(max(n_results, 1), 10)),
        "safe": "active" if cfg.safe == "active" else "off",
    }
    if time_range:
        params["dateRestrict"] = time_range
    resp = _http_get(cfg.api_base, params=params)
    status = _resp_status(resp)
    if status and status >= 400:
        return {
            "engine": "google_api",
            "query": query,
            "error": f"HTTP {status}",
            "results": [],
        }
    payload = _resp_json(resp)
    items = payload.get("items") or []
    out_items: list[dict[str, tp.Any]] = []
    for it in items[:n_results]:
        out_items.append(
            {
                "title": it.get("title", ""),
                "url": it.get("link", ""),
                "snippet": it.get("snippet", ""),
                "displayed_url": it.get("displayLink", ""),
            }
        )
    return {
        "engine": "google_api",
        "query": query,
        "count": len(out_items),
        "results": out_items,
        "search_information": payload.get("searchInformation", {}),
    }


_RESULT_RE = re.compile(
    r'<a[^>]+href="(?P<url>https?://[^"#&]+)"[^>]*>'
    r".*?<h3[^>]*>(?P<title>.*?)</h3>"
    r'(?:.*?<div[^>]*class="VwiC3b[^"]*"[^>]*>(?P<snippet>.*?)</div>)?',
    re.DOTALL,
)


def _strip_tags(html: str) -> str:
    """Remove HTML tags from string.

    Args:
        html: HTML string to strip.

    Returns:
        Plain text with HTML tags removed.
    """
    return re.sub(r"<[^>]+>", "", html or "").strip()


def _search_via_scrape(
    query: str,
    *,
    n_results: int,
    cfg: GoogleSearchConfig,
    site: str | None,
    time_range: str | None,
) -> dict[str, tp.Any]:
    """Search by scraping Google HTML.

    Args:
        query: Search query string.
        n_results: Number of results to return.
        cfg: Search configuration.
        site: Optional site restriction.
        time_range: Optional time range filter.

    Returns:
        Dictionary with search results and metadata.
    """
    q = (f"site:{site} " if site else "") + query
    params: dict[str, str] = {
        "q": q,
        "num": str(min(max(n_results, 1), 30)),
        "hl": "en",
    }
    if cfg.safe in ("active", "on"):
        params["safe"] = "active"
    if time_range:
        params["tbs"] = f"qdr:{time_range[0]}"
    headers = {
        "User-Agent": cfg.user_agent,
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    try:
        resp = _http_get(cfg.scrape_base, headers=headers, params=params)
    except Exception as exc:
        return {"engine": "google_scrape", "query": query, "error": str(exc), "results": []}
    status = _resp_status(resp)
    html = _resp_text(resp)
    if status and status >= 400:
        return {
            "engine": "google_scrape",
            "query": query,
            "error": f"HTTP {status} (Google likely blocked the scrape — set GOOGLE_API_KEY+GOOGLE_CSE_ID)",
            "results": [],
        }
    out: list[dict[str, str]] = []
    _BeautifulSoup: type | None = None
    try:
        from bs4 import BeautifulSoup, Tag

        _BeautifulSoup = BeautifulSoup
    except ImportError:
        pass
    if _BeautifulSoup is not None:
        soup = _BeautifulSoup(html, "html.parser")
        for h3 in soup.find_all("h3")[: n_results * 4]:
            link = h3.find_parent("a")
            if not link:
                continue
            tag_link = tp.cast(Tag, link)
            url = str(tag_link.get("href", ""))
            if not url.startswith(("http://", "https://")):
                continue
            host = urllib.parse.urlparse(url).netloc.lower()
            if host.endswith(("google.com", "googleusercontent.com", "gstatic.com")) or host.startswith("webcache."):
                continue
            title = h3.get_text(" ", strip=True)
            snippet = ""
            container = tag_link.find_parent(class_=lambda c: bool(c and "g" in c.split()))
            if container is None:
                container = tag_link
            tag_container = tp.cast(Tag, container)
            for cand in tag_container.find_all(class_=lambda c: bool(c and ("VwiC3b" in c or "MUxGbd" in c))):
                txt = cand.get_text(" ", strip=True)
                if txt and len(txt) > 20:
                    snippet = txt[:300]
                    break
            out.append({"title": title, "url": url, "snippet": snippet})
            if len(out) >= n_results:
                break
    if not out:
        for m in _RESULT_RE.finditer(html):
            url = m.group("url")
            if "google.com" in url or url.startswith("https://webcache"):
                continue
            title = _strip_tags(m.group("title"))
            snippet = _strip_tags(m.group("snippet") or "")
            out.append({"title": title, "url": url, "snippet": snippet[:300]})
            if len(out) >= n_results:
                break
    return {
        "engine": "google_scrape",
        "query": query,
        "count": len(out),
        "results": out,
        "warning": (
            "Anonymous scrapes are rate-limited; set GOOGLE_API_KEY + GOOGLE_CSE_ID for a quota-backed path."
            if not out
            else ""
        ),
    }


class GoogleSearch(AgentBaseFn):
    """Perform Google searches using API or scraping.

    Uses Google Custom Search API when credentials are configured,
    falls back to HTML scraping otherwise.

    Example:
        >>> results = GoogleSearch.static_call(
        ...     query="Python tutorials",
        ...     n_results=10
        ... )
    """

    @staticmethod
    def static_call(
        query: str,
        n_results: int = 5,
        site: str | None = None,
        time_range: str | None = None,
        **context_variables: tp.Any,
    ) -> dict[str, tp.Any]:
        """Perform a Google search.

        Args:
            query: Search query string.
            n_results: Number of results to return. Defaults to 5.
            site: Optional site to restrict search to.
            time_range: Optional time filter ('d' for day, 'w' for week, 'm' for month, 'y' for year).
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Dictionary with search results containing 'results' list and metadata.

        Example:
            >>> GoogleSearch.static_call("AI research", n_results=10)
        """
        if isinstance(n_results, str):
            try:
                n_results = int(n_results)
            except ValueError:
                n_results = 5
        return GoogleSearch._execute(query, n_results, site, time_range)

    @staticmethod
    def _execute(
        query: str,
        n_results: int = 5,
        site: str | None = None,
        time_range: str | None = None,
    ) -> dict[str, tp.Any]:
        """Internal search execution with backend selection.

        Args:
            query: Search query.
            n_results: Number of results.
            site: Optional site filter.
            time_range: Optional time filter.

        Returns:
            Search results dictionary.
        """
        cfg = get_google_search_config()
        if cfg.api_key and cfg.cse_id:
            return _search_via_api(query, n_results=n_results, cfg=cfg, site=site, time_range=time_range)
        return _search_via_scrape(query, n_results=n_results, cfg=cfg, site=site, time_range=time_range)


__all__ = [
    "GoogleSearch",
    "GoogleSearchConfig",
    "configure_google_search",
    "get_google_search_config",
    "set_google_search_client",
]
