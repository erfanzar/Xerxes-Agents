# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Google web-search agent tool.

Two-mode resolution per call:

1. **Custom Search JSON API** — used when ``GOOGLE_API_KEY`` and
   ``GOOGLE_CSE_ID`` are set. Quota: 100 free queries/day. Returns
   structured JSON, never blocked. This is the recommended mode.
2. **HTML scrape fallback** — straight ``GET https://www.google.com/search``
   with a browser User-Agent. No key required, but Google rate-limits
   anonymous scraping aggressively (HTTP 429, sometimes a CAPTCHA).
   Use this for quick local development or tests; not for production.

The tool always returns the same envelope shape so the agent can
chain it the same way regardless of which path was taken.

Example::

    from xerxes_agent.tools.google_search import GoogleSearch
    out = GoogleSearch.static_call("ai news 2026", n_results=5)
    for hit in out["results"]:
        print(hit["title"], hit["url"])
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
    """Endpoint + auth used by the Google search tool."""

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
    """Update the process-wide Google search configuration."""
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
    """Return the currently active Google search configuration."""
    return _config


def set_google_search_client(client: tp.Any | None) -> None:
    """Install (or clear) an injected HTTP client for tests.

    The client must expose ``get(url, headers=None, params=None)`` and
    return either an object with ``.text`` and ``.json()``, or a dict.
    """
    global _http_client
    _http_client = client


def _http_get(url: str, *, headers: dict[str, str] | None = None, params: dict[str, str] | None = None) -> tp.Any:
    """Issue a GET using the injected test client or fall back to ``httpx``."""
    if _http_client is not None:
        return _http_client.get(url, headers=headers, params=params)
    try:
        import httpx  # type: ignore
    except ImportError as exc:
        raise RuntimeError("httpx required for GoogleSearch HTTP fallback") from exc
    return httpx.get(url, headers=headers or {}, params=params or {}, timeout=20.0, follow_redirects=True)


def _resp_text(resp: tp.Any) -> str:
    """Extract the response body as text, accepting both ``.text`` and ``.body``."""
    text = getattr(resp, "text", None)
    if isinstance(text, str):
        return text
    body = getattr(resp, "body", "") or ""
    if isinstance(body, bytes):
        body = body.decode(errors="replace")
    return body


def _resp_json(resp: tp.Any) -> dict[str, tp.Any]:
    """Decode the response as JSON, trying ``resp.json()`` then raw text parsing."""
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
    """Return the HTTP status code, or ``0`` when the response lacks one."""
    return int(getattr(resp, "status_code", 0) or 0)


def _search_via_api(
    query: str,
    *,
    n_results: int,
    cfg: GoogleSearchConfig,
    site: str | None,
    time_range: str | None,
) -> dict[str, tp.Any]:
    """Execute the search against the Google Custom Search JSON API.

    Args:
        query: Free-text query.
        n_results: Number of hits to return (clamped to 1-10 by the API).
        cfg: Resolved configuration carrying the API key and CSE id.
        site: Optional ``site:`` restriction prepended to the query.
        time_range: Optional ``dateRestrict`` filter (e.g. ``"d1"``, ``"w"``).

    Returns:
        Normalised result dict with ``engine="google_api"`` and a list of
        ``{title, url, snippet, displayed_url}`` entries.
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
    """Remove HTML tags from *html* and return the stripped inner text."""
    return re.sub(r"<[^>]+>", "", html or "").strip()


def _search_via_scrape(
    query: str,
    *,
    n_results: int,
    cfg: GoogleSearchConfig,
    site: str | None,
    time_range: str | None,
) -> dict[str, tp.Any]:
    """Execute the search by fetching and parsing ``google.com/search`` HTML.

    This path is used when no API credentials are configured; it is
    aggressively rate-limited by Google. When BeautifulSoup is
    unavailable, a regex fallback on ``<a>`` / ``<h3>`` is used.

    Args:
        query: Free-text query.
        n_results: Maximum number of hits to return.
        cfg: Resolved configuration (user-agent, safe-search, base URL).
        site: Optional ``site:`` restriction prepended to the query.
        time_range: Optional recency filter — first letter maps to Google's
            ``tbs=qdr:`` parameter (e.g. ``"d"`` → ``qdr:d``).

    Returns:
        Normalised result dict with ``engine="google_scrape"``.
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
    try:
        from bs4 import BeautifulSoup  # type: ignore
    except ImportError:
        BeautifulSoup = None  # type: ignore[assignment]
    if BeautifulSoup is not None:
        soup = BeautifulSoup(html, "html.parser")
        for h3 in soup.find_all("h3")[: n_results * 4]:
            link = h3.find_parent("a")
            if not link:
                continue
            url = link.get("href", "")
            if not url.startswith(("http://", "https://")):
                continue
            host = urllib.parse.urlparse(url).netloc.lower()
            if host.endswith(("google.com", "googleusercontent.com", "gstatic.com")) or host.startswith("webcache."):
                continue
            title = h3.get_text(" ", strip=True)
            snippet = ""
            container = link.find_parent(class_=lambda c: bool(c and "g" in c.split()))
            if container is None:
                container = link
            for cand in container.find_all(class_=lambda c: bool(c and ("VwiC3b" in c or "MUxGbd" in c))):
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
    """Search Google and return ranked results."""

    @staticmethod
    def static_call(
        query: str,
        n_results: int = 5,
        site: str | None = None,
        time_range: str | None = None,
        **context_variables: tp.Any,
    ) -> dict[str, tp.Any]:
        """Run a Google web search and return up to *n_results* hits.

        Use this whenever the user wants live web results — it is the
        primary search tool and should be preferred over any other
        engine. The tool resolves to the Google Custom Search JSON
        API when ``GOOGLE_API_KEY`` and ``GOOGLE_CSE_ID`` are set,
        otherwise it falls back to a direct HTML fetch of
        ``google.com/search`` (which Google rate-limits aggressively).

        Args:
            query: Free-text search phrase.
            n_results: Max number of results (1-10 in API mode, 1-30
                in scrape mode).
            site: Optional ``site:`` restriction (e.g. ``"github.com"``).
            time_range: Optional recency filter — one of ``"d"`` (day),
                ``"w"`` (week), ``"m"`` (month), ``"y"`` (year).

        Returns:
            ``{"engine", "query", "count", "results": [{"title", "url",
            "snippet"}], "warning"|"error"}``.
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
