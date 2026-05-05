# Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
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
"""Google search module for Xerxes.

Exports:
    - logger
    - SCRAPE_USER_AGENT
    - GoogleSearchConfig
    - configure_google_search
    - get_google_search_config
    - set_google_search_client
    - GoogleSearch"""

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
    """Google search config.

    Attributes:
        api_key (str): api key.
        cse_id (str): cse id.
        api_base (str): api base.
        scrape_base (str): scrape base.
        safe (str): safe.
        user_agent (str): user agent."""

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
    """Configure google search.

    Args:
        api_key (str | None, optional): IN: api key. Defaults to None. OUT: Consumed during execution.
        cse_id (str | None, optional): IN: cse id. Defaults to None. OUT: Consumed during execution.
        safe (str | None, optional): IN: safe. Defaults to None. OUT: Consumed during execution.
        user_agent (str | None, optional): IN: user agent. Defaults to None. OUT: Consumed during execution.
    Returns:
        GoogleSearchConfig: OUT: Result of the operation."""

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
    """Retrieve the google search config.

    Returns:
        GoogleSearchConfig: OUT: Result of the operation."""

    return _config


def set_google_search_client(client: tp.Any | None) -> None:
    """Set the google search client.

    Args:
        client (tp.Any | None): IN: client. OUT: Consumed during execution."""

    global _http_client
    _http_client = client


def _http_get(url: str, *, headers: dict[str, str] | None = None, params: dict[str, str] | None = None) -> tp.Any:
    """Internal helper to http get.

    Args:
        url (str): IN: url. OUT: Consumed during execution.
        headers (dict[str, str] | None, optional): IN: headers. Defaults to None. OUT: Consumed during execution.
        params (dict[str, str] | None, optional): IN: params. Defaults to None. OUT: Consumed during execution.
    Returns:
        tp.Any: OUT: Result of the operation."""

    if _http_client is not None:
        return _http_client.get(url, headers=headers, params=params)
    try:
        import httpx
    except ImportError as exc:
        raise RuntimeError("httpx required for GoogleSearch HTTP fallback") from exc
    return httpx.get(url, headers=headers or {}, params=params or {}, timeout=20.0, follow_redirects=True)


def _resp_text(resp: tp.Any) -> str:
    """Internal helper to resp text.

    Args:
        resp (tp.Any): IN: resp. OUT: Consumed during execution.
    Returns:
        str: OUT: Result of the operation."""

    text = getattr(resp, "text", None)
    if isinstance(text, str):
        return text
    body = getattr(resp, "body", "") or ""
    if isinstance(body, bytes):
        body = body.decode(errors="replace")
    return body


def _resp_json(resp: tp.Any) -> dict[str, tp.Any]:
    """Internal helper to resp json.

    Args:
        resp (tp.Any): IN: resp. OUT: Consumed during execution.
    Returns:
        dict[str, tp.Any]: OUT: Result of the operation."""

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
    """Internal helper to resp status.

    Args:
        resp (tp.Any): IN: resp. OUT: Consumed during execution.
    Returns:
        int: OUT: Result of the operation."""

    return int(getattr(resp, "status_code", 0) or 0)


def _search_via_api(
    query: str,
    *,
    n_results: int,
    cfg: GoogleSearchConfig,
    site: str | None,
    time_range: str | None,
) -> dict[str, tp.Any]:
    """Internal helper to search via api.

    Args:
        query (str): IN: query. OUT: Consumed during execution.
        n_results (int): IN: n results. OUT: Consumed during execution.
        cfg (GoogleSearchConfig): IN: cfg. OUT: Consumed during execution.
        site (str | None): IN: site. OUT: Consumed during execution.
        time_range (str | None): IN: time range. OUT: Consumed during execution.
    Returns:
        dict[str, tp.Any]: OUT: Result of the operation."""

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
    """Internal helper to strip tags.

    Args:
        html (str): IN: html. OUT: Consumed during execution.
    Returns:
        str: OUT: Result of the operation."""

    return re.sub(r"<[^>]+>", "", html or "").strip()


def _search_via_scrape(
    query: str,
    *,
    n_results: int,
    cfg: GoogleSearchConfig,
    site: str | None,
    time_range: str | None,
) -> dict[str, tp.Any]:
    """Internal helper to search via scrape.

    Args:
        query (str): IN: query. OUT: Consumed during execution.
        n_results (int): IN: n results. OUT: Consumed during execution.
        cfg (GoogleSearchConfig): IN: cfg. OUT: Consumed during execution.
        site (str | None): IN: site. OUT: Consumed during execution.
        time_range (str | None): IN: time range. OUT: Consumed during execution.
    Returns:
        dict[str, tp.Any]: OUT: Result of the operation."""

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
    """Google search.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    def static_call(
        query: str,
        n_results: int = 5,
        site: str | None = None,
        time_range: str | None = None,
        **context_variables: tp.Any,
    ) -> dict[str, tp.Any]:
        """Static call.

        Args:
            query (str): IN: query. OUT: Consumed during execution.
            n_results (int, optional): IN: n results. Defaults to 5. OUT: Consumed during execution.
            site (str | None, optional): IN: site. Defaults to None. OUT: Consumed during execution.
            time_range (str | None, optional): IN: time range. Defaults to None. OUT: Consumed during execution.
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

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
        """Internal helper to execute.

        Args:
            query (str): IN: query. OUT: Consumed during execution.
            n_results (int, optional): IN: n results. Defaults to 5. OUT: Consumed during execution.
            site (str | None, optional): IN: site. Defaults to None. OUT: Consumed during execution.
            time_range (str | None, optional): IN: time range. Defaults to None. OUT: Consumed during execution.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

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
