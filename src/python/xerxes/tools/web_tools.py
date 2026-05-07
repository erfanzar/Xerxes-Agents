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
"""Web tools for scraping, API requests, RSS feeds, and URL analysis.

This module provides comprehensive web interaction tools including web scraping,
HTTP API clients, RSS feed readers, and URL analysis utilities.

Example:
    >>> from xerxes.tools.web_tools import WebScraper, APIClient, RSSReader
    >>> WebScraper.static_call(url="https://example.com")
    >>> APIClient.static_call(url="https://api.example.com/data")
"""

from __future__ import annotations

import re
from typing import Any, cast
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import Tag

from ..core.utils import run_sync
from ..types import AgentBaseFn


class WebScraper(AgentBaseFn):
    """Scrape web pages with CSS selector support and content extraction.

    Provides web scraping capabilities with optional CSS selector filtering,
    link extraction, image extraction, and metadata parsing.

    Example:
        >>> WebScraper.static_call(url="https://news.ycombinator.com")
        >>> WebScraper.static_call(url="https://example.com", selector="article")
    """

    @staticmethod
    async def async_call(
        url: str,
        selector: str | None = None,
        extract_links: bool = False,
        extract_images: bool = False,
        clean_text: bool = True,
        timeout: int = 30,
        **context_variables,
    ) -> dict[str, Any]:
        """Asynchronously scrape a web page.

        Args:
            url: URL to scrape.
            selector: Optional CSS selector to extract specific elements.
            extract_links: Include links in results. Defaults to False.
            extract_images: Include image metadata in results. Defaults to False.
            clean_text: Remove extra whitespace. Defaults to True.
            timeout: Request timeout in seconds. Defaults to 30.
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Dictionary with scraped content, metadata, and optional extracted data.
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            return {"error": "beautifulsoup4 is required but missing from the environment."}

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, timeout=timeout, follow_redirects=True)
                response.raise_for_status()
            except Exception as e:
                return {"error": f"Failed to fetch URL: {e!s}"}

        soup = BeautifulSoup(response.text, "html.parser")
        result: dict[str, Any] = {
            "url": str(response.url),
            "status_code": response.status_code,
            "title": soup.title.string if soup.title else None,
        }

        if selector:
            elements = soup.select(selector)
            result["selected_content"] = [elem.get_text(strip=True) for elem in elements]
        else:
            content = soup.get_text(separator=" ", strip=True) if clean_text else response.text
            result["content"] = content[:10000]

        if extract_links:
            links = []
            for link in soup.find_all("a", href=True):
                link_tag = cast(Tag, link)
                href = str(link_tag["href"])
                absolute_url = urljoin(url, href)
                links.append({"text": link_tag.get_text(strip=True), "url": absolute_url})
            result["links"] = links[:100]

        if extract_images:
            images = []
            for img in soup.find_all("img", src=True):
                img_tag = cast(Tag, img)
                src = str(img_tag["src"])
                absolute_url = urljoin(url, src)
                images.append({"alt": img_tag.get("alt", ""), "src": absolute_url})
            result["images"] = images[:50]

        meta_tags = {}
        for meta in soup.find_all("meta"):
            meta_tag = cast(Tag, meta)
            if meta_tag.get("name"):
                meta_tags[meta_tag["name"]] = meta_tag.get("content", "")
            elif meta_tag.get("property"):
                meta_tags[meta_tag["property"]] = meta_tag.get("content", "")
        result["meta"] = meta_tags

        return result

    @staticmethod
    def static_call(
        url: str,
        selector: str | None = None,
        extract_links: bool = False,
        extract_images: bool = False,
        clean_text: bool = True,
        timeout: int = 30,
        **context_variables,
    ) -> dict[str, Any]:
        """Scrape a web page.

        Args:
            url: URL to scrape.
            selector: Optional CSS selector to extract specific elements.
            extract_links: Include links in results. Defaults to False.
            extract_images: Include image metadata in results. Defaults to False.
            clean_text: Remove extra whitespace. Defaults to True.
            timeout: Request timeout in seconds. Defaults to 30.
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Dictionary with scraped content, metadata, and optional extracted data.
        """
        return run_sync(
            WebScraper.async_call(url, selector, extract_links, extract_images, clean_text, timeout, **context_variables)
        )


class APIClient(AgentBaseFn):
    """Make HTTP API requests with support for various methods and data formats.

    Provides flexible HTTP client capabilities for interacting with REST APIs.

    Example:
        >>> APIClient.static_call(url="https://api.example.com/users", method="GET")
        >>> APIClient.static_call(url="https://api.example.com/users", method="POST", json_data={"name": "Alice"})
    """

    @staticmethod
    async def async_call(
        url: str,
        method: str = "GET",
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        json_data: dict[str, Any] | None = None,
        data: str | None = None,
        timeout: int = 30,
        **context_variables,
    ) -> dict[str, Any]:
        """Asynchronously make an API request.

        Args:
            url: Full URL to request.
            method: HTTP method. Defaults to "GET".
            headers: Optional request headers.
            params: Optional query parameters.
            json_data: Optional JSON body for POST/PUT/PATCH.
            data: Optional raw string body.
            timeout: Request timeout in seconds. Defaults to 30.
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Dictionary with status_code, headers, response data, and URL.
        """
        method = method.upper()
        if method not in ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]:
            return {"error": f"Invalid HTTP method: {method}"}

        async with httpx.AsyncClient() as client:
            try:
                kwargs: dict[str, Any] = {
                    "timeout": timeout,
                    "follow_redirects": True,
                }

                if headers:
                    kwargs["headers"] = headers
                if params:
                    kwargs["params"] = params
                if json_data:
                    kwargs["json"] = json_data
                elif data:
                    kwargs["data"] = data

                response = await client.request(method, url, **kwargs)

                result = {
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "url": str(response.url),
                }

                try:
                    result["json"] = response.json()
                except (ValueError, TypeError):
                    result["text"] = response.text[:10000]

                return result

            except Exception as e:
                return {"error": f"API request failed: {e!s}"}

    @staticmethod
    def static_call(
        url: str,
        method: str = "GET",
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        json_data: dict[str, Any] | None = None,
        data: str | None = None,
        timeout: int = 30,
        **context_variables,
    ) -> dict[str, Any]:
        """Make an API request.

        Args:
            url: Full URL to request.
            method: HTTP method. Defaults to "GET".
            headers: Optional request headers.
            params: Optional query parameters.
            json_data: Optional JSON body for POST/PUT/PATCH.
            data: Optional raw string body.
            timeout: Request timeout in seconds. Defaults to 30.
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Dictionary with status_code, headers, response data, and URL.
        """
        return run_sync(
            APIClient.async_call(url, method, headers, params, json_data, data, timeout, **context_variables)
        )


class RSSReader(AgentBaseFn):
    """Read and parse RSS/Atom feeds.

    Provides RSS feed parsing with optional content extraction.

    Example:
        >>> RSSReader.static_call(feed_url="https://example.com/feed.xml")
        >>> RSSReader.static_call(feed_url="https://news.ycombinator.com/rss", max_items=20)
    """

    @staticmethod
    async def async_call(
        feed_url: str,
        max_items: int = 10,
        include_content: bool = True,
        **context_variables,
    ) -> dict[str, Any]:
        """Asynchronously read an RSS feed.

        Args:
            feed_url: URL of the RSS or Atom feed.
            max_items: Maximum number of items to return. Defaults to 10.
            include_content: Include full content. Defaults to True.
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Dictionary with feed metadata and list of items.
        """
        try:
            import feedparser
        except ImportError:
            return {"error": "feedparser is required. Install with: pip install feedparser"}

        try:
            feed = feedparser.parse(feed_url)

            if feed.bozo:
                return {"error": f"Feed parsing error: {feed.bozo_exception}"}

            result: dict[str, Any] = {
                "title": feed.feed.get("title", ""),
                "description": feed.feed.get("description", ""),
                "link": feed.feed.get("link", ""),
                "updated": feed.feed.get("updated", ""),
                "items": [],
            }

            for entry in feed.entries[:max_items]:
                item: dict[str, Any] = {
                    "title": entry.get("title", ""),
                    "link": entry.get("link", ""),
                    "published": entry.get("published", ""),
                    "author": entry.get("author", ""),
                    "tags": [tag.term for tag in entry.get("tags", [])],
                }

                if include_content:
                    content = entry.get("content", [{}])[0].get("value", "") if "content" in entry else ""
                    if not content:
                        content = entry.get("summary", "")
                    item["content"] = content[:5000]

                result["items"].append(item)

            return result

        except Exception as e:
            return {"error": f"Failed to read RSS feed: {e!s}"}

    @staticmethod
    def static_call(
        feed_url: str,
        max_items: int = 10,
        include_content: bool = True,
        **context_variables,
    ) -> dict[str, Any]:
        """Read an RSS feed.

        Args:
            feed_url: URL of the RSS or Atom feed.
            max_items: Maximum number of items to return. Defaults to 10.
            include_content: Include full content. Defaults to True.
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Dictionary with feed metadata and list of items.
        """
        return run_sync(RSSReader.async_call(feed_url, max_items, include_content, **context_variables))


class URLAnalyzer(AgentBaseFn):
    """Analyze URLs and extract metadata.

    Provides URL parsing and optional availability checking.

    Example:
        >>> URLAnalyzer.static_call(url="https://example.com")
        >>> URLAnalyzer.static_call(url="https://example.com", check_availability=True)
    """

    @staticmethod
    def static_call(
        url: str,
        check_availability: bool = False,
        extract_metadata: bool = True,
        **context_variables,
    ) -> dict[str, Any]:
        """Analyze a URL.

        Args:
            url: URL to analyze.
            check_availability: Check if URL is accessible. Defaults to False.
            extract_metadata: Extract page metadata. Defaults to True.
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Dictionary with URL components and optional metadata.
        """
        parsed = urlparse(url)

        result: dict[str, Any] = {
            "url": url,
            "scheme": parsed.scheme,
            "domain": parsed.netloc,
            "path": parsed.path,
            "params": parsed.params,
            "query": parsed.query,
            "fragment": parsed.fragment,
            "is_valid": bool(parsed.scheme and parsed.netloc),
        }

        if parsed.netloc:
            parts = parsed.netloc.split(".")
            if len(parts) >= 2:
                result["tld"] = parts[-1]
                result["domain_name"] = ".".join(parts[-2:])
                if len(parts) > 2:
                    result["subdomain"] = ".".join(parts[:-2])

        if check_availability and result["is_valid"]:
            try:
                import httpx

                response = httpx.head(url, timeout=5, follow_redirects=True)
                result["is_available"] = response.status_code < 400
                result["status_code"] = response.status_code
                result["final_url"] = str(response.url)
            except (httpx.RequestError, httpx.HTTPStatusError, Exception):
                result["is_available"] = False

        if extract_metadata and result.get("is_available"):
            try:
                import httpx
                from bs4 import BeautifulSoup

                response = httpx.get(url, timeout=10)
                soup = BeautifulSoup(response.text, "html.parser")

                result["title"] = soup.title.string if soup.title else None

                og_tags: dict[str, str] = {}
                for meta in soup.find_all("meta", property=re.compile(r"^og:")):
                    meta_tag = cast(Tag, meta)
                    og_tags[meta_tag["property"]] = meta_tag.get("content", "")
                if og_tags:
                    result["open_graph"] = og_tags

                description = soup.find("meta", attrs={"name": "description"})
                if description:
                    desc_tag = cast(Tag, description)
                    result["description"] = desc_tag.get("content", "")

            except Exception:
                pass

        return result
