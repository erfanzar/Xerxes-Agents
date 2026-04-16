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


"""Web-related tools for fetching, scraping, and processing web content.

This module provides a collection of web tools for agents to interact
with web resources. It includes:
- Web scraping with content extraction and CSS selectors
- Generic HTTP API client for RESTful interactions
- RSS/Atom feed reading and parsing
- URL analysis and validation

All tools support both synchronous and asynchronous execution modes,
with comprehensive error handling and configurable timeouts.

Example:
    >>> scraper = WebScraper()
    >>> result = scraper.static_call("https://example.com", extract_links=True)
    >>> print(result["title"])
"""

from __future__ import annotations

import asyncio
import re
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx

from ..types import AgentBaseFn


class WebScraper(AgentBaseFn):
    """Advanced web scraper with content extraction.

    Provides comprehensive web scraping capabilities including HTML parsing,
    CSS selector-based content extraction, link and image extraction,
    and metadata parsing. Supports both async and sync execution modes.

    The scraper uses BeautifulSoup for HTML parsing and httpx for
    HTTP requests with automatic redirect following.

    Attributes:
        Inherits from AgentBaseFn for agent integration.

    Example:
        >>> result = WebScraper.static_call(
        ...     "https://example.com",
        ...     selector="article",
        ...     extract_links=True
        ... )
        >>> print(result["title"])
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
        """Scrape web content with advanced extraction options.

        Fetches the specified URL and extracts content based on the
        provided options. Supports CSS selector-based extraction,
        link/image extraction, and automatic text cleaning.

        Args:
            url: The URL to scrape.
            selector: CSS selector for specific content (requires beautifulsoup4).
            extract_links: Whether to extract all links from the page.
            extract_images: Whether to extract all images from the page.
            clean_text: Whether to clean and format extracted text.
            timeout: Request timeout in seconds (default: 30).
            **context_variables: Additional context passed from the agent.

        Returns:
            Dictionary containing:
                - url: Final URL after redirects
                - status_code: HTTP response status code
                - title: Page title if available
                - content/selected_content: Extracted text content
                - links: List of extracted links (if requested)
                - images: List of extracted images (if requested)
                - meta: Dictionary of meta tags
                - error: Error message if request failed

        Note:
            Requires beautifulsoup4 for HTML parsing.
        """
        try:
            from bs4 import BeautifulSoup  # type:ignore
        except ImportError:
            return {"error": "beautifulsoup4 is required but missing from the environment."}

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, timeout=timeout, follow_redirects=True)
                response.raise_for_status()
            except Exception as e:
                return {"error": f"Failed to fetch URL: {e!s}"}

        soup = BeautifulSoup(response.text, "html.parser")
        result = {
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
                href = link["href"]
                absolute_url = urljoin(url, href)
                links.append({"text": link.get_text(strip=True), "url": absolute_url})
            result["links"] = links[:100]

        if extract_images:
            images = []
            for img in soup.find_all("img", src=True):
                src = img["src"]
                absolute_url = urljoin(url, src)
                images.append({"alt": img.get("alt", ""), "src": absolute_url})
            result["images"] = images[:50]

        meta_tags = {}
        for meta in soup.find_all("meta"):
            if meta.get("name"):
                meta_tags[meta["name"]] = meta.get("content", "")
            elif meta.get("property"):
                meta_tags[meta["property"]] = meta.get("content", "")
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
        """Synchronous wrapper for async web scraping.

        Provides a blocking interface to the async scraping functionality.
        Uses asyncio.run() to execute the async method.

        Args:
            url: The URL to scrape.
            selector: CSS selector for specific content.
            extract_links: Whether to extract all links from the page.
            extract_images: Whether to extract all images from the page.
            clean_text: Whether to clean and format extracted text.
            timeout: Request timeout in seconds.
            **context_variables: Additional context passed from the agent.

        Returns:
            Dictionary with scraped content and metadata.
        """
        return asyncio.run(
            WebScraper.async_call(url, selector, extract_links, extract_images, clean_text, timeout, **context_variables)
        )


class APIClient(AgentBaseFn):
    """Generic API client for making HTTP requests.

    Provides a flexible HTTP client for interacting with REST APIs.
    Supports all standard HTTP methods, custom headers, query parameters,
    JSON payloads, and raw data bodies.

    The client automatically handles JSON response parsing and provides
    comprehensive error handling for network and HTTP errors.

    Attributes:
        Inherits from AgentBaseFn for agent integration.

    Example:
        >>> result = APIClient.static_call(
        ...     "https://api.example.com/data",
        ...     method="POST",
        ...     json_data={"key": "value"}
        ... )
        >>> print(result["status_code"])
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
        """Make HTTP API requests with various methods and payloads.

        Sends an HTTP request to the specified URL with the given
        method and payload options. Automatically follows redirects
        and parses JSON responses when available.

        Args:
            url: The API endpoint URL.
            method: HTTP method (GET, POST, PUT, DELETE, PATCH, HEAD, OPTIONS).
            headers: Dictionary of request headers.
            params: Dictionary of URL query parameters.
            json_data: Dictionary to be sent as JSON in request body.
            data: Raw string data for request body.
            timeout: Request timeout in seconds (default: 30).
            **context_variables: Additional context passed from the agent.

        Returns:
            Dictionary containing:
                - status_code: HTTP response status code
                - headers: Response headers as dictionary
                - url: Final URL after redirects
                - json: Parsed JSON response (if applicable)
                - text: Response text (if not JSON, truncated to 10000 chars)
                - error: Error message if request failed

        Raises:
            Returns error dict if HTTP method is invalid or request fails.
        """
        method = method.upper()
        if method not in ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]:
            return {"error": f"Invalid HTTP method: {method}"}

        async with httpx.AsyncClient() as client:
            try:
                kwargs = {
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
        """Synchronous wrapper for API requests.

        Provides a blocking interface to the async API client.
        Uses asyncio.run() to execute the async method.

        Args:
            url: The API endpoint URL.
            method: HTTP method (GET, POST, PUT, DELETE, PATCH, HEAD, OPTIONS).
            headers: Dictionary of request headers.
            params: Dictionary of URL query parameters.
            json_data: Dictionary to be sent as JSON in request body.
            data: Raw string data for request body.
            timeout: Request timeout in seconds.
            **context_variables: Additional context passed from the agent.

        Returns:
            Dictionary with response data and metadata.
        """
        return asyncio.run(
            APIClient.async_call(url, method, headers, params, json_data, data, timeout, **context_variables)
        )


class RSSReader(AgentBaseFn):
    """RSS/Atom feed reader and parser.

    Provides functionality for reading and parsing RSS and Atom feeds.
    Extracts feed metadata and individual entries with configurable
    content inclusion and item limits.

    Uses the feedparser library for robust feed parsing that handles
    various feed formats and edge cases.

    Attributes:
        Inherits from AgentBaseFn for agent integration.

    Example:
        >>> result = RSSReader.static_call(
        ...     "https://example.com/feed.xml",
        ...     max_items=5
        ... )
        >>> for item in result["items"]:
        ...     print(item["title"])
    """

    @staticmethod
    async def async_call(
        feed_url: str,
        max_items: int = 10,
        include_content: bool = True,
        **context_variables,
    ) -> dict[str, Any]:
        """Read and parse RSS/Atom feeds.

        Fetches and parses an RSS or Atom feed from the specified URL,
        extracting feed metadata and individual entries.

        Args:
            feed_url: URL of the RSS/Atom feed.
            max_items: Maximum number of items to return (default: 10).
            include_content: Whether to include full article content.
            **context_variables: Additional context passed from the agent.

        Returns:
            Dictionary containing:
                - title: Feed title
                - description: Feed description
                - link: Feed website link
                - updated: Last update timestamp
                - items: List of feed entries with title, link, published,
                  author, tags, and optionally content
                - error: Error message if parsing failed

        Note:
            Requires feedparser library. Install with: pip install feedparser
        """
        try:
            import feedparser  # type:ignore
        except ImportError:
            return {"error": "feedparser is required. Install with: pip install feedparser"}

        try:
            feed = feedparser.parse(feed_url)

            if feed.bozo:
                return {"error": f"Feed parsing error: {feed.bozo_exception}"}

            result = {
                "title": feed.feed.get("title", ""),
                "description": feed.feed.get("description", ""),
                "link": feed.feed.get("link", ""),
                "updated": feed.feed.get("updated", ""),
                "items": [],
            }

            for entry in feed.entries[:max_items]:
                item = {
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
        """Synchronous wrapper for RSS reading.

        Provides a blocking interface to the async RSS reader.
        Uses asyncio.run() to execute the async method.

        Args:
            feed_url: URL of the RSS/Atom feed.
            max_items: Maximum number of items to return.
            include_content: Whether to include full article content.
            **context_variables: Additional context passed from the agent.

        Returns:
            Dictionary with parsed feed data and articles.
        """
        return asyncio.run(RSSReader.async_call(feed_url, max_items, include_content, **context_variables))


class URLAnalyzer(AgentBaseFn):
    """Analyze and extract information from URLs.

    Provides URL parsing, validation, and optional availability checking.
    Extracts URL components like scheme, domain, path, and query parameters.
    Can also fetch and extract page metadata including Open Graph tags.

    Attributes:
        Inherits from AgentBaseFn for agent integration.

    Example:
        >>> result = URLAnalyzer.static_call(
        ...     "https://example.com/path?query=value",
        ...     check_availability=True
        ... )
        >>> print(result["domain"])
    """

    @staticmethod
    def static_call(
        url: str,
        check_availability: bool = False,
        extract_metadata: bool = True,
        **context_variables,
    ) -> dict[str, Any]:
        """Analyze URL structure and optionally check availability.

        Parses the URL to extract its components and optionally checks
        if the URL is accessible and extracts page metadata.

        Args:
            url: URL to analyze.
            check_availability: Whether to check if URL is accessible.
            extract_metadata: Whether to extract page metadata (requires availability).
            **context_variables: Additional context passed from the agent.

        Returns:
            Dictionary containing:
                - url: Original URL
                - scheme: URL scheme (http, https, etc.)
                - domain: Full domain/netloc
                - path: URL path
                - params: URL parameters
                - query: Query string
                - fragment: URL fragment
                - is_valid: Whether URL has valid scheme and domain
                - tld: Top-level domain (if extractable)
                - domain_name: Primary domain name
                - subdomain: Subdomain if present
                - is_available: Whether URL responds (if checked)
                - status_code: HTTP status code (if checked)
                - final_url: URL after redirects (if checked)
                - title: Page title (if metadata extracted)
                - open_graph: Open Graph tags (if present)
                - description: Meta description (if present)
        """
        parsed = urlparse(url)

        result = {
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
                from bs4 import BeautifulSoup  # type:ignore

                response = httpx.get(url, timeout=10)
                soup = BeautifulSoup(response.text, "html.parser")

                result["title"] = soup.title.string if soup.title else None

                og_tags = {}
                for meta in soup.find_all("meta", property=re.compile(r"^og:")):
                    og_tags[meta["property"]] = meta.get("content", "")
                if og_tags:
                    result["open_graph"] = og_tags

                description = soup.find("meta", attrs={"name": "description"})
                if description:
                    result["description"] = description.get("content", "")

            except Exception:
                pass

        return result
