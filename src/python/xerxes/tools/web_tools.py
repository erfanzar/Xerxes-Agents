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
"""Web tools module for Xerxes.

Exports:
    - WebScraper
    - APIClient
    - RSSReader
    - URLAnalyzer"""

from __future__ import annotations

import re
from typing import Any, cast
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import Tag

from ..core.utils import run_sync
from ..types import AgentBaseFn


class WebScraper(AgentBaseFn):
    """Web scraper.

    Inherits from: AgentBaseFn
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
        """Asynchronously Async call.

        Args:
            url (str): IN: url. OUT: Consumed during execution.
            selector (str | None, optional): IN: selector. Defaults to None. OUT: Consumed during execution.
            extract_links (bool, optional): IN: extract links. Defaults to False. OUT: Consumed during execution.
            extract_images (bool, optional): IN: extract images. Defaults to False. OUT: Consumed during execution.
            clean_text (bool, optional): IN: clean text. Defaults to True. OUT: Consumed during execution.
            timeout (int, optional): IN: timeout. Defaults to 30. OUT: Consumed during execution.
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            dict[str, Any]: OUT: Result of the operation."""

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
                tag = cast(Tag, link)
                href = str(tag["href"])
                absolute_url = urljoin(url, href)
                links.append({"text": tag.get_text(strip=True), "url": absolute_url})
            result["links"] = links[:100]

        if extract_images:
            images = []
            for img in soup.find_all("img", src=True):
                tag = cast(Tag, img)
                src = str(tag["src"])
                absolute_url = urljoin(url, src)
                images.append({"alt": tag.get("alt", ""), "src": absolute_url})
            result["images"] = images[:50]

        meta_tags = {}
        for meta in soup.find_all("meta"):
            tag = cast(Tag, meta)
            if tag.get("name"):
                meta_tags[tag["name"]] = tag.get("content", "")
            elif tag.get("property"):
                meta_tags[tag["property"]] = tag.get("content", "")
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
        """Static call.

        Args:
            url (str): IN: url. OUT: Consumed during execution.
            selector (str | None, optional): IN: selector. Defaults to None. OUT: Consumed during execution.
            extract_links (bool, optional): IN: extract links. Defaults to False. OUT: Consumed during execution.
            extract_images (bool, optional): IN: extract images. Defaults to False. OUT: Consumed during execution.
            clean_text (bool, optional): IN: clean text. Defaults to True. OUT: Consumed during execution.
            timeout (int, optional): IN: timeout. Defaults to 30. OUT: Consumed during execution.
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            dict[str, Any]: OUT: Result of the operation."""

        return run_sync(
            WebScraper.async_call(url, selector, extract_links, extract_images, clean_text, timeout, **context_variables)
        )


class APIClient(AgentBaseFn):
    """Apiclient.

    Inherits from: AgentBaseFn
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
        """Asynchronously Async call.

        Args:
            url (str): IN: url. OUT: Consumed during execution.
            method (str, optional): IN: method. Defaults to 'GET'. OUT: Consumed during execution.
            headers (dict[str, str] | None, optional): IN: headers. Defaults to None. OUT: Consumed during execution.
            params (dict[str, Any] | None, optional): IN: params. Defaults to None. OUT: Consumed during execution.
            json_data (dict[str, Any] | None, optional): IN: json data. Defaults to None. OUT: Consumed during execution.
            data (str | None, optional): IN: data. Defaults to None. OUT: Consumed during execution.
            timeout (int, optional): IN: timeout. Defaults to 30. OUT: Consumed during execution.
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            dict[str, Any]: OUT: Result of the operation."""

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
        """Static call.

        Args:
            url (str): IN: url. OUT: Consumed during execution.
            method (str, optional): IN: method. Defaults to 'GET'. OUT: Consumed during execution.
            headers (dict[str, str] | None, optional): IN: headers. Defaults to None. OUT: Consumed during execution.
            params (dict[str, Any] | None, optional): IN: params. Defaults to None. OUT: Consumed during execution.
            json_data (dict[str, Any] | None, optional): IN: json data. Defaults to None. OUT: Consumed during execution.
            data (str | None, optional): IN: data. Defaults to None. OUT: Consumed during execution.
            timeout (int, optional): IN: timeout. Defaults to 30. OUT: Consumed during execution.
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            dict[str, Any]: OUT: Result of the operation."""

        return run_sync(
            APIClient.async_call(url, method, headers, params, json_data, data, timeout, **context_variables)
        )


class RSSReader(AgentBaseFn):
    """Rssreader.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    async def async_call(
        feed_url: str,
        max_items: int = 10,
        include_content: bool = True,
        **context_variables,
    ) -> dict[str, Any]:
        """Asynchronously Async call.

        Args:
            feed_url (str): IN: feed url. OUT: Consumed during execution.
            max_items (int, optional): IN: max items. Defaults to 10. OUT: Consumed during execution.
            include_content (bool, optional): IN: include content. Defaults to True. OUT: Consumed during execution.
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            dict[str, Any]: OUT: Result of the operation."""

        try:
            import feedparser
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
        """Static call.

        Args:
            feed_url (str): IN: feed url. OUT: Consumed during execution.
            max_items (int, optional): IN: max items. Defaults to 10. OUT: Consumed during execution.
            include_content (bool, optional): IN: include content. Defaults to True. OUT: Consumed during execution.
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            dict[str, Any]: OUT: Result of the operation."""

        return run_sync(RSSReader.async_call(feed_url, max_items, include_content, **context_variables))


class URLAnalyzer(AgentBaseFn):
    """Urlanalyzer.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    def static_call(
        url: str,
        check_availability: bool = False,
        extract_metadata: bool = True,
        **context_variables,
    ) -> dict[str, Any]:
        """Static call.

        Args:
            url (str): IN: url. OUT: Consumed during execution.
            check_availability (bool, optional): IN: check availability. Defaults to False. OUT: Consumed during execution.
            extract_metadata (bool, optional): IN: extract metadata. Defaults to True. OUT: Consumed during execution.
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            dict[str, Any]: OUT: Result of the operation."""

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
                from bs4 import BeautifulSoup

                response = httpx.get(url, timeout=10)
                soup = BeautifulSoup(response.text, "html.parser")

                result["title"] = soup.title.string if soup.title else None

                og_tags = {}
                for meta in soup.find_all("meta", property=re.compile(r"^og:")):
                    tag = cast(Tag, meta)
                    og_tags[tag["property"]] = tag.get("content", "")
                if og_tags:
                    result["open_graph"] = og_tags

                description = soup.find("meta", attrs={"name": "description"})
                if description:
                    result["description"] = cast(Tag, description).get("content", "")

            except Exception:
                pass

        return result
