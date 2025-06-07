# Copyright 2023 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

from typing import Literal
from urllib.parse import quote_plus

from bs4 import BeautifulSoup

from ..tools import ComposedTool, CortexTool, ToolExecutionContext, ToolParameter, ToolSignature, WebTool


class WebSearchTool(WebTool):
    """
    Advanced web search tool with multiple search engines support
    """

    def __init__(
        self,
        name: str = "web_search",
        description: str = "Search the web for information using multiple search engines",
        api_keys: dict[str, str] | None = None,
        default_engine: str = "duckduckgo",
        max_results: int = 10,
        **kwargs,
    ):
        super().__init__(name, description, **kwargs)
        self.api_keys = api_keys or {}
        self.default_engine = default_engine
        self.max_results = max_results

        # Define examples for better LLM understanding
        self.examples = [
            {"query": "latest developments in quantum computing 2024", "engine": "google", "max_results": 5},
            {"query": "OpenAI GPT-4 technical specifications", "engine": "duckduckgo", "max_results": 3},
        ]

    def _extract_signature(self) -> ToolSignature:
        """Override to provide detailed signature"""
        return ToolSignature(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="query", type="string", description="The search query to look up on the web", required=True
                ),
                ToolParameter(
                    name="engine",
                    type="string",
                    description="Search engine to use",
                    required=False,
                    default=self.default_engine,
                    enum=["google", "duckduckgo", "bing", "serpapi"],
                ),
                ToolParameter(
                    name="max_results",
                    type="integer",
                    description="Maximum number of results to return",
                    required=False,
                    default=self.max_results,
                ),
                ToolParameter(
                    name="include_snippets",
                    type="boolean",
                    description="Whether to include text snippets from results",
                    required=False,
                    default=True,
                ),
            ],
            returns={
                "type": "object",
                "properties": {
                    "results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "url": {"type": "string"},
                                "snippet": {"type": "string"},
                                "position": {"type": "integer"},
                            },
                        },
                    },
                    "query": {"type": "string"},
                    "engine": {"type": "string"},
                    "total_results": {"type": "integer"},
                },
            },
            examples=self.examples,
        )

    async def _run(
        self,
        query: str,
        engine: Literal["google", "duckduckgo", "bing", "serpapi"] | None = None,
        max_results: int | None = None,
        include_snippets: bool = True,
    ) -> dict:
        """
        Execute web search

        Args:
            query: The search query
            engine: Search engine to use
            max_results: Maximum number of results
            include_snippets: Whether to include text snippets

        Returns:
            Dictionary containing search results
        """
        engine = engine or self.default_engine
        max_results = max_results or self.max_results
        await self._rate_limit_check()

        if engine == "duckduckgo":
            results = await self._search_duckduckgo(query, max_results, include_snippets)
        elif engine == "google" and "google" in self.api_keys:
            results = await self._search_google(query, max_results, include_snippets)
        elif engine == "serpapi" and "serpapi" in self.api_keys:
            results = await self._search_serpapi(query, max_results, include_snippets)
        elif engine == "bing" and "bing" in self.api_keys:
            results = await self._search_bing(query, max_results, include_snippets)
        else:
            # Fallback to DuckDuckGo if requested engine unavailable
            results = await self._search_duckduckgo(query, max_results, include_snippets)
            engine = "duckduckgo"
        output = {"results": results, "query": query, "engine": engine, "total_results": len(results)}

        return output

    async def _search_duckduckgo(self, query: str, max_results: int, include_snippets: bool) -> list[dict]:
        """Search using DuckDuckGo (no API key required)"""
        from duckduckgo_search import DDGS

        results = []
        try:
            ddgs = DDGS()
            for r in ddgs.text(query, max_results=max_results):
                result = {"title": r.get("title", ""), "url": r.get("link", ""), "position": len(results) + 1}
                if include_snippets:
                    result["snippet"] = r.get("body", "")
                results.append(result)
        except Exception:
            results = await self._search_duckduckgo_html(query, max_results, include_snippets)

        return results

    async def _search_duckduckgo_html(self, query: str, max_results: int, include_snippets: bool) -> list[dict]:
        """Fallback DuckDuckGo search using HTML parsing"""
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"

        session = await self._get_async_session()
        async with session.get(url) as response:
            html = await response.text()

        soup = BeautifulSoup(html, "html.parser")
        results = []

        for i, result in enumerate(soup.select(".result__body")[:max_results]):
            title_elem = result.select_one(".result__title")
            url_elem = result.select_one(".result__url")
            snippet_elem = result.select_one(".result__snippet")

            if title_elem and url_elem:
                result_dict = {
                    "title": title_elem.get_text(strip=True),
                    "url": url_elem.get("href", ""),
                    "position": i + 1,
                }
                if include_snippets and snippet_elem:
                    result_dict["snippet"] = snippet_elem.get_text(strip=True)
                results.append(result_dict)

        return results

    async def _search_google(self, query: str, max_results: int, include_snippets: bool) -> list[dict]:
        """Search using Google Custom Search API"""
        if "google" not in self.api_keys or "google_cx" not in self.api_keys:
            raise ValueError("Google API key and CX (Custom Search Engine ID) required")

        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self.api_keys["google"],
            "cx": self.api_keys["google_cx"],
            "q": query,
            "num": min(max_results, 10),  # Google API limit
        }

        session = await self._get_async_session()
        async with session.get(url, params=params) as response:
            data = await response.json()

        results = []
        for i, item in enumerate(data.get("items", [])):
            result = {"title": item.get("title", ""), "url": item.get("link", ""), "position": i + 1}
            if include_snippets:
                result["snippet"] = item.get("snippet", "")
            results.append(result)

        return results

    async def _search_serpapi(self, query: str, max_results: int, include_snippets: bool) -> list[dict]:
        """Search using SerpAPI (supports multiple search engines)"""
        if "serpapi" not in self.api_keys:
            raise ValueError("SerpAPI key required")

        url = "https://serpapi.com/search"
        params = {
            "api_key": self.api_keys["serpapi"],
            "q": query,
            "num": max_results,
            "engine": "google",  # Can be changed to bing, yahoo, etc.
        }

        session = await self._get_async_session()
        async with session.get(url, params=params) as response:
            data = await response.json()

        results = []
        for i, result in enumerate(data.get("organic_results", [])[:max_results]):
            result_dict = {
                "title": result.get("title", ""),
                "url": result.get("link", ""),
                "position": result.get("position", i + 1),
            }
            if include_snippets:
                result_dict["snippet"] = result.get("snippet", "")
            results.append(result_dict)

        return results

    async def _search_bing(self, query: str, max_results: int, include_snippets: bool) -> list[dict]:
        """Search using Bing Search API"""
        if "bing" not in self.api_keys:
            raise ValueError("Bing API key required")

        url = "https://api.bing.microsoft.com/v7.0/search"
        headers = {"Ocp-Apim-Subscription-Key": self.api_keys["bing"]}
        params = {"q": query, "count": max_results}

        session = await self._get_async_session()
        async with session.get(url, headers=headers, params=params) as response:
            data = await response.json()

        results = []
        for i, result in enumerate(data.get("webPages", {}).get("value", [])):
            result_dict = {"title": result.get("name", ""), "url": result.get("url", ""), "position": i + 1}
            if include_snippets:
                result_dict["snippet"] = result.get("snippet", "")
            results.append(result_dict)

        return results


class WebScraperTool(WebTool):
    """Tool for scraping web page content"""

    def __init__(
        self, name: str = "web_scraper", description: str = "Extract and parse content from web pages", **kwargs
    ):
        super().__init__(name, description, **kwargs)
        self.examples = [{"url": "https://example.com/article", "extract": "text", "selector": "article"}]

    async def _run(
        self,
        url: str,
        extract: Literal["text", "html", "markdown", "links", "images"] = "text",
        selector: str | None = None,
        max_length: int = 5000,
    ) -> dict:
        """
        Scrape content from a web page

        Args:
            url: URL to scrape
            extract: Type of content to extract
            selector: CSS selector to target specific content
            max_length: Maximum length of extracted content

        Returns:
            Dictionary with extracted content
        """
        await self._rate_limit_check()

        session = await self._get_async_session()
        async with session.get(url) as response:
            html = await response.text()

        soup = BeautifulSoup(html, "html.parser")

        # Apply selector if provided
        if selector:
            elements = soup.select(selector)
            if elements:
                soup = BeautifulSoup(str(elements[0]), "html.parser")

        result = {"url": url, "extract_type": extract}

        if extract == "text":
            text = soup.get_text(separator="\n", strip=True)
            result["content"] = text[:max_length]
            result["truncated"] = len(text) > max_length

        elif extract == "html":
            html_content = str(soup)
            result["content"] = html_content[:max_length]
            result["truncated"] = len(html_content) > max_length

        elif extract == "markdown":
            # Simple markdown conversion
            import html2text  # type:ignore

            h = html2text.HTML2Text()
            h.ignore_links = False
            markdown = h.handle(str(soup))
            result["content"] = markdown[:max_length]
            result["truncated"] = len(markdown) > max_length

        elif extract == "links":
            links = []
            for link in soup.find_all("a", href=True):
                links.append({"text": link.get_text(strip=True), "url": link["href"]})
            result["links"] = links[:50]  # Limit to 50 links
            result["total_links"] = len(links)
        elif extract == "images":
            images = []
            for img in soup.find_all("img", src=True):
                images.append({"src": img["src"], "alt": img.get("alt", ""), "title": img.get("title", "")})
            result["images"] = images[:50]  # Limit to 50 images
            result["total_images"] = len(images)

        return result


class WebResearchTool(ComposedTool):
    """
    Advanced web research tool that combines search and scraping
    """

    def __init__(
        self,
        name: str = "web_research",
        description: str = "Comprehensive web research tool that searches and extracts content",
        api_keys: dict[str, str] | None = None,
        **kwargs,
    ):
        search_tool = WebSearchTool(api_keys=api_keys)
        scraper_tool = WebScraperTool()

        super().__init__(name=name, description=description, tools=[search_tool, scraper_tool])

        self.examples = [
            {
                "query": "latest AI developments",
                "max_results": 3,
                "extract_content": True,
                "content_type": "text",
            }
        ]

    async def _run(
        self,
        query: str,
        max_results: int = 5,
        extract_content: bool = True,
        content_type: Literal["text", "markdown"] = "text",
        max_content_length: int = 2000,
    ) -> dict:
        """
        Perform comprehensive web research

        Args:
            query: Search query
            max_results: Number of search results
            extract_content: Whether to extract content from results
            content_type: Type of content to extract
            max_content_length: Max length per page

        Returns:
            Dictionary with search results and extracted content
        """
        # First, search the web
        search_result = await self.tools[0].execute_with_context(
            ToolExecutionContext(), query=query, max_results=max_results
        )

        if not search_result.success:
            return {"error": search_result.error}

        research_results = {"query": query, "search_results": search_result.value["results"], "extracted_content": []}

        # Then, extract content from top results if requested
        if extract_content and search_result.value["results"]:
            scraper = self.tools[1]

            for i, result in enumerate(search_result.value["results"][:3]):  # Limit to top 3
                try:
                    scrape_result = await scraper.execute_with_context(
                        ToolExecutionContext(),
                        url=result["url"],
                        extract=content_type,
                        max_length=max_content_length,
                    )

                    if scrape_result.success:
                        research_results["extracted_content"].append(
                            {
                                "url": result["url"],
                                "title": result["title"],
                                "content": scrape_result.value["content"],
                                "position": i + 1,
                            }
                        )
                except Exception:
                    continue

        return research_results


def create_web_tools(api_keys: dict[str, str] | None = None) -> dict[str, CortexTool]:
    """
    Create a standard set of web tools

    Args:
        api_keys: Dictionary of API keys for various services
            - google: Google Custom Search API key
            - google_cx: Google Custom Search Engine ID
            - serpapi: SerpAPI key
            - bing: Bing Search API key

    Returns:
        Dictionary of tool name to tool instance
    """
    tools = {
        "web_search": WebSearchTool(api_keys=api_keys),
        "web_scraper": WebScraperTool(),
        "web_research": WebResearchTool(api_keys=api_keys),
    }

    for tool in tools.values():
        tool.category = "Web Tools"

    return tools
