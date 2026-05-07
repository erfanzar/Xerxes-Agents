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
"""DuckDuckGo search integration for web searches, images, videos, news, and maps.

This module provides a comprehensive DuckDuckGo search interface that supports multiple
search types and filtering options. Useful for gathering information from the web.

Example:
    >>> from xerxes.tools.duckduckgo_engine import DuckDuckGoSearch
    >>> results = DuckDuckGoSearch.static_call(query="Python tutorial")
"""

import typing as tp
from datetime import datetime
from typing import Any, Literal

from ..types import AgentBaseFn

_DDGS = None
_DDGS_AVAILABLE = None


def _get_ddgs():
    """Get or initialize the DuckDuckGo search module.

    Returns:
        The DDGS class for creating search instances.

    Raises:
        ImportError: If the ddgs package is not installed.
    """
    global _DDGS, _DDGS_AVAILABLE
    if _DDGS_AVAILABLE is None:
        try:
            from ddgs import DDGS

            _DDGS = DDGS
            _DDGS_AVAILABLE = True
        except ModuleNotFoundError:
            _DDGS_AVAILABLE = False
    if not _DDGS_AVAILABLE:
        raise ImportError("`ddgs` package is required but missing from the environment.")
    return _DDGS


class DuckDuckGoSearch(AgentBaseFn):
    """Perform searches using DuckDuckGo with support for text, images, videos, news, and maps.

    This class provides a unified interface to DuckDuckGo's search capabilities with
    configurable filtering, result limits, and metadata options.

    Attributes:
        SearchType: Type alias for supported search types.
        TimeFilter: Type alias for time-based filtering.
        SafeSearch: Type alias for safe search levels.

    Example:
        >>> results = DuckDuckGoSearch.static_call(query="AI news", n_results=10)
    """

    SearchType = Literal["text", "images", "videos", "news", "maps"]

    TimeFilter = Literal["day", "week", "month", "year", None]

    SafeSearch = Literal["strict", "moderate", "off"]

    @staticmethod
    def _maybe_truncate(text: str, limit: int | None) -> str:
        """Truncate text to specified length if limit is set.

        Args:
            text: Text to potentially truncate.
            limit: Maximum length, or None for no truncation.

        Returns:
            Original text or truncated version.
        """
        return text if limit is None else text[:limit]

    @staticmethod
    def _filter_by_domain(results: list[dict], domains: list[str] | None) -> list[dict]:
        """Filter search results by allowed domains.

        Args:
            results: List of search result dictionaries.
            domains: List of allowed domain names.

        Returns:
            Filtered results containing only specified domains.
        """
        if not domains:
            return results

        filtered = []
        for result in results:
            url = result.get("url", "")
            if any(domain in url for domain in domains):
                filtered.append(result)
        return filtered

    @staticmethod
    def _filter_by_keywords(results: list[dict], keywords: list[str] | None, exclude: bool = False) -> list[dict]:
        """Filter search results by keyword inclusion or exclusion.

        Args:
            results: List of search result dictionaries.
            keywords: Keywords to include or exclude.
            exclude: If True, exclude results with keywords. If False, include only results with keywords.

        Returns:
            Filtered results based on keyword criteria.
        """
        if not keywords:
            return results

        filtered = []
        for result in results:
            text = (result.get("title", "") + " " + result.get("snippet", "")).lower()
            has_keyword = any(keyword.lower() in text for keyword in keywords)

            if (has_keyword and not exclude) or (not has_keyword and exclude):
                filtered.append(result)
        return filtered

    @staticmethod
    def _append_text_results(
        results: list[dict],
        search_results: tp.Iterable[dict],
        n_results: int | None,
        title_length_limit: int | None,
        snippet_length_limit: int | None,
    ) -> None:
        """Append text search results to results list with formatting.

        Args:
            results: List to append formatted results to.
            search_results: Raw search results from DuckDuckGo.
            n_results: Maximum number of results to include.
            title_length_limit: Maximum title length.
            snippet_length_limit: Maximum snippet length.
        """
        for r in search_results:
            results.append(
                {
                    "title": DuckDuckGoSearch._maybe_truncate(r.get("title", ""), title_length_limit),
                    "url": r.get("href", ""),
                    "snippet": DuckDuckGoSearch._maybe_truncate(r.get("body", ""), snippet_length_limit),
                    "source": "DuckDuckGo",
                }
            )
            if n_results and len(results) >= n_results:
                break

    @staticmethod
    def _is_no_results_error(error: Exception) -> bool:
        """Check if an error indicates no search results found.

        Args:
            error: Exception to check.

        Returns:
            True if error indicates no results found.
        """
        return "no results found" in str(error).lower()

    @staticmethod
    def static_call(
        query: str,
        search_type: SearchType = "text",
        n_results: int | None = 5,
        title_length_limit: int | None = 200,
        snippet_length_limit: int | None = 1_000,
        region: str = "us-en",
        safesearch: SafeSearch = "moderate",
        timelimit: TimeFilter = None,
        allowed_domains: list[str] | None = None,
        excluded_domains: list[str] | None = None,
        must_include_keywords: list[str] | None = None,
        exclude_keywords: list[str] | None = None,
        file_type: str | None = None,
        return_metadata: bool = False,
        **context_variables,
    ) -> list[dict] | dict:
        """Perform a DuckDuckGo search.

        Args:
            query: Search query string.
            search_type: Type of search to perform. Defaults to 'text'.
            n_results: Maximum results to return (1-30). Defaults to 5.
            title_length_limit: Maximum character length for titles.
            snippet_length_limit: Maximum character length for snippets.
            region: Geographic region for search. Defaults to 'us-en'.
            safesearch: Safe search level. Defaults to 'moderate'.
            timelimit: Time filter for results ('day', 'week', 'month', 'year').
            allowed_domains: Restrict results to these domains.
            excluded_domains: Exclude results from these domains.
            must_include_keywords: Only include results containing these keywords.
            exclude_keywords: Exclude results containing these keywords.
            file_type: Filter to specific file types.
            return_metadata: Include search metadata in response.
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            List of result dictionaries, or dict with 'results' and 'metadata' if return_metadata is True.

        Raises:
            ValueError: If query is empty or n_results is out of range.
        """
        if not query.strip():
            raise ValueError("Query string must be non-empty")
        if isinstance(n_results, str):
            try:
                n_results = int(n_results)
            except ValueError:
                n_results = 5
        if n_results is not None and not (1 <= n_results <= 30):
            raise ValueError("n_results must be 1-30")

        if file_type:
            query = f"{query} filetype:{file_type}"

        if allowed_domains:
            site_query = " OR ".join(f"site:{domain}" for domain in allowed_domains)
            query = f"{query} ({site_query})"

        if excluded_domains:
            for domain in excluded_domains:
                query = f"{query} -site:{domain}"

        results: list[dict] = []
        search_metadata: dict[str, Any] = {
            "query": query,
            "search_type": search_type,
            "timestamp": datetime.now().isoformat(),
            "filters_applied": {
                "region": region,
                "safesearch": safesearch,
                "timelimit": timelimit,
                "file_type": file_type,
                "allowed_domains": allowed_domains,
                "excluded_domains": excluded_domains,
            },
        }

        with _get_ddgs()() as ddgs:
            if search_type == "text":
                search_results = ddgs.text(
                    query,
                    region=region,
                    safesearch=safesearch.capitalize() if safesearch else "Moderate",
                    timelimit=timelimit,
                )
                DuckDuckGoSearch._append_text_results(
                    results,
                    search_results,
                    n_results=n_results,
                    title_length_limit=title_length_limit,
                    snippet_length_limit=snippet_length_limit,
                )

            elif search_type == "images":
                search_results = ddgs.images(
                    query,
                    region=region,
                    safesearch=safesearch.capitalize() if safesearch else "Moderate",
                    timelimit=timelimit,
                )
                for r in search_results:
                    results.append(
                        {
                            "title": DuckDuckGoSearch._maybe_truncate(r.get("title", ""), title_length_limit),
                            "url": r.get("url", ""),
                            "image_url": r.get("image", ""),
                            "thumbnail": r.get("thumbnail", ""),
                            "source": r.get("source", ""),
                            "width": r.get("width", 0),
                            "height": r.get("height", 0),
                        }
                    )
                    if n_results and len(results) >= n_results:
                        break

            elif search_type == "videos":
                search_results = ddgs.videos(
                    query,
                    region=region,
                    safesearch=safesearch.capitalize() if safesearch else "Moderate",
                    timelimit=timelimit,
                )
                for r in search_results:
                    results.append(
                        {
                            "title": DuckDuckGoSearch._maybe_truncate(r.get("title", ""), title_length_limit),
                            "url": r.get("content", ""),
                            "description": DuckDuckGoSearch._maybe_truncate(
                                r.get("description", ""), snippet_length_limit
                            ),
                            "duration": r.get("duration", ""),
                            "uploader": r.get("uploader", ""),
                            "published": r.get("published", ""),
                            "thumbnail": r.get("thumbnail", ""),
                        }
                    )
                    if n_results and len(results) >= n_results:
                        break

            elif search_type == "news":
                news_safesearch = safesearch.lower() if safesearch else "moderate"
                if news_safesearch == "strict" and timelimit:
                    news_safesearch = "moderate"

                news_failed_with_no_results = False
                try:
                    search_results = ddgs.news(
                        query,
                        region=region,
                        safesearch=news_safesearch,
                        timelimit=timelimit,
                    )
                    for r in search_results:
                        results.append(
                            {
                                "title": DuckDuckGoSearch._maybe_truncate(r.get("title", ""), title_length_limit),
                                "url": r.get("url", ""),
                                "snippet": DuckDuckGoSearch._maybe_truncate(r.get("body", ""), snippet_length_limit),
                                "source": r.get("source", ""),
                                "date": r.get("date", ""),
                                "image": r.get("image", ""),
                            }
                        )
                        if n_results and len(results) >= n_results:
                            break
                except Exception as exc:
                    if not DuckDuckGoSearch._is_no_results_error(exc):
                        raise
                    news_failed_with_no_results = True

                if news_failed_with_no_results or not results:
                    search_metadata["fallback_applied"] = "news_to_text"
                    search_metadata["effective_search_type"] = "text"
                    search_results = ddgs.text(
                        query,
                        region=region,
                        safesearch=safesearch.capitalize() if safesearch else "Moderate",
                        timelimit=timelimit,
                    )
                    DuckDuckGoSearch._append_text_results(
                        results,
                        search_results,
                        n_results=n_results,
                        title_length_limit=title_length_limit,
                        snippet_length_limit=snippet_length_limit,
                    )

            elif search_type == "maps":
                search_results = ddgs.maps(query, place=region.split("-")[0] if region else None)
                for r in search_results:
                    results.append(
                        {
                            "title": DuckDuckGoSearch._maybe_truncate(r.get("title", ""), title_length_limit),
                            "address": r.get("address", ""),
                            "country": r.get("country", ""),
                            "city": r.get("city", ""),
                            "phone": r.get("phone", ""),
                            "latitude": r.get("latitude", ""),
                            "longitude": r.get("longitude", ""),
                            "url": r.get("url", ""),
                            "desc": DuckDuckGoSearch._maybe_truncate(r.get("desc", ""), snippet_length_limit),
                            "hours": r.get("hours", {}),
                        }
                    )
                    if n_results and len(results) >= n_results:
                        break

        if must_include_keywords:
            results = DuckDuckGoSearch._filter_by_keywords(results, must_include_keywords, exclude=False)

        if exclude_keywords:
            results = DuckDuckGoSearch._filter_by_keywords(results, exclude_keywords, exclude=True)

        search_metadata.setdefault("effective_search_type", search_type)
        search_metadata["total_results"] = len(results)
        search_metadata["filters_applied"]["keyword_filters"] = {
            "must_include": must_include_keywords,
            "exclude": exclude_keywords,
        }

        if return_metadata:
            return {"results": results, "metadata": search_metadata}

        return results

    @staticmethod
    def search_multiple_sources(
        query: str,
        sources: list[SearchType] | None = None,
        n_results_per_source: int = 3,
        **kwargs,
    ) -> dict[str, Any]:
        """Search across multiple source types in parallel.

        Args:
            query: Search query string.
            sources: List of search types to query. Defaults to ['text', 'news'].
            n_results_per_source: Results per source type.
            **kwargs: Additional arguments passed to static_call.

        Returns:
            Dictionary mapping source types to their results.

        Example:
            >>> multi = DuckDuckGoSearch.search_multiple_sources(
            ...     query="Python",
            ...     sources=["text", "images", "news"]
            ... )
        """
        if sources is None:
            sources = ["text", "news"]
        all_results: dict[str, Any] = {}

        for source in sources:
            try:
                results = DuckDuckGoSearch.static_call(
                    query=query, search_type=source, n_results=n_results_per_source, **kwargs
                )
                all_results[source] = results
            except Exception as e:
                all_results[source] = {"error": str(e)}

        return all_results

    @staticmethod
    def get_suggestions(query: str, region: str = "us-en", **context_variables) -> list[str]:
        """Get search suggestions for a query.

        Args:
            query: Partial search query.
            region: Geographic region for suggestions.
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            List of suggested search phrases.

        Example:
            >>> DuckDuckGoSearch.get_suggestions("Pyth")
            ['Python tutorial', 'Python download', ...]
        """
        suggestions = []

        with _get_ddgs()() as ddgs:
            try:
                results = ddgs.suggestions(query, region=region)
                suggestions = [r.get("phrase", "") for r in results if r.get("phrase")]
            except Exception as e:
                import logging

                logging.getLogger(__name__).debug(f"Failed to get suggestions for '{query}': {e}")

        return suggestions

    @staticmethod
    def translate_query(query: str, to_language: str = "en", **context_variables) -> str:
        """Translate a search query to another language.

        Args:
            query: Query to translate.
            to_language: Target language code. Defaults to 'en'.
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Translated query, or original if translation fails.

        Example:
            >>> DuckDuckGoSearch.translate_query("Comment programmer en Python", "en")
            'How to program in Python'
        """
        with _get_ddgs()() as ddgs:
            try:
                result = ddgs.translate(query, to=to_language)
                return result.get("translated", query)
            except Exception as e:
                import logging

                logging.getLogger(__name__).debug(f"Failed to translate '{query}' to {to_language}: {e}")
                return query


__all__ = ("DuckDuckGoSearch",)
