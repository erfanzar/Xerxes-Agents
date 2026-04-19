# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# distributed under the License is distributed on an "AS IS" BASIS,
# See the License for the specific language governing permissions and
# limitations under the License.


"""DuckDuckGo search engine integration for Xerxes agents.

This module provides a comprehensive DuckDuckGo search tool for Xerxes agents,
enabling web searches with advanced filtering and customization options. It includes:
- Text, image, video, news, and map searches
- Domain and keyword filtering
- Safe search and time range filtering
- Multi-source search across different content types
- Search suggestions and query translation
- Lazy loading of dependencies to avoid import errors

The search tool is implemented as an AgentBaseFn subclass for seamless
integration with Xerxes agents. It depends on the `ddgs` package, which is
included in Xerxes's core runtime dependencies.

Example:
    >>> from xerxes.tools.duckduckgo_engine import DuckDuckGoSearch
    >>> results = DuckDuckGoSearch.static_call("Python programming", n_results=5)
    >>> news = DuckDuckGoSearch.static_call("AI news", search_type="news")
"""

import typing as tp
from datetime import datetime
from typing import Literal

from ..types import AgentBaseFn

_DDGS = None
_DDGS_AVAILABLE = None


def _get_ddgs():
    """Lazy import of DDGS to avoid crashing if the environment is incomplete.

    Returns:
        The DDGS class from the ddgs package.

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
    """DuckDuckGo search tool for web, image, video, news, and map searches.

    Provides comprehensive search capabilities through the DuckDuckGo API
    with support for filtering, safe search, time limits, and domain
    restrictions. Implements lazy loading of the ddgs package.

    Attributes:
        SearchType: Literal type for search categories (text, images, videos, news, maps).
        TimeFilter: Literal type for time range filtering (day, week, month, year, None).
        SafeSearch: Literal type for safe search levels (strict, moderate, off).

    Methods:
        static_call: Perform a search with full filtering options.
        search_multiple_sources: Search across multiple content types.
        get_suggestions: Get search query suggestions.
        translate_query: Translate a query to another language.

    Example:
        >>> results = DuckDuckGoSearch.static_call(
        ...     query="machine learning",
        ...     search_type="text",
        ...     n_results=10,
        ...     timelimit="month"
        ... )
    """

    SearchType = Literal["text", "images", "videos", "news", "maps"]

    TimeFilter = Literal["day", "week", "month", "year", None]

    SafeSearch = Literal["strict", "moderate", "off"]

    @staticmethod
    def _maybe_truncate(text: str, limit: int | None) -> str:
        """Return the full text if limit is None, else the first `limit` chars.

        Args:
            text: The text to potentially truncate.
            limit: Maximum character limit, or None for no limit.

        Returns:
            The original text or truncated version.
        """
        return text if limit is None else text[:limit]

    @staticmethod
    def _filter_by_domain(results: list[dict], domains: list[str] | None) -> list[dict]:
        """Filter results to only include specified domains.

        Args:
            results: List of search result dictionaries.
            domains: List of domain strings to filter by, or None.

        Returns:
            Filtered list containing only results from specified domains.
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
        """Filter results by keywords in title or snippet.

        Args:
            results: List of search result dictionaries.
            keywords: List of keywords to filter by, or None.
            exclude: If True, exclude results containing keywords.

        Returns:
            Filtered list based on keyword presence.
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
        """Normalize DuckDuckGo text results into the shared result shape."""
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
        """Return True when the provider error means an empty result set."""
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
        """
        Perform an enhanced DuckDuckGo search with multiple options and filters.

        Use this tool when the model needs fresh public-web information rather
        than local workspace context. It supports regular text search plus
        images, videos, news, and maps. The tool normalizes provider output into
        compact result dictionaries so a model can scan titles, snippets, URLs,
        and metadata without scraping a page first.

        Args:
            query (str):
                Search keywords. The query can be plain language or can include
                search-style qualifiers. If ``file_type`` or domain filters are
                provided, they are merged into the outgoing query automatically.
            search_type (SearchType):
                Search vertical to use: ``"text"``, ``"images"``, ``"videos"``,
                ``"news"``, or ``"maps"``.
            n_results (int, optional):
                Number of results to return. Must be between 1 and 30.
            title_length_limit (int | None):
                Maximum number of characters kept from the result title. Set to
                ``None`` to keep titles in full.
            snippet_length_limit (int | None):
                Maximum number of characters kept from result body text or
                summary fields. Set to ``None`` to keep snippets in full.
            region (str):
                Region code such as ``"us-en"``, ``"uk-en"``, or ``"fr-fr"``.
            safesearch (SafeSearch):
                Safe-search level: ``"strict"``, ``"moderate"``, or ``"off"``.
            timelimit (TimeFilter):
                Optional recency filter such as ``"day"``, ``"week"``,
                ``"month"``, or ``"year"``. This is especially useful for news
                and fast-moving topics.
            allowed_domains (list[str] | None):
                Restrict results to these domains. This is implemented both by
                expanding the query and by filtering the returned URLs.
            excluded_domains (list[str] | None):
                Remove results from these domains.
            must_include_keywords (list[str] | None):
                Keep only results whose title or snippet contains at least one of
                the provided keywords.
            exclude_keywords (list[str] | None):
                Remove results whose title or snippet contains any of the
                provided keywords.
            file_type (str | None):
                Add a file-type constraint such as ``"pdf"`` or ``"doc"`` to
                the search query.
            return_metadata (bool):
                When ``True``, return a dictionary with ``results`` and
                additional metadata such as the final query string, timestamp,
                and filters applied. When ``False``, return only the results
                list.

        Returns:
            Union[list[dict], dict]:
                Either a list of result dictionaries or a metadata wrapper
                containing ``results`` and search context. Result items typically
                include keys such as ``title``, ``snippet``, ``url``, and
                type-specific fields like image source or publication date.
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
        search_metadata = {
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
    ) -> dict[str, list[dict]]:
        """Search across multiple source types and return categorized results.

        Performs separate searches for each specified source type and
        aggregates the results into a single dictionary keyed by source.
        Errors for individual sources are captured without failing the
        entire operation.

        Args:
            query: The search query string to use across all sources.
            sources: List of search types to query. Each must be one of
                "text", "images", "videos", "news", "maps". Defaults to
                ["text", "news"] if None.
            n_results_per_source: Maximum number of results to return per
                source type. Defaults to 3.
            **kwargs: Additional keyword arguments forwarded to
                ``static_call`` (e.g., region, safesearch, timelimit).

        Returns:
            A dictionary mapping source type names to their respective
            result lists. If a source fails, its value is a dict with
            an "error" key describing the failure.

        Example:
            >>> results = DuckDuckGoSearch.search_multiple_sources(
            ...     "Python programming",
            ...     sources=["text", "news"],
            ...     n_results_per_source=3
            ... )
            >>> print(len(results["text"]))
            3
        """
        if sources is None:
            sources = ["text", "news"]
        all_results = {}

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
        """Get search query suggestions (autocomplete) for a partial query.

        Retrieves search suggestions from DuckDuckGo's suggestion API,
        useful for expanding or refining queries before performing a full
        search.

        Args:
            query: Partial or full search query to get suggestions for.
            region: Region code for localized suggestions (e.g., "us-en",
                "uk-en", "de-de"). Defaults to "us-en".
            **context_variables: Runtime context from the agent (unused).

        Returns:
            A list of suggested search query strings. Returns an empty
            list if no suggestions are available or if the request fails.

        Example:
            >>> suggestions = DuckDuckGoSearch.get_suggestions("python prog")
            >>> print(suggestions)
            ['python programming', 'python programming language', ...]
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
        """Translate a search query to another language using DuckDuckGo.

        Uses DuckDuckGo's translation service to convert a query from
        its detected language to the specified target language. Falls back
        to returning the original query if translation fails.

        Args:
            query: The original search query to translate.
            to_language: Target language code (e.g., "en" for English,
                "es" for Spanish, "fr" for French, "de" for German).
                Defaults to "en".
            **context_variables: Runtime context from the agent (unused).

        Returns:
            The translated query string. If translation fails, returns
            the original query unchanged.

        Example:
            >>> translated = DuckDuckGoSearch.translate_query("hola mundo", to_language="en")
            >>> print(translated)
            'hello world'
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
