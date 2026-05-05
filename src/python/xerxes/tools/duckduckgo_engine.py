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
"""Duckduckgo engine module for Xerxes.

Exports:
    - DuckDuckGoSearch"""

import typing as tp
from datetime import datetime
from typing import Any, Literal

from ..types import AgentBaseFn

_DDGS = None
_DDGS_AVAILABLE = None


def _get_ddgs():
    """Internal helper to get ddgs.

    Returns:
        Any: OUT: Result of the operation."""

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
    """Duck duck go search.

    Inherits from: AgentBaseFn

    Attributes:
        SearchType (Any): search type.
        TimeFilter (Any): time filter.
        SafeSearch (Any): safe search."""

    SearchType = Literal["text", "images", "videos", "news", "maps"]

    TimeFilter = Literal["day", "week", "month", "year", None]

    SafeSearch = Literal["strict", "moderate", "off"]

    @staticmethod
    def _maybe_truncate(text: str, limit: int | None) -> str:
        """Internal helper to maybe truncate.

        Args:
            text (str): IN: text. OUT: Consumed during execution.
            limit (int | None): IN: limit. OUT: Consumed during execution.
        Returns:
            str: OUT: Result of the operation."""

        return text if limit is None else text[:limit]

    @staticmethod
    def _filter_by_domain(results: list[dict], domains: list[str] | None) -> list[dict]:
        """Internal helper to filter by domain.

        Args:
            results (list[dict]): IN: results. OUT: Consumed during execution.
            domains (list[str] | None): IN: domains. OUT: Consumed during execution.
        Returns:
            list[dict]: OUT: Result of the operation."""

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
        """Internal helper to filter by keywords.

        Args:
            results (list[dict]): IN: results. OUT: Consumed during execution.
            keywords (list[str] | None): IN: keywords. OUT: Consumed during execution.
            exclude (bool, optional): IN: exclude. Defaults to False. OUT: Consumed during execution.
        Returns:
            list[dict]: OUT: Result of the operation."""

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
        """Internal helper to append text results.

        Args:
            results (list[dict]): IN: results. OUT: Consumed during execution.
            search_results (tp.Iterable[dict]): IN: search results. OUT: Consumed during execution.
            n_results (int | None): IN: n results. OUT: Consumed during execution.
            title_length_limit (int | None): IN: title length limit. OUT: Consumed during execution.
            snippet_length_limit (int | None): IN: snippet length limit. OUT: Consumed during execution."""

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
        """Internal helper to is no results error.

        Args:
            error (Exception): IN: error. OUT: Consumed during execution.
        Returns:
            bool: OUT: Result of the operation."""

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
        """Static call.

        Args:
            query (str): IN: query. OUT: Consumed during execution.
            search_type (SearchType, optional): IN: search type. Defaults to 'text'. OUT: Consumed during execution.
            n_results (int | None, optional): IN: n results. Defaults to 5. OUT: Consumed during execution.
            title_length_limit (int | None, optional): IN: title length limit. Defaults to 200. OUT: Consumed during execution.
            snippet_length_limit (int | None, optional): IN: snippet length limit. Defaults to 1000. OUT: Consumed during execution.
            region (str, optional): IN: region. Defaults to 'us-en'. OUT: Consumed during execution.
            safesearch (SafeSearch, optional): IN: safesearch. Defaults to 'moderate'. OUT: Consumed during execution.
            timelimit (TimeFilter, optional): IN: timelimit. Defaults to None. OUT: Consumed during execution.
            allowed_domains (list[str] | None, optional): IN: allowed domains. Defaults to None. OUT: Consumed during execution.
            excluded_domains (list[str] | None, optional): IN: excluded domains. Defaults to None. OUT: Consumed during execution.
            must_include_keywords (list[str] | None, optional): IN: must include keywords. Defaults to None. OUT: Consumed during execution.
            exclude_keywords (list[str] | None, optional): IN: exclude keywords. Defaults to None. OUT: Consumed during execution.
            file_type (str | None, optional): IN: file type. Defaults to None. OUT: Consumed during execution.
            return_metadata (bool, optional): IN: return metadata. Defaults to False. OUT: Consumed during execution.
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            list[dict] | dict: OUT: Result of the operation."""

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
        """Search for multiple sources.

        Args:
            query (str): IN: query. OUT: Consumed during execution.
            sources (list[SearchType] | None, optional): IN: sources. Defaults to None. OUT: Consumed during execution.
            n_results_per_source (int, optional): IN: n results per source. Defaults to 3. OUT: Consumed during execution.
            **kwargs: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            dict[str, Any]: OUT: Result of the operation."""

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
        """Retrieve the suggestions.

        Args:
            query (str): IN: query. OUT: Consumed during execution.
            region (str, optional): IN: region. Defaults to 'us-en'. OUT: Consumed during execution.
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            list[str]: OUT: Result of the operation."""

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
        """Translate query.

        Args:
            query (str): IN: query. OUT: Consumed during execution.
            to_language (str, optional): IN: to language. Defaults to 'en'. OUT: Consumed during execution.
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            str: OUT: Result of the operation."""

        with _get_ddgs()() as ddgs:
            try:
                result = ddgs.translate(query, to=to_language)
                return result.get("translated", query)
            except Exception as e:
                import logging

                logging.getLogger(__name__).debug(f"Failed to translate '{query}' to {to_language}: {e}")
                return query


__all__ = ("DuckDuckGoSearch",)
