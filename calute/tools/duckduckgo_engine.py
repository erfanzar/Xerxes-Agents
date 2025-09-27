# Copyright 2025 The EasyDeL/Calute Author @erfanzar (Erfan Zare Chavoshi).
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


from datetime import datetime
from typing import Literal

from ..types import AgentBaseFn


class DuckDuckGoSearch(AgentBaseFn):
    try:
        from ddgs import DDGS
    except ModuleNotFoundError as e:
        print("`ddgs` package not found please install with calute[search]")
        raise e

    SearchType = Literal["text", "images", "videos", "news", "maps"]

    TimeFilter = Literal["day", "week", "month", "year", None]

    SafeSearch = Literal["strict", "moderate", "off"]

    @staticmethod
    def _maybe_truncate(text: str, limit: int | None) -> str:
        """Return the full text if limit is None, else the first `limit` chars."""
        return text if limit is None else text[:limit]

    @staticmethod
    def _filter_by_domain(results: list[dict], domains: list[str] | None) -> list[dict]:
        """Filter results to only include specified domains."""
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
        """Filter results by keywords in title or snippet."""
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

        Args:
            query (str): Search keywords.
            search_type (SearchType): Type of search - "text", "images", "videos", "news", "maps".
            n_results (int, optional): 1-30 results (default 5).
            title_length_limit (int | None): Max chars for title; None = no cut.
            snippet_length_limit (int | None): Max chars for snippet; None = no cut.
            region (str): Region code (e.g., "us-en", "uk-en", "fr-fr").
            safesearch (SafeSearch): Safe search level - "strict", "moderate", "off".
            timelimit (TimeFilter): Time filter - "day", "week", "month", "year", None.
            allowed_domains (list[str] | None): Only return results from these domains.
            excluded_domains (list[str] | None): Exclude results from these domains.
            must_include_keywords (list[str] | None): Results must contain these keywords.
            exclude_keywords (list[str] | None): Results must not contain these keywords.
            file_type (str | None): Filter by file type (e.g., "pdf", "doc", "xls").
            return_metadata (bool): Return metadata about the search.

        Returns:
            Union[list[dict], dict]: Search results or dict with results and metadata.
        """
        if not query.strip():
            raise ValueError("Query string must be non-empty")
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

        with DuckDuckGoSearch.DDGS() as ddgs:
            if search_type == "text":
                search_results = ddgs.text(
                    query,
                    region=region,
                    safesearch=safesearch.capitalize() if safesearch else "Moderate",
                    timelimit=timelimit,
                )
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
        """
        Search across multiple source types and return categorized results.

        Args:
            query (str): Search query.
            sources (list[SearchType]): List of search types to query.
            n_results_per_source (int): Number of results per source type.
            **kwargs: Additional arguments passed to static_call.

        Returns:
            dict[str, list[dict]]: Results categorized by source type.
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
        """
        Get search suggestions for a query.

        Args:
            query (str): Partial search query.
            region (str): Region code.

        Returns:
            list[str]: List of suggested search queries.
        """
        suggestions = []

        with DuckDuckGoSearch.DDGS() as ddgs:
            try:
                results = ddgs.suggestions(query, region=region)
                suggestions = [r.get("phrase", "") for r in results if r.get("phrase")]
            except Exception:
                pass

        return suggestions

    @staticmethod
    def translate_query(query: str, to_language: str = "en", **context_variables) -> str:
        """
        Translate a search query to another language.

        Args:
            query (str): Original query.
            to_language (str): Target language code.

        Returns:
            str: Translated query.
        """
        with DuckDuckGoSearch.DDGS() as ddgs:
            try:
                result = ddgs.translate(query, to=to_language)
                return result.get("translated", query)
            except Exception:
                return query


__all__ = ("DuckDuckGoSearch",)
