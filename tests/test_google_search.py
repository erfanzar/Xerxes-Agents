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
"""Tests for GoogleSearch (API + scrape paths) with injected HTTP."""

from __future__ import annotations

import pytest
from xerxes.tools.google_search import (
    GoogleSearch,
    configure_google_search,
    get_google_search_config,
    set_google_search_client,
)


@pytest.fixture(autouse=True)
def _restore_config():
    cfg = get_google_search_config()
    saved = (cfg.api_key, cfg.cse_id, cfg.safe, cfg.user_agent)
    yield
    configure_google_search(api_key=saved[0], cse_id=saved[1], safe=saved[2], user_agent=saved[3])
    set_google_search_client(None)


# ── Helpers ───────────────────────────────────────────────────────────


class _Resp:
    def __init__(self, payload=None, text=None, status_code=200):
        self._payload = payload
        self._text = text
        self.status_code = status_code

    def json(self):
        if self._payload is None:
            raise ValueError("no JSON")
        return self._payload

    @property
    def text(self):
        return self._text or ""


class _FakeHTTP:
    def __init__(self, response):
        self.response = response
        self.calls: list[dict] = []

    def get(self, url, headers=None, params=None):
        self.calls.append({"url": url, "headers": headers, "params": params})
        return self.response


# ── API mode ──────────────────────────────────────────────────────────


class TestApiMode:
    def test_resolves_to_api_when_configured(self):
        configure_google_search(api_key="KEY", cse_id="CSE")
        http = _FakeHTTP(
            _Resp(
                payload={
                    "items": [
                        {
                            "title": "Xerxes",
                            "link": "https://github.com/erfanzar/Xerxes",
                            "snippet": "An agent runtime",
                            "displayLink": "github.com",
                        },
                        {
                            "title": "Xerxes docs",
                            "link": "https://example.com/xerxes",
                            "snippet": "Hermes parity",
                            "displayLink": "example.com",
                        },
                    ],
                    "searchInformation": {"totalResults": "2"},
                }
            )
        )
        set_google_search_client(http)
        out = GoogleSearch.static_call(query="xerxes agent runtime", n_results=2)
        assert out["engine"] == "google_api"
        assert out["count"] == 2
        assert out["results"][0]["url"] == "https://github.com/erfanzar/Xerxes"
        assert "totalResults" in out["search_information"]

    def test_api_passes_query_and_quota_args(self):
        configure_google_search(api_key="KEY", cse_id="CSE")
        http = _FakeHTTP(_Resp(payload={"items": []}))
        set_google_search_client(http)
        GoogleSearch.static_call(query="weather tokyo", n_results=3, site="bbc.co.uk", time_range="d")
        params = http.calls[0]["params"]
        assert params["key"] == "KEY"
        assert params["cx"] == "CSE"
        assert params["q"].startswith("site:bbc.co.uk weather tokyo")
        assert params["num"] == "3"
        assert params["dateRestrict"] == "d"

    def test_api_caps_num_at_10(self):
        configure_google_search(api_key="KEY", cse_id="CSE")
        http = _FakeHTTP(_Resp(payload={"items": []}))
        set_google_search_client(http)
        GoogleSearch.static_call(query="x", n_results=50)
        assert http.calls[0]["params"]["num"] == "10"

    def test_api_error_returns_envelope(self):
        configure_google_search(api_key="KEY", cse_id="CSE")
        set_google_search_client(_FakeHTTP(_Resp(text="forbidden", status_code=403)))
        out = GoogleSearch.static_call(query="x")
        assert out["engine"] == "google_api"
        assert "error" in out
        assert out["results"] == []


# ── Scrape mode ───────────────────────────────────────────────────────


_FAKE_HTML = """
<html><body>
<div class="g">
  <a href="https://github.com/erfanzar/Xerxes"><h3>Xerxes on GitHub</h3></a>
  <div class="VwiC3b">An autonomous AI agent runtime by erfanzar with Hermes-parity tools.</div>
</div>
<div class="g">
  <a href="https://hermes.example/news"><h3>Hermes news index</h3></a>
  <div class="VwiC3b">Latest headlines from the Hermes example domain.</div>
</div>
<div class="g">
  <a href="https://www.google.com/policies"><h3>Should be filtered</h3></a>
  <div class="VwiC3b">internal google link</div>
</div>
</body></html>
"""


class TestScrapeMode:
    def test_falls_back_to_scrape_without_api_key(self):
        configure_google_search(api_key="", cse_id="")
        http = _FakeHTTP(_Resp(text=_FAKE_HTML))
        set_google_search_client(http)
        out = GoogleSearch.static_call(query="xerxes github", n_results=5)
        assert out["engine"] == "google_scrape"
        urls = {r["url"] for r in out["results"]}
        assert "https://github.com/erfanzar/Xerxes" in urls
        assert "https://hermes.example/news" in urls
        assert all("google.com" not in u for u in urls)

    def test_scrape_sends_browser_user_agent(self):
        configure_google_search(api_key="", cse_id="")
        http = _FakeHTTP(_Resp(text="<html></html>"))
        set_google_search_client(http)
        GoogleSearch.static_call(query="x")
        ua = http.calls[0]["headers"]["User-Agent"]
        assert "Mozilla" in ua and "Safari" in ua

    def test_scrape_passes_site_filter(self):
        configure_google_search(api_key="", cse_id="")
        http = _FakeHTTP(_Resp(text="<html></html>"))
        set_google_search_client(http)
        GoogleSearch.static_call(query="memes", site="reddit.com")
        assert http.calls[0]["params"]["q"].startswith("site:reddit.com memes")

    def test_scrape_time_range_translates_to_qdr(self):
        configure_google_search(api_key="", cse_id="")
        http = _FakeHTTP(_Resp(text="<html></html>"))
        set_google_search_client(http)
        GoogleSearch.static_call(query="news", time_range="w")
        assert http.calls[0]["params"]["tbs"] == "qdr:w"

    def test_scrape_429_returns_error_envelope(self):
        configure_google_search(api_key="", cse_id="")
        set_google_search_client(_FakeHTTP(_Resp(text="rate limited", status_code=429)))
        out = GoogleSearch.static_call(query="x")
        assert out["engine"] == "google_scrape"
        assert "HTTP 429" in out["error"]
        assert out["results"] == []

    def test_scrape_handles_empty_html(self):
        configure_google_search(api_key="", cse_id="")
        set_google_search_client(_FakeHTTP(_Resp(text="<html></html>")))
        out = GoogleSearch.static_call(query="x")
        assert out["count"] == 0
        assert "warning" in out

    def test_scrape_caps_num_at_30(self):
        configure_google_search(api_key="", cse_id="")
        http = _FakeHTTP(_Resp(text="<html></html>"))
        set_google_search_client(http)
        GoogleSearch.static_call(query="x", n_results=999)
        assert http.calls[0]["params"]["num"] == "30"


# ── Public registry ───────────────────────────────────────────────────


class TestRegistryWiring:
    def test_exported_from_tools_package(self):
        from xerxes.tools import GoogleSearch as G

        assert G is GoogleSearch

    def test_listed_in_web_category(self):
        from xerxes.tools import TOOL_CATEGORIES

        assert "GoogleSearch" in TOOL_CATEGORIES["web"]
        # GoogleSearch should be the first/canonical web search tool.
        assert TOOL_CATEGORIES["web"][0] == "GoogleSearch"
