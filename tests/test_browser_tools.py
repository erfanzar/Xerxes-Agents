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
"""Tests for browser-control tools (httpx fallback path)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from xerxes.tools.browser_tools import (
    BrowserSession,
    browser_back,
    browser_click,
    browser_console,
    browser_get_images,
    browser_navigate,
    browser_scroll,
    browser_snapshot,
    browser_vision,
)


@dataclass
class _FakeResp:
    text: str
    url: str
    status_code: int = 200


class _FakeHTTP:
    def __init__(self, pages: dict[str, str]):
        self.pages = pages
        self.calls: list[str] = []

    def get(self, url: str, follow_redirects: bool = True) -> _FakeResp:
        self.calls.append(url)
        return _FakeResp(text=self.pages.get(url, "<html><title>404</title></html>"), url=url)


@pytest.fixture
def session():
    pages = {
        "https://example.com": (
            """
            <html>
              <head><title>Example Domain</title></head>
              <body>
                <h1>Hello world</h1>
                <p>An <a id="more" href="/more">more info</a> link.</p>
                <input id="q" name="q" placeholder="search" />
                <button id="go">Go</button>
                <img src="/logo.png" alt="logo" />
                <img src="https://cdn.example.com/banner.jpg" alt="banner" />
              </body>
            </html>
        """
        ),
        "https://example.com/more": (
            """
            <html><head><title>More</title></head>
            <body><p>More page content</p></body></html>
        """
        ),
    }
    s = BrowserSession.install_for_test(_FakeHTTP(pages))
    yield s
    BrowserSession.reset()


class TestNavigate:
    def test_loads_and_returns_summary(self, session):
        out = browser_navigate.static_call(url="https://example.com")
        assert out["url"] == "https://example.com"
        assert out["title"] == "Example Domain"
        assert out["elements"] >= 3

    def test_rejects_non_http_scheme(self, session):
        with pytest.raises(ValueError):
            browser_navigate.static_call(url="file:///etc/passwd")


class TestSnapshot:
    def test_returns_text_and_elements(self, session):
        browser_navigate.static_call(url="https://example.com")
        snap = browser_snapshot.static_call()
        assert snap["url"] == "https://example.com"
        assert "Hello world" in snap["text"]
        names = {e["name"] for e in snap["elements"]}
        assert "more info" in names
        assert "Go" in names

    def test_each_element_has_ref_and_role(self, session):
        browser_navigate.static_call(url="https://example.com")
        snap = browser_snapshot.static_call()
        for elem in snap["elements"]:
            assert elem["ref"].startswith("e")
            assert elem["role"]
            assert elem["tag"]


class TestImagesAndConsole:
    def test_get_images_resolves_relative(self, session):
        browser_navigate.static_call(url="https://example.com")
        out = browser_get_images.static_call()
        srcs = {img["src"] for img in out["images"]}
        assert "https://example.com/logo.png" in srcs
        assert "https://cdn.example.com/banner.jpg" in srcs

    def test_console_returns_empty_in_fallback(self, session):
        browser_navigate.static_call(url="https://example.com")
        assert browser_console.static_call()["console"] == []


class TestNavigationActions:
    def test_click_link_navigates(self, session):
        browser_navigate.static_call(url="https://example.com")
        snap = browser_snapshot.static_call()
        link = next(e for e in snap["elements"] if e["name"] == "more info")
        result = browser_click.static_call(ref=link["ref"])
        assert result["ok"] is True
        assert browser_snapshot.static_call()["url"] == "https://example.com/more"

    def test_back_returns_to_prior_page(self, session):
        browser_navigate.static_call(url="https://example.com")
        browser_navigate.static_call(url="https://example.com/more")
        out = browser_back.static_call()
        assert out["ok"] is True
        assert browser_snapshot.static_call()["url"] == "https://example.com"

    def test_back_with_no_history_fails_safely(self, session):
        out = browser_back.static_call()
        assert out["ok"] is False

    def test_click_unknown_ref_fails(self, session):
        browser_navigate.static_call(url="https://example.com")
        out = browser_click.static_call(ref="ezzz")
        assert out["ok"] is False


class TestScrollAndVision:
    def test_scroll_increments_offset(self, session):
        browser_navigate.static_call(url="https://example.com")
        out = browser_scroll.static_call(dy=300)
        assert out["scroll_y"] == 300
        out2 = browser_scroll.static_call(dy=100)
        assert out2["scroll_y"] == 400

    def test_vision_falls_back_to_text(self, session):
        browser_navigate.static_call(url="https://example.com")
        out = browser_vision.static_call()
        assert out["url"] == "https://example.com"
        assert out["format"] in {"png", "none"}
        if out["format"] == "none":
            assert "Hello world" in out.get("text", "")
