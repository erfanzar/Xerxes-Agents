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
"""Playwright-backed browser manager used by the operator ``web.*`` tools.

Owns a single chromium browser process and context, and tracks every page
opened during the session by a stable ``ref_id``. The manager is lazy: the
browser is not launched until the first ``open`` call, so sessions that
never touch the web pay no startup cost.
"""

from __future__ import annotations

import re
import tempfile
import typing as tp
import uuid
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BrowserPageState:
    """Bookkeeping for one tracked Playwright page.

    Attributes:
        ref_id: Stable identifier used by ``web.click`` /``web.find`` /
            ``web.screenshot`` to address the page across tool calls.
        url: Last-known URL of the page (updated on every ``open``).
        title: Page ``<title>`` captured during the most recent navigation.
        link_map: Dictionary mapping numeric link ids to absolute URLs so
            the model can click discovered links by id without writing a
            CSS selector.
    """

    ref_id: str
    url: str
    title: str = ""
    link_map: dict[int, str] = field(default_factory=dict)


class BrowserManager:
    """Lazy Playwright wrapper that backs the operator ``web.*`` tools.

    A single instance owns the chromium browser, the default context, and
    every :class:`BrowserPageState` registered during the session. The
    browser is only launched on the first ``open`` call; all later calls
    reuse the running context.
    """

    def __init__(self, *, headless: bool = True, screenshot_dir: str | None = None) -> None:
        """Configure browser launch options without starting Playwright yet.

        Args:
            headless: Launch chromium without a visible window.
            screenshot_dir: Directory used by ``screenshot`` when the caller
                doesn't supply an explicit path; a tempdir is used if this
                is ``None``.
        """

        self._headless = headless
        self._screenshot_dir = screenshot_dir
        self._playwright: tp.Any = None
        self._browser: tp.Any = None
        self._context: tp.Any = None
        self._pages: dict[str, tp.Any] = {}
        self._page_state: dict[str, BrowserPageState] = {}

    async def open(self, *, url: str | None = None, ref_id: str | None = None, wait_ms: int = 500) -> dict[str, tp.Any]:
        """Navigate to ``url`` (or revisit ``ref_id``) and return page state.

        When ``ref_id`` is omitted a brand-new tab is opened and assigned a
        fresh id. When ``url`` is omitted the existing page is re-inspected
        without navigating. The returned payload includes the rendered
        title, a 2 KB content preview and the click-able link map.
        """

        page, state = await self._resolve_page(url=url, ref_id=ref_id)
        if url is not None:
            await page.goto(url, wait_until="domcontentloaded")
            await page.wait_for_timeout(wait_ms)
            state.url = page.url
        state.title = await page.title()
        state.link_map = await self._extract_link_map(page)
        content = await page.locator("body").inner_text()
        return {
            "ref_id": state.ref_id,
            "url": page.url,
            "title": state.title,
            "content_preview": content[:2000],
            "links": [{"id": idx, "url": href} for idx, href in sorted(state.link_map.items())],
        }

    async def click(
        self,
        ref_id: str,
        *,
        link_id: int | None = None,
        selector: str | None = None,
        text: str | None = None,
        wait_ms: int = 500,
    ) -> dict[str, tp.Any]:
        """Interact with a tracked page and return its updated state.

        Exactly one of ``link_id``, ``selector`` or ``text`` must be set:

        * ``link_id`` — navigate to ``link_map[link_id]`` discovered by the
          last :meth:`open`.
        * ``selector`` — click the first element matching the CSS selector.
        * ``text`` — click the first element containing the literal text.

        Raises:
            ValueError: When no targeting argument is given, or when
                ``link_id`` has no mapping for the page.
        """

        page = self._require_page(ref_id)
        state = self._page_state[ref_id]
        if link_id is not None:
            href = state.link_map.get(link_id)
            if href is None:
                raise ValueError(f"Link id {link_id} not found for page {ref_id}")
            await page.goto(href, wait_until="domcontentloaded")
        elif selector:
            await page.locator(selector).first.click()
        elif text:
            await page.get_by_text(text).first.click()
        else:
            raise ValueError("click requires link_id, selector, or text")
        await page.wait_for_timeout(wait_ms)
        return await self.open(ref_id=ref_id)

    async def find(self, ref_id: str, pattern: str) -> dict[str, tp.Any]:
        """Search the page's visible body text for a case-insensitive regex.

        Returns at most 20 matches plus the total match count so the model
        can decide whether to refine the pattern.
        """

        page = self._require_page(ref_id)
        body_text = await page.locator("body").inner_text()
        regex = re.compile(pattern, re.IGNORECASE)
        matches = regex.findall(body_text)
        return {
            "ref_id": ref_id,
            "pattern": pattern,
            "match_count": len(matches),
            "matches": matches[:20],
        }

    async def screenshot(self, ref_id: str, *, path: str | None = None, full_page: bool = True) -> dict[str, tp.Any]:
        """Capture a screenshot of the tracked page and persist it to disk.

        Args:
            ref_id: Identifier returned by a previous :meth:`open`.
            path: Output filename; auto-generated under the screenshot
                directory when ``None``.
            full_page: Capture the entire scrollable canvas (default) or
                just the current viewport.
        """

        page = self._require_page(ref_id)
        screenshot_path = path or self._default_screenshot_path(ref_id)
        await page.screenshot(path=screenshot_path, full_page=full_page)
        return {"ref_id": ref_id, "path": screenshot_path, "full_page": full_page}

    def list_pages(self) -> list[dict[str, str]]:
        """Return a wire-safe summary of every tracked page."""

        return [
            {"ref_id": ref_id, "url": state.url, "title": state.title}
            for ref_id, state in sorted(self._page_state.items())
        ]

    async def _resolve_page(self, *, url: str | None, ref_id: str | None) -> tuple[tp.Any, BrowserPageState]:
        """Return ``(page, state)`` for an existing ref or open a new tab."""

        await self._ensure_browser()
        if ref_id is not None:
            return self._require_page(ref_id), self._page_state[ref_id]
        if url is None:
            raise ValueError("open requires url or ref_id")
        page = await self._context.new_page()
        ref_id = f"page_{uuid.uuid4().hex[:10]}"
        state = BrowserPageState(ref_id=ref_id, url=url)
        self._pages[ref_id] = page
        self._page_state[ref_id] = state
        return page, state

    def _require_page(self, ref_id: str) -> tp.Any:
        """Return the Playwright page for ``ref_id`` or raise ``ValueError``."""

        if ref_id not in self._pages:
            raise ValueError(f"Browser page not found: {ref_id}")
        return self._pages[ref_id]

    async def _ensure_browser(self) -> None:
        """Lazily launch Playwright and the chromium browser/context."""

        if self._browser is not None:
            return
        try:
            from playwright.async_api import async_playwright
        except ImportError as exc:
            raise RuntimeError("Playwright is required for browser operator tools") from exc

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=self._headless)
        self._context = await self._browser.new_context()

    async def _extract_link_map(self, page: tp.Any) -> dict[int, str]:
        """Enumerate ``<a href>`` URLs on the page into an id-indexed map."""

        links = await page.locator("a[href]").evaluate_all("(els) => els.map((el) => el.href).filter(Boolean)")
        return {index: href for index, href in enumerate(links)}

    def _default_screenshot_path(self, ref_id: str) -> str:
        """Return an auto-generated screenshot filename for ``ref_id``."""

        if self._screenshot_dir:
            directory = Path(self._screenshot_dir)
            directory.mkdir(parents=True, exist_ok=True)
        else:
            directory = Path(tempfile.mkdtemp(prefix="xerxes-browser-"))
        return str(directory / f"{ref_id}.png")
