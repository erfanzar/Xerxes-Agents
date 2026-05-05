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
"""Browser module for Xerxes.

Exports:
    - BrowserPageState
    - BrowserManager"""

from __future__ import annotations

import re
import tempfile
import typing as tp
import uuid
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BrowserPageState:
    """Browser page state.

    Attributes:
        ref_id (str): ref id.
        url (str): url.
        title (str): title.
        link_map (dict[int, str]): link map."""

    ref_id: str
    url: str
    title: str = ""
    link_map: dict[int, str] = field(default_factory=dict)


class BrowserManager:
    """Browser manager."""

    def __init__(self, *, headless: bool = True, screenshot_dir: str | None = None) -> None:
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            headless (bool, optional): IN: headless. Defaults to True. OUT: Consumed during execution.
            screenshot_dir (str | None, optional): IN: screenshot dir. Defaults to None. OUT: Consumed during execution."""

        self._headless = headless
        self._screenshot_dir = screenshot_dir
        self._playwright: tp.Any = None
        self._browser: tp.Any = None
        self._context: tp.Any = None
        self._pages: dict[str, tp.Any] = {}
        self._page_state: dict[str, BrowserPageState] = {}

    async def open(self, *, url: str | None = None, ref_id: str | None = None, wait_ms: int = 500) -> dict[str, tp.Any]:
        """Asynchronously Open.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            url (str | None, optional): IN: url. Defaults to None. OUT: Consumed during execution.
            ref_id (str | None, optional): IN: ref id. Defaults to None. OUT: Consumed during execution.
            wait_ms (int, optional): IN: wait ms. Defaults to 500. OUT: Consumed during execution.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

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
        """Asynchronously Click.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            ref_id (str): IN: ref id. OUT: Consumed during execution.
            link_id (int | None, optional): IN: link id. Defaults to None. OUT: Consumed during execution.
            selector (str | None, optional): IN: selector. Defaults to None. OUT: Consumed during execution.
            text (str | None, optional): IN: text. Defaults to None. OUT: Consumed during execution.
            wait_ms (int, optional): IN: wait ms. Defaults to 500. OUT: Consumed during execution.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

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
        """Asynchronously Find.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            ref_id (str): IN: ref id. OUT: Consumed during execution.
            pattern (str): IN: pattern. OUT: Consumed during execution.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

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
        """Asynchronously Screenshot.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            ref_id (str): IN: ref id. OUT: Consumed during execution.
            path (str | None, optional): IN: path. Defaults to None. OUT: Consumed during execution.
            full_page (bool, optional): IN: full page. Defaults to True. OUT: Consumed during execution.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

        page = self._require_page(ref_id)
        screenshot_path = path or self._default_screenshot_path(ref_id)
        await page.screenshot(path=screenshot_path, full_page=full_page)
        return {"ref_id": ref_id, "path": screenshot_path, "full_page": full_page}

    def list_pages(self) -> list[dict[str, str]]:
        """List pages.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            list[dict[str, str]]: OUT: Result of the operation."""

        return [
            {"ref_id": ref_id, "url": state.url, "title": state.title}
            for ref_id, state in sorted(self._page_state.items())
        ]

    async def _resolve_page(self, *, url: str | None, ref_id: str | None) -> tuple[tp.Any, BrowserPageState]:
        """Asynchronously Internal helper to resolve page.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            url (str | None): IN: url. OUT: Consumed during execution.
            ref_id (str | None): IN: ref id. OUT: Consumed during execution.
        Returns:
            tuple[tp.Any, BrowserPageState]: OUT: Result of the operation."""

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
        """Internal helper to require page.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            ref_id (str): IN: ref id. OUT: Consumed during execution.
        Returns:
            tp.Any: OUT: Result of the operation."""

        if ref_id not in self._pages:
            raise ValueError(f"Browser page not found: {ref_id}")
        return self._pages[ref_id]

    async def _ensure_browser(self) -> None:
        """Asynchronously Internal helper to ensure browser.

        Args:
            self: IN: The instance. OUT: Used for attribute access."""

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
        """Asynchronously Internal helper to extract link map.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            page (tp.Any): IN: page. OUT: Consumed during execution.
        Returns:
            dict[int, str]: OUT: Result of the operation."""

        links = await page.locator("a[href]").evaluate_all("(els) => els.map((el) => el.href).filter(Boolean)")
        return {index: href for index, href in enumerate(links)}

    def _default_screenshot_path(self, ref_id: str) -> str:
        """Internal helper to default screenshot path.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            ref_id (str): IN: ref id. OUT: Consumed during execution.
        Returns:
            str: OUT: Result of the operation."""

        if self._screenshot_dir:
            directory = Path(self._screenshot_dir)
            directory.mkdir(parents=True, exist_ok=True)
        else:
            directory = Path(tempfile.mkdtemp(prefix="xerxes-browser-"))
        return str(directory / f"{ref_id}.png")
