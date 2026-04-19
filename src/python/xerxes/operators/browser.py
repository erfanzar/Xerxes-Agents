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


"""Playwright-backed browser state manager for operator tooling.

Provides :class:`BrowserManager`, which lazily initialises a Chromium
browser via Playwright and manages a pool of tracked pages.  Each page
is represented by a lightweight :class:`BrowserPageState` dataclass that
records the page reference ID, current URL, title, and extracted links.
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
    """Tracked state for an opened browser page.

    Attributes:
        ref_id: Unique reference identifier used to address this page
            across operator tool calls.
        url: The last-known URL loaded by this page.
        title: The page title extracted after navigation.
        link_map: Mapping of numeric link IDs to their ``href`` values,
            populated after each page load or refresh.
    """

    ref_id: str
    url: str
    title: str = ""
    link_map: dict[int, str] = field(default_factory=dict)


class BrowserManager:
    """Manage a shared Playwright browser and tracked pages.

    The manager lazily starts a Chromium browser on the first call that
    requires a live page.  All pages opened through the manager are
    tracked by a generated ``ref_id`` so that subsequent operator tool
    calls (click, find, screenshot) can address them without re-opening.

    Attributes:
        _headless: Whether the browser runs in headless mode.
        _screenshot_dir: Optional directory for screenshot output.
        _playwright: Playwright instance, created lazily.
        _browser: Chromium browser instance, created lazily.
        _context: Default browser context.
        _pages: Mapping of ``ref_id`` to live Playwright page objects.
        _page_state: Mapping of ``ref_id`` to :class:`BrowserPageState`.
    """

    def __init__(self, *, headless: bool = True, screenshot_dir: str | None = None) -> None:
        """Initialise the browser manager.

        Args:
            headless: If ``True``, the Chromium browser is launched
                without a visible window.  Defaults to ``True``.
            screenshot_dir: Optional directory path where screenshots
                are saved.  When ``None``, a temporary directory is
                created per screenshot call.
        """
        self._headless = headless
        self._screenshot_dir = screenshot_dir
        self._playwright: tp.Any = None
        self._browser: tp.Any = None
        self._context: tp.Any = None
        self._pages: dict[str, tp.Any] = {}
        self._page_state: dict[str, BrowserPageState] = {}

    async def open(self, *, url: str | None = None, ref_id: str | None = None, wait_ms: int = 500) -> dict[str, tp.Any]:
        """Open a URL or inspect an existing tracked page.

        Either ``url`` or ``ref_id`` must be provided.  When ``url`` is
        given, a new page is created (or an existing page navigated) and
        its metadata is returned.  When only ``ref_id`` is given, the
        currently loaded page is re-inspected.

        Args:
            url: URL to navigate to.  A new tracked page is created when
                no ``ref_id`` is supplied alongside the URL.
            ref_id: Reference identifier of a previously opened page to
                re-inspect without navigating.
            wait_ms: Milliseconds to wait after navigation before
                extracting page metadata.  Defaults to ``500``.

        Returns:
            A dictionary containing the page ``ref_id``, current URL,
            title, a truncated content preview (first 2000 characters),
            and a list of extracted links with numeric IDs.

        Raises:
            ValueError: If neither ``url`` nor ``ref_id`` is provided,
                or if the given ``ref_id`` does not match any tracked
                page.
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
        """Click an element on a tracked page.

        Exactly one of ``link_id``, ``selector``, or ``text`` must be
        provided to identify the target element.

        Args:
            ref_id: Reference identifier of the tracked page.
            link_id: Numeric link identifier from the page's
                :attr:`BrowserPageState.link_map`.  When provided, the
                browser navigates to the corresponding ``href``.
            selector: CSS selector of the element to click.
            text: Visible text used to locate the element via
                Playwright's ``get_by_text``.
            wait_ms: Milliseconds to wait after the click before
                refreshing the page metadata.  Defaults to ``500``.

        Returns:
            The refreshed page metadata dictionary (same shape as
            :meth:`open`).

        Raises:
            ValueError: If the ``ref_id`` is unknown, the ``link_id``
                is not found, or none of the three target parameters
                is provided.
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
        """Find text matches on a tracked page.

        Performs a case-insensitive regular expression search across the
        visible body text of the referenced page.

        Args:
            ref_id: Reference identifier of the tracked page to search.
            pattern: Regular expression pattern to match against the
                page's visible text content.

        Returns:
            A dictionary with the ``ref_id``, the ``pattern`` used, the
            total ``match_count``, and up to 20 matching strings.

        Raises:
            ValueError: If the ``ref_id`` does not correspond to a
                tracked page.
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
        """Capture a screenshot of a tracked page.

        Args:
            ref_id: Reference identifier of the tracked page to
                capture.
            path: Optional file path for the screenshot.  If omitted, a
                default path inside the configured screenshot directory
                (or a temporary directory) is used.
            full_page: When ``True``, capture the entire scrollable page
                instead of just the visible viewport.  Defaults to
                ``True``.

        Returns:
            A dictionary containing the ``ref_id``, saved file ``path``,
            and the ``full_page`` flag.

        Raises:
            ValueError: If the ``ref_id`` is not tracked.
        """
        page = self._require_page(ref_id)
        screenshot_path = path or self._default_screenshot_path(ref_id)
        await page.screenshot(path=screenshot_path, full_page=full_page)
        return {"ref_id": ref_id, "path": screenshot_path, "full_page": full_page}

    def list_pages(self) -> list[dict[str, str]]:
        """Return summaries for tracked pages.

        Returns:
            A list of dictionaries, each containing the ``ref_id``,
            ``url``, and ``title`` of a tracked page, sorted by
            ``ref_id``.
        """
        return [
            {"ref_id": ref_id, "url": state.url, "title": state.title}
            for ref_id, state in sorted(self._page_state.items())
        ]

    async def _resolve_page(self, *, url: str | None, ref_id: str | None) -> tuple[tp.Any, BrowserPageState]:
        """Resolve or create a Playwright page and its tracking state.

        If ``ref_id`` is given, the existing page and state are returned.
        If only ``url`` is given, a new page is created in the shared
        browser context.

        Args:
            url: URL for which a new page should be created when no
                ``ref_id`` is provided.
            ref_id: Reference identifier of an existing tracked page.

        Returns:
            A tuple of ``(page, state)`` where *page* is the Playwright
            page object and *state* is the :class:`BrowserPageState`.

        Raises:
            ValueError: If neither ``url`` nor ``ref_id`` is provided,
                or the ``ref_id`` is unknown.
        """
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
        """Return a tracked Playwright page or raise.

        Args:
            ref_id: Reference identifier of the page to retrieve.

        Returns:
            The live Playwright page object.

        Raises:
            ValueError: If no page with the given ``ref_id`` exists.
        """
        if ref_id not in self._pages:
            raise ValueError(f"Browser page not found: {ref_id}")
        return self._pages[ref_id]

    async def _ensure_browser(self) -> None:
        """Lazily start the Playwright Chromium browser.

        Called automatically before any page operation.  If the browser
        is already running this method is a no-op.

        Raises:
            RuntimeError: If the ``playwright`` package is not
                installed.
        """
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
        """Extract an ordered mapping of link hrefs from the page.

        Args:
            page: Playwright page object to inspect.

        Returns:
            A dictionary mapping sequential integer IDs to the ``href``
            values of all ``<a>`` elements found on the page.
        """
        links = await page.locator("a[href]").evaluate_all("(els) => els.map((el) => el.href).filter(Boolean)")
        return {index: href for index, href in enumerate(links)}

    def _default_screenshot_path(self, ref_id: str) -> str:
        """Compute a default screenshot file path.

        Uses the configured :attr:`_screenshot_dir` when set, otherwise
        creates a temporary directory.

        Args:
            ref_id: Page reference identifier used to construct the
                filename.

        Returns:
            Absolute path string for the screenshot file.
        """
        if self._screenshot_dir:
            directory = Path(self._screenshot_dir)
            directory.mkdir(parents=True, exist_ok=True)
        else:
            directory = Path(tempfile.mkdtemp(prefix="xerxes-browser-"))
        return str(directory / f"{ref_id}.png")
