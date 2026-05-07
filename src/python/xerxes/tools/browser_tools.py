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
"""Browser automation tools for navigating, interacting with, and capturing web pages.

This module provides browser automation capabilities using Playwright with an HTTP fallback.
Agents can use these tools to scrape web content, interact with web forms, and capture
screenshots of web pages.

Example:
    >>> from xerxes.tools.browser_tools import browser_navigate, browser_snapshot
    >>> browser_navigate.static_call("https://example.com")
    >>> page = browser_snapshot.static_call()
"""

from __future__ import annotations

import base64
import logging
import re
import threading
import typing as tp
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse

from ..types import AgentBaseFn

logger = logging.getLogger(__name__)


@dataclass
class _Element:
    """Internal representation of a DOM element discovered during page parsing.

    Attributes:
        ref: Unique reference identifier for the element.
        tag: HTML tag name (e.g., 'a', 'button', 'input').
        role: ARIA role of the element (e.g., 'link', 'button', 'textbox').
        name: Display name extracted from text, placeholder, or aria-label.
        href: Resolved absolute URL for anchor elements.
        attrs: Dictionary of HTML attributes.
    """

    ref: str
    tag: str
    role: str
    name: str
    href: str = ""
    attrs: dict[str, str] = field(default_factory=dict)


@dataclass
class _Page:
    """Internal representation of the current browser page state.

    Attributes:
        url: Current page URL.
        title: Page title text.
        text: Extracted visible text content.
        elements: List of interactive elements found on the page.
        images: List of image metadata (src, alt).
        console: Console log messages captured during page execution.
        history: Navigation history stack.
        scroll_y: Current vertical scroll position in pixels.
    """

    url: str = ""
    title: str = ""
    text: str = ""
    elements: list[_Element] = field(default_factory=list)
    images: list[dict[str, str]] = field(default_factory=list)
    console: list[str] = field(default_factory=list)
    history: list[str] = field(default_factory=list)
    scroll_y: int = 0


class BrowserSession:
    """Manages a singleton browser session for web automation tasks.

    This class provides a thread-safe interface to browser automation, supporting
    both Playwright (for full JavaScript support) and HTTP fallback (for simple
    scraping). The session maintains page state including navigation history,
    parsed elements, and console logs.

    The session is shared across all tool calls to avoid resource exhaustion.
    Use reset() to clear state between independent browsing tasks.

    Attributes:
        _instance: Class-level singleton instance.
        _instance_lock: Thread synchronization lock for singleton access.

    Example:
        >>> session = BrowserSession.instance()
        >>> session.navigate("https://example.com")
        >>> session.click("e1")
    """

    _instance: BrowserSession | None = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        """Initialize a new browser session with empty state."""
        self._lock = threading.RLock()
        self._page = _Page()
        self._http_client: tp.Any | None = None
        self._playwright_page: tp.Any | None = None
        self._playwright: tp.Any | None = None
        self._playwright_browser: tp.Any | None = None

    @classmethod
    def instance(cls) -> BrowserSession:
        """Get the singleton BrowserSession instance, creating it if necessary.

        Returns:
            The shared BrowserSession instance.
        """
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance, closing any open browser and clearing state.

        Use this between independent browsing tasks to ensure clean state.
        """
        with cls._instance_lock:
            if cls._instance is not None:
                cls._instance.close()
                cls._instance = None

    @classmethod
    def install_for_test(cls, http_client: tp.Any) -> BrowserSession:
        """Install a test HTTP client and reset the singleton.

        Used for testing without requiring actual HTTP requests.

        Args:
            http_client: Mock HTTP client to use for requests.

        Returns:
            The new BrowserSession instance with test client installed.
        """
        with cls._instance_lock:
            if cls._instance is not None:
                cls._instance.close()
            inst = cls()
            inst._http_client = http_client
            cls._instance = inst
            return inst

    def close(self) -> None:
        """Close the browser, playwright process, and clear all state."""
        try:
            if self._playwright_page is not None:
                self._playwright_page.close()
            if self._playwright_browser is not None:
                self._playwright_browser.close()
            if self._playwright is not None:
                self._playwright.stop()
        except Exception:
            pass
        self._page = _Page()
        self._http_client = None
        self._playwright_page = None
        self._playwright_browser = None
        self._playwright = None

    def _ensure_playwright(self) -> bool:
        """Ensure Playwright is initialized and running.

        Initializes Playwright browser if not already done. Falls back to HTTP
        client if Playwright fails to start.

        Returns:
            True if Playwright is available and running, False if using fallback.
        """
        if self._http_client is not None:
            return False
        if self._playwright_page is not None:
            return True
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            return False
        try:
            self._playwright = sync_playwright().start()
            self._playwright_browser = self._playwright.chromium.launch(headless=True)
            self._playwright_page = self._playwright_browser.new_page()
            self._playwright_page.on("console", lambda m: self._page.console.append(f"[{m.type}] {m.text}"))
            return True
        except Exception:
            logger.warning("Playwright failed to start; falling back to httpx", exc_info=True)
            try:
                if self._playwright_browser is not None:
                    self._playwright_browser.close()
            except Exception:
                pass
            try:
                if self._playwright is not None:
                    self._playwright.stop()
            except Exception:
                pass
            self._playwright = None
            self._playwright_browser = None
            self._playwright_page = None
            return False

    def _http_get(self, url: str) -> tuple[str, str, int]:
        """Fetch URL content using the HTTP client.

        Args:
            url: The URL to fetch.

        Returns:
            Tuple of (final_url, html_content, status_code).
        """
        if self._http_client is not None:
            resp = self._http_client.get(url, follow_redirects=True)
            return str(getattr(resp, "url", url)), resp.text, int(resp.status_code)
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("httpx required for browser fallback") from exc
        resp = httpx.get(url, follow_redirects=True, timeout=30.0)
        resp.raise_for_status()
        return str(resp.url), resp.text, resp.status_code

    def navigate(self, url: str) -> dict[str, tp.Any]:
        """Navigate to a URL and parse the resulting page.

        Loads the page, waits for DOM content to be ready, extracts text and
        interactive elements, and clears console logs.

        Args:
            url: The URL to navigate to.

        Returns:
            Dictionary with 'url', 'title', and 'elements' count.
        """
        with self._lock:
            if self._ensure_playwright():
                self._page.console.clear()
                assert self._playwright_page is not None
                self._playwright_page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                final_url = self._playwright_page.url
                title = self._playwright_page.title()
                content = self._playwright_page.content()
            else:
                final_url, content, _status = self._http_get(url)
                title = ""
            self._page.url = final_url
            self._page.title = title
            self._page.history.append(final_url)
            self._page.scroll_y = 0
            self._parse_html(content)
            return {"url": final_url, "title": self._page.title, "elements": len(self._page.elements)}

    def back(self) -> dict[str, tp.Any]:
        """Navigate back in browser history.

        Returns:
            Result of navigating to the previous URL in history.
        """
        with self._lock:
            if len(self._page.history) < 2:
                return {"ok": False, "reason": "no history"}
            self._page.history.pop()
            target = self._page.history[-1]
            return self.navigate(target) | {"ok": True}

    def snapshot(self) -> dict[str, tp.Any]:
        """Capture current page state without making a network request.

        Returns:
            Dictionary with url, title, text (truncated to 4000 chars),
            element list, and scroll position.
        """
        with self._lock:
            return {
                "url": self._page.url,
                "title": self._page.title,
                "text": self._page.text[:4000],
                "elements": [
                    {"ref": e.ref, "tag": e.tag, "role": e.role, "name": e.name, "href": e.href}
                    for e in self._page.elements[:200]
                ],
                "scroll_y": self._page.scroll_y,
            }

    def vision(self) -> dict[str, tp.Any]:
        """Capture a screenshot of the current page.

        Returns:
            Dictionary with url, base64-encoded PNG image, format,
            and summary text. Returns empty image if Playwright unavailable.
        """
        with self._lock:
            if self._ensure_playwright():
                assert self._playwright_page is not None
                png = self._playwright_page.screenshot(full_page=False)
                return {
                    "url": self._page.url,
                    "image_b64": base64.b64encode(png).decode(),
                    "format": "png",
                    "summary": self._page.title or self._page.url,
                }
            return {
                "url": self._page.url,
                "image_b64": "",
                "format": "none",
                "summary": "Playwright not installed; vision returns text summary only",
                "text": self._page.text[:1000],
            }

    def get_images(self) -> dict[str, tp.Any]:
        """Retrieve metadata for images on the current page.

        Returns:
            Dictionary with url and list of image objects (src, alt).
        """
        with self._lock:
            return {"url": self._page.url, "images": list(self._page.images[:200])}

    def console_log(self) -> dict[str, tp.Any]:
        """Retrieve JavaScript console messages captured during page execution.

        Returns:
            Dictionary with url and list of console messages.
        """
        with self._lock:
            return {"url": self._page.url, "console": list(self._page.console[-200:])}

    def click(self, ref: str) -> dict[str, tp.Any]:
        """Click an element by its reference ID.

        Args:
            ref: The element reference to click.

        Returns:
            Dictionary with 'ok' status and 'ref' or 'reason' for errors.
        """
        with self._lock:
            elem = self._find(ref)
            if elem is None:
                return {"ok": False, "reason": f"unknown ref {ref!r}"}
            if self._playwright_page is not None:
                try:
                    selector = self._element_selector(elem)
                    self._playwright_page.click(selector, timeout=5_000)
                    self._sync_from_playwright()
                except Exception as exc:
                    return {"ok": False, "reason": str(exc)}
                return {"ok": True, "ref": ref}
            if elem.tag == "a" and elem.href:
                return self.navigate(elem.href) | {"ok": True, "ref": ref}
            return {"ok": True, "ref": ref, "note": "click had no effect (no JS)"}

    def type_text(self, ref: str, text: str, *, submit: bool = False) -> dict[str, tp.Any]:
        """Type text into an input element.

        Args:
            ref: The element reference to type into.
            text: The text string to type.
            submit: Whether to press Enter after typing.

        Returns:
            Dictionary with 'ok' status, 'ref', and 'submitted' flag.
        """
        with self._lock:
            elem = self._find(ref)
            if elem is None:
                return {"ok": False, "reason": f"unknown ref {ref!r}"}
            if self._playwright_page is not None:
                try:
                    selector = self._element_selector(elem)
                    self._playwright_page.fill(selector, text, timeout=5_000)
                    if submit:
                        self._playwright_page.press(selector, "Enter")
                    self._sync_from_playwright()
                except Exception as exc:
                    return {"ok": False, "reason": str(exc)}
                return {"ok": True, "ref": ref, "submitted": submit}
            elem.attrs["value"] = text
            elem.name = text
            return {"ok": True, "ref": ref, "submitted": submit, "note": "no JS in fallback"}

    def press(self, key: str) -> dict[str, tp.Any]:
        """Press a keyboard key.

        Args:
            key: Key name (e.g., 'Enter', 'Escape', 'ArrowDown').

        Returns:
            Dictionary with 'ok' status and 'key'.
        """
        with self._lock:
            if self._playwright_page is not None:
                try:
                    self._playwright_page.keyboard.press(key)
                    self._sync_from_playwright()
                except Exception as exc:
                    return {"ok": False, "reason": str(exc)}
                return {"ok": True, "key": key}
            return {"ok": True, "key": key, "note": "no JS in fallback"}

    def scroll(self, dy: int) -> dict[str, tp.Any]:
        """Scroll the page vertically.

        Args:
            dy: Number of pixels to scroll. Positive scrolls down, negative up.

        Returns:
            Dictionary with 'ok' status and new 'scroll_y' position.
        """
        with self._lock:
            if self._playwright_page is not None:
                try:
                    self._playwright_page.mouse.wheel(0, dy)
                    self._page.scroll_y = max(0, self._page.scroll_y + dy)
                except Exception as exc:
                    return {"ok": False, "reason": str(exc)}
            else:
                self._page.scroll_y = max(0, self._page.scroll_y + dy)
            return {"ok": True, "scroll_y": self._page.scroll_y}

    def _sync_from_playwright(self) -> None:
        """Synchronize internal page state from Playwright after interactions."""
        try:
            assert self._playwright_page is not None
            self._page.url = self._playwright_page.url
            self._page.title = self._playwright_page.title()
            content = self._playwright_page.content()
            self._parse_html(content)
        except Exception:
            pass

    def _find(self, ref: str) -> _Element | None:
        """Find an element by its reference ID.

        Args:
            ref: The element reference to search for.

        Returns:
            The matching _Element or None if not found.
        """
        for e in self._page.elements:
            if e.ref == ref:
                return e
        return None

    def _element_selector(self, elem: _Element) -> str:
        """Build a CSS selector for an element.

        Args:
            elem: The element to create a selector for.

        Returns:
            A CSS selector string targeting the element.
        """
        if elem.attrs.get("id"):
            return f"#{elem.attrs['id']}"
        if elem.attrs.get("name"):
            return f'{elem.tag}[name="{elem.attrs["name"]}"]'
        if elem.tag == "a" and elem.href:
            return f'a[href="{elem.href}"]'
        return elem.tag

    def _parse_html(self, html: str) -> None:
        """Parse HTML content and extract page information.

        Populates self._page with title, text, elements, and images.

        Args:
            html: Raw HTML content to parse.
        """
        try:
            from bs4 import BeautifulSoup, Tag
        except ImportError:
            self._page.text = re.sub(r"<[^>]+>", " ", html or "")
            self._page.text = re.sub(r"\s+", " ", self._page.text).strip()
            self._page.elements = []
            self._page.images = []
            return
        soup = BeautifulSoup(html or "", "html.parser")
        title_tag = soup.find("title")
        if title_tag and title_tag.get_text(strip=True):
            self._page.title = title_tag.get_text(strip=True)
        self._page.text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))[:8000]
        elements: list[_Element] = []
        counter = 0
        for tag in soup.find_all(["a", "button", "input", "textarea", "select"]):
            if not isinstance(tag, Tag):
                continue
            counter += 1
            ref = f"e{counter}"
            attrs = {k: " ".join(v) if isinstance(v, list) else str(v) for k, v in (tag.attrs or {}).items()}
            href = ""
            if tag.name == "a" and attrs.get("href"):
                href = urljoin(self._page.url or "", attrs["href"])
                attrs["href"] = href
            role = attrs.get("role") or {
                "a": "link",
                "button": "button",
                "input": attrs.get("type", "textbox") if tag.name == "input" else "textbox",
                "textarea": "textbox",
                "select": "combobox",
            }.get(tag.name or "", "generic")
            name = (
                attrs.get("aria-label")
                or tag.get_text(strip=True)
                or attrs.get("placeholder")
                or attrs.get("value")
                or ""
            )[:120]
            elements.append(_Element(ref=ref, tag=tag.name or "", role=role, name=name, href=href, attrs=attrs))
        self._page.elements = elements
        images: list[dict[str, str]] = []
        for img in soup.find_all("img"):
            if not isinstance(img, Tag):
                continue
            src_val = img.get("src")
            if isinstance(src_val, list):
                src_val = src_val[0] if src_val else ""
            src = str(src_val or "")
            if src:
                src = urljoin(self._page.url or "", src)
            images.append({"src": src, "alt": str(img.get("alt", ""))[:120]})
        self._page.images = images


def _ensure_url(url: str) -> str:
    """Validate that a URL uses http or https scheme.

    Args:
        url: The URL string to validate.

    Returns:
        The validated URL.

    Raises:
        ValueError: If the URL doesn't use http/https scheme.
    """
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"browser tools only accept http(s) URLs; got {url!r}")
    return url


class browser_navigate(AgentBaseFn):
    """Navigate to a URL and load the page content.

    Navigates to the specified URL, waits for the DOM to load, and parses
    the page to extract interactive elements and text content.

    Example:
        >>> browser_navigate.static_call("https://news.ycombinator.com")
    """

    @staticmethod
    def static_call(url: str, **context_variables: tp.Any) -> dict[str, tp.Any]:
        """Navigate to a URL and parse the resulting page.

        Args:
            url: The URL to navigate to. Must use http or https scheme.
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Dictionary containing url, title, and element count.
        """
        return BrowserSession.instance().navigate(_ensure_url(url))


class browser_back(AgentBaseFn):
    """Navigate back to the previous URL in browser history.

    Example:
        >>> browser_back.static_call()
    """

    @staticmethod
    def static_call(**context_variables: tp.Any) -> dict[str, tp.Any]:
        """Navigate to the previous URL in history.

        Args:
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Dictionary with 'ok' status, or 'reason' if no history available.
        """
        return BrowserSession.instance().back()


class browser_snapshot(AgentBaseFn):
    """Capture the current page state without making a network request.

    Returns the URL, title, text content, and list of interactive elements.

    Example:
        >>> state = browser_snapshot.static_call()
        >>> print(state["title"])
    """

    @staticmethod
    def static_call(**context_variables: tp.Any) -> dict[str, tp.Any]:
        """Get current page state.

        Args:
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Dictionary with url, title, text (truncated), elements, and scroll_y.
        """
        return BrowserSession.instance().snapshot()


class browser_vision(AgentBaseFn):
    """Take a screenshot of the current page.

    Returns a base64-encoded PNG image of the viewport. Requires Playwright
    for actual screenshot capture; returns text summary otherwise.

    Example:
        >>> result = browser_vision.static_call()
        >>> image_data = result["image_b64"]
    """

    @staticmethod
    def static_call(**context_variables: tp.Any) -> dict[str, tp.Any]:
        """Capture a screenshot of the current page.

        Args:
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Dictionary with url, image_b64 (base64 PNG), format, and summary.
        """
        return BrowserSession.instance().vision()


class browser_get_images(AgentBaseFn):
    """Retrieve metadata for all images on the current page.

    Returns a list of image objects with src and alt attributes.

    Example:
        >>> images = browser_get_images.static_call()
        >>> for img in images["images"]:
        ...     print(img["src"])
    """

    @staticmethod
    def static_call(**context_variables: tp.Any) -> dict[str, tp.Any]:
        """Get image metadata from current page.

        Args:
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Dictionary with url and list of image objects (src, alt).
        """
        return BrowserSession.instance().get_images()


class browser_console(AgentBaseFn):
    """Retrieve JavaScript console messages from page execution.

    Useful for debugging page scripts and understanding client-side behavior.

    Example:
        >>> logs = browser_console.static_call()
        >>> for msg in logs["console"]:
        ...     print(msg)
    """

    @staticmethod
    def static_call(**context_variables: tp.Any) -> dict[str, tp.Any]:
        """Get console messages from page execution.

        Args:
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Dictionary with url and list of console message strings.
        """
        return BrowserSession.instance().console_log()


class browser_click(AgentBaseFn):
    """Click an element on the current page by its reference ID.

    For links, triggers navigation. For buttons/inputs, triggers click handlers.

    Example:
        >>> browser_click.static_call("e1")
    """

    @staticmethod
    def static_call(ref: str, **context_variables: tp.Any) -> dict[str, tp.Any]:
        """Click an element by its reference ID.

        Args:
            ref: Element reference from snapshot (e.g., 'e1', 'e2').
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Dictionary with 'ok' status, 'ref', or 'reason' for errors.
        """
        return BrowserSession.instance().click(ref)


class browser_type(AgentBaseFn):
    """Type text into an input field.

    Supports textareas and input elements. Can optionally submit by pressing Enter.

    Example:
        >>> browser_type.static_call("e5", "search query")
    """

    @staticmethod
    def static_call(
        ref: str,
        text: str,
        submit: bool = False,
        **context_variables: tp.Any,
    ) -> dict[str, tp.Any]:
        """Type text into an element.

        Args:
            ref: Element reference from snapshot.
            text: The text to type.
            submit: Whether to press Enter after typing.
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Dictionary with 'ok' status, 'ref', and 'submitted' flag.
        """
        return BrowserSession.instance().type_text(ref, text, submit=submit)


class browser_press(AgentBaseFn):
    """Press a keyboard key on the page.

    Example:
        >>> browser_press.static_call("Escape")
    """

    @staticmethod
    def static_call(key: str, **context_variables: tp.Any) -> dict[str, tp.Any]:
        """Press a keyboard key.

        Args:
            key: Key name (e.g., 'Enter', 'Escape', 'ArrowDown', 'Tab').
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Dictionary with 'ok' status and 'key'.
        """
        return BrowserSession.instance().press(key)


class browser_scroll(AgentBaseFn):
    """Scroll the page vertically.

    Use to reveal content below the viewport or load lazy-loaded content.

    Example:
        >>> browser_scroll.static_call(dy=500)  # Scroll down 500px
    """

    @staticmethod
    def static_call(dy: int = 400, **context_variables: tp.Any) -> dict[str, tp.Any]:
        """Scroll the page vertically.

        Args:
            dy: Pixels to scroll. Positive = down, negative = up. Defaults to 400.
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Dictionary with 'ok' status and new 'scroll_y' position.
        """
        return BrowserSession.instance().scroll(int(dy))


__all__ = [
    "BrowserSession",
    "browser_back",
    "browser_click",
    "browser_console",
    "browser_get_images",
    "browser_navigate",
    "browser_press",
    "browser_scroll",
    "browser_snapshot",
    "browser_type",
    "browser_vision",
]
