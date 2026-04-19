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

"""Browser-control agent tools.

Ten ``browser_*`` tools backed by a tiny in-process
:class:`BrowserSession` driver. Navigation is performed by Playwright
when it is installed and a browser binary is available; otherwise the
session falls back to ``httpx`` plus a lightweight HTML DOM emulation.

The fallback never executes JavaScript — it only fetches the rendered
HTML — so JS-heavy SPAs require Playwright. The public tool surface
is identical either way so agents and tests do not have to know
which backend is active.

Tools:

- :class:`browser_navigate`
- :class:`browser_back`
- :class:`browser_click`
- :class:`browser_type`
- :class:`browser_press`
- :class:`browser_scroll`
- :class:`browser_snapshot`
- :class:`browser_vision`
- :class:`browser_get_images`
- :class:`browser_console`
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
    """One interactive node in the rendered page accessibility tree.

    Attributes:
        ref: Stable ``ref`` ID surfaced to the agent (e.g. ``"e7"``).
        tag: HTML tag (``"a"``, ``"button"``, ``"input"``, etc.).
        role: ARIA role inferred from tag/attrs.
        name: Best-effort accessible name (text / aria-label / value).
        href: For links, the resolved URL.
        attrs: Raw attribute dict for advanced consumers.
    """

    ref: str
    tag: str
    role: str
    name: str
    href: str = ""
    attrs: dict[str, str] = field(default_factory=dict)


@dataclass
class _Page:
    """Snapshot of the current page state.

    Attributes:
        url: Resolved current URL.
        title: Document title.
        text: Plain-text content (cleaned + collapsed).
        elements: Interactive accessibility tree (list).
        images: ``[{src, alt}]`` list for ``browser_get_images``.
        console: Captured ``console.*`` lines (Playwright only).
        history: Visited URL stack.
        scroll_y: Current vertical scroll offset (best-effort).
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
    """Process-wide singleton holding the active page state.

    The tools below operate on this shared session so that a sequence
    of agent calls (``navigate → snapshot → click → type → snapshot``)
    composes naturally without the agent having to thread a session
    handle through every call.

    Use :meth:`reset` between tests for isolation.
    """

    _instance: BrowserSession | None = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        """Initialise an empty session with a fresh page and no backend."""
        self._lock = threading.RLock()
        self._page = _Page()
        self._http_client: tp.Any | None = None
        self._playwright_page: tp.Any | None = None
        self._playwright: tp.Any | None = None
        self._playwright_browser: tp.Any | None = None

    @classmethod
    def instance(cls) -> BrowserSession:
        """Return the process-wide singleton, creating it on first use."""
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Tear down the global session (mainly for tests)."""
        with cls._instance_lock:
            if cls._instance is not None:
                cls._instance.close()
                cls._instance = None

    @classmethod
    def install_for_test(cls, http_client: tp.Any) -> BrowserSession:
        """Install a fake HTTP client and return a fresh session.

        The client must support ``client.get(url, follow_redirects=True)``
        returning an object with ``status_code``, ``url``, and ``text``.
        """
        with cls._instance_lock:
            if cls._instance is not None:
                cls._instance.close()
            inst = cls()
            inst._http_client = http_client
            cls._instance = inst
            return inst

    def close(self) -> None:
        """Shut down Playwright (if running) and reset page/client state."""
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
        """Lazily start Playwright; returns ``True`` when the real browser is usable.

        Returns ``False`` when an injected HTTP client is present (tests),
        when the ``playwright`` package is missing, or when launching the
        chromium binary fails — in which case the session uses the httpx
        fallback instead.
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
        """Fetch *url* via injected client or httpx; returns (final_url, html, status)."""
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
        """Load *url* and return ``{url, title, elements}`` for the new page."""
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
        """Navigate to the previous URL in history (no-op with a single entry)."""
        with self._lock:
            if len(self._page.history) < 2:
                return {"ok": False, "reason": "no history"}
            self._page.history.pop()
            target = self._page.history[-1]
            return self.navigate(target) | {"ok": True}

    def snapshot(self) -> dict[str, tp.Any]:
        """Return a structured page view (text + up to 200 accessible elements)."""
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
        """Return a base64 PNG screenshot (Playwright) or a text summary fallback."""
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
        """Return up to 200 ``{src, alt}`` records for images on the current page."""
        with self._lock:
            return {"url": self._page.url, "images": list(self._page.images[:200])}

    def console_log(self) -> dict[str, tp.Any]:
        """Return the most recent 200 captured console lines (Playwright only)."""
        with self._lock:
            return {"url": self._page.url, "console": list(self._page.console[-200:])}

    def click(self, ref: str) -> dict[str, tp.Any]:
        """Click the element identified by *ref*; follows hrefs in the httpx fallback."""
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
        """Type *text* into the element at *ref*, optionally pressing Enter to submit."""
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
        """Send a raw keypress (e.g. ``"Enter"``); no-op in the httpx fallback."""
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
        """Scroll the page vertically by *dy* pixels and return the new offset."""
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
        """Refresh cached url/title/elements from the live Playwright page."""
        try:
            assert self._playwright_page is not None
            self._page.url = self._playwright_page.url
            self._page.title = self._playwright_page.title()
            content = self._playwright_page.content()
            self._parse_html(content)
        except Exception:
            pass

    def _find(self, ref: str) -> _Element | None:
        """Return the accessibility tree entry with the given ref id, or ``None``."""
        for e in self._page.elements:
            if e.ref == ref:
                return e
        return None

    def _element_selector(self, elem: _Element) -> str:
        """Build a best-effort CSS selector for *elem* (id > name > href > tag)."""
        if elem.attrs.get("id"):
            return f"#{elem.attrs['id']}"
        if elem.attrs.get("name"):
            return f'{elem.tag}[name="{elem.attrs["name"]}"]'
        if elem.tag == "a" and elem.href:
            return f'a[href="{elem.href}"]'
        return elem.tag

    def _parse_html(self, html: str) -> None:
        """Populate page text, elements, and images from the raw HTML string.

        Uses BeautifulSoup when available for accessibility tree extraction;
        otherwise strips tags with a regex to produce plain-text content only.
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
    """Validate + normalise an http(s) URL; raise on bad scheme."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"browser tools only accept http(s) URLs; got {url!r}")
    return url


class browser_navigate(AgentBaseFn):
    """Open a URL in the headless browser session."""

    @staticmethod
    def static_call(url: str, **context_variables: tp.Any) -> dict[str, tp.Any]:
        """Load ``url`` and return a short summary of the new page.

        Use this when the agent needs to fetch live web content,
        navigate a multi-page flow, or set up the session for a
        subsequent ``browser_*`` action. Resets per-page state
        (elements, images, console). Only ``http`` and ``https``
        schemes are accepted.

        Args:
            url: Absolute URL to load (e.g. ``https://example.com``).

        Returns:
            ``{"url": <final>, "title": <str>, "elements": <int>}``.
        """
        return BrowserSession.instance().navigate(_ensure_url(url))


class browser_back(AgentBaseFn):
    """Navigate one step back in the session's history."""

    @staticmethod
    def static_call(**context_variables: tp.Any) -> dict[str, tp.Any]:
        """Pop the last URL off the history stack and reload the previous page.

        Returns ``{"ok": False, "reason": "no history"}`` when the
        agent is already at the first page of the session.
        """
        return BrowserSession.instance().back()


class browser_snapshot(AgentBaseFn):
    """Read the rendered page as text + accessibility tree."""

    @staticmethod
    def static_call(**context_variables: tp.Any) -> dict[str, tp.Any]:
        """Return the current page's text body and a list of interactive elements.

        Each element carries a stable ``ref`` ID (e.g. ``"e7"``) that
        the agent passes back to ``browser_click`` / ``browser_type``
        instead of CSS selectors.

        Returns:
            ``{"url", "title", "text", "elements": [...], "scroll_y"}``.
        """
        return BrowserSession.instance().snapshot()


class browser_vision(AgentBaseFn):
    """Take a screenshot and return base64 PNG."""

    @staticmethod
    def static_call(**context_variables: tp.Any) -> dict[str, tp.Any]:
        """Capture a screenshot of the current viewport.

        Use this when the agent must reason about layout, charts,
        canvas content, or any visual element that the accessibility
        tree cannot represent. When Playwright is not installed the
        tool returns a text summary in lieu of an image.

        Returns:
            ``{"url", "image_b64", "format", "summary"}``.
        """
        return BrowserSession.instance().vision()


class browser_get_images(AgentBaseFn):
    """List images on the current page."""

    @staticmethod
    def static_call(**context_variables: tp.Any) -> dict[str, tp.Any]:
        """Return ``[{src, alt}]`` for every ``<img>`` on the current page.

        URLs are resolved against the page URL so they are always
        absolute.
        """
        return BrowserSession.instance().get_images()


class browser_console(AgentBaseFn):
    """Read recent JavaScript console output."""

    @staticmethod
    def static_call(**context_variables: tp.Any) -> dict[str, tp.Any]:
        """Return the last 200 ``console.*`` lines captured this session.

        Only populated when Playwright is the active backend; the
        ``httpx`` fallback never executes JS.
        """
        return BrowserSession.instance().console_log()


class browser_click(AgentBaseFn):
    """Click an element identified by ``ref``."""

    @staticmethod
    def static_call(ref: str, **context_variables: tp.Any) -> dict[str, tp.Any]:
        """Activate the element whose ``ref`` was returned by ``browser_snapshot``.

        Args:
            ref: A ref ID like ``"e7"`` from the most recent snapshot.

        Returns:
            ``{"ok": True, "ref": ref}`` on success, ``{"ok": False, "reason": ...}`` otherwise.
            Clicking an ``<a>`` link in the fallback backend triggers a
            navigation; clicking a ``<button>`` requires Playwright.
        """
        return BrowserSession.instance().click(ref)


class browser_type(AgentBaseFn):
    """Type text into an input field identified by ``ref``."""

    @staticmethod
    def static_call(
        ref: str,
        text: str,
        submit: bool = False,
        **context_variables: tp.Any,
    ) -> dict[str, tp.Any]:
        """Fill the input ``ref`` with ``text`` and (optionally) press Enter.

        Args:
            ref: A ref ID for an ``<input>`` / ``<textarea>``.
            text: Plain text to type into the field.
            submit: When ``True``, press Enter after typing — useful
                for triggering search forms.
        """
        return BrowserSession.instance().type_text(ref, text, submit=submit)


class browser_press(AgentBaseFn):
    """Press a keyboard key (Playwright backend only)."""

    @staticmethod
    def static_call(key: str, **context_variables: tp.Any) -> dict[str, tp.Any]:
        """Send a single key press to the active element.

        Args:
            key: Key name in Playwright syntax (``"Enter"``,
                ``"Escape"``, ``"ArrowDown"``, ``"Control+a"``).
        """
        return BrowserSession.instance().press(key)


class browser_scroll(AgentBaseFn):
    """Scroll the page vertically."""

    @staticmethod
    def static_call(dy: int = 400, **context_variables: tp.Any) -> dict[str, tp.Any]:
        """Scroll by ``dy`` pixels (negative = up, positive = down).

        Returns the new ``scroll_y`` so the agent can plan further
        scrolling. The session keeps its own counter even when the
        ``httpx`` fallback is in use, which lets simple single-page
        scenarios still reason about scroll depth.
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
