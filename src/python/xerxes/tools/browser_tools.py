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
"""Browser tools module for Xerxes.

Exports:
    - logger
    - BrowserSession
    - browser_navigate
    - browser_back
    - browser_snapshot
    - browser_vision
    - browser_get_images
    - browser_console
    - browser_click
    - browser_type
    - ... and 2 more."""

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
    """Element.

    Attributes:
        ref (str): ref.
        tag (str): tag.
        role (str): role.
        name (str): name.
        href (str): href.
        attrs (dict[str, str]): attrs."""

    ref: str
    tag: str
    role: str
    name: str
    href: str = ""
    attrs: dict[str, str] = field(default_factory=dict)


@dataclass
class _Page:
    """Page.

    Attributes:
        url (str): url.
        title (str): title.
        text (str): text.
        elements (list[_Element]): elements.
        images (list[dict[str, str]]): images.
        console (list[str]): console.
        history (list[str]): history.
        scroll_y (int): scroll y."""

    url: str = ""
    title: str = ""
    text: str = ""
    elements: list[_Element] = field(default_factory=list)
    images: list[dict[str, str]] = field(default_factory=list)
    console: list[str] = field(default_factory=list)
    history: list[str] = field(default_factory=list)
    scroll_y: int = 0


class BrowserSession:
    """Browser session.

    Attributes:
        _instance (BrowserSession | None): instance."""

    _instance: BrowserSession | None = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access."""

        self._lock = threading.RLock()
        self._page = _Page()
        self._http_client: tp.Any | None = None
        self._playwright_page: tp.Any | None = None
        self._playwright: tp.Any | None = None
        self._playwright_browser: tp.Any | None = None

    @classmethod
    def instance(cls) -> BrowserSession:
        """Instance.

        Args:
            cls: IN: The class. OUT: Used for class-level operations.
        Returns:
            BrowserSession: OUT: Result of the operation."""

        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset.

        Args:
            cls: IN: The class. OUT: Used for class-level operations."""

        with cls._instance_lock:
            if cls._instance is not None:
                cls._instance.close()
                cls._instance = None

    @classmethod
    def install_for_test(cls, http_client: tp.Any) -> BrowserSession:
        """Install for test.

        Args:
            cls: IN: The class. OUT: Used for class-level operations.
            http_client (tp.Any): IN: http client. OUT: Consumed during execution.
        Returns:
            BrowserSession: OUT: Result of the operation."""

        with cls._instance_lock:
            if cls._instance is not None:
                cls._instance.close()
            inst = cls()
            inst._http_client = http_client
            cls._instance = inst
            return inst

    def close(self) -> None:
        """Close.

        Args:
            self: IN: The instance. OUT: Used for attribute access."""

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
        """Internal helper to ensure playwright.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            bool: OUT: Result of the operation."""

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
        """Internal helper to http get.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            url (str): IN: url. OUT: Consumed during execution.
        Returns:
            tuple[str, str, int]: OUT: Result of the operation."""

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
        """Navigate.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            url (str): IN: url. OUT: Consumed during execution.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

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
        """Back.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

        with self._lock:
            if len(self._page.history) < 2:
                return {"ok": False, "reason": "no history"}
            self._page.history.pop()
            target = self._page.history[-1]
            return self.navigate(target) | {"ok": True}

    def snapshot(self) -> dict[str, tp.Any]:
        """Snapshot.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

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
        """Vision.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

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
        """Retrieve the images.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

        with self._lock:
            return {"url": self._page.url, "images": list(self._page.images[:200])}

    def console_log(self) -> dict[str, tp.Any]:
        """Console log.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

        with self._lock:
            return {"url": self._page.url, "console": list(self._page.console[-200:])}

    def click(self, ref: str) -> dict[str, tp.Any]:
        """Click.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            ref (str): IN: ref. OUT: Consumed during execution.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

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
        """Type text.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            ref (str): IN: ref. OUT: Consumed during execution.
            text (str): IN: text. OUT: Consumed during execution.
            submit (bool, optional): IN: submit. Defaults to False. OUT: Consumed during execution.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

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
        """Press.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            key (str): IN: key. OUT: Consumed during execution.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

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
        """Scroll.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            dy (int): IN: dy. OUT: Consumed during execution.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

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
        """Internal helper to sync from playwright.

        Args:
            self: IN: The instance. OUT: Used for attribute access."""

        try:
            assert self._playwright_page is not None
            self._page.url = self._playwright_page.url
            self._page.title = self._playwright_page.title()
            content = self._playwright_page.content()
            self._parse_html(content)
        except Exception:
            pass

    def _find(self, ref: str) -> _Element | None:
        """Internal helper to find.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            ref (str): IN: ref. OUT: Consumed during execution.
        Returns:
            _Element | None: OUT: Result of the operation."""

        for e in self._page.elements:
            if e.ref == ref:
                return e
        return None

    def _element_selector(self, elem: _Element) -> str:
        """Internal helper to element selector.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            elem (_Element): IN: elem. OUT: Consumed during execution.
        Returns:
            str: OUT: Result of the operation."""

        if elem.attrs.get("id"):
            return f"#{elem.attrs['id']}"
        if elem.attrs.get("name"):
            return f'{elem.tag}[name="{elem.attrs["name"]}"]'
        if elem.tag == "a" and elem.href:
            return f'a[href="{elem.href}"]'
        return elem.tag

    def _parse_html(self, html: str) -> None:
        """Internal helper to parse html.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            html (str): IN: html. OUT: Consumed during execution."""

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
    """Internal helper to ensure url.

    Args:
        url (str): IN: url. OUT: Consumed during execution.
    Returns:
        str: OUT: Result of the operation."""

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"browser tools only accept http(s) URLs; got {url!r}")
    return url


class browser_navigate(AgentBaseFn):
    """Browser navigate.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    def static_call(url: str, **context_variables: tp.Any) -> dict[str, tp.Any]:
        """Static call.

        Args:
            url (str): IN: url. OUT: Consumed during execution.
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

        return BrowserSession.instance().navigate(_ensure_url(url))


class browser_back(AgentBaseFn):
    """Browser back.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    def static_call(**context_variables: tp.Any) -> dict[str, tp.Any]:
        """Static call.

        Args:
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

        return BrowserSession.instance().back()


class browser_snapshot(AgentBaseFn):
    """Browser snapshot.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    def static_call(**context_variables: tp.Any) -> dict[str, tp.Any]:
        """Static call.

        Args:
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

        return BrowserSession.instance().snapshot()


class browser_vision(AgentBaseFn):
    """Browser vision.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    def static_call(**context_variables: tp.Any) -> dict[str, tp.Any]:
        """Static call.

        Args:
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

        return BrowserSession.instance().vision()


class browser_get_images(AgentBaseFn):
    """Browser get images.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    def static_call(**context_variables: tp.Any) -> dict[str, tp.Any]:
        """Static call.

        Args:
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

        return BrowserSession.instance().get_images()


class browser_console(AgentBaseFn):
    """Browser console.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    def static_call(**context_variables: tp.Any) -> dict[str, tp.Any]:
        """Static call.

        Args:
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

        return BrowserSession.instance().console_log()


class browser_click(AgentBaseFn):
    """Browser click.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    def static_call(ref: str, **context_variables: tp.Any) -> dict[str, tp.Any]:
        """Static call.

        Args:
            ref (str): IN: ref. OUT: Consumed during execution.
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

        return BrowserSession.instance().click(ref)


class browser_type(AgentBaseFn):
    """Browser type.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    def static_call(
        ref: str,
        text: str,
        submit: bool = False,
        **context_variables: tp.Any,
    ) -> dict[str, tp.Any]:
        """Static call.

        Args:
            ref (str): IN: ref. OUT: Consumed during execution.
            text (str): IN: text. OUT: Consumed during execution.
            submit (bool, optional): IN: submit. Defaults to False. OUT: Consumed during execution.
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

        return BrowserSession.instance().type_text(ref, text, submit=submit)


class browser_press(AgentBaseFn):
    """Browser press.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    def static_call(key: str, **context_variables: tp.Any) -> dict[str, tp.Any]:
        """Static call.

        Args:
            key (str): IN: key. OUT: Consumed during execution.
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

        return BrowserSession.instance().press(key)


class browser_scroll(AgentBaseFn):
    """Browser scroll.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    def static_call(dy: int = 400, **context_variables: tp.Any) -> dict[str, tp.Any]:
        """Static call.

        Args:
            dy (int, optional): IN: dy. Defaults to 400. OUT: Consumed during execution.
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

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
