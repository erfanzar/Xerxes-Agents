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
"""Cua-driver backend (macOS only).

Speaks MCP over stdio to `cua-driver`. The Python `mcp` SDK is async, so we
run a dedicated asyncio event loop on a background thread and marshal sync
calls through it.

Install: `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/trycua/cua/main/libs/cua-driver/scripts/install.sh)"`

After install, `cua-driver` is on $PATH and supports `cua-driver mcp` (stdio
transport) which is what we invoke.

The private SkyLight SPIs cua-driver uses (SLEventPostToPid, SLPSPostEvent-
RecordTo, _AXObserverAddNotificationAndCheckRemote) are not Apple-public and
can break on OS updates. Pin the installed version via `XERXES_CUA_DRIVER_
VERSION` if you want reproducibility across an OS bump.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import re
import shutil
import sys
import threading
from typing import Any

from .backend import (
    ActionResult,
    CaptureResult,
    ComputerUseBackend,
    UIElement,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Version pinning
# ---------------------------------------------------------------------------

PINNED_CUA_DRIVER_VERSION = os.environ.get("XERXES_CUA_DRIVER_VERSION", "0.5.0")

_CUA_DRIVER_CMD = os.environ.get("XERXES_CUA_DRIVER_CMD", "cua-driver")
_CUA_DRIVER_ARGS = ["mcp"]  # stdio MCP transport

# Regex to parse list_windows text output lines:
#   "- AppName (pid 12345) "Title" [window_id: 67890]"
_WINDOW_LINE_RE = re.compile(
    r'^-\s+(.+?)\s+\(pid\s+(\d+)\)\s+.*\[window_id:\s+(\d+)\]',
    re.MULTILINE,
)

# Regex to parse element lines from get_window_state AX tree markdown.
#
# Handles two output formats from different cua-driver versions:
#   Classic:  "  - [N] AXRole \"label\""
#   New:       "[N] AXRole (order) id=Label"
#
# Group 1: element index
# Group 2: AX role
# Group 3: quoted label (classic format)
# Group 4: id= label (new format)
_ELEMENT_LINE_RE = re.compile(
    r'^\s*(?:-\s+)?\[(\d+)\]\s+(\w+)(?:\s+"([^"]*)"|(?:\s+\(\d+\))?\s+id=([^\s\[\]]*))?' ,
    re.MULTILINE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_macos() -> bool:
    return sys.platform == "darwin"


def cua_driver_binary_available() -> bool:
    """True if `cua-driver` is on $PATH or XERXES_CUA_DRIVER_CMD resolves."""
    return bool(shutil.which(_CUA_DRIVER_CMD))


def cua_driver_install_hint() -> str:
    return (
        "cua-driver is not installed. Install with:\n"
        "  /bin/bash -c \"$(curl -fsSL "
        "https://raw.githubusercontent.com/trycua/cua/main/libs/cua-driver/scripts/install.sh)\"\n"
        "Or set XERXES_CUA_DRIVER_CMD to the binary path."
    )


def _parse_windows_from_text(text: str) -> list[dict[str, Any]]:
    """Parse window records from list_windows text output."""
    windows = []
    for m in _WINDOW_LINE_RE.finditer(text):
        windows.append({
            "app": m.group(1).strip(),
            "pid": int(m.group(2)),
            "window_id": int(m.group(3)),
        })
    return windows


def _parse_elements_from_text(text: str) -> list[UIElement]:
    """Parse AX element records from get_window_state markdown."""
    elements = []
    for m in _ELEMENT_LINE_RE.finditer(text):
        idx = int(m.group(1))
        role = m.group(2)
        label = (m.group(3) or m.group(4) or "").strip()
        elements.append(UIElement(index=idx, role=role, label=label))
    return elements


def _b64_from_png_bytes(png_bytes: bytes) -> str:
    return base64.b64encode(png_bytes).decode("ascii")


# ---------------------------------------------------------------------------
# CuaBackend
# ---------------------------------------------------------------------------

class CuaBackend(ComputerUseBackend):
    """Sync wrapper around the async cua-driver MCP client.

    A private asyncio event loop runs on a background thread so that every
    public method can be called synchronously from the agent's tool executor.
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._session: Any = None  # mcp.ClientSession
        self._tools: list[dict[str, Any]] = []
        self._started = False

    # ── Lifecycle ───────────────────────────────────────────────────

    def start(self) -> None:
        if self._started:
            return
        if not _is_macos():
            raise RuntimeError("CuaBackend is macOS-only")
        if not cua_driver_binary_available():
            raise RuntimeError(cua_driver_install_hint())

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

        # Connect to cua-driver via MCP stdio
        future = asyncio.run_coroutine_threadsafe(self._connect(), self._loop)
        future.result(timeout=15)
        self._started = True

    def stop(self) -> None:
        if not self._started or self._loop is None:
            return
        future = asyncio.run_coroutine_threadsafe(self._disconnect(), self._loop)
        try:
            future.result(timeout=5)
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=2)
        self._started = False

    def is_available(self) -> bool:
        return _is_macos() and cua_driver_binary_available()

    # ── Internal async plumbing ─────────────────────────────────────

    async def _connect(self) -> None:
        """Spawn cua-driver and initialize the MCP session."""
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError as exc:
            raise RuntimeError(
                "mcp SDK not installed. Run: uv pip install 'mcp>=1.0.0'"
            ) from exc

        server_params = StdioServerParameters(
            command=_CUA_DRIVER_CMD,
            args=_CUA_DRIVER_ARGS,
            env=None,
        )

        self._stdio_ctx = stdio_client(server_params)
        read, write = await self._stdio_ctx.__aenter__()

        self._session = ClientSession(read, write)
        await self._session.__aenter__()
        await self._session.initialize()

        # Discover available tools
        tools_result = await self._session.list_tools()
        self._tools = [
            {"name": t.name, "description": t.description}
            for t in (tools_result.tools if hasattr(tools_result, "tools") else [])
        ]
        logger.debug("cua-driver tools: %s", [t["name"] for t in self._tools])

    async def _disconnect(self) -> None:
        if self._session:
            await self._session.__aexit__(None, None, None)
            self._session = None
        if hasattr(self, "_stdio_ctx"):
            await self._stdio_ctx.__aexit__(None, None, None)

    async def _call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call a cua-driver tool by name and return the parsed result."""
        if self._session is None:
            raise RuntimeError("CuaBackend not started")

        result = await self._session.call_tool(name, arguments=arguments)

        # Parse content — can be text or image
        content = result.content if hasattr(result, "content") else []
        text_parts = []
        image_parts = []

        for part in content:
            if hasattr(part, "text"):
                text_parts.append(part.text)
            elif hasattr(part, "image_url"):
                image_parts.append(part.image_url)
            elif hasattr(part, "data") and hasattr(part, "mimeType"):
                # Binary image data
                if "png" in part.mimeType:
                    image_parts.append(_b64_from_png_bytes(part.data))

        return {
            "text": "\n".join(text_parts),
            "images": image_parts,
            "is_error": getattr(result, "isError", False),
        }

    # ── Public sync API ───────────────────────────────────────────

    def _run(self, coro: asyncio.Coroutine) -> Any:
        """Marshal an async call onto the background loop."""
        if self._loop is None:
            raise RuntimeError("CuaBackend not started")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=30)

    # ── Capture ─────────────────────────────────────────────────────

    def capture(self, mode: str = "som", app: str | None = None) -> CaptureResult:
        args: dict[str, Any] = {"mode": mode}
        if app:
            args["app"] = app

        result = self._run(self._call_tool("capture", args))
        text = result.get("text", "")
        images = result.get("images", [])

        png_b64 = images[0] if images else None

        # Parse elements from text response
        elements = _parse_elements_from_text(text)

        # Try to infer dimensions from the image if available
        width, height = 0, 0
        if png_b64:
            try:
                png_bytes = base64.b64decode(png_b64)
                # PNG header: width at bytes 16-20, height at 20-24
                if len(png_bytes) >= 24:
                    width = int.from_bytes(png_bytes[16:20], "big")
                    height = int.from_bytes(png_bytes[20:24], "big")
            except Exception:
                pass

        return CaptureResult(
            mode=mode,
            width=width,
            height=height,
            png_b64=png_b64,
            elements=elements,
            png_bytes_len=len(base64.b64decode(png_b64)) if png_b64 else 0,
        )

    # ── Pointer actions ─────────────────────────────────────────────

    def click(
        self,
        *,
        element: int | None = None,
        x: int | None = None,
        y: int | None = None,
        button: str = "left",
        click_count: int = 1,
        capture_after: bool = False,
    ) -> ActionResult:
        args: dict[str, Any] = {"button": button, "click_count": click_count}
        if element is not None:
            args["element"] = element
        elif x is not None and y is not None:
            args["x"] = x
            args["y"] = y
        else:
            return ActionResult(ok=False, action="click", message="Need element or x,y")

        if capture_after:
            args["capture_after"] = True

        result = self._run(self._call_tool("click", args))
        return ActionResult(
            ok=not result.get("is_error", False),
            action="click",
            message=result.get("text", ""),
        )

    def double_click(
        self,
        *,
        element: int | None = None,
        x: int | None = None,
        y: int | None = None,
        capture_after: bool = False,
    ) -> ActionResult:
        return self.click(
            element=element,
            x=x,
            y=y,
            button="left",
            click_count=2,
            capture_after=capture_after,
        )

    def right_click(
        self,
        *,
        element: int | None = None,
        x: int | None = None,
        y: int | None = None,
        capture_after: bool = False,
    ) -> ActionResult:
        return self.click(
            element=element,
            x=x,
            y=y,
            button="right",
            click_count=1,
            capture_after=capture_after,
        )

    def middle_click(
        self,
        *,
        element: int | None = None,
        x: int | None = None,
        y: int | None = None,
        capture_after: bool = False,
    ) -> ActionResult:
        return self.click(
            element=element,
            x=x,
            y=y,
            button="middle",
            click_count=1,
            capture_after=capture_after,
        )

    def drag(
        self,
        *,
        start_element: int | None = None,
        start_x: int | None = None,
        start_y: int | None = None,
        end_element: int | None = None,
        end_x: int | None = None,
        end_y: int | None = None,
        capture_after: bool = False,
    ) -> ActionResult:
        args: dict[str, Any] = {}
        if start_element is not None:
            args["start_element"] = start_element
        elif start_x is not None and start_y is not None:
            args["start_x"] = start_x
            args["start_y"] = start_y
        else:
            return ActionResult(ok=False, action="drag", message="Need start_element or start_x,start_y")

        if end_element is not None:
            args["end_element"] = end_element
        elif end_x is not None and end_y is not None:
            args["end_x"] = end_x
            args["end_y"] = end_y
        else:
            return ActionResult(ok=False, action="drag", message="Need end_element or end_x,end_y")

        if capture_after:
            args["capture_after"] = True

        result = self._run(self._call_tool("drag", args))
        return ActionResult(
            ok=not result.get("is_error", False),
            action="drag",
            message=result.get("text", ""),
        )

    def scroll(
        self,
        *,
        element: int | None = None,
        x: int | None = None,
        y: int | None = None,
        dx: int = 0,
        dy: int = 0,
        capture_after: bool = False,
    ) -> ActionResult:
        args: dict[str, Any] = {"dx": dx, "dy": dy}
        if element is not None:
            args["element"] = element
        elif x is not None and y is not None:
            args["x"] = x
            args["y"] = y

        if capture_after:
            args["capture_after"] = True

        result = self._run(self._call_tool("scroll", args))
        return ActionResult(
            ok=not result.get("is_error", False),
            action="scroll",
            message=result.get("text", ""),
        )

    # ── Keyboard actions ────────────────────────────────────────────

    def type(self, text: str, capture_after: bool = False) -> ActionResult:
        args: dict[str, Any] = {"text": text}
        if capture_after:
            args["capture_after"] = True
        result = self._run(self._call_tool("type", args))
        return ActionResult(
            ok=not result.get("is_error", False),
            action="type",
            message=result.get("text", ""),
        )

    def key(self, key: str, capture_after: bool = False) -> ActionResult:
        args: dict[str, Any] = {"key": key}
        if capture_after:
            args["capture_after"] = True
        result = self._run(self._call_tool("key", args))
        return ActionResult(
            ok=not result.get("is_error", False),
            action="key",
            message=result.get("text", ""),
        )

    def set_value(self, value: str, element: int | None = None, capture_after: bool = False) -> ActionResult:
        args: dict[str, Any] = {"value": value}
        if element is not None:
            args["element"] = element
        if capture_after:
            args["capture_after"] = True
        result = self._run(self._call_tool("set_value", args))
        return ActionResult(
            ok=not result.get("is_error", False),
            action="set_value",
            message=result.get("text", ""),
        )

    # ── Wait / App management ─────────────────────────────────────

    def wait(self, ms: int = 1000) -> ActionResult:
        result = self._run(self._call_tool("wait", {"ms": ms}))
        return ActionResult(
            ok=not result.get("is_error", False),
            action="wait",
            message=result.get("text", ""),
        )

    def list_apps(self) -> ActionResult:
        result = self._run(self._call_tool("list_apps", {}))
        text = result.get("text", "")
        windows = _parse_windows_from_text(text)
        return ActionResult(
            ok=not result.get("is_error", False),
            action="list_apps",
            message=text,
            meta={"windows": windows},
        )

    def focus_app(self, app: str) -> ActionResult:
        result = self._run(self._call_tool("focus_app", {"app": app}))
        return ActionResult(
            ok=not result.get("is_error", False),
            action="focus_app",
            message=result.get("text", ""),
        )
