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
"""Abstract backend interface for computer use.

Any implementation (cua-driver over MCP, pyautogui, noop, future Linux/Windows)
must return the shape described below. All methods synchronous; async is
handled inside the backend implementation if needed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class UIElement:
    """One interactable element on the current screen."""

    index: int  # 1-based SOM index
    role: str  # AX role (AXButton, AXTextField, ...)
    label: str = ""  # AXTitle / AXDescription / AXValue snippet
    bounds: tuple[int, int, int, int] = (0, 0, 0, 0)  # x, y, w, h (logical px)
    app: str = ""  # owning bundle ID or app name
    pid: int = 0  # owning process PID
    window_id: int = 0  # SkyLight / CG window ID
    attributes: dict[str, Any] = field(default_factory=dict)

    def center(self) -> tuple[int, int]:
        x, y, w, h = self.bounds
        return x + w // 2, y + h // 2


@dataclass
class CaptureResult:
    """Result of a screen capture call.

    At least one of png_b64 / elements is populated depending on capture mode:
      * mode="vision" → png_b64 only
      * mode="ax"     → elements only
      * mode="som"    → both (default): PNG already has numbered overlays
                         drawn by the backend, and `elements` holds the
                         matching index → element mapping.
    """

    mode: str
    width: int  # screenshot width (logical px, pre-Anthropic-scale)
    height: int
    png_b64: str | None = None
    elements: list[UIElement] = field(default_factory=list)
    # Optional: the target app/window the elements were captured for.
    app: str = ""
    window_title: str = ""
    # Raw bytes we sent to Anthropic, for token estimation.
    png_bytes_len: int = 0


@dataclass
class ActionResult:
    """Result of any action (click / type / scroll / drag / key / wait)."""

    ok: bool
    action: str
    message: str = ""  # human-readable summary
    # Optional trailing screenshot — set when the caller asked for a
    # post-action capture or the backend always returns one.
    capture: CaptureResult | None = None
    # Arbitrary extra fields for debugging / telemetry.
    meta: dict[str, Any] = field(default_factory=dict)


class ComputerUseBackend(ABC):
    """Lifecycle: `start()` before first use, `stop()` at shutdown."""

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the backend can be used on this host right now.

        Used by check_fn gating and by the post-setup wizard.
        """

    @abstractmethod
    def capture(self, mode: str = "som", app: str | None = None) -> CaptureResult: ...

    @abstractmethod
    def click(
        self,
        *,
        element: int | None = None,
        x: int | None = None,
        y: int | None = None,
        button: str = "left",  # left | right | middle
        click_count: int = 1,
        capture_after: bool = False,
    ) -> ActionResult: ...

    @abstractmethod
    def double_click(
        self,
        *,
        element: int | None = None,
        x: int | None = None,
        y: int | None = None,
        capture_after: bool = False,
    ) -> ActionResult: ...

    @abstractmethod
    def right_click(
        self,
        *,
        element: int | None = None,
        x: int | None = None,
        y: int | None = None,
        capture_after: bool = False,
    ) -> ActionResult: ...

    @abstractmethod
    def middle_click(
        self,
        *,
        element: int | None = None,
        x: int | None = None,
        y: int | None = None,
        capture_after: bool = False,
    ) -> ActionResult: ...

    @abstractmethod
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
    ) -> ActionResult: ...

    @abstractmethod
    def scroll(
        self,
        *,
        element: int | None = None,
        x: int | None = None,
        y: int | None = None,
        dx: int = 0,
        dy: int = 0,
        capture_after: bool = False,
    ) -> ActionResult: ...

    @abstractmethod
    def type(self, text: str, capture_after: bool = False) -> ActionResult: ...

    @abstractmethod
    def key(self, key: str, capture_after: bool = False) -> ActionResult: ...

    @abstractmethod
    def set_value(self, value: str, element: int | None = None, capture_after: bool = False) -> ActionResult: ...

    @abstractmethod
    def wait(self, ms: int = 1000) -> ActionResult: ...

    @abstractmethod
    def list_apps(self) -> ActionResult: ...

    @abstractmethod
    def focus_app(self, app: str) -> ActionResult: ...
