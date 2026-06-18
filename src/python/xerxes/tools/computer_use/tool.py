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
"""Entry point for the `computer_use` tool.

Universal (any-model) macOS desktop control via cua-driver's background
computer-use primitive. The schema here is standard OpenAI function-calling
so every tool-capable model can drive it.

Return contract
---------------
For text-only results (wait, key, list_apps, focus_app, failures, etc.):
  JSON string.

For captures / actions with `capture_after=True`:
  Returns a dict with multimodal content:

      {
        "_multimodal": True,
        "content": [
            {"type": "text", "text": "<human-readable summary + SOM index>"},
            {"type": "image_url",
             "image_url": {"url": "data:image/png;base64,<b64>"}},
        ],
        "text_summary": "<text used for fallback string content>",
      }

  The streaming loop's tool-message builder inspects `_multimodal` and emits a
  list-shaped `content` for OpenAI-compatible providers.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .backend import ActionResult, CaptureResult
from .cua_backend import CuaBackend, cua_driver_binary_available, cua_driver_install_hint
from .schema import COMPUTER_USE_SCHEMA

logger = logging.getLogger(__name__)

# Singleton backend instance
_backend: CuaBackend | None = None


def _get_backend() -> CuaBackend:
    """Return the singleton CuaBackend, starting it if needed."""
    global _backend
    if _backend is None:
        _backend = CuaBackend()
        _backend.start()
    return _backend


def _format_capture_result(capture: CaptureResult, max_elements: int = 100) -> dict[str, Any]:
    """Format a CaptureResult into a multimodal response dict."""
    elements = capture.elements[:max_elements]
    truncated = len(capture.elements) > max_elements

    # Build text summary
    lines = [
        f"Screen capture: {capture.width}x{capture.height}",
        f"Mode: {capture.mode}",
    ]
    if capture.app:
        lines.append(f"App: {capture.app}")
    if capture.window_title:
        lines.append(f"Window: {capture.window_title}")

    if elements:
        lines.append(f"\nElements ({len(elements)} shown{'+' if truncated else ''}):")
        for el in elements:
            label = f' "{el.label}"' if el.label else ""
            lines.append(f"  [{el.index}] {el.role}{label}")
        if truncated:
            lines.append(f"  ... ({len(capture.elements) - max_elements} more elements)")
    else:
        lines.append("\nNo interactable elements detected.")

    text_summary = "\n".join(lines)

    if capture.png_b64:
        return {
            "_multimodal": True,
            "content": [
                {"type": "text", "text": text_summary},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{capture.png_b64}"},
                },
            ],
            "text_summary": text_summary,
        }

    # Fallback: no image, just text
    return {"result": text_summary}


def _format_action_result(result: ActionResult, capture_after: bool = False) -> dict[str, Any] | str:
    """Format an ActionResult into a response."""
    if capture_after and result.capture:
        capture_dict = _format_capture_result(result.capture)
        if isinstance(capture_dict, dict) and capture_dict.get("_multimodal"):
            # Prepend action message to the text part
            for item in capture_dict.get("content", []):
                if item.get("type") == "text":
                    item["text"] = f"Action: {result.action}\n{result.message}\n\n{item['text']}"
                    break
            return capture_dict

    return json.dumps({
        "ok": result.ok,
        "action": result.action,
        "message": result.message,
        "meta": result.meta,
    }, ensure_ascii=False)


class computer_use:
    """Drive the macOS desktop via cua-driver.

    Requires cua-driver to be installed. Install with:
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/trycua/cua/main/libs/cua-driver/scripts/install.sh)"

    Or set XERXES_CUA_DRIVER_CMD to the binary path.
    """

    @staticmethod
    def check_fn() -> bool:
        """Return True if cua-driver is available on this host."""
        if cua_driver_binary_available():
            return True
        # Try to auto-install
        try:
            computer_use._auto_install()
            return cua_driver_binary_available()
        except Exception:
            return False

    @staticmethod
    def _auto_install() -> None:
        """Auto-install cua-driver if not present."""
        if cua_driver_binary_available():
            return
        logger.info("cua-driver not found, attempting auto-install...")
        import subprocess
        install_script = "https://raw.githubusercontent.com/trycua/cua/main/libs/cua-driver/scripts/install.sh"
        try:
            result = subprocess.run(
                ["/bin/bash", "-c", f'"$(curl -fsSL {install_script})"'],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                logger.info("cua-driver auto-installed successfully")
            else:
                logger.warning("cua-driver auto-install failed: %s", result.stderr)
        except Exception as e:
            logger.warning("cua-driver auto-install error: %s", e)

    @staticmethod
    def get_schema() -> dict[str, Any]:
        """Return the OpenAI-compatible function schema."""
        return COMPUTER_USE_SCHEMA

    @staticmethod
    def static_call(
        action: str,
        mode: str = "som",
        app: str | None = None,
        max_elements: int = 100,
        element: int | None = None,
        x: int | None = None,
        y: int | None = None,
        start_element: int | None = None,
        start_x: int | None = None,
        start_y: int | None = None,
        end_element: int | None = None,
        end_x: int | None = None,
        end_y: int | None = None,
        dx: int = 0,
        dy: int = 0,
        text: str = "",
        key: str = "",
        value: str = "",
        ms: int = 1000,
        capture_after: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any] | str:
        """Execute a computer-use action.

        Args:
            action: Which action to perform (capture, click, type, etc.)
            mode: Capture mode (som, vision, ax)
            app: Target app name or bundle ID
            max_elements: Cap on AX elements returned
            element: Element index (1-based) for click/scroll/etc
            x, y: Pixel coordinates
            start_element/start_x/start_y: Drag start target
            end_element/end_x/end_y: Drag end target
            dx, dy: Scroll amounts
            text: Text to type
            key: Key to press
            value: Value to set
            ms: Milliseconds to wait
            capture_after: Capture screenshot after action
        """
        if not computer_use.check_fn():
            return json.dumps({"error": cua_driver_install_hint()})

        backend = _get_backend()

        try:
            if action == "capture":
                capture = backend.capture(mode=mode, app=app)
                return _format_capture_result(capture, max_elements=max_elements)

            elif action == "click":
                result = backend.click(
                    element=element, x=x, y=y,
                    capture_after=capture_after,
                )
                return _format_action_result(result, capture_after)

            elif action == "double_click":
                result = backend.double_click(
                    element=element, x=x, y=y,
                    capture_after=capture_after,
                )
                return _format_action_result(result, capture_after)

            elif action == "right_click":
                result = backend.right_click(
                    element=element, x=x, y=y,
                    capture_after=capture_after,
                )
                return _format_action_result(result, capture_after)

            elif action == "middle_click":
                result = backend.middle_click(
                    element=element, x=x, y=y,
                    capture_after=capture_after,
                )
                return _format_action_result(result, capture_after)

            elif action == "drag":
                result = backend.drag(
                    start_element=start_element,
                    start_x=start_x, start_y=start_y,
                    end_element=end_element,
                    end_x=end_x, end_y=end_y,
                    capture_after=capture_after,
                )
                return _format_action_result(result, capture_after)

            elif action == "scroll":
                result = backend.scroll(
                    element=element, x=x, y=y,
                    dx=dx, dy=dy,
                    capture_after=capture_after,
                )
                return _format_action_result(result, capture_after)

            elif action == "type":
                result = backend.type(text=text, capture_after=capture_after)
                return _format_action_result(result, capture_after)

            elif action == "key":
                result = backend.key(key=key, capture_after=capture_after)
                return _format_action_result(result, capture_after)

            elif action == "set_value":
                result = backend.set_value(
                    value=value, element=element,
                    capture_after=capture_after,
                )
                return _format_action_result(result, capture_after)

            elif action == "wait":
                result = backend.wait(ms=ms)
                return _format_action_result(result)

            elif action == "list_apps":
                result = backend.list_apps()
                return _format_action_result(result)

            elif action == "focus_app":
                if not app:
                    return json.dumps({"error": "focus_app requires 'app' parameter"})
                result = backend.focus_app(app=app)
                return _format_action_result(result)

            else:
                return json.dumps({"error": f"Unknown action: {action}"})

        except Exception as e:
            logger.exception("computer_use action failed: %s", action)
            return json.dumps({"error": str(e), "action": action})

    @staticmethod
    def async_call(*args: Any, **kwargs: Any) -> dict[str, Any] | str:
        """Async wrapper — computer_use is sync under the hood."""
        return computer_use.static_call(*args, **kwargs)


def cleanup_backend() -> None:
    """Shutdown the singleton backend. Call at process exit."""
    global _backend
    if _backend is not None:
        try:
            _backend.stop()
        except Exception:
            pass
        _backend = None
