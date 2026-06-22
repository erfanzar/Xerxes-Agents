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
"""Schema for the generic `computer_use` tool.

Model-agnostic. Any tool-calling model can drive this. Vision-capable models
should prefer `capture(mode='som')` then `click(element=N)` — much more
reliable than pixel coordinates. Pixel coordinates remain supported for
models that were trained on them (e.g. Claude's computer-use RL).
"""

from __future__ import annotations

from typing import Any

# One consolidated tool with an `action` discriminator. Keeps the schema
# compact and the per-turn token cost low.
COMPUTER_USE_SCHEMA: dict[str, Any] = {
    "name": "computer_use",
    "description": (
        "Drive the macOS desktop in the background — screenshots, mouse, "
        "keyboard, scroll, drag — without stealing the user's cursor, "
        "keyboard focus, or Space. Preferred workflow: call with "
        "action='capture' (mode='som' gives numbered element overlays), "
        "then click by `element` index for reliability. Pixel coordinates "
        "are supported for models trained on them. Works on any window — "
        "hidden, minimized, on another Space, or behind another app. "
        "macOS only; requires cua-driver to be installed."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "capture",
                    "click",
                    "double_click",
                    "right_click",
                    "middle_click",
                    "drag",
                    "scroll",
                    "type",
                    "key",
                    "set_value",
                    "wait",
                    "list_apps",
                    "focus_app",
                ],
                "description": (
                    "Which action to perform. `capture` is free (no side "
                    "effects). All other actions require approval unless "
                    "auto-approved. Use `set_value` for select/popup elements "
                    "and sliders — it selects the matching option directly "
                    "without opening the native menu (no focus steal)."
                ),
            },
            "mode": {
                "type": "string",
                "enum": ["som", "vision", "ax"],
                "description": (
                    "Capture mode. `som` (default) is a screenshot with "
                    "numbered overlays on every interactable element plus "
                    "the AX tree — best for vision models, lets you click "
                    "by element index. `vision` is a plain screenshot. "
                    "`ax` is the accessibility tree only (no image; useful "
                    "for text-only models)."
                ),
            },
            "app": {
                "type": "string",
                "description": (
                    "Optional. Limit capture/action to a specific app "
                    "(by name, e.g. 'Safari', or bundle ID, "
                    "'com.apple.Safari'). If omitted, operates on the "
                    "frontmost app's window or the whole screen."
                ),
            },
            "max_elements": {
                "type": "integer",
                "description": (
                    "Optional cap on the AX `elements` array returned by "
                    "`action='capture'`. Default 100, hard maximum 1000. "
                    "Dense UIs (Electron apps such as Obsidian or VS Code, "
                    "JetBrains IDEs) can publish 500+ AX nodes — capping "
                    "prevents a single capture from blowing session "
                    "context. When the cap trims the response, "
                    "`total_elements` and `truncated_elements` are "
                    "surfaced in the result so you can re-call with "
                    "`app=` to narrow scope or raise `max_elements` when "
                    "the full tree is required. Has no effect on "
                    "`mode='som'` / `mode='vision'` when a screenshot is "
                    "included in the response; only the rare image-"
                    "missing fallback returns an `elements` array and is "
                    "subject to the cap."
                ),
                "default": 100,
                "minimum": 1,
                "maximum": 1000,
            },
            "element": {
                "type": "integer",
                "description": (
                    "Element index (1-based) from the most recent capture. Preferred over x/y for reliability."
                ),
            },
            "x": {
                "type": "integer",
                "description": "Pixel X coordinate (logical px).",
            },
            "y": {
                "type": "integer",
                "description": "Pixel Y coordinate (logical px).",
            },
            "start_element": {
                "type": "integer",
                "description": "Start element index for drag.",
            },
            "start_x": {
                "type": "integer",
                "description": "Start X coordinate for drag.",
            },
            "start_y": {
                "type": "integer",
                "description": "Start Y coordinate for drag.",
            },
            "end_element": {
                "type": "integer",
                "description": "End element index for drag.",
            },
            "end_x": {
                "type": "integer",
                "description": "End X coordinate for drag.",
            },
            "end_y": {
                "type": "integer",
                "description": "End Y coordinate for drag.",
            },
            "dx": {
                "type": "integer",
                "description": "Horizontal scroll amount (positive = right).",
                "default": 0,
            },
            "dy": {
                "type": "integer",
                "description": "Vertical scroll amount (positive = down).",
                "default": 0,
            },
            "text": {
                "type": "string",
                "description": "Text to type (for action='type').",
            },
            "key": {
                "type": "string",
                "description": (
                    "Key to press (for action='key'). Special keys: "
                    "'return', 'enter', 'tab', 'escape', 'space', "
                    "'backspace', 'delete', 'arrowup', 'arrowdown', "
                    "'arrowleft', 'arrowright', 'home', 'end', "
                    "'pageup', 'pagedown'. Also supports modifiers: "
                    "'command+a', 'shift+tab', etc."
                ),
            },
            "value": {
                "type": "string",
                "description": "Value to set (for action='set_value').",
            },
            "ms": {
                "type": "integer",
                "description": "Milliseconds to wait (for action='wait').",
                "default": 1000,
            },
            "capture_after": {
                "type": "boolean",
                "description": (
                    "If true, capture a screenshot after the action and "
                    "return it in the result. Useful for verifying the "
                    "action had the intended effect."
                ),
                "default": False,
            },
        },
        "required": ["action"],
    },
}
