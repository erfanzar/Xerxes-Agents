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
"""Live context-window accounting for UI and daemon status."""

from __future__ import annotations

from typing import Any

from .token_counter import SmartTokenCounter


def estimate_context_tokens(messages: list[dict[str, Any]], *, model: str) -> int:
    """Return the estimated token count for messages currently in the prompt window."""
    if not messages:
        return 0
    try:
        return max(0, SmartTokenCounter(model=model).count_tokens(messages))
    except Exception:
        text = "\n".join(f"{message.get('role', '')}: {message.get('content', '')}" for message in messages)
        return max(0, len(text) // 4)
