# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).
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


"""Streaming event protocol for Xerxes agent loops.

This package provides a typed event protocol for streaming agent execution,
a neutral message format with bidirectional provider converters, a permission
modes system, and a generator-based agent loop.

Modules:
    - :mod:`events`: Typed streaming events (TextChunk, ToolStart, etc.)
    - :mod:`messages`: Neutral message format with Anthropic/OpenAI converters
    - :mod:`permissions`: Permission modes (auto/accept-all/manual)
    - :mod:`loop`: Generator-based streaming agent loop
"""

from .events import (
    AgentState,
    PermissionRequest,
    StreamEvent,
    TextChunk,
    ThinkingChunk,
    ToolEnd,
    ToolStart,
    TurnDone,
)
from .loop import run as run_agent_loop
from .messages import (
    NeutralMessage,
    messages_to_anthropic,
    messages_to_openai,
)
from .permissions import PermissionMode, check_permission

__all__ = [
    "AgentState",
    "NeutralMessage",
    "PermissionMode",
    "PermissionRequest",
    "StreamEvent",
    "TextChunk",
    "ThinkingChunk",
    "ToolEnd",
    "ToolStart",
    "TurnDone",
    "check_permission",
    "messages_to_anthropic",
    "messages_to_openai",
    "run_agent_loop",
]
