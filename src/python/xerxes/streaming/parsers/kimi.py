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
"""Kimi-K2 parser."""

from __future__ import annotations

from . import ParsedToolCall, ToolCallParser
from .common import parse_tool_call_blocks


class KimiK2Parser(ToolCallParser):
    """Parser for the Kimi-K2 ``<|tool_call|>...<|/tool_call|>`` format."""

    name = "kimi_k2"

    def parse(self, text: str) -> list[ParsedToolCall]:
        """Extract Kimi-K2 tool calls delimited by ``<|tool_call|>...<|/tool_call|>``."""
        return parse_tool_call_blocks(text, open_tag="<|tool_call|>", close_tag="<|/tool_call|>")
