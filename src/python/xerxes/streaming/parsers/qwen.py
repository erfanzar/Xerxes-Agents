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
"""Qwen and Qwen3-Coder parsers."""

from __future__ import annotations

from . import ParsedToolCall, ToolCallParser
from .common import parse_tool_call_blocks


class QwenParser(ToolCallParser):
    """Parser for the Qwen 2.5 chat ``<tool_call>...</tool_call>`` format."""

    name = "qwen"

    def parse(self, text: str) -> list[ParsedToolCall]:
        """Extract Qwen tool calls delimited by ``<tool_call>...</tool_call>``."""
        return parse_tool_call_blocks(text, open_tag="<tool_call>", close_tag="</tool_call>")


class Qwen3CoderParser(ToolCallParser):
    """Parser for the Qwen3-Coder ``|<function_call_start|>...|<function_call_end|>`` format."""

    name = "qwen3_coder"

    def parse(self, text: str) -> list[ParsedToolCall]:
        """Extract Qwen3-Coder tool calls delimited by ``|<function_call_start|>...|<function_call_end|>``."""
        return parse_tool_call_blocks(text, open_tag="|<function_call_start|>", close_tag="|<function_call_end|>")
