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
"""DeepSeek v3 / v3.1 tool-call parsers."""

from __future__ import annotations

from . import ParsedToolCall, ToolCallParser
from .common import parse_tool_call_blocks

# DeepSeek-V3 tokenizer markers use two specific non-ASCII characters:
#   U+FF5C  FULLWIDTH VERTICAL LINE
#   U+2581  LOWER ONE EIGHTH BLOCK
# These are the bytes the model literally emits; the parser must match
# byte-for-byte. We build the markers via chr() so the source file stays
# pure ASCII and ruff's RUF001/RUF003 lints don't flag the literals.
_FW_BAR = chr(0xFF5C)
_LOW_BAR = chr(0x2581)
_DEEPSEEK_V3_OPEN = f"<{_FW_BAR}tool{_LOW_BAR}call{_LOW_BAR}begin{_FW_BAR}>"
_DEEPSEEK_V3_CLOSE = f"<{_FW_BAR}tool{_LOW_BAR}call{_LOW_BAR}end{_FW_BAR}>"


class DeepSeekV3Parser(ToolCallParser):
    """Parser for DeepSeek-V3's full-width-bar / lower-block tokenizer tags.

    The model emits the literal sequence ``<U+FF5C tool U+2581 call U+2581
    begin U+FF5C>...<U+FF5C tool U+2581 call U+2581 end U+FF5C>`` which we
    match byte-for-byte.
    """

    name = "deepseek_v3"

    def parse(self, text: str) -> list[ParsedToolCall]:
        """Extract DeepSeek-V3 tool calls between the full-width-bar tags."""
        return parse_tool_call_blocks(text, open_tag=_DEEPSEEK_V3_OPEN, close_tag=_DEEPSEEK_V3_CLOSE)


class DeepSeekV31Parser(ToolCallParser):
    """Parser for the simplified DeepSeek-V3.1 ``<tool>...</tool>`` format."""

    name = "deepseek_v3_1"

    def parse(self, text: str) -> list[ParsedToolCall]:
        """Extract DeepSeek-V3.1 tool calls delimited by ``<tool>...</tool>``."""
        return parse_tool_call_blocks(text, open_tag="<tool>", close_tag="</tool>")
