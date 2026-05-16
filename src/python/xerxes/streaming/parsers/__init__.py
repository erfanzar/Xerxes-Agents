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
"""Tool-call parsers for self-hosted models.

When a vLLM / Ollama / SGLang server returns raw model text instead of
structured ``tool_calls``, the loop falls back to one of these parsers.
Each parser implements ``parse(text) -> list[ParsedToolCall]`` and tolerates
partial / streaming text by matching only completed tag blocks.

The registry maps parser ``name`` to instance; :func:`detect_format` guesses
the matching parser from a model identifier substring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParsedToolCall:
    """A single tool call extracted from raw model text.

    Attributes:
        name: Tool identifier.
        arguments: Parsed argument dict (empty when arguments were missing or
            failed to JSON-decode).
        raw_id: Model-supplied id when the format carries one (Mistral does;
            most do not).
    """

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw_id: str = ""


class ToolCallParser:
    """Base class for raw-text tool-call parsers.

    Subclasses set :attr:`name` and override :meth:`parse`. The default
    implementation raises ``NotImplementedError``.

    Attributes:
        name: Registry key used by :func:`get_parser`.
    """

    name: str = ""

    def parse(self, text: str) -> list[ParsedToolCall]:  # pragma: no cover — abstract
        """Extract every complete tool call found in ``text``."""
        raise NotImplementedError


# Eager subclass imports so the registry populates.
from .common import LlamaParser, XmlToolCallParser
from .deepseek import DeepSeekV3Parser, DeepSeekV31Parser
from .glm import GLM45Parser, GLM47Parser
from .kimi import KimiK2Parser
from .longcat import LongCatParser
from .mistral import MistralParser
from .qwen import Qwen3CoderParser, QwenParser

REGISTRY: dict[str, ToolCallParser] = {
    p.name: p
    for p in (
        XmlToolCallParser(),
        LlamaParser(),
        MistralParser(),
        QwenParser(),
        Qwen3CoderParser(),
        DeepSeekV3Parser(),
        DeepSeekV31Parser(),
        GLM45Parser(),
        GLM47Parser(),
        KimiK2Parser(),
        LongCatParser(),
    )
}


def get_parser(name: str) -> ToolCallParser | None:
    """Return the registered parser for ``name``, or ``None`` if unknown."""
    return REGISTRY.get(name)


def detect_format(model: str) -> str | None:
    """Guess the parser ``name`` matching a model identifier.

    Matches substrings case-insensitively. Returns ``None`` if no known family
    matches, in which case the loop should fall back to structured tool_calls.
    """
    m = model.lower()
    if "hermes" in m:
        return "xml_tool_call"
    if "llama" in m or "llama-3" in m:
        return "llama"
    if "mistral" in m or "mixtral" in m:
        return "mistral"
    if "qwen3-coder" in m or "qwen-coder" in m:
        return "qwen3_coder"
    if "qwen" in m:
        return "qwen"
    if "deepseek-v3.1" in m or "deepseek-v3-1" in m:
        return "deepseek_v3_1"
    if "deepseek" in m:
        return "deepseek_v3"
    if "glm-4.7" in m or "glm47" in m:
        return "glm47"
    if "glm-4.5" in m or "glm45" in m:
        return "glm45"
    if "kimi-k2" in m or "kimi" in m:
        return "kimi_k2"
    if "longcat" in m:
        return "longcat"
    return None


__all__ = [
    "REGISTRY",
    "ParsedToolCall",
    "ToolCallParser",
    "detect_format",
    "get_parser",
]
