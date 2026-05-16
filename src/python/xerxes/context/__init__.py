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
"""Conversation-context utilities: pruning, compaction, and overflow storage.

Composition of this package, in order of how aggressively they reduce
context:

* :mod:`tool_result_pruner` — cheap pre-pruning of oversize tool outputs.
* :mod:`tool_result_storage` — overflow huge tool outputs to disk and
  replace them with reference placeholders.
* :mod:`compaction_strategies` — pluggable rolling strategies
  (truncate, sliding window, priority, LLM summarization).
* :mod:`compressor` — the orchestrator that decides when to compact
  and which summarizer to use.
* :mod:`advanced_compressor` — the multi-stage prune-then-summarise stack.
* :mod:`token_counter` — model-aware token estimation.
"""

from .advanced_compressor import AdvancedCompressionStrategy
from .compaction_strategies import (
    BaseCompactionStrategy,
    PriorityBasedStrategy,
    SlidingWindowStrategy,
    SummarizationStrategy,
    TruncateStrategy,
    get_compaction_strategy,
)
from .compressor import (
    COMPACTION_REFERENCE_PREFIX,
    CompressionResult,
    ContextCompressor,
    Summarizer,
    naive_summarizer,
)
from .token_counter import SmartTokenCounter
from .tool_result_pruner import (
    DEFAULT_HEAD_LINES,
    DEFAULT_MAX_CHARS,
    DEFAULT_TAIL_LINES,
    prune_messages,
    prune_tool_result,
)
from .tool_result_storage import DEFAULT_INLINE_LIMIT_CHARS, ToolResultStorage

__all__ = [
    "COMPACTION_REFERENCE_PREFIX",
    "DEFAULT_HEAD_LINES",
    "DEFAULT_INLINE_LIMIT_CHARS",
    "DEFAULT_MAX_CHARS",
    "DEFAULT_TAIL_LINES",
    "AdvancedCompressionStrategy",
    "BaseCompactionStrategy",
    "CompressionResult",
    "ContextCompressor",
    "PriorityBasedStrategy",
    "SlidingWindowStrategy",
    "SmartTokenCounter",
    "SummarizationStrategy",
    "Summarizer",
    "ToolResultStorage",
    "TruncateStrategy",
    "get_compaction_strategy",
    "naive_summarizer",
    "prune_messages",
    "prune_tool_result",
]
