# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# distributed under the License is distributed on an "AS IS" BASIS,
# See the License for the specific language governing permissions and
# limitations under the License.


"""Auto-compaction config holder for CortexAgent."""

from __future__ import annotations

from typing import Any

from ..context.token_counter import SmartTokenCounter


class AutoCompactAgent:
    """Holds compaction configuration and exposes it to CortexAgent.

    This is not an agent — it's a config object that CortexAgent reads
    during its execution loop to decide when and how to compact context.
    """

    def __init__(
        self,
        llm_client: Any = None,
        model: str = "",
        auto_compact: bool = True,
        compact_threshold: float = 0.8,
        compact_target: float = 0.5,
        max_context_tokens: int = 8000,
        compaction_strategy: str = "summarize",
        preserve_system_prompt: bool = True,
        preserve_recent_messages: int = 5,
        **_kwargs: Any,
    ) -> None:
        self.llm_client = llm_client
        self.model = model
        self.auto_compact = auto_compact
        self.compact_threshold = compact_threshold
        self.compact_target = compact_target
        self.max_context_tokens = max_context_tokens
        self.compaction_strategy = compaction_strategy
        self.preserve_system_prompt = preserve_system_prompt
        self.preserve_recent_messages = preserve_recent_messages
        self.token_counter = SmartTokenCounter(model=model)
        self.threshold_tokens = int(max_context_tokens * compact_threshold)
        self.target_tokens = int(max_context_tokens * compact_target)
        self._compaction_count = 0
        self._tokens_saved = 0

    def get_statistics(self) -> dict[str, Any]:
        return {
            "compaction_count": self._compaction_count,
            "tokens_saved": self._tokens_saved,
            "max_context_tokens": self.max_context_tokens,
            "threshold_tokens": self.threshold_tokens,
            "target_tokens": self.target_tokens,
            "strategy": self.compaction_strategy,
        }

    def check_usage(self) -> dict[str, Any]:
        return {
            "max_context_tokens": self.max_context_tokens,
            "threshold_tokens": self.threshold_tokens,
            "compact_threshold": self.compact_threshold,
            "compact_target": self.compact_target,
        }

    def record_compaction(self, tokens_before: int, tokens_after: int) -> None:
        self._compaction_count += 1
        self._tokens_saved += tokens_before - tokens_after

    def compact(self, messages: list[dict[str, str]]) -> tuple[list[dict[str, str]], dict[str, Any]]:
        """Compact messages using CompactionAgent.

        Delegates to :class:`CompactionAgent` for LLM-based summarization,
        unifying the two compaction implementations.

        Args:
            messages: List of message dictionaries with 'role' and 'content'.

        Returns:
            Tuple of (compacted_messages, stats).
        """
        from ..agents.compaction_agent import CompactionAgent

        agent = CompactionAgent(llm_client=self.llm_client, target_length="concise")
        compacted = agent.summarize_messages(
            messages,
            preserve_recent=self.preserve_recent_messages,
        )
        return compacted, {}
