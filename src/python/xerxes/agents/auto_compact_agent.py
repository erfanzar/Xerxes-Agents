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
"""Auto-compaction agent that monitors context usage and triggers compaction.

This module provides :class:`AutoCompactAgent`, which watches token counts
and automatically compacts conversation history when thresholds are exceeded.
"""

from __future__ import annotations

from typing import Any

from ..context.token_counter import SmartTokenCounter


class AutoCompactAgent:
    """Monitors conversation context size and triggers compaction automatically.

    Uses token-count thresholds to decide when to summarize older messages
    while preserving recent context and system prompts.
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
        """Initialize the auto-compaction agent.

        Args:
            llm_client (Any): IN: LLM client used for summarization. OUT: Stored
                for use during compaction.
            model (str): IN: Model name for token counting. OUT: Passed to the
                token counter.
            auto_compact (bool): IN: Whether compaction is enabled. OUT: Stored
                as an instance flag.
            compact_threshold (float): IN: Fraction of max tokens that triggers
                compaction (e.g., 0.8). OUT: Used to compute ``threshold_tokens``.
            compact_target (float): IN: Fraction of max tokens to aim for after
                compaction (e.g., 0.5). OUT: Used to compute ``target_tokens``.
            max_context_tokens (int): IN: Maximum context token budget. OUT: Used
                to compute threshold and target token counts.
            compaction_strategy (str): IN: Strategy name (e.g., ``"summarize"``).
                OUT: Stored for future strategy selection.
            preserve_system_prompt (bool): IN: Whether to keep system messages. OUT:
                Stored and used during compaction.
            preserve_recent_messages (int): IN: Number of recent messages to keep.
                OUT: Passed to the compaction logic.
            **_kwargs (Any): IN: Additional keyword arguments for extensibility. OUT: Ignored.
        """
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
        """Return compaction statistics.

        Returns:
            dict[str, Any]: OUT: Metrics including compaction count, tokens saved,
                thresholds, and strategy.
        """
        return {
            "compaction_count": self._compaction_count,
            "tokens_saved": self._tokens_saved,
            "max_context_tokens": self.max_context_tokens,
            "threshold_tokens": self.threshold_tokens,
            "target_tokens": self.target_tokens,
            "strategy": self.compaction_strategy,
        }

    def check_usage(self) -> dict[str, Any]:
        """Return current usage thresholds.

        Returns:
            dict[str, Any]: OUT: Threshold and target configuration values.
        """
        return {
            "max_context_tokens": self.max_context_tokens,
            "threshold_tokens": self.threshold_tokens,
            "compact_threshold": self.compact_threshold,
            "compact_target": self.compact_target,
        }

    def record_compaction(self, tokens_before: int, tokens_after: int) -> None:
        """Record the results of a compaction operation.

        Args:
            tokens_before (int): IN: Token count before compaction. OUT: Used to
                compute tokens saved.
            tokens_after (int): IN: Token count after compaction. OUT: Used to
                compute tokens saved.
        """
        self._compaction_count += 1
        self._tokens_saved += tokens_before - tokens_after

    def compact(self, messages: list[dict[str, str]]) -> tuple[list[dict[str, str]], dict[str, Any]]:
        """Compact a message list by summarizing older messages.

        Args:
            messages (list[dict[str, str]]): IN: Full conversation history. OUT:
                Passed to the compaction agent for summarization.

        Returns:
            tuple[list[dict[str, str]], dict[str, Any]]: OUT: Compacted messages
                and an empty metadata dict.
        """
        from ..agents.compaction_agent import CompactionAgent

        agent = CompactionAgent(llm_client=self.llm_client, target_length="concise")
        compacted = agent.summarize_messages(
            messages,
            preserve_recent=self.preserve_recent_messages,
        )
        return compacted, {}
