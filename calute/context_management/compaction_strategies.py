# Copyright 2025 The EasyDeL/Calute Author @erfanzar (Erfan Zare Chavoshi).
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


"""Context compaction strategies for managing conversation history."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from ..types.function_execution_types import CompactionStrategy
from .token_counter import SmartTokenCounter


class BaseCompactionStrategy(ABC):
    """Base class for context compaction strategies."""

    def __init__(
        self,
        target_tokens: int,
        model: str = "gpt-4",
        preserve_system: bool = True,
        preserve_recent: int = 3,
    ):
        """Initialize the compaction strategy.

        Args:
            target_tokens: Target number of tokens after compaction
            model: Model name for token counting
            preserve_system: Whether to preserve system messages
            preserve_recent: Number of recent messages to preserve
        """
        self.target_tokens = target_tokens
        self.model = model
        self.preserve_system = preserve_system
        self.preserve_recent = preserve_recent
        self.token_counter = SmartTokenCounter(model=model)

    @abstractmethod
    def compact(
        self,
        messages: list[dict[str, str]],
        metadata: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, str]], dict[str, Any]]:
        """Compact the message history.

        Args:
            messages: List of message dictionaries
            metadata: Optional metadata about messages

        Returns:
            Tuple of (compacted_messages, compaction_stats)
        """
        pass

    def _separate_messages(
        self, messages: list[dict[str, str]]
    ) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
        """Separate messages into system, preserved, and compactable.

        Args:
            messages: List of all messages

        Returns:
            Tuple of (system_messages, preserved_messages, compactable_messages)
        """
        system_messages = []
        preserved_messages = []
        compactable_messages = []

        for msg in messages:
            if msg.get("role") == "system" and self.preserve_system:
                system_messages.append(msg)
                break

        non_system = [m for m in messages if m.get("role") != "system"]

        if self.preserve_recent > 0 and len(non_system) > self.preserve_recent:
            preserved_messages = non_system[-self.preserve_recent :]
            compactable_messages = non_system[: -self.preserve_recent]
        else:
            preserved_messages = non_system
            compactable_messages = []

        return system_messages, preserved_messages, compactable_messages


class SummarizationStrategy(BaseCompactionStrategy):
    """Compaction strategy that uses LLM to summarize older messages."""

    def __init__(self, llm_client: Any | None = None, **kwargs):
        """Initialize summarization strategy.

        Args:
            llm_client: LLM client for generating summaries
            **kwargs: Arguments for base class
        """
        super().__init__(**kwargs)
        self.llm_client = llm_client

        self.compaction_agent = None
        if llm_client:
            from ..agents.compaction_agent import create_compaction_agent

            self.compaction_agent = create_compaction_agent(llm_client, target_length="concise")

    def compact(
        self,
        messages: list[dict[str, str]],
        metadata: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, str]], dict[str, Any]]:
        """Compact messages using summarization.

        Args:
            messages: List of message dictionaries
            metadata: Optional metadata

        Returns:
            Compacted messages and statistics
        """
        system_msgs, preserved_msgs, compactable_msgs = self._separate_messages(messages)

        stats = {
            "original_count": len(messages),
            "strategy": "summarization",
        }

        if not compactable_msgs and len(preserved_msgs) == 1:
            single_msg = preserved_msgs[0]
            content = single_msg.get("content", "")

            if self.compaction_agent and len(content) > 500:
                try:
                    summary = self.compaction_agent.summarize_context(content)
                    compacted = [*system_msgs, {"role": single_msg.get("role", "user"), "content": summary}]
                    stats["compacted_count"] = len(compacted)
                    stats["summary_created"] = True
                    stats["messages_summarized"] = 1
                    return compacted, stats
                except Exception as e:
                    print(f"Error summarizing single message: {e}")

        if not compactable_msgs:
            stats["compacted_count"] = len(messages)
            stats["summary_created"] = False
            return messages, stats

        if self.compaction_agent:
            compacted = self.compaction_agent.summarize_messages(messages=messages, preserve_recent=self.preserve_recent)
            stats["compacted_count"] = len(compacted)
            stats["summary_created"] = True
            stats["messages_summarized"] = len(compactable_msgs)
            return compacted, stats
        else:
            conversation_text = self._format_conversation(compactable_msgs)
            summary = self._generate_summary(conversation_text)

            summary_message = {"role": "system", "content": f"[Previous conversation summary]\n{summary}"}

        compacted = [*system_msgs, summary_message, *preserved_msgs]

        stats["compacted_count"] = len(compacted)
        stats["summary_created"] = True
        stats["messages_summarized"] = len(compactable_msgs)

        return compacted, stats

    def _format_conversation(self, messages: list[dict[str, str]]) -> str:
        """Format messages as conversation text."""
        lines = []
        for msg in messages:
            role = msg.get("role", "unknown").capitalize()
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n\n".join(lines)

    def _generate_summary(self, conversation: str) -> str:
        """Generate summary using LLM or fallback method."""
        if self.llm_client:
            pass

        lines = conversation.split("\n")
        if len(lines) > 10:
            summary_parts = ["Earlier discussion covered:", *lines[:5], "...", "Recent points:", *lines[-5:]]
            return "\n".join(summary_parts)
        return conversation


class SlidingWindowStrategy(BaseCompactionStrategy):
    """Compaction strategy that keeps only recent messages."""

    def compact(
        self,
        messages: list[dict[str, str]],
        metadata: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, str]], dict[str, Any]]:
        """Compact messages using sliding window.

        Args:
            messages: List of message dictionaries
            metadata: Optional metadata

        Returns:
            Compacted messages and statistics
        """
        stats = {
            "original_count": len(messages),
            "strategy": "sliding_window",
        }

        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]

        compacted = system_msgs.copy() if self.preserve_system else []

        if self.preserve_recent > 0 and len(non_system) > 0:
            recent_to_keep = min(self.preserve_recent, len(non_system))
            recent_messages = non_system[-recent_to_keep:]
            remaining_messages = non_system[:-recent_to_keep] if recent_to_keep < len(non_system) else []
        else:
            recent_messages = []
            remaining_messages = non_system

        test_compacted = system_msgs.copy() if self.preserve_system else []
        test_compacted.extend(recent_messages)
        tokens_used = self.token_counter.count_tokens(test_compacted)

        if tokens_used > self.target_tokens:
            compacted = system_msgs.copy() if self.preserve_system else []
            for msg in recent_messages:
                content = msg.get("content", "")
                if len(content) > 500:
                    truncated_msg = msg.copy()
                    truncated_msg["content"] = content[:500] + "... [truncated for context limit]"
                    compacted.append(truncated_msg)
                else:
                    compacted.append(msg)
            tokens_used = self.token_counter.count_tokens(compacted)
        else:
            compacted = test_compacted

        messages_to_add = []
        for msg in reversed(remaining_messages):
            msg_tokens = self.token_counter.count_tokens([msg])
            if tokens_used + msg_tokens <= self.target_tokens:
                messages_to_add.insert(0, msg)
                tokens_used += msg_tokens
            else:
                break

        if messages_to_add:
            insert_pos = len(system_msgs) if self.preserve_system else 0
            compacted[insert_pos:insert_pos] = messages_to_add

        stats["compacted_count"] = len(compacted)
        stats["messages_removed"] = len(messages) - len(compacted)
        stats["final_tokens"] = tokens_used

        return compacted, stats


class PriorityBasedStrategy(BaseCompactionStrategy):
    """Compaction strategy based on message priority/importance."""

    def __init__(self, priority_scorer: Callable | None = None, **kwargs):
        """Initialize priority-based strategy.

        Args:
            priority_scorer: Function to score message priority
            **kwargs: Arguments for base class
        """
        super().__init__(**kwargs)
        self.priority_scorer = priority_scorer or self._default_scorer

    def compact(
        self,
        messages: list[dict[str, str]],
        metadata: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, str]], dict[str, Any]]:
        """Compact messages based on priority.

        Args:
            messages: List of message dictionaries
            metadata: Optional metadata with priority info

        Returns:
            Compacted messages and statistics
        """
        stats = {
            "original_count": len(messages),
            "strategy": "priority_based",
        }

        system_msgs, preserved_msgs, compactable_msgs = self._separate_messages(messages)

        if not compactable_msgs:
            stats["compacted_count"] = len(messages)
            return messages, stats

        scored_messages = [(msg, self.priority_scorer(msg, i, metadata)) for i, msg in enumerate(compactable_msgs)]

        scored_messages.sort(key=lambda x: x[1], reverse=True)

        compacted = system_msgs.copy()
        tokens_used = self.token_counter.count_tokens(compacted)

        kept_messages = []
        for msg, _score in scored_messages:
            msg_tokens = self.token_counter.count_tokens([msg])
            if tokens_used + msg_tokens <= self.target_tokens:
                kept_messages.append(msg)
                tokens_used += msg_tokens

        original_order = {id(msg): i for i, msg in enumerate(compactable_msgs)}
        kept_messages.sort(key=lambda m: original_order.get(id(m), float("inf")))

        compacted.extend(kept_messages)
        compacted.extend(preserved_msgs)

        stats["compacted_count"] = len(compacted)
        stats["messages_removed"] = len(messages) - len(compacted)
        stats["final_tokens"] = tokens_used

        return compacted, stats

    def _default_scorer(self, message: dict[str, str], index: int, metadata: dict[str, Any] | None) -> float:
        """Default message priority scorer.

        Args:
            message: Message to score
            index: Message index
            metadata: Optional metadata

        Returns:
            Priority score (0-1)
        """
        score = 0.5

        if message.get("role") == "system":
            score += 0.3

        if "function_call" in message or "tool_calls" in message:
            score += 0.2

        content_length = len(message.get("content", ""))
        if content_length > 500:
            score += 0.1

        recency_bonus = 0.1 * (index / 100)
        score += min(recency_bonus, 0.1)

        return min(score, 1.0)


class SmartCompactionStrategy(BaseCompactionStrategy):
    """Hybrid strategy combining multiple approaches."""

    def __init__(self, llm_client: Any | None = None, **kwargs):
        """Initialize smart compaction strategy.

        Args:
            llm_client: LLM client for intelligent compaction
            **kwargs: Arguments for base class
        """
        super().__init__(**kwargs)
        self.llm_client = llm_client
        self.summarizer = SummarizationStrategy(llm_client, **kwargs)
        self.windower = SlidingWindowStrategy(**kwargs)
        self.prioritizer = PriorityBasedStrategy(**kwargs)
        self.truncator = TruncateStrategy(**kwargs)

    def compact(
        self,
        messages: list[dict[str, str]],
        metadata: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, str]], dict[str, Any]]:
        """Compact messages using smart hybrid approach.

        Args:
            messages: List of message dictionaries
            metadata: Optional metadata

        Returns:
            Compacted messages and statistics
        """
        stats = {"original_count": len(messages), "strategy": "smart"}

        total_tokens = self.token_counter.count_tokens(messages)

        compression_ratio = self.target_tokens / total_tokens if total_tokens > 0 else 1.0

        if compression_ratio < 0.3:
            if self.llm_client:
                compacted, substats = self.summarizer.compact(messages, metadata)
                stats["substrategy"] = "summarization"
            else:
                compacted, substats = self.truncator.compact(messages, metadata)
                stats["substrategy"] = "truncate"
        elif compression_ratio < 0.5:
            compacted, substats = self.windower.compact(messages, metadata)
            stats["substrategy"] = "sliding_window"
        elif compression_ratio < 0.8:
            compacted, substats = self.prioritizer.compact(messages, metadata)
            stats["substrategy"] = "priority"
        else:
            if self.llm_client:
                compacted, substats = self.summarizer.compact(messages, metadata)
                stats["substrategy"] = "summarization_light"
            else:
                compacted, substats = self.truncator.compact(messages, metadata)
                stats["substrategy"] = "truncate_light"

        stats.update(substats)
        return compacted, stats


class TruncateStrategy(BaseCompactionStrategy):
    """Simple truncation strategy."""

    def compact(
        self,
        messages: list[dict[str, str]],
        metadata: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, str]], dict[str, Any]]:
        """Compact messages using simple truncation.

        Args:
            messages: List of message dictionaries
            metadata: Optional metadata

        Returns:
            Compacted messages and statistics
        """
        stats = {
            "original_count": len(messages),
            "strategy": "truncate",
        }

        system_msgs, preserved_msgs, compactable_msgs = self._separate_messages(messages)

        current_tokens = self.token_counter.count_tokens(messages)
        tokens_to_save = max(0, current_tokens - self.target_tokens)

        if tokens_to_save > 0:
            compacted = []

            compacted.extend(system_msgs)

            for msg in preserved_msgs:
                content = msg.get("content", "")
                if len(content) > 1000:
                    truncated_msg = msg.copy()
                    truncated_msg["content"] = content[:1000] + "... [truncated]"
                    compacted.append(truncated_msg)
                else:
                    compacted.append(msg)

            if compactable_msgs:
                tokens_used = self.token_counter.count_tokens(compacted)
                tokens_available = self.target_tokens - tokens_used

                if tokens_available > 100:
                    summary = f"[Previous {len(compactable_msgs)} messages truncated. "
                    if compactable_msgs:
                        last_content = compactable_msgs[-1].get("content", "")[:200]
                        summary += f"Last message preview: {last_content}...]"

                    compacted.append({"role": "system", "content": summary})
        else:
            compacted = system_msgs + compactable_msgs + preserved_msgs

        stats["compacted_count"] = len(compacted)
        stats["messages_removed"] = len(messages) - len(compacted)

        return compacted, stats


def get_compaction_strategy(
    strategy: CompactionStrategy, target_tokens: int, model: str = "gpt-4", llm_client: Any | None = None, **kwargs
) -> BaseCompactionStrategy:
    """Factory function to get a compaction strategy.

    Args:
        strategy: The compaction strategy enum
        target_tokens: Target number of tokens
        model: Model name for token counting
        llm_client: Optional LLM client
        **kwargs: Additional strategy-specific arguments

    Returns:
        Compaction strategy instance
    """
    strategy_map = {
        CompactionStrategy.SUMMARIZE: SummarizationStrategy,
        CompactionStrategy.SLIDING_WINDOW: SlidingWindowStrategy,
        CompactionStrategy.PRIORITY_BASED: PriorityBasedStrategy,
        CompactionStrategy.SMART: SmartCompactionStrategy,
        CompactionStrategy.TRUNCATE: TruncateStrategy,
    }

    strategy_class = strategy_map.get(strategy, SmartCompactionStrategy)

    if strategy in [CompactionStrategy.SUMMARIZE, CompactionStrategy.SMART]:
        return strategy_class(llm_client=llm_client, target_tokens=target_tokens, model=model, **kwargs)
    else:
        return strategy_class(target_tokens=target_tokens, model=model, **kwargs)
