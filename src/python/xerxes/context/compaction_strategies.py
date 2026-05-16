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
"""Pluggable conversation compaction strategies.

All concrete strategies extend :class:`BaseCompactionStrategy` and
return ``(messages, stats)``. Available strategies:

* :class:`SummarizationStrategy` — LLM summary of the older slice.
* :class:`SlidingWindowStrategy` — keep the last N messages.
* :class:`PriorityBasedStrategy` — score and keep the highest-priority.
* :class:`TruncateStrategy` — truncate or drop old turns by char count.
* :class:`SmartCompactionStrategy` — dispatch to one of the above
  based on the current/target token ratio.

Pick a strategy with :func:`get_compaction_strategy`.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from ..types.function_execution_types import CompactionStrategy
from .token_counter import SmartTokenCounter


class BaseCompactionStrategy(ABC):
    """Common scaffolding for compaction strategies.

    Subclasses must implement :meth:`compact`. The base class provides
    a :class:`SmartTokenCounter` keyed by ``model`` and a helper that
    splits messages into ``(system, recent-preserved, compactable)``.
    """

    def __init__(
        self,
        target_tokens: int,
        model: str = "gpt-4",
        preserve_system: bool = True,
        preserve_recent: int = 3,
    ):
        """Configure budget and preservation knobs."""
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
        """Return compacted messages and a statistics dict.

        ``metadata`` is opaque and may be used by individual strategies
        (e.g. the priority scorer receives it).
        """
        pass

    def _separate_messages(
        self, messages: list[dict[str, str]]
    ) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
        """Partition ``messages`` into ``(system, preserved, compactable)``."""
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
    """Replace the older slice with an LLM-generated summary.

    Uses :func:`agents.compaction_agent.create_compaction_agent` when
    an ``llm_client`` is supplied; otherwise falls back to a plain
    ``generate_completion`` call against the client (or an extractive
    excerpt when no client is configured at all).
    """

    def __init__(self, llm_client: Any | None = None, **kwargs):
        """Wire the summary LLM and forward base kwargs."""
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
        """Summarize the compactable slice and reinsert a single system message.

        When only a single preserved message remains and it's large
        enough, that one message is summarized in-place instead.
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
        """Render messages as ``Role: content`` blocks separated by blank lines."""
        lines = []
        for msg in messages:
            role = msg.get("role", "unknown").capitalize()
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n\n".join(lines)

    def _generate_summary(self, conversation: str) -> str:
        """Summarize ``conversation`` via the configured LLM or an excerpt fallback.

        Uses whatever async event-loop state is available so callers
        can invoke from sync code or nested async contexts.
        """
        if self.llm_client:
            try:
                import asyncio

                prompt = (
                    "Summarize the following conversation concisely. "
                    "Preserve key facts, decisions, and outcomes. "
                    "Remove redundant information.\n\n"
                    f"CONVERSATION:\n{conversation}\n\nSUMMARY:"
                )

                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        from ..core.utils import run_sync

                        response = run_sync(
                            self.llm_client.generate_completion(
                                prompt=prompt, temperature=0.3, max_tokens=1024, stream=False
                            )
                        )
                    else:
                        response = loop.run_until_complete(
                            self.llm_client.generate_completion(
                                prompt=prompt, temperature=0.3, max_tokens=1024, stream=False
                            )
                        )
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    response = loop.run_until_complete(
                        self.llm_client.generate_completion(
                            prompt=prompt, temperature=0.3, max_tokens=1024, stream=False
                        )
                    )

                if hasattr(self.llm_client, "extract_content"):
                    return self.llm_client.extract_content(response)
                elif hasattr(response, "choices") and response.choices:
                    return response.choices[0].message.content
                elif isinstance(response, str):
                    return response
                return str(response)
            except Exception:
                pass

        lines = conversation.split("\n")
        if len(lines) > 10:
            summary_parts = ["Earlier discussion covered:", *lines[:5], "...", "Recent points:", *lines[-5:]]
            return "\n".join(summary_parts)
        return conversation


class SlidingWindowStrategy(BaseCompactionStrategy):
    """Keep system messages plus the most recent N up to ``target_tokens``."""

    def compact(
        self,
        messages: list[dict[str, str]],
        metadata: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, str]], dict[str, Any]]:
        """Trim from the head until the result fits the token budget.

        If the preserved-recent window already overflows, each
        oversize message is char-truncated rather than dropped.
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

        messages_to_add: list[dict[str, str]] = []
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
    """Keep the highest-priority messages until the budget is exhausted."""

    def __init__(self, priority_scorer: Callable | None = None, **kwargs):
        """Bind a ``(message, index, metadata) -> float`` scorer (default included)."""
        super().__init__(**kwargs)
        self.priority_scorer = priority_scorer or self._default_scorer

    def compact(
        self,
        messages: list[dict[str, str]],
        metadata: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, str]], dict[str, Any]]:
        """Score every compactable message, keep the best within budget, re-order.

        Selected messages are returned in their original positions so
        the conversation flow is preserved.
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
        """Heuristic scorer favouring system, tool-call, long, and recent messages."""
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


class TruncateStrategy(BaseCompactionStrategy):
    """Drop old messages and char-truncate long preserved ones."""

    def compact(
        self,
        messages: list[dict[str, str]],
        metadata: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, str]], dict[str, Any]]:
        """Drop the older slice and append a short note referencing it.

        Preserved-recent messages longer than 1000 chars are truncated
        with a ``[truncated]`` marker rather than dropped.
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


class SmartCompactionStrategy(BaseCompactionStrategy):
    """Pick a sub-strategy adaptively based on the overflow ratio.

    Routing table:

    * ratio < 1: sliding window
    * ratio < 1.8: truncate (light)
    * ratio < 4.0: summarization
    * ratio >= 4.0: truncate (heavy)
    """

    def __init__(self, llm_client: Any | None = None, **kwargs):
        """Forward kwargs; remember the LLM client for sub-strategies."""
        super().__init__(**kwargs)
        self.llm_client = llm_client

    def compact(
        self,
        messages: list[dict[str, str]],
        metadata: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, str]], dict[str, Any]]:
        """Run the adaptively chosen sub-strategy and tag the stats dict."""
        current_tokens = self.token_counter.count_tokens(messages)

        if current_tokens <= self.target_tokens:
            substrategy = "sliding_window"
            strategy: BaseCompactionStrategy = SlidingWindowStrategy(
                target_tokens=self.target_tokens,
                model=self.model,
                preserve_system=self.preserve_system,
                preserve_recent=self.preserve_recent,
            )
        else:
            compression_ratio = current_tokens / max(self.target_tokens, 1)
            if compression_ratio >= 4.0:
                substrategy = "truncate"
                strategy = TruncateStrategy(
                    target_tokens=self.target_tokens,
                    model=self.model,
                    preserve_system=self.preserve_system,
                    preserve_recent=self.preserve_recent,
                )
            elif compression_ratio >= 1.8:
                substrategy = "summarization"
                strategy = SummarizationStrategy(
                    llm_client=self.llm_client,
                    target_tokens=self.target_tokens,
                    model=self.model,
                    preserve_system=self.preserve_system,
                    preserve_recent=self.preserve_recent,
                )
            else:
                substrategy = "truncate_light"
                strategy = TruncateStrategy(
                    target_tokens=self.target_tokens,
                    model=self.model,
                    preserve_system=self.preserve_system,
                    preserve_recent=self.preserve_recent,
                )

        compacted, stats = strategy.compact(messages, metadata=metadata)
        stats["strategy"] = "smart"
        stats["substrategy"] = substrategy
        stats.setdefault("original_count", len(messages))
        stats.setdefault("compacted_count", len(compacted))
        return compacted, stats


def get_compaction_strategy(
    strategy: CompactionStrategy, target_tokens: int, model: str = "gpt-4", llm_client: Any | None = None, **kwargs
) -> BaseCompactionStrategy:
    """Return a configured strategy for a :class:`CompactionStrategy` enum value.

    ``CompactionStrategy.ADVANCED`` is handled specially and constructs
    an :class:`AdvancedCompressionStrategy` from ``advanced_compressor``.
    The summarization-capable strategies also receive ``llm_client``.
    """
    strategy_map = {
        CompactionStrategy.SUMMARIZE: SummarizationStrategy,
        CompactionStrategy.SLIDING_WINDOW: SlidingWindowStrategy,
        CompactionStrategy.PRIORITY_BASED: PriorityBasedStrategy,
        CompactionStrategy.SMART: SmartCompactionStrategy,
        CompactionStrategy.TRUNCATE: TruncateStrategy,
    }

    if strategy == CompactionStrategy.ADVANCED:
        from .advanced_compressor import AdvancedCompressionStrategy

        return AdvancedCompressionStrategy(llm_client=llm_client, target_tokens=target_tokens, model=model, **kwargs)

    strategy_class = strategy_map.get(strategy, SummarizationStrategy)

    if strategy in {CompactionStrategy.SUMMARIZE, CompactionStrategy.SMART}:
        return strategy_class(llm_client=llm_client, target_tokens=target_tokens, model=model, **kwargs)
    else:
        return strategy_class(target_tokens=target_tokens, model=model, **kwargs)
