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
"""Agent-backed conversation compaction strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from ..types.function_execution_types import CompactionStrategy
from .compaction_provisioner import (
    CompactionProvisioner,
    CompactionSummaryAgent,
    render_messages_for_summary,
)
from .token_counter import SmartTokenCounter


class BaseCompactionStrategy(ABC):
    """Common scaffolding for provisioner-backed strategies."""

    strategy_name = "agent_backed"

    def __init__(
        self,
        target_tokens: int,
        model: str = "gpt-4",
        preserve_system: bool = True,
        summary_agent: CompactionSummaryAgent | None = None,
        **_legacy_kwargs: Any,
    ):
        """Configure the token budget and optional summary agent."""
        self.target_tokens = max(1, int(target_tokens))
        self.model = model
        self.preserve_system = preserve_system
        self.summary_agent = summary_agent
        self.token_counter = SmartTokenCounter(model=model)

    @abstractmethod
    def compact(
        self,
        messages: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Return compacted messages and a statistics dict."""
        pass

    def _no_agent_result(self, messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Return a no-op result when no compaction agent is available."""
        return messages, {
            "original_count": len(messages),
            "compacted_count": len(messages),
            "strategy": self.strategy_name,
            "summary_created": False,
            "reason": "no_summary_agent",
        }

    def _run_provisioner(
        self,
        messages: list[dict[str, Any]],
        *,
        strategy: str | None = None,
        force: bool = True,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Delegate compaction to :class:`CompactionProvisioner`."""
        if self.summary_agent is None:
            return self._no_agent_result(messages)

        provisioner = CompactionProvisioner(
            model=self.model,
            max_context_tokens=max(self.target_tokens * 2, self.target_tokens + 1),
            threshold_tokens=1,
            target_tokens=self.target_tokens,
            summary_agent=self.summary_agent,
        )
        result = provisioner.compact(messages, force=force)
        if not result.compacted:
            return messages, {
                "original_count": len(messages),
                "compacted_count": len(messages),
                "strategy": strategy or self.strategy_name,
                "summary_created": False,
                "reason": result.reason,
                "tokens_before": result.tokens_before,
                "tokens_after": result.tokens_after,
            }

        return result.messages, {
            "original_count": len(messages),
            "compacted_count": len(result.messages),
            "strategy": strategy or self.strategy_name,
            "summary_created": True,
            "messages_summarized": result.summarized_count,
            "messages_kept": result.kept_count,
            "tokens_before": result.tokens_before,
            "tokens_after": result.tokens_after,
        }


def _summary_agent_from_llm_client(llm_client: Any | None) -> CompactionSummaryAgent | None:
    """Adapt an LLM client with ``generate_completion`` into a summary agent."""
    if llm_client is None:
        return None

    def summary_agent(messages: list[dict[str, Any]], _previous_summary: str | None) -> str:
        from ..agents.compaction_agent import CompactionAgent

        agent = CompactionAgent(llm_client=llm_client, target_length="concise")
        return agent.summarize_context(render_messages_for_summary(messages))

    return summary_agent


class SummarizationStrategy(BaseCompactionStrategy):
    """Replace compactable history with an agent-generated summary."""

    strategy_name = "summarization"

    def __init__(self, llm_client: Any | None = None, **kwargs: Any):
        """Wire the summary LLM and forward base kwargs."""
        summary_agent = kwargs.pop("summary_agent", None) or _summary_agent_from_llm_client(llm_client)
        super().__init__(summary_agent=summary_agent, **kwargs)
        self.llm_client = llm_client

    def compact(
        self,
        messages: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Summarize via the configured compaction agent."""
        return self._run_provisioner(messages)


class SlidingWindowStrategy(BaseCompactionStrategy):
    """Legacy name retained; compaction is now agent-backed only."""

    strategy_name = "sliding_window"

    def compact(
        self,
        messages: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Delegate to the provisioner when a summary agent is configured."""
        return self._run_provisioner(messages)


class PriorityBasedStrategy(BaseCompactionStrategy):
    """Legacy name retained; priority scoring no longer drops messages."""

    strategy_name = "priority_based"

    def __init__(self, priority_scorer: Callable | None = None, **kwargs: Any):
        """Bind the optional scorer for callers that still inspect it."""
        super().__init__(**kwargs)
        self.priority_scorer = priority_scorer or self._default_scorer

    def compact(
        self,
        messages: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Delegate to the provisioner when a summary agent is configured."""
        return self._run_provisioner(messages)

    def _default_scorer(self, message: dict[str, Any], index: int, metadata: dict[str, Any] | None) -> float:
        """Return a bounded relevance hint without affecting compaction."""
        score = 0.5
        if message.get("role") == "system":
            score += 0.3
        if "function_call" in message or "tool_calls" in message:
            score += 0.2
        if len(str(message.get("content", ""))) > 500:
            score += 0.1
        score += min(0.1 * (index / 100), 0.1)
        return min(score, 1.0)


class TruncateStrategy(BaseCompactionStrategy):
    """Legacy name retained; truncation has been replaced by agent summary."""

    strategy_name = "truncate"

    def compact(
        self,
        messages: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Delegate to the provisioner when a summary agent is configured."""
        return self._run_provisioner(messages)


class SmartCompactionStrategy(BaseCompactionStrategy):
    """Adaptive entry point that always selects agent summarization."""

    strategy_name = "smart"

    def __init__(self, llm_client: Any | None = None, **kwargs: Any):
        """Wire the summary LLM and forward base kwargs."""
        summary_agent = kwargs.pop("summary_agent", None) or _summary_agent_from_llm_client(llm_client)
        super().__init__(summary_agent=summary_agent, **kwargs)
        self.llm_client = llm_client

    def compact(
        self,
        messages: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Run the agent-backed compaction path and tag the selected strategy."""
        compacted, stats = self._run_provisioner(messages, strategy="smart")
        stats["substrategy"] = "summarization" if stats.get("summary_created") else "no_summary_agent"
        return compacted, stats


def get_compaction_strategy(
    strategy: CompactionStrategy,
    target_tokens: int,
    model: str = "gpt-4",
    llm_client: Any | None = None,
    **kwargs: Any,
) -> BaseCompactionStrategy:
    """Return a configured compaction strategy for the enum value."""
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
    if strategy_class in {SummarizationStrategy, SmartCompactionStrategy}:
        return strategy_class(llm_client=llm_client, target_tokens=target_tokens, model=model, **kwargs)
    return strategy_class(
        target_tokens=target_tokens,
        model=model,
        summary_agent=_summary_agent_from_llm_client(llm_client),
        **kwargs,
    )
