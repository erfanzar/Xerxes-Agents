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
"""Query engine module for Xerxes.

Exports:
    - logger
    - QueryEngineConfig
    - TurnResult
    - QueryEngine"""

from __future__ import annotations

import logging
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from .cost_tracker import CostTracker
from .execution_registry import ExecutionRegistry
from .history import HistoryLog
from .transcript import TranscriptStore

logger = logging.getLogger(__name__)


@dataclass
class QueryEngineConfig:
    """Query engine config.

    Attributes:
        max_turns (int): max turns.
        max_budget_tokens (int): max budget tokens.
        compact_after_turns (int): compact after turns.
        compact_keep_last (int): compact keep last.
        model (str): model.
        system_prompt (str): system prompt.
        permission_mode (str): permission mode.
        max_tokens (int): max tokens.
        thinking (bool): thinking.
        thinking_budget (int): thinking budget."""

    max_turns: int = 50
    max_budget_tokens: int = 500_000
    compact_after_turns: int = 20
    compact_keep_last: int = 10
    model: str = "gpt-4o"
    system_prompt: str = ""
    permission_mode: str = "auto"
    max_tokens: int = 8192
    thinking: bool = False
    thinking_budget: int = 10000


@dataclass
class TurnResult:
    """Turn result.

    Attributes:
        prompt (str): prompt.
        output (str): output.
        tool_calls (tuple[str, ...]): tool calls.
        in_tokens (int): in tokens.
        out_tokens (int): out tokens.
        stop_reason (str): stop reason."""

    prompt: str
    output: str
    tool_calls: tuple[str, ...] = ()
    in_tokens: int = 0
    out_tokens: int = 0
    stop_reason: str = "complete"


class QueryEngine:
    """Query engine."""

    def __init__(
        self,
        config: QueryEngineConfig,
        registry: ExecutionRegistry | None = None,
        session_id: str | None = None,
    ) -> None:
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            config (QueryEngineConfig): IN: config. OUT: Consumed during execution.
            registry (ExecutionRegistry | None, optional): IN: registry. Defaults to None. OUT: Consumed during execution.
            session_id (str | None, optional): IN: session id. Defaults to None. OUT: Consumed during execution."""
        self.config = config
        self.session_id = session_id or uuid4().hex
        self.registry = registry or ExecutionRegistry()
        self.transcript = TranscriptStore()
        self.history = HistoryLog()
        self.cost_tracker = CostTracker()
        self._turn_count = 0
        self._total_in_tokens = 0
        self._total_out_tokens = 0

    @classmethod
    def create(
        cls,
        model: str = "gpt-4o",
        system_prompt: str = "",
        registry: ExecutionRegistry | None = None,
        **config_kwargs: Any,
    ) -> QueryEngine:
        """Create.

        Args:
            cls: IN: The class. OUT: Used for class-level operations.
            model (str, optional): IN: model. Defaults to 'gpt-4o'. OUT: Consumed during execution.
            system_prompt (str, optional): IN: system prompt. Defaults to ''. OUT: Consumed during execution.
            registry (ExecutionRegistry | None, optional): IN: registry. Defaults to None. OUT: Consumed during execution.
            **config_kwargs: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            QueryEngine: OUT: Result of the operation."""

        config = QueryEngineConfig(model=model, system_prompt=system_prompt, **config_kwargs)
        return cls(config=config, registry=registry)

    def submit(
        self,
        prompt: str,
        tool_executor: Any = None,
        tool_schemas: list[dict[str, Any]] | None = None,
    ) -> TurnResult:
        """Submit.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            prompt (str): IN: prompt. OUT: Consumed during execution.
            tool_executor (Any, optional): IN: tool executor. Defaults to None. OUT: Consumed during execution.
            tool_schemas (list[dict[str, Any]] | None, optional): IN: tool schemas. Defaults to None. OUT: Consumed during execution.
        Returns:
            TurnResult: OUT: Result of the operation."""

        if self._turn_count >= self.config.max_turns:
            return TurnResult(
                prompt=prompt,
                output=f"Max turns ({self.config.max_turns}) reached.",
                stop_reason="max_turns",
            )
        if self._total_in_tokens + self._total_out_tokens >= self.config.max_budget_tokens:
            return TurnResult(
                prompt=prompt,
                output=f"Token budget ({self.config.max_budget_tokens:,}) exhausted.",
                stop_reason="budget_exhausted",
            )

        if self.transcript.turn_count >= self.config.compact_after_turns:
            removed = self.transcript.compact(keep_last=self.config.compact_keep_last)
            if removed > 0:
                self.history.add("compaction", f"Removed {removed} old messages")

        self._turn_count += 1

        from xerxes.streaming.events import AgentState, TextChunk, ToolEnd, ToolStart, TurnDone

        state = AgentState(messages=self.transcript.to_messages())
        config = {
            "model": self.config.model,
            "permission_mode": self.config.permission_mode,
            "max_tokens": self.config.max_tokens,
            "thinking": self.config.thinking,
            "thinking_budget": self.config.thinking_budget,
        }

        output_parts: list[str] = []
        tool_names: list[str] = []
        in_tok = out_tok = 0

        from xerxes.streaming.loop import run

        for event in run(
            user_message=prompt,
            state=state,
            config=config,
            system_prompt=self.config.system_prompt,
            tool_executor=tool_executor,
            tool_schemas=tool_schemas,
        ):
            if isinstance(event, TextChunk):
                output_parts.append(event.text)
            elif isinstance(event, ToolStart):
                tool_names.append(event.name)
                self.history.add_tool_call(event.name)
            elif isinstance(event, ToolEnd):
                pass
            elif isinstance(event, TurnDone):
                in_tok += event.input_tokens
                out_tok += event.output_tokens
                self.cost_tracker.record_turn(
                    self.config.model,
                    event.input_tokens,
                    event.output_tokens,
                    label=f"turn_{self._turn_count}",
                )

        output = "".join(output_parts)
        self._total_in_tokens += in_tok
        self._total_out_tokens += out_tok
        self.history.add_turn(self.config.model, in_tok, out_tok)

        self.transcript.entries.clear()
        for msg in state.messages:
            self.transcript.append(
                msg["role"], msg.get("content", ""), **{k: v for k, v in msg.items() if k not in ("role", "content")}
            )

        return TurnResult(
            prompt=prompt,
            output=output,
            tool_calls=tuple(tool_names),
            in_tokens=in_tok,
            out_tokens=out_tok,
        )

    def submit_stream(
        self,
        prompt: str,
        tool_executor: Any = None,
        tool_schemas: list[dict[str, Any]] | None = None,
    ) -> Generator[Any, None, TurnResult]:
        """Submit stream.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            prompt (str): IN: prompt. OUT: Consumed during execution.
            tool_executor (Any, optional): IN: tool executor. Defaults to None. OUT: Consumed during execution.
            tool_schemas (list[dict[str, Any]] | None, optional): IN: tool schemas. Defaults to None. OUT: Consumed during execution.
        Returns:
            Generator[Any, None, TurnResult]: OUT: Result of the operation."""

        from xerxes.streaming.events import AgentState, TextChunk, ToolStart, TurnDone
        from xerxes.streaming.loop import run

        if self._turn_count >= self.config.max_turns:
            result = TurnResult(prompt=prompt, output="Max turns reached.", stop_reason="max_turns")
            return result

        if self.transcript.turn_count >= self.config.compact_after_turns:
            self.transcript.compact(keep_last=self.config.compact_keep_last)

        self._turn_count += 1

        state = AgentState(messages=self.transcript.to_messages())
        config = {
            "model": self.config.model,
            "permission_mode": self.config.permission_mode,
            "max_tokens": self.config.max_tokens,
        }

        output_parts: list[str] = []
        tool_names: list[str] = []
        in_tok = out_tok = 0

        for event in run(
            user_message=prompt,
            state=state,
            config=config,
            system_prompt=self.config.system_prompt,
            tool_executor=tool_executor,
            tool_schemas=tool_schemas,
        ):
            yield event
            if isinstance(event, TextChunk):
                output_parts.append(event.text)
            elif isinstance(event, ToolStart):
                tool_names.append(event.name)
            elif isinstance(event, TurnDone):
                in_tok += event.input_tokens
                out_tok += event.output_tokens
                self.cost_tracker.record_turn(
                    self.config.model,
                    event.input_tokens,
                    event.output_tokens,
                )

        output = "".join(output_parts)
        self._total_in_tokens += in_tok
        self._total_out_tokens += out_tok
        self.transcript.entries.clear()
        for msg in state.messages:
            self.transcript.append(
                msg["role"], msg.get("content", ""), **{k: v for k, v in msg.items() if k not in ("role", "content")}
            )

        return TurnResult(
            prompt=prompt,
            output=output,
            tool_calls=tuple(tool_names),
            in_tokens=in_tok,
            out_tokens=out_tok,
        )

    @property
    def turn_count(self) -> int:
        """Return Turn count.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            int: OUT: Result of the operation."""
        return self._turn_count

    @property
    def total_cost(self) -> float:
        """Return Total cost.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            float: OUT: Result of the operation."""
        return self.cost_tracker.total_cost_usd

    def as_markdown(self) -> str:
        """As markdown.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            str: OUT: Result of the operation."""

        lines = [
            "# Query Engine Session",
            "",
            f"Session ID: {self.session_id}",
            f"Model: {self.config.model}",
            f"Turns: {self._turn_count}",
            f"Total tokens: {self._total_in_tokens + self._total_out_tokens:,}",
            f"Total cost: ${self.cost_tracker.total_cost_usd:.4f}",
            "",
            self.transcript.as_markdown(),
            "",
            self.history.as_markdown(),
            "",
            self.cost_tracker.summary(),
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """To dict.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            dict[str, Any]: OUT: Result of the operation."""

        return {
            "session_id": self.session_id,
            "model": self.config.model,
            "turn_count": self._turn_count,
            "total_in_tokens": self._total_in_tokens,
            "total_out_tokens": self._total_out_tokens,
            "messages": self.transcript.to_messages(),
            "history": self.history.as_dicts(),
            "costs": self.cost_tracker.as_dicts(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], **kwargs: Any) -> QueryEngine:
        """From dict.

        Args:
            cls: IN: The class. OUT: Used for class-level operations.
            data (dict[str, Any]): IN: data. OUT: Consumed during execution.
            **kwargs: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            QueryEngine: OUT: Result of the operation."""

        config = QueryEngineConfig(model=data.get("model", "gpt-4o"), **kwargs)
        engine = cls(config=config, session_id=data.get("session_id"))
        engine._turn_count = data.get("turn_count", 0)
        engine._total_in_tokens = data.get("total_in_tokens", 0)
        engine._total_out_tokens = data.get("total_out_tokens", 0)
        for msg in data.get("messages", []):
            role = msg.pop("role", "user")
            content = msg.pop("content", "")
            engine.transcript.append(role, content, **msg)
        return engine


__all__ = [
    "QueryEngine",
    "QueryEngineConfig",
    "TurnResult",
]
