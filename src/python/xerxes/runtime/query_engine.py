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
"""High-level driver that turns user prompts into completed agent turns.

:class:`QueryEngine` owns the per-session transcript, history log, and cost
tracker, and drives :func:`xerxes.streaming.loop.run` to actually talk to the
LLM. It enforces per-session limits (max turns, token budget, automatic
compaction) and exposes both a blocking :meth:`QueryEngine.submit` and a
streaming :meth:`QueryEngine.submit_stream` interface for the TUI and bridge.
"""

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
    """Tunable parameters of a :class:`QueryEngine` instance.

    Attributes:
        max_turns: Hard cap on user → assistant turns before the engine refuses
            new prompts with ``stop_reason="max_turns"``.
        max_budget_tokens: Combined input+output token ceiling enforced across
            the whole session.
        compact_after_turns: Trigger transcript compaction once the stored
            turn count reaches this value.
        compact_keep_last: Number of most-recent transcript entries preserved
            when compaction runs.
        model: LLM identifier passed to the streaming loop.
        system_prompt: System prompt injected ahead of every turn.
        permission_mode: Permission policy forwarded to the streaming loop
            (``"auto"``, ``"plan"``, ``"manual"``, etc.).
        max_tokens: Maximum tokens the LLM may emit per response.
        thinking: Whether to request reasoning content from the provider.
        thinking_budget: Reasoning-token budget when ``thinking`` is enabled.
    """

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
    """Outcome of a single :meth:`QueryEngine.submit` call.

    Attributes:
        prompt: The user message that initiated this turn.
        output: Assistant-visible text, concatenated from every ``TextChunk``.
        tool_calls: Names of tools invoked during the turn, in order.
        in_tokens: Input tokens consumed.
        out_tokens: Output tokens produced.
        stop_reason: ``"complete"``, ``"max_turns"``, or ``"budget_exhausted"``.
    """

    prompt: str
    output: str
    tool_calls: tuple[str, ...] = ()
    in_tokens: int = 0
    out_tokens: int = 0
    stop_reason: str = "complete"


class QueryEngine:
    """Per-session driver that turns prompts into completed agent turns."""

    def __init__(
        self,
        config: QueryEngineConfig,
        registry: ExecutionRegistry | None = None,
        session_id: str | None = None,
    ) -> None:
        """Construct a new engine.

        Args:
            config: Behavioural knobs (limits, model, prompt, etc.).
            registry: Optional pre-populated execution registry; when omitted
                an empty one is created.
            session_id: Stable session identifier; a random hex id is
                generated when ``None``.
        """
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
        """Build a :class:`QueryEngine` with a freshly constructed config.

        Convenience wrapper around ``QueryEngine(QueryEngineConfig(...))`` that
        forwards extra ``config_kwargs`` straight into the dataclass.
        """

        config = QueryEngineConfig(model=model, system_prompt=system_prompt, **config_kwargs)
        return cls(config=config, registry=registry)

    def submit(
        self,
        prompt: str,
        tool_executor: Any = None,
        tool_schemas: list[dict[str, Any]] | None = None,
    ) -> TurnResult:
        """Run one full turn synchronously and return its aggregated result.

        Blocks until the streaming loop emits ``TurnDone`` (or the configured
        limits are hit). Updates the session transcript, history log, and
        cost tracker as a side-effect. Triggers transcript compaction when the
        stored turn count crosses ``compact_after_turns``.

        Args:
            prompt: User message to send for this turn.
            tool_executor: Callable forwarded to the streaming loop to actually
                dispatch tool calls; ``None`` disables tool execution.
            tool_schemas: Optional list of tool schemas exposed to the LLM.

        Returns:
            A :class:`TurnResult` summarising the turn; ``stop_reason`` is
            ``"max_turns"`` or ``"budget_exhausted"`` when limits aborted the
            call before the LLM was contacted.
        """

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
        """Run a turn while yielding every underlying streaming-loop event.

        Like :meth:`submit` but exposes the raw event stream so callers (TUI,
        bridge) can render text chunks, tool starts, etc. live. Returns the
        completed :class:`TurnResult` via ``StopIteration.value`` once the
        generator is exhausted.
        """

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
        """Number of turns submitted so far this session."""
        return self._turn_count

    @property
    def total_cost(self) -> float:
        """Cumulative USD cost recorded by the :class:`CostTracker`."""
        return self.cost_tracker.total_cost_usd

    def as_markdown(self) -> str:
        """Render the full session (transcript, history, costs) as Markdown."""

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
        """Serialise the engine's persistent state for save/load round-trips."""

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
        """Rehydrate a :class:`QueryEngine` from :meth:`to_dict` output.

        Extra ``kwargs`` are forwarded to :class:`QueryEngineConfig` so callers
        can override model/system-prompt at restore time.
        """

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
