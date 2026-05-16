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
"""Long-lived session record that survives one user prompt across a turn.

:class:`RuntimeContext` snapshots the environment at session start (cwd,
Python version, platform, git branch, model, provider). :class:`RuntimeSession`
glues together the transcript, history log, cost tracker, raw streaming
events, and per-tool execution records. Sessions can be persisted as JSON via
:meth:`RuntimeSession.save` and reloaded with :meth:`RuntimeSession.load`,
powering ``/sessions``, replay, and audit features.
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from .cost_tracker import CostTracker
from .history import HistoryLog
from .transcript import TranscriptStore


@dataclass
class RuntimeContext:
    """Environment snapshot captured when a session starts.

    Attributes:
        cwd: Working directory at capture time.
        python_version: Interpreter version as ``major.minor.micro``.
        platform_name: ``platform.platform()`` string.
        git_branch: Git branch in ``cwd`` (or ``""`` when not a repo).
        model: LLM identifier the session was opened against.
        provider: Provider name resolved from ``model`` (e.g. ``"openai"``).
        timestamp: ISO-8601 timestamp captured at construction.
    """

    cwd: str = ""
    python_version: str = ""
    platform_name: str = ""
    git_branch: str = ""
    model: str = ""
    provider: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @classmethod
    def capture(cls, model: str = "", provider: str = "") -> RuntimeContext:
        """Snapshot the current cwd, Python, platform, and git branch.

        Args:
            model: Model identifier to embed on the snapshot.
            provider: Provider name to embed on the snapshot.
        """

        import subprocess

        git_branch = ""
        try:
            git_branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except Exception:
            pass

        return cls(
            cwd=str(Path.cwd()),
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            platform_name=platform.platform(),
            git_branch=git_branch,
            model=model,
            provider=provider,
        )


@dataclass
class RuntimeSession:
    """All persistent state belonging to one Xerxes session.

    Attributes:
        session_id: Stable hex identifier (random uuid4 by default).
        prompt: Initiating user prompt; may be ``""`` for interactive sessions.
        context: Captured :class:`RuntimeContext`.
        transcript: Conversation transcript shared with the query engine.
        history: Compact :class:`HistoryLog` of turns/tool calls.
        cost_tracker: Per-session :class:`CostTracker`.
        stream_events: Raw streaming events captured for replay.
        tool_executions: Per-tool-call records with timing and permission flag.
        metadata: Free-form key/value bag for caller use.
    """

    session_id: str = field(default_factory=lambda: uuid4().hex)
    prompt: str = ""
    context: RuntimeContext = field(default_factory=RuntimeContext)
    transcript: TranscriptStore = field(default_factory=TranscriptStore)
    history: HistoryLog = field(default_factory=HistoryLog)
    cost_tracker: CostTracker = field(default_factory=CostTracker)
    stream_events: list[dict[str, Any]] = field(default_factory=list)
    tool_executions: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        model: str = "",
        prompt: str = "",
        **metadata: Any,
    ) -> RuntimeSession:
        """Build a new session whose context is freshly captured."""

        from xerxes.llms.registry import detect_provider

        provider = detect_provider(model) if model else ""
        return cls(
            prompt=prompt,
            context=RuntimeContext.capture(model=model, provider=provider),
            metadata=metadata,
        )

    def record_tool_execution(
        self,
        tool_name: str,
        inputs: Any = None,
        result: str = "",
        duration_ms: float = 0.0,
        permitted: bool = True,
    ) -> None:
        """Append a tool-execution record and mirror it into the history log.

        ``result`` is truncated to 500 chars for the structured record and to
        100 chars for the history entry to keep persisted payloads small.
        """

        self.tool_executions.append(
            {
                "tool": tool_name,
                "inputs": inputs,
                "result": str(result)[:500],
                "duration_ms": duration_ms,
                "permitted": permitted,
                "timestamp": datetime.now().isoformat(),
            }
        )
        self.history.add_tool_call(tool_name, str(result)[:100], duration_ms)

    def record_stream_event(self, event_type: str, **data: Any) -> None:
        """Append a typed streaming-loop event with a wall-clock timestamp."""

        self.stream_events.append(
            {
                "type": event_type,
                "timestamp": datetime.now().isoformat(),
                **data,
            }
        )

    def record_turn(self, model: str, in_tokens: int, out_tokens: int) -> None:
        """Record one completed turn into both cost tracker and history."""

        self.cost_tracker.record_turn(model, in_tokens, out_tokens)
        self.history.add_turn(model, in_tokens, out_tokens)

    def as_markdown(self) -> str:
        """Render a Markdown overview of the session for ``/sessions`` and replay."""

        lines = [
            "# Runtime Session",
            "",
            f"Session ID: `{self.session_id}`",
            f"Prompt: {self.prompt}",
            f"Started: {self.context.timestamp}",
            "",
            "## Context",
            f"- CWD: `{self.context.cwd}`",
            f"- Python: {self.context.python_version}",
            f"- Platform: {self.context.platform_name}",
            f"- Git branch: {self.context.git_branch or 'N/A'}",
            f"- Model: {self.context.model}",
            f"- Provider: {self.context.provider}",
            "",
            "## Tool Executions",
        ]
        if self.tool_executions:
            for te in self.tool_executions:
                status = "OK" if te["permitted"] else "DENIED"
                lines.append(f"- `{te['tool']}` [{status}] ({te['duration_ms']:.1f}ms)")
        else:
            lines.append("- none")

        lines.extend(
            [
                "",
                "## Stream Events",
                f"Total events: {len(self.stream_events)}",
                "",
                self.cost_tracker.summary(),
                "",
                self.history.as_markdown(),
                "",
                self.transcript.as_markdown(),
            ]
        )
        return "\n".join(lines)

    def save(self, directory: str | Path = ".xerxes_sessions") -> Path:
        """Persist the session as ``<directory>/<session_id>.json``.

        Truncates ``stream_events`` to the first 100 entries to keep file
        sizes manageable. Returns the path written. Marks the transcript as
        flushed.
        """

        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        path = target / f"{self.session_id}.json"

        data = {
            "session_id": self.session_id,
            "prompt": self.prompt,
            "context": {
                "cwd": self.context.cwd,
                "python_version": self.context.python_version,
                "platform_name": self.context.platform_name,
                "git_branch": self.context.git_branch,
                "model": self.context.model,
                "provider": self.context.provider,
                "timestamp": self.context.timestamp,
            },
            "messages": self.transcript.to_messages(),
            "history": self.history.as_dicts(),
            "costs": self.cost_tracker.as_dicts(),
            "tool_executions": self.tool_executions,
            "stream_events": self.stream_events[:100],
            "metadata": self.metadata,
        }

        path.write_text(json.dumps(data, indent=2, default=str))
        self.transcript.flush()
        return path

    @classmethod
    def load(cls, path: str | Path) -> RuntimeSession:
        """Rehydrate a session previously written by :meth:`save`."""

        data = json.loads(Path(path).read_text())
        ctx_data = data.get("context", {})

        session = cls(
            session_id=data.get("session_id", ""),
            prompt=data.get("prompt", ""),
            context=RuntimeContext(**ctx_data),
            metadata=data.get("metadata", {}),
        )

        for msg in data.get("messages", []):
            role = msg.pop("role", "user")
            content = msg.pop("content", "")
            session.transcript.append(role, content, **msg)

        session.tool_executions = data.get("tool_executions", [])
        session.stream_events = data.get("stream_events", [])

        return session


__all__ = [
    "RuntimeContext",
    "RuntimeSession",
]
