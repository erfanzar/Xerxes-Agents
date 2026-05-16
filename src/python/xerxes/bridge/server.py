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
"""Stdio JSON-RPC bridge hosting the agent runtime in-process.

:class:`BridgeServer` reads newline-delimited JSON-RPC messages from
``stdin``, dispatches them to handlers (``init``, ``query``, ``slash``,
``cancel``, provider/profile CRUD, ...), and emits events back on ``stdout``
under one of two protocols:

* **Legacy** ``{event, data}`` flat objects — what older Xerxes clients expect.
* **Wire** mode — Kimi-Code-compatible JSON-RPC 2.0 NDJSON with
  ``method=event`` / ``method=request`` frames so the same TUI can talk to
  Kimi and Xerxes interchangeably.

The bridge owns the same streaming loop the daemon uses, plus per-session
JSON persistence under ``$XERXES_HOME/sessions``, the skill authoring
pipeline, and inline auto-compaction when the conversation exceeds 75% of
the model context window. :func:`main` is the console-script entry point.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import queue
import sys
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..core.paths import xerxes_subdir
from ..extensions.skill_authoring.pipeline import SkillAuthoringPipeline
from ..extensions.skills import SkillRegistry, activate_skill
from ..llms.registry import calc_cost, detect_provider, get_context_limit
from ..runtime.bootstrap import bootstrap
from ..runtime.bridge import build_tool_executor, populate_registry
from ..runtime.config_context import set_config as set_global_config
from ..runtime.config_context import set_event_callback
from ..runtime.cost_tracker import CostTracker
from ..streaming.events import (
    AgentState,
    PermissionRequest,
    TextChunk,
    ThinkingChunk,
    ToolEnd,
    ToolStart,
    TurnDone,
)
from ..streaming.loop import run as run_agent_loop
from ..streaming.wire_events import to_kimi_event_name
from ..tools.agent_meta_tools import set_skill_registry
from ..tools.claude_tools import set_ask_user_question_callback
from . import profiles

logger = logging.getLogger(__name__)


def _normalize_interaction_mode(mode: Any, *, plan_mode: bool = False) -> str:
    """Coerce assorted mode labels (``coder``, ``research``, ...) to ``code``/``researcher``/``plan``."""
    if plan_mode:
        return "plan"
    value = str(mode or "code").strip().lower()
    aliases = {
        "": "code",
        "coding": "code",
        "coder": "code",
        "research": "researcher",
        "researcher": "researcher",
        "plan": "plan",
        "planner": "plan",
    }
    return aliases.get(value, "code")


def _agent_name_for_mode(mode: str) -> str:
    """Map an interaction mode to its YAML agent definition (``planner``/``researcher``/``coder``)."""
    if mode == "plan":
        return "planner"
    if mode == "researcher":
        return "researcher"
    return "coder"


class BridgeServer:
    """Stdio JSON-RPC server hosting one in-process agent session.

    Owns the conversation state (:class:`AgentState`), the tool executor and
    schema list, the cost tracker, the active skill registry, the sub-agent
    event aggregator, and the per-session resume / replay machinery. Frame
    emission is dual-mode: the legacy ``{event, data}`` shape and the Kimi
    wire-protocol JSON-RPC frames are toggled by ``wire_mode``.
    """

    SESSIONS_DIR = xerxes_subdir("sessions")

    def __init__(self, wire_mode: bool = False) -> None:
        """Build state, registries, and IO locks.

        Args:
            wire_mode: When true, emit Kimi-compatible JSON-RPC 2.0 NDJSON
                instead of the legacy flat ``{event, data}`` shape.
        """
        self.config: dict[str, Any] = {}
        self.state = AgentState()
        self.cost_tracker = CostTracker()
        self.system_prompt = ""
        self.tool_executor = None
        self.tool_schemas: list[dict[str, Any]] = []
        self._initialized = False
        self._running = True
        self._cancel = False
        self._out_lock = threading.Lock()
        self._stdout = sys.stdout
        self._suppressing_tag = False
        self._suppress_buf: list[str] = []
        self._session_id = str(uuid.uuid4())[:8]
        self._session_cwd = os.getcwd()
        self.SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        self._wire_mode = wire_mode
        self._current_tool_call_id = ""
        self._step_count = 0

        self._subagent_text_buffers: dict[str, str] = {}
        self._subagent_thinking_buffers: dict[str, str] = {}
        self._subagent_buffer_lock = threading.Lock()
        self._subagent_parent_tool: dict[str, str] = {}
        self._subagent_tool_id_fifo: dict[str, list[str]] = {}
        self._SUBAGENT_FLUSH_THRESHOLD = 400

        self._pending_resume_replays: list[dict[str, Any]] = []

        self._skill_registry = SkillRegistry()
        self._skills_dir = xerxes_subdir("skills")
        self._skills_dir.mkdir(parents=True, exist_ok=True)

        import xerxes as _xerxes_pkg

        _bundled_skills_dir = Path(_xerxes_pkg.__file__).parent / "skills"
        discover_dirs = [str(self._skills_dir), str(Path.cwd() / "skills")]
        if _bundled_skills_dir.is_dir():
            discover_dirs.insert(0, str(_bundled_skills_dir))

        self._skill_registry.discover(*discover_dirs)
        set_skill_registry(self._skill_registry)

        self._pending_skill_name: str = ""
        self._query_thread: threading.Thread | None = None
        self._permission_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._question_queue: queue.Queue[dict[str, Any]] = queue.Queue()

        set_ask_user_question_callback(self._ask_question)

        self._authoring_pipeline = SkillAuthoringPipeline(
            skills_dir=self._skills_dir,
            skill_registry=self._skill_registry,
        )
        self._pending_tool_inputs: dict[str, Any] | None = None

    def _emit(self, event: str, data: dict[str, Any] | None = None) -> None:
        """Emit a legacy ``{event, data}`` line on stdout.

        In wire mode, ``slash_result`` events are reflected as
        ``category="slash"`` notifications instead; other event names are
        dropped (wire mode already emits them through ``_emit_wire_*``).
        """
        if self._wire_mode:
            if event == "slash_result":
                self._emit_wire_notification(
                    notification_id=str(uuid.uuid4()),
                    category="slash",
                    type_="slash_result",
                    severity="info",
                    title="",
                    body=str((data or {}).get("output", "")),
                )
            return
        msg = {"event": event, "data": data or {}}
        line = json.dumps(msg, ensure_ascii=False, default=str)
        with self._out_lock:
            self._stdout.write(line + "\n")
            self._stdout.flush()

    def _emit_error(self, message: str) -> None:
        """Emit ``message`` as an error event (or wire ``severity=error`` notification)."""
        if self._wire_mode:
            self._emit_wire_notification(
                notification_id=str(uuid.uuid4()),
                category="bridge",
                type_="error",
                severity="error",
                title="Bridge error",
                body=message,
            )
            return
        self._emit("error", {"message": message})

    def _emit_state(self) -> None:
        """Emit a legacy ``state`` event summarising tokens, cost, and message count."""
        model = self.config.get("model", "")
        context_limit = get_context_limit(model)
        total_tokens = self.state.total_input_tokens + self.state.total_output_tokens
        remaining = max(0, context_limit - total_tokens)
        self._emit(
            "state",
            {
                "turn_count": self.state.turn_count,
                "total_input_tokens": self.state.total_input_tokens,
                "total_output_tokens": self.state.total_output_tokens,
                "message_count": len(self.state.messages),
                "tool_execution_count": len(self.state.tool_executions),
                "context_limit": context_limit,
                "remaining_context": remaining,
                "used_context": total_tokens,
                "cost_usd": calc_cost(
                    model,
                    self.state.total_input_tokens,
                    self.state.total_output_tokens,
                ),
            },
        )

    def _emit_text(self, text: str) -> None:
        """Emit a legacy ``text_chunk`` event, eliding any ``<function=...>...</function>`` payloads.

        Tool calls some providers stream as inline ``<function>`` blocks would
        otherwise leak into the rendered transcript; this stateful filter
        buffers across chunks until the closing tag arrives.
        """
        if self._suppressing_tag:
            self._suppress_buf.append(text)
            joined = "".join(self._suppress_buf)
            if "</function>" in joined:
                after = joined.split("</function>", 1)[1]
                self._suppressing_tag = False
                self._suppress_buf.clear()
                if after.strip():
                    self._emit("text_chunk", {"text": after})
            return

        if "<function=" in text:
            before, _, rest = text.partition("<function=")
            if before.strip():
                self._emit("text_chunk", {"text": before})
            self._suppressing_tag = True
            self._suppress_buf.clear()
            self._suppress_buf.append("<function=" + rest)
            joined = "".join(self._suppress_buf)
            if "</function>" in joined:
                after = joined.split("</function>", 1)[1]
                self._suppressing_tag = False
                self._suppress_buf.clear()
                if after.strip():
                    self._emit("text_chunk", {"text": after})
            return

        stripped = text.strip()
        if not stripped:
            return
        if stripped.startswith('{"name":') and '"arguments"' in stripped:
            return

        self._emit("text_chunk", {"text": text})

    def _on_agent_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Handle a runtime callback event (mode change, sub-agent activity, ...).

        Re-emits the event on both protocols and, for ``agent_*`` events in
        wire mode, calls :meth:`_emit_subagent_summary` so the TUI sees the
        chatty stream folded into compact preview notifications.
        """
        if event_type == "interaction_mode_changed":
            mode = _normalize_interaction_mode(data.get("mode"), plan_mode=bool(data.get("plan_mode", False)))
            self.config["mode"] = mode
            self.config["plan_mode"] = mode == "plan"
            set_global_config(self.config)
            if self._wire_mode:
                self._emit_wire_status()
            self._emit(event_type, {**data, "mode": mode, "plan_mode": mode == "plan"})
            return
        self._emit(event_type, data)
        if self._wire_mode and event_type.startswith("agent_"):
            self._emit_subagent_summary(event_type, data)

    def _replay_history_to_wire(self) -> None:
        """Emit prior turns as ``category="history"`` wire notifications, ending with a ``resumed`` marker."""
        count = 0
        for msg in self.state.messages:
            role = (msg.get("role") or "").lower()
            content = msg.get("content")

            if isinstance(content, list):
                parts = []
                for p in content:
                    if isinstance(p, dict):
                        parts.append(str(p.get("text", "")))
                    else:
                        parts.append(str(p))
                text = "\n".join(p for p in parts if p)
            elif isinstance(content, dict):
                text = str(content.get("text", ""))
            else:
                text = str(content or "")

            text = text.strip()
            if not text:
                continue

            if role == "user":
                body = f"✨ {text}"
            elif role == "assistant":
                body = text
            else:
                continue

            self._emit_wire_notification(
                notification_id=str(uuid.uuid4()),
                category="history",
                type_=f"replay_{role}",
                severity="info",
                title="",
                body=body,
            )
            count += 1

        self._emit_wire_notification(
            notification_id=str(uuid.uuid4()),
            category="history",
            type_="resumed",
            severity="info",
            title="",
            body=f"── resumed session {self._session_id} ({count} messages) ──",
        )

    def _emit_subagent_summary(self, event_type: str, data: dict[str, Any]) -> None:
        """Fold ``agent_*`` events into compact wire summaries.

        Text and thinking chunks accumulate in per-task rolling buffers and
        flush on paragraph breaks, threshold size, tool boundaries, or
        completion so the TUI sees coherent prose instead of per-token noise.
        Sub-agent tool calls are additionally wrapped in ``subagent_event`` so
        they nest under the parent ``AgentTool`` block.
        """
        agent_name = data.get("agent_name") or data.get("agent_type") or "subagent"
        agent_type = data.get("agent_type") or ""
        task_id = data.get("task_id", "")
        short_id = (task_id[:8] + "…") if len(task_id) > 8 else task_id
        prefix = f"{agent_name}#{short_id}" if short_id else agent_name

        if event_type == "agent_spawn":
            if task_id and self._current_tool_call_id:
                with self._subagent_buffer_lock:
                    self._subagent_parent_tool[task_id] = self._current_tool_call_id
            body = f"{prefix} spawned (depth={data.get('depth', '?')}): {data.get('prompt', '')[:140]}"
            self._emit_wire_notification(
                notification_id=str(uuid.uuid4()),
                category="subagent",
                type_=event_type,
                severity="info",
                title="",
                body=body,
            )
            self._emit_subagent_stream(task_id, prefix, "starting…")
            return

        if event_type == "agent_text":
            self._stream_subagent_chunk(task_id, prefix, data.get("text") or "", kind="text")
            return

        if event_type == "agent_thinking":
            self._stream_subagent_chunk(task_id, prefix, data.get("text") or "", kind="thinking")
            return

        if event_type == "agent_tool_start":
            self._emit_subagent_tool_event(task_id, agent_type, data, kind="start")
            inputs = data.get("inputs") or {}
            key = next(iter(inputs.values()), "") if isinstance(inputs, dict) else ""
            self._emit_subagent_stream(
                task_id,
                prefix,
                f"◐ {data.get('tool_name', 'tool')}({str(key)[:80]})",
            )
            return

        if event_type == "agent_tool_end":
            self._emit_subagent_tool_event(task_id, agent_type, data, kind="end")
            mark = "✓" if data.get("permitted", True) else "✗"
            self._emit_subagent_stream(
                task_id,
                prefix,
                f"{mark} {data.get('tool_name', 'tool')} — {data.get('duration_ms', 0):.0f}ms",
            )
            return

        if event_type == "agent_done":
            with self._subagent_buffer_lock:
                self._subagent_parent_tool.pop(task_id, None)
                self._subagent_tool_id_fifo.pop(task_id, None)
                self._subagent_text_buffers.pop(task_id, None)
                self._subagent_thinking_buffers.pop(task_id, None)
            self._emit_subagent_stream(task_id, prefix, "")
            return

    SUBAGENT_PREVIEW_CHARS = 100

    def _stream_subagent_chunk(self, task_id: str, prefix: str, text: str, *, kind: str) -> None:
        """Update the rolling preview tail shown for ``task_id``.

        Per-task buffers are capped at twice :attr:`SUBAGENT_PREVIEW_CHARS`;
        emits a single ``subagent_stream`` notification with the latest tail.
        ``kind`` is ``"text"`` or ``"thinking"`` and selects which buffer to
        update.
        """
        if not text or not task_id:
            return
        buffers = self._subagent_text_buffers if kind == "text" else self._subagent_thinking_buffers
        cap = self.SUBAGENT_PREVIEW_CHARS
        with self._subagent_buffer_lock:
            merged = buffers.get(task_id, "") + text
            if len(merged) > cap * 2:
                merged = merged[-cap * 2 :]
            buffers[task_id] = merged
            tail = " ".join(merged.split())
        if not tail:
            return
        if len(tail) > cap:
            tail = "…" + tail[-cap:]
        label_suffix = " (thinking)" if kind == "thinking" else ""
        self._emit_subagent_stream(task_id, f"{prefix}{label_suffix}", tail)

    def _emit_subagent_stream(self, task_id: str, label: str, body: str) -> None:
        """Emit one transient preview notification for ``task_id`` (empty ``body`` clears it)."""
        self._emit_wire_notification(
            notification_id=str(uuid.uuid4()),
            category="subagent_stream",
            type_="subagent_stream",
            severity="info",
            title="",
            body=body,
            payload={"task_id": task_id, "label": label},
        )

    def _emit_subagent_tool_event(
        self,
        task_id: str,
        agent_type: str,
        data: dict[str, Any],
        *,
        kind: str,
    ) -> bool:
        """Wrap a sub-agent's inner tool call/result in a nested ``subagent_event``.

        The TUI uses ``parent_tool_call_id`` to render the inner call inside
        the parent ``AgentTool`` block. ``kind`` is ``"start"`` or ``"end"``.
        Returns ``True`` when the nested event was emitted with both ids; the
        caller can then suppress the flat fallback notification.
        """
        with self._subagent_buffer_lock:
            parent_id = self._subagent_parent_tool.get(task_id) or self._current_tool_call_id
            raw_inner = data.get("tool_call_id")
            if kind == "start":
                if raw_inner:
                    inner_id = str(raw_inner)
                else:
                    inner_id = f"sub_{uuid.uuid4().hex[:12]}"
                self._subagent_tool_id_fifo.setdefault(task_id, []).append(inner_id)
            else:
                if raw_inner:
                    inner_id = str(raw_inner)
                    fifo = self._subagent_tool_id_fifo.get(task_id) or []
                    if inner_id in fifo:
                        fifo.remove(inner_id)
                else:
                    fifo = self._subagent_tool_id_fifo.get(task_id) or []
                    inner_id = fifo.pop(0) if fifo else ""
        if not parent_id or not inner_id:
            return False
        if kind == "start":
            inputs = data.get("inputs") or {}
            try:
                arguments_str = json.dumps(inputs, default=str)
            except Exception:
                arguments_str = ""
            inner = {
                "type": "ToolCall",
                "payload": {
                    "id": inner_id,
                    "name": data.get("tool_name", ""),
                    "arguments": arguments_str,
                },
            }
        else:
            inner = {
                "type": "ToolResult",
                "payload": {
                    "tool_call_id": inner_id,
                    "return_value": data.get("result") or "",
                    "display_blocks": [],
                },
            }
        self._emit_wire_event(
            "subagent_event",
            {
                "id": str(uuid.uuid4()),
                "parent_tool_call_id": parent_id,
                "agent_id": task_id,
                "subagent_type": agent_type,
                "event": inner,
            },
        )
        return True

    def _emit_wire_event(self, event_type: str, payload: dict[str, Any]) -> None:
        """Write one ``method=event`` JSON-RPC frame, mapping ``event_type`` through :func:`to_kimi_event_name`."""
        if not self._wire_mode:
            return
        msg = {
            "jsonrpc": "2.0",
            "method": "event",
            "params": {"type": to_kimi_event_name(event_type), "payload": payload},
        }
        line = json.dumps(msg, ensure_ascii=False, default=str)
        with self._out_lock:
            try:
                self._stdout.write(line + "\n")
                self._stdout.flush()
            except Exception:
                pass

    def _emit_wire_request(
        self,
        request_id: str,
        request_type: str,
        payload: dict[str, Any],
    ) -> None:
        """Write one ``method=request`` frame the TUI is expected to respond to."""
        if not self._wire_mode:
            return
        msg = {
            "jsonrpc": "2.0",
            "method": "request",
            "id": request_id,
            "params": {"type": to_kimi_event_name(request_type), "payload": payload},
        }
        line = json.dumps(msg, ensure_ascii=False, default=str)
        with self._out_lock:
            try:
                self._stdout.write(line + "\n")
                self._stdout.flush()
            except Exception:
                pass

    def _emit_wire_turn_begin(self, user_input: str) -> None:
        """Mark the start of a new turn carrying ``user_input`` as a text part."""
        self._emit_wire_event("turn_begin", {"user_input": [{"type": "text", "text": user_input}]})
        self._step_count = 0

    def _emit_wire_step_begin(self, n: int) -> None:
        """Mark the start of step ``n`` and remember it for subsequent events."""
        self._emit_wire_event("step_begin", {"n": n})
        self._step_count = n

    def _emit_wire_text(self, text: str) -> None:
        """Emit one ``text_part`` event, eliding inline ``<function>`` payloads."""
        if self._suppressing_tag:
            self._suppress_buf.append(text)
            joined = "".join(self._suppress_buf)
            if "</function>" in joined:
                after = joined.split("</function>", 1)[1]
                self._suppressing_tag = False
                self._suppress_buf.clear()
                if after.strip():
                    self._emit_wire_event("text_part", {"text": after})
            return

        if "<function=" in text:
            before, _, rest = text.partition("<function=")
            if before.strip():
                self._emit_wire_event("text_part", {"text": before})
            self._suppressing_tag = True
            self._suppress_buf.clear()
            self._suppress_buf.append("<function=" + rest)
            joined = "".join(self._suppress_buf)
            if "</function>" in joined:
                after = joined.split("</function>", 1)[1]
                self._suppressing_tag = False
                self._suppress_buf.clear()
                if after.strip():
                    self._emit_wire_event("text_part", {"text": after})
            return

        stripped = text.strip()
        if not stripped:
            return
        if stripped.startswith('{"name":') and '"arguments"' in stripped:
            return

        self._emit_wire_event("text_part", {"text": text})

    def _emit_wire_think(self, think: str) -> None:
        """Emit one ``think_part`` event carrying a thinking-stream chunk."""
        self._emit_wire_event("think_part", {"think": think})

    def _emit_wire_tool_start(self, tool_call_id: str, name: str, arguments: dict[str, Any]) -> None:
        """Emit a ``tool_call`` start event, synthesising an id if the provider omits one.

        Some upstreams (e.g. raw OpenAI streams) don't supply a tool call id;
        in that case a stable ``call_<uuid12>`` is generated so the TUI's tool
        blocks and the sub-agent nester can correlate later events.
        """
        if not tool_call_id:
            tool_call_id = f"call_{uuid.uuid4().hex[:12]}"
        self._current_tool_call_id = tool_call_id
        self._emit_wire_event(
            "tool_call",
            {"id": tool_call_id, "name": name, "arguments": json.dumps(arguments)},
        )

    def _emit_wire_tool_args_part(self, arguments_part: str) -> None:
        """Emit a streaming ``tool_call_part`` delta with an argument fragment."""
        self._emit_wire_event("tool_call_part", {"arguments_part": arguments_part})

    def _emit_wire_tool_result(
        self,
        tool_call_id: str,
        return_value: str,
        permitted: bool = True,
        duration_ms: float = 0.0,
    ) -> None:
        """Emit a ``tool_result`` event finishing the call with ``return_value`` and a duration."""
        if not tool_call_id:
            tool_call_id = self._current_tool_call_id
        self._emit_wire_event(
            "tool_result",
            {
                "tool_call_id": tool_call_id,
                "return_value": return_value,
                "duration_ms": duration_ms,
                "display_blocks": [],
            },
        )

    def _emit_wire_permission_request(
        self,
        tool_call_id: str,
        name: str,
        description: str,
    ) -> str:
        """Emit an ``approval_request`` and return the generated request id the TUI must echo."""
        request_id = str(uuid.uuid4())
        self._emit_wire_request(
            request_id,
            "approval_request",
            {
                "id": request_id,
                "tool_call_id": tool_call_id,
                "action": name,
                "description": description,
            },
        )
        return request_id

    def _emit_wire_question_request(
        self,
        questions: list[dict[str, Any]],
    ) -> str:
        """Emit a ``question_request`` and return its id; ``questions`` is the wire shape."""
        request_id = str(uuid.uuid4())
        self._emit_wire_request(
            request_id,
            "question_request",
            {"id": request_id, "questions": questions},
        )
        return request_id

    def _emit_wire_init_done(self) -> None:
        """Emit ``init_done`` summarising model, session id, cwd, branch, and discovered skills."""
        model = self.config.get("model", "")
        agent_defs = self.config.get("_agent_definitions") or []
        agent_name = "default" if "default" in agent_defs else (agent_defs[0] if agent_defs else "agent")
        self._emit_wire_event(
            "init_done",
            {
                "model": model,
                "session_id": self._session_id,
                "cwd": str(self._session_cwd) if self._session_cwd else os.getcwd(),
                "git_branch": self._git_branch(),
                "context_limit": get_context_limit(model),
                "agent_name": agent_name,
                "skills": sorted(self._skill_registry.skill_names),
            },
        )

    @staticmethod
    def _git_branch() -> str:
        """Return the current branch name, or empty string when not in a git repo."""
        try:
            import subprocess

            return subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=2,
            ).strip()
        except Exception:
            return ""

    def _emit_wire_status(self) -> None:
        """Emit a ``status_update`` summarising token usage and the active mode."""
        model = self.config.get("model", "")
        context_limit = get_context_limit(model)
        total_tokens = self.state.total_input_tokens + self.state.total_output_tokens
        self._emit_wire_event(
            "status_update",
            {
                "context_tokens": total_tokens,
                "max_context": context_limit,
                "mcp_status": {},
                "plan_mode": self.config.get("plan_mode", False),
                "mode": self.config.get("mode", "code"),
            },
        )

    def _agent_definition_for_mode(self, mode: str) -> Any:
        """Look up the YAML agent definition for ``mode`` or return ``None``."""
        try:
            from ..agents.definitions import get_agent_definition

            return get_agent_definition(_agent_name_for_mode(mode))
        except Exception:
            return None

    def _system_prompt_for_mode(self, mode: str) -> str:
        """Return the mode-specific system prompt with the mode-switching hint appended."""
        agent_def = self._agent_definition_for_mode(mode)
        if agent_def is not None and getattr(agent_def, "system_prompt", ""):
            prompt = str(agent_def.system_prompt).rstrip()
        else:
            prompt = self.system_prompt.rstrip()
        return prompt + "\n\n" + self._mode_switch_hint(mode) + "\n"

    @staticmethod
    def _mode_switch_hint(mode: str) -> str:
        """Return the ``[Mode control]`` paragraph appended to system prompts.

        The hint tells the model which ``SetInteractionModeTool`` call moves
        forward from the current mode (plan/research/code).
        """
        if mode == "plan":
            return (
                "[Mode control]\n"
                "If the plan is complete and implementation should begin in a later turn, "
                'call SetInteractionModeTool(mode="code").'
            )
        if mode == "researcher":
            return (
                "[Mode control]\n"
                "If implementation is needed after your findings, "
                'call SetInteractionModeTool(mode="code").'
            )
        return (
            "[Mode control]\n"
            "If this task should first be researched or planned, call "
            'SetInteractionModeTool(mode="researcher") or SetInteractionModeTool(mode="plan").'
        )

    def _emit_wire_notification(
        self,
        notification_id: str,
        category: str,
        type_: str,
        severity: str,
        title: str,
        body: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Emit one ``notification`` event with the canonical TUI-render fields.

        ``payload`` carries structured side-channel data (e.g. ``task_id``
        for sub-agent stream updates); it defaults to ``{}`` on the wire.
        """
        self._emit_wire_event(
            "notification",
            {
                "id": notification_id,
                "category": category,
                "type": type_,
                "severity": severity,
                "title": title,
                "body": body,
                "payload": payload or {},
            },
        )

    def _emit_wire_compaction_begin(self) -> None:
        """Signal the TUI to render its compaction-in-progress indicator."""
        self._emit_wire_event("compaction_begin", {})

    def _emit_wire_compaction_end(self) -> None:
        """Signal the TUI to clear the compaction indicator."""
        self._emit_wire_event("compaction_end", {})

    def _emit_wire_turn_end(self) -> None:
        """Mark the end of the current turn on the wire."""
        self._emit_wire_event("turn_end", {})

    def _save_session(self) -> None:
        """Persist the current session state to ``$XERXES_HOME/sessions/<id>.json``."""
        data = {
            "session_id": self._session_id,
            "model": self.config.get("model", ""),
            "cwd": self._session_cwd,
            "created_at": getattr(self, "_created_at", datetime.now(UTC).isoformat()),
            "updated_at": datetime.now(UTC).isoformat(),
            "messages": self.state.messages,
            "turn_count": self.state.turn_count,
            "total_input_tokens": self.state.total_input_tokens,
            "total_output_tokens": self.state.total_output_tokens,
            "thinking_content": self.state.thinking_content,
            "tool_executions": self.state.tool_executions,
        }
        path = self.SESSIONS_DIR / f"{self._session_id}.json"
        path.write_text(json.dumps(data, indent=2, default=str, ensure_ascii=False))

    def _load_session(self, session_id: str) -> bool:
        """Restore in-memory state from the saved record; return ``False`` if it can't be parsed.

        Interrupted tool calls in the saved history are sanitised via
        :meth:`_sanitize_resumed_messages`; the resulting stubs will be
        replayed by :meth:`_replay_pending_tool_calls` once the executor is up.
        """
        path = self.SESSIONS_DIR / f"{session_id}.json"
        if not path.exists():
            return False
        try:
            data = json.loads(path.read_text())
            self._session_id = data["session_id"]
            self._created_at = data.get("created_at", "")
            sanitized, replays = self._sanitize_resumed_messages(data.get("messages", []))
            self.state.messages = sanitized
            self._pending_resume_replays = replays
            self.state.turn_count = data.get("turn_count", 0)
            self.state.total_input_tokens = data.get("total_input_tokens", 0)
            self.state.total_output_tokens = data.get("total_output_tokens", 0)
            self.state.thinking_content = data.get("thinking_content", [])
            self.state.tool_executions = data.get("tool_executions", [])
            return True
        except (json.JSONDecodeError, KeyError):
            return False

    _RESUME_STUB_CONTENT = "[interrupted: pending replay]"

    @classmethod
    def _sanitize_resumed_messages(
        cls,
        messages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Repair an interrupted history so providers accept the resumed transcript.

        Ensures every ``tool_calls`` entry on an assistant message has a
        matching ``role="tool"`` reply before the next non-tool message.
        Missing replies are filled with stubs carrying
        :attr:`_RESUME_STUB_CONTENT`; :meth:`_replay_pending_tool_calls`
        later swaps them for real results. Orphan tool replies (no matching
        prior call) are dropped.

        Returns ``(sanitized_messages, pending_replays)`` where each pending
        replay is ``{"tool_call_id", "name", "arguments"}``.
        """
        if not messages:
            return [], []

        outstanding: dict[str, dict[str, str]] = {}
        repaired: list[dict[str, Any]] = []
        replays: list[dict[str, Any]] = []

        def _flush_outstanding() -> None:
            """Drain ``outstanding`` into repaired stubs + pending replays."""
            for tid, meta in list(outstanding.items()):
                repaired.append(
                    {
                        "role": "tool",
                        "tool_call_id": tid,
                        "content": cls._RESUME_STUB_CONTENT,
                    }
                )
                replays.append({"tool_call_id": tid, **meta})
            outstanding.clear()

        for msg in messages:
            role = msg.get("role")

            if role == "assistant":
                if outstanding:
                    _flush_outstanding()
                repaired.append(msg)
                for tc in msg.get("tool_calls") or []:
                    tid = tc.get("id") or ""
                    if not tid:
                        continue
                    fn = tc.get("function") or {}
                    name = fn.get("name", "") or tc.get("name", "")
                    raw_args = fn.get("arguments", "")
                    if not raw_args:
                        tc_input = tc.get("input")
                        if isinstance(tc_input, dict):
                            try:
                                raw_args = json.dumps(tc_input)
                            except Exception:
                                raw_args = ""
                        elif isinstance(tc_input, str):
                            raw_args = tc_input
                    outstanding[tid] = {"name": name, "arguments": raw_args or ""}
                continue

            if role == "tool":
                tid = msg.get("tool_call_id", "")
                if tid and tid in outstanding:
                    outstanding.pop(tid, None)
                    repaired.append(msg)
                continue

            if outstanding:
                _flush_outstanding()
            repaired.append(msg)

        if outstanding:
            _flush_outstanding()

        return repaired, replays

    def _replay_pending_tool_calls(self) -> None:
        """Re-run tool calls whose results were lost during the previous session.

        Walks :attr:`_pending_resume_replays`, invokes the live tool executor
        for each, and replaces the matching :attr:`_RESUME_STUB_CONTENT`
        message with the real result. Honours ``XERXES_NO_RESUME_REPLAY=1``
        (stubs stay in place — history is still structurally valid, just
        opaque). Per-call errors land as ``[replay error: ...]`` in the
        message so one failing tool can't poison the whole resume.
        """
        if os.environ.get("XERXES_NO_RESUME_REPLAY") == "1":
            self._pending_resume_replays = []
            return
        if self.tool_executor is None:
            return

        replays = self._pending_resume_replays
        self._pending_resume_replays = []

        if self._wire_mode:
            self._emit_wire_notification(
                notification_id=str(uuid.uuid4()),
                category="subagent",
                type_="resume_replay_begin",
                severity="info",
                title="",
                body=f"Replaying {len(replays)} interrupted tool call(s) from previous session…",
            )

        by_tid: dict[str, dict[str, Any]] = {}
        for entry in replays:
            tid = entry.get("tool_call_id") or ""
            if tid:
                by_tid[tid] = entry

        for msg in self.state.messages:
            if msg.get("role") != "tool":
                continue
            if msg.get("content") != self._RESUME_STUB_CONTENT:
                continue
            tid = msg.get("tool_call_id", "")
            entry = by_tid.get(tid)
            if entry is None:
                continue

            name = entry.get("name", "")
            raw_args = entry.get("arguments", "")
            try:
                tool_input = json.loads(raw_args) if isinstance(raw_args, str) and raw_args else (raw_args or {})
                if not isinstance(tool_input, dict):
                    tool_input = {"value": tool_input}
            except Exception as exc:
                msg["content"] = f"[replay error: invalid arguments — {exc}]"
                continue

            try:
                result = self.tool_executor(name, tool_input)
            except Exception as exc:
                msg["content"] = f"[replay error: {exc}]"
                continue

            msg["content"] = str(result) if result is not None else ""

            if self._wire_mode:
                self._emit_wire_notification(
                    notification_id=str(uuid.uuid4()),
                    category="subagent",
                    type_="resume_replay_done",
                    severity="info",
                    title="",
                    body=f"replayed {name}({tid[:12]}…) — {len(str(result or ''))} chars",
                )

    def _list_sessions(self) -> list[dict[str, Any]]:
        """Return saved sessions sorted by mtime (most recent first), each with a preview line."""
        sessions = []
        for path in sorted(self.SESSIONS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                data = json.loads(path.read_text())
                preview = ""
                for msg in data.get("messages", []):
                    if msg.get("role") == "user":
                        content = msg.get("content", "")
                        if isinstance(content, str):
                            preview = content[:60]
                            break
                sessions.append(
                    {
                        "session_id": data.get("session_id", path.stem),
                        "model": data.get("model", ""),
                        "cwd": data.get("cwd", ""),
                        "updated_at": data.get("updated_at", ""),
                        "turns": data.get("turn_count", 0),
                        "preview": preview,
                    }
                )
            except (json.JSONDecodeError, KeyError):
                continue
        return sessions

    def handle_init(self, params: dict[str, Any]) -> None:
        """Bootstrap the runtime and emit ``ready``/``init_done``.

        ``params`` may carry ``permission_mode``, ``verbose``, ``thinking``,
        ``debug``, ``mode``, ``model``, ``base_url``, ``api_key``,
        ``resume_session_id``, and ``mcp_servers``. The active provider
        profile fills any gaps, the bootstrap is run, tools and skills are
        loaded, agent definitions are resolved, and MCP servers (if any) are
        connected in background threads.
        """
        self.config = {
            "permission_mode": params.get("permission_mode", "auto"),
            "verbose": params.get("verbose", False),
            "thinking": params.get("thinking", True),
            "debug": params.get("debug", False),
            "mode": _normalize_interaction_mode(params.get("mode")),
            "plan_mode": False,
        }

        model = params.get("model", "")
        base_url = params.get("base_url", "")
        api_key = params.get("api_key", "")

        resume_id = params.get("resume_session_id") or ""
        if resume_id:
            self._load_session(resume_id)

        if not model and not base_url:
            profile = profiles.get_active_profile()
            if profile:
                model = profile.get("model", "")
                base_url = profile.get("base_url", "")
                api_key = profile.get("api_key", "")
                for k, v in profile.get("sampling", {}).items():
                    self.config[k] = v

        has_profile = bool(model)
        self.config["model"] = model if model else ""
        if base_url:
            self.config["base_url"] = base_url
        if api_key:
            self.config["api_key"] = api_key

        if base_url and not params.get("model", ""):
            try:
                available = profiles.fetch_models(base_url, api_key)
            except Exception:
                available = []
            self._auto_switch_stale_model(available)

        boot = bootstrap(model=self.config["model"])
        self.system_prompt = boot.system_prompt

        agent_skill = self._skill_registry.get("xerxes-agent")
        if agent_skill is not None:
            self.system_prompt += "\n\n" + agent_skill.to_prompt_section()

        registry = populate_registry()
        self.tool_executor = build_tool_executor(registry=registry)
        self.tool_schemas = registry.tool_schemas()

        try:
            from ..agents.definitions import (
                get_agent_definition,
                list_agent_definition_load_errors,
                list_agent_definitions,
            )

            agent_defs = list_agent_definitions()
            agent_errors = list_agent_definition_load_errors()
            self.config["_agent_definitions"] = [d.name for d in agent_defs]
            default_agent = get_agent_definition("default")
            if default_agent:
                self.tool_schemas = self._filter_tool_schemas_for_agent(self.tool_schemas, default_agent)
        except Exception as exc:
            agent_defs = []
            agent_errors = [f"Failed to load agent definitions: {exc}"]

        if self._wire_mode:
            for error in agent_errors:
                self._emit_wire_notification(
                    notification_id=str(uuid.uuid4()),
                    category="agents",
                    type_="agent_spec_error",
                    severity="error",
                    title="agent.yaml ignored",
                    body=error,
                )

        mcp_servers = params.get("mcp_servers", [])
        for server_config in mcp_servers:
            server_name = server_config.get("name", "unknown")
            if self._wire_mode:
                self._emit_wire_event(
                    "mcp_loading_begin",
                    {"server_name": server_name},
                )
            try:
                from ..mcp import MCPManager, MCPServerConfig

                manager = MCPManager()
                cfg = MCPServerConfig(
                    name=server_config.get("name", ""),
                    command=server_config.get("command", ""),
                    args=server_config.get("args", []),
                    env=server_config.get("env"),
                    url=server_config.get("url"),
                    transport=server_config.get("transport", "stdio"),
                    enabled=server_config.get("enabled", True),
                )

                def _connect(
                    _manager: MCPManager = manager,
                    _cfg: MCPServerConfig = cfg,
                ) -> None:
                    """Spin up a dedicated event loop and connect the MCP server."""
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(_manager.add_server(_cfg))
                    finally:
                        loop.close()

                import threading

                t = threading.Thread(target=_connect, daemon=True)
                t.start()
                success = True
            except Exception as exc:
                logger.warning("Failed to connect MCP server %s: %s", server_name, exc)
                success = False

            if self._wire_mode:
                self._emit_wire_event(
                    "mcp_loading_end",
                    {"server_name": server_name, "success": success},
                )

        self._initialized = True

        set_global_config(self.config)
        set_event_callback(self._on_agent_event)

        if self._pending_resume_replays:
            self._replay_pending_tool_calls()

        active_model = self.config.get("model", "")
        provider = detect_provider(active_model)
        skill_names = sorted(self._skill_registry.skill_names)

        self._emit_wire_init_done()

        if resume_id and self._wire_mode and self.state.messages:
            self._replay_history_to_wire()
        self._emit(
            "ready",
            {
                "model": active_model,
                "provider": provider,
                "tools": len(self.tool_schemas),
                "permission_mode": self.config["permission_mode"],
                "has_profile": has_profile,
                "skills": skill_names,
                "agents": [d.name for d in agent_defs],
            },
        )

    def handle_query(self, params: dict[str, Any], override_tool_schemas: list[dict[str, Any]] | None = None) -> None:
        """Run one streaming-loop turn against the model.

        ``params["text"]`` carries the user input (string or list of content
        parts). ``params["mode"]``/``params["plan_mode"]`` switch the
        interaction mode for this turn. If a skill scaffold is pending from
        ``/skill-create``, this turn generates the SKILL.md instead of
        running the agent. ``override_tool_schemas`` narrows the toolset
        exposed to the model (used when a skill suppresses ``SkillTool``).
        """
        if not self._initialized:
            self._emit_error("Not initialized. Send 'init' first.")
            return

        raw_text = params.get("text", "")
        if isinstance(raw_text, list):
            parts: list[str] = []
            for item in raw_text:
                if isinstance(item, dict):
                    parts.append(str(item.get("text", item)))
                else:
                    parts.append(str(item))
            raw_text = "\n".join(parts)
        text = str(raw_text).strip()
        if not text:
            self._emit_error("Empty query.")
            return

        if "plan_mode" in params:
            self.config["plan_mode"] = bool(params.get("plan_mode"))
            set_global_config(self.config)
        interaction_mode = _normalize_interaction_mode(
            params.get("mode", self.config.get("mode")),
            plan_mode=bool(self.config.get("plan_mode", False)),
        )
        self.config["mode"] = interaction_mode
        self.config["plan_mode"] = interaction_mode == "plan"
        set_global_config(self.config)

        if self._pending_skill_name:
            skill_name = self._pending_skill_name
            self._pending_skill_name = ""
            result = self._generate_skill(skill_name, text)
            self._emit("slash_result", {"output": result})
            self._emit("query_done", {})
            return

        self._cancel = False
        self._suppressing_tag = False
        self._suppress_buf.clear()
        self._pending_tool_inputs = None

        if self._wire_mode:
            self._maybe_compact_context_wire()
        else:
            self._maybe_compact_context()

        self._authoring_pipeline.begin_turn(
            agent_id="default",
            user_prompt=text,
        )

        if self._wire_mode:
            self._emit_wire_turn_begin(text)

        schemas = override_tool_schemas if override_tool_schemas is not None else self.tool_schemas
        system_prompt = self._system_prompt_for_mode(interaction_mode)
        mode_agent_def = self._agent_definition_for_mode(interaction_mode)
        if mode_agent_def is not None:
            schemas = self._filter_tool_schemas_for_agent(schemas, mode_agent_def)

        for event in run_agent_loop(
            user_message=text,
            state=self.state,
            config=self.config,
            system_prompt=system_prompt,
            tool_executor=self.tool_executor,
            tool_schemas=schemas,
            cancel_check=lambda: self._cancel,
        ):
            if self._cancel:
                break

            if isinstance(event, TextChunk):
                if self._wire_mode:
                    self._emit_wire_text(event.text)
                self._emit_text(event.text)

            elif isinstance(event, ThinkingChunk):
                if self._wire_mode:
                    self._emit_wire_think(event.text)
                self._emit("thinking_chunk", {"text": event.text})

            elif isinstance(event, ToolStart):
                self._suppressing_tag = False
                self._suppress_buf.clear()
                self._pending_tool_inputs = event.inputs
                if self._wire_mode:
                    self._emit_wire_tool_start(event.tool_call_id, event.name, event.inputs)
                self._emit(
                    "tool_start",
                    {
                        "name": event.name,
                        "inputs": event.inputs,
                        "tool_call_id": event.tool_call_id,
                    },
                )

            elif isinstance(event, ToolEnd):
                self._authoring_pipeline.record_call(
                    tool_name=event.name,
                    arguments=self._pending_tool_inputs or {},
                    status="success" if event.permitted else "blocked",
                    duration_ms=event.duration_ms,
                )
                self._pending_tool_inputs = None
                if self._wire_mode:
                    self._emit_wire_tool_result(
                        event.tool_call_id,
                        event.result,
                        event.permitted,
                        event.duration_ms,
                    )
                self._emit(
                    "tool_end",
                    {
                        "name": event.name,
                        "result": event.result,
                        "permitted": event.permitted,
                        "tool_call_id": event.tool_call_id,
                        "duration_ms": event.duration_ms,
                    },
                )

            elif isinstance(event, PermissionRequest):
                if self._wire_mode:
                    self._emit_wire_permission_request(self._current_tool_call_id, event.tool_name, event.description)

                    event.granted = self._wait_for_permission()
                else:
                    self._emit(
                        "permission_request",
                        {
                            "tool_name": event.tool_name,
                            "description": event.description,
                            "inputs": event.inputs,
                        },
                    )
                    event.granted = self._wait_for_permission()

            elif isinstance(event, TurnDone):
                self.cost_tracker.record_turn(
                    self.config.get("model", ""),
                    event.input_tokens,
                    event.output_tokens,
                )

                self._emit(
                    "turn_done",
                    {
                        "input_tokens": event.input_tokens,
                        "output_tokens": event.output_tokens,
                        "tool_calls_count": event.tool_calls_count,
                        "model": event.model,
                    },
                )

        final_response = ""
        for msg in reversed(self.state.messages):
            if msg.get("role") == "assistant":
                final_response = msg.get("content", "")
                break

        if not self._cancel:
            authoring_result = self._authoring_pipeline.on_turn_end(final_response=final_response)
            if authoring_result.authored and authoring_result.skill_path:
                self._emit(
                    "skill_suggested",
                    {
                        "skill_name": authoring_result.skill_name,
                        "version": authoring_result.version,
                        "source_path": str(authoring_result.skill_path),
                        "tool_count": len(authoring_result.candidate.events),
                        "unique_tools": authoring_result.candidate.unique_tools,
                    },
                )

        try:
            self._save_session()
        except Exception as exc:
            logger.warning("Failed to save session %s: %s", self._session_id, exc)

        if self._wire_mode:
            self._emit_wire_turn_end()
            self._emit_wire_status()

        self._emit("query_done", {})
        self._emit_state()

    @staticmethod
    def _filter_tool_schemas_for_agent(tool_schemas: list[dict[str, Any]], agent_def: Any) -> list[dict[str, Any]]:
        """Return ``tool_schemas`` restricted to ``agent_def``'s ``tools``/``allowed_tools``/``exclude_tools``."""
        names = {schema.get("name", "") for schema in tool_schemas}
        allowed = set(getattr(agent_def, "tools", None) or names)
        explicit_allowed = getattr(agent_def, "allowed_tools", None)
        if explicit_allowed:
            allowed &= set(explicit_allowed)
        allowed -= set(getattr(agent_def, "exclude_tools", None) or [])
        return [schema for schema in tool_schemas if schema.get("name", "") in allowed]

    def _maybe_compact_context(self) -> None:
        """Auto-compact older messages via an LLM summary when usage exceeds 75% of the limit.

        Legacy-protocol variant: emits a ``slash_result`` with the outcome.
        Falls back silently when the summariser call fails.
        """
        model = self.config.get("model", "")
        if not model:
            return
        context_limit = get_context_limit(model)
        if context_limit <= 0:
            return
        total_tokens = self.state.total_input_tokens + self.state.total_output_tokens
        threshold = int(context_limit * 0.75)
        if total_tokens < threshold:
            return

        messages = self.state.messages
        if len(messages) < 4:
            return

        system_msgs = [m for m in messages if m.get("role") == "system"]
        conv_msgs = [m for m in messages if m.get("role") != "system"]
        if len(conv_msgs) < 3:
            return

        preserve_recent = 2
        older = conv_msgs[:-preserve_recent]
        recent = conv_msgs[-preserve_recent:]

        conv_text = []
        for msg in older:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            if isinstance(content, list):
                parts = []
                for p in content:
                    if isinstance(p, dict):
                        parts.append(p.get("text", str(p)))
                    else:
                        parts.append(str(p))
                content = "\n".join(parts)
            if len(content) > 800:
                content = content[:800] + "..."
            conv_text.append(f"[{role}]: {content}")

        conversation = "\n\n".join(conv_text)

        try:
            from openai import OpenAI

            from ..llms.registry import PROVIDERS, get_api_key

            provider_name = detect_provider(model)
            api_key = self.config.get("api_key") or get_api_key(provider_name, self.config)
            prov = PROVIDERS.get(provider_name, PROVIDERS.get("openai"))
            base_url = (
                self.config.get("base_url")
                or self.config.get("custom_base_url")
                or (prov.base_url if prov else None)
                or "https://api.openai.com/v1"
            )
            client = OpenAI(api_key=api_key or "dummy", base_url=base_url)

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a conversation summarizer. Summarize the following conversation "
                            "into a concise summary that preserves all key information: decisions made, "
                            "files discussed, code changes, tool results, errors encountered and their "
                            "solutions, and any important context. Be factual and specific. "
                            "Output only the summary, no preamble."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Summarize this conversation ({len(older)} messages):\n\n{conversation}",
                    },
                ],
                max_tokens=8192,
                temperature=0.2,
            )

            summary = response.choices[0].message.content or ""
            if not summary.strip():
                return

        except Exception:
            logger.warning("Auto-compaction LLM call failed", exc_info=True)
            return

        len(self.state.messages)
        self.state.messages = [
            *system_msgs,
            {
                "role": "user",
                "content": f"[Previous conversation summary — {len(older)} messages compacted]\n\n{summary}",
            },
            *recent,
        ]
        new_count = len(self.state.messages)
        self._emit(
            "slash_result",
            {
                "output": (
                    f"[Auto-compact] Context at {total_tokens:,}/{context_limit:,} tokens. "
                    f"Summarized {len(older)} messages → kept {new_count} messages."
                ),
            },
        )

    def _maybe_compact_context_wire(self) -> None:
        """Wire-protocol twin of :meth:`_maybe_compact_context` (emits ``compaction_*`` events)."""
        model = self.config.get("model", "")
        if not model:
            return
        context_limit = get_context_limit(model)
        if context_limit <= 0:
            return
        total_tokens = self.state.total_input_tokens + self.state.total_output_tokens
        threshold = int(context_limit * 0.75)
        if total_tokens < threshold:
            return
        messages = self.state.messages
        if len(messages) < 4:
            return
        system_msgs = [m for m in messages if m.get("role") == "system"]
        conv_msgs = [m for m in messages if m.get("role") != "system"]
        if len(conv_msgs) < 3:
            return

        self._emit_wire_compaction_begin()

        preserve_recent = 2
        older = conv_msgs[:-preserve_recent]
        recent = conv_msgs[-preserve_recent:]

        conv_text = []
        for msg in older:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            if isinstance(content, list):
                parts = []
                for p in content:
                    if isinstance(p, dict):
                        parts.append(p.get("text", str(p)))
                    else:
                        parts.append(str(p))
                content = "\n".join(parts)
            if len(content) > 800:
                content = content[:800] + "..."
            conv_text.append(f"[{role}]: {content}")

        conversation = "\n\n".join(conv_text)

        try:
            from openai import OpenAI

            from ..llms.registry import PROVIDERS, get_api_key

            provider_name = detect_provider(model)
            api_key = self.config.get("api_key") or get_api_key(provider_name, self.config)
            prov = PROVIDERS.get(provider_name, PROVIDERS.get("openai"))
            base_url = (
                self.config.get("base_url")
                or self.config.get("custom_base_url")
                or (prov.base_url if prov else None)
                or "https://api.openai.com/v1"
            )
            client = OpenAI(api_key=api_key or "dummy", base_url=base_url)

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a conversation summarizer. Summarize the following conversation "
                            "into a concise summary that preserves all key information: decisions made, "
                            "files discussed, code changes, tool results, errors encountered and their "
                            "solutions, and any important context. Be factual and specific. "
                            "Output only the summary, no preamble."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Summarize this conversation ({len(older)} messages):\n\n{conversation}",
                    },
                ],
                max_tokens=8192,
                temperature=0.2,
            )

            summary = response.choices[0].message.content or ""
            if not summary.strip():
                self._emit_wire_compaction_end()
                return

        except Exception:
            self._emit_wire_compaction_end()
            return

        self.state.messages = [
            *system_msgs,
            {
                "role": "user",
                "content": f"[Previous conversation summary — {len(older)} messages compacted]\n\n{summary}",
            },
            *recent,
        ]

        self._emit_wire_compaction_end()
        self._emit_wire_notification(
            notification_id=str(uuid.uuid4()),
            category="context",
            type_="compacted",
            severity="info",
            title="Context compacted",
            body=f"Summarized {len(older)} messages → kept {len(self.state.messages)} messages.",
        )

    def _wait_for_permission(self) -> bool:
        """Block the query thread until the TUI answers; ``True`` means approved (incl. for-session)."""
        while True:
            if self._cancel:
                return False
            try:
                msg = self._permission_queue.get(timeout=0.1)
                params = msg.get("params", {})
                response = params.get("response")
                if response is not None:
                    return response in ("approve", "approve_for_session")
                return params.get("granted", False)
            except queue.Empty:
                continue

    def _ask_question(self, question: str) -> str:
        """Synchronously prompt the user for an answer; returns ``"[cancelled]"`` on cancel."""
        if self._wire_mode:
            self._emit_wire_question_request(
                [
                    {
                        "id": str(uuid.uuid4()),
                        "question": question,
                        "options": [],
                        "allow_free_form": True,
                    }
                ]
            )
        else:
            self._emit("question_request", {"question": question})
        return self._wait_for_question_response()

    def _ask_free_form(self, question: str) -> str:
        """Wire-protocol free-form prompt; returns the user's answer or ``"[cancelled]"``."""
        self._emit_wire_question_request(
            [
                {
                    "id": str(uuid.uuid4()),
                    "question": question,
                    "options": [],
                    "allow_free_form": True,
                }
            ]
        )
        return self._wait_for_question_response()

    def _provider_create_interactive(self) -> str:
        """Walk the user through the wire-mode ``/provider`` profile wizard."""
        name = self._ask_free_form("Profile name (e.g. 'kimi', 'openai-prod')").strip()
        if not name or name == "[cancelled]":
            return "Cancelled."
        if any(p.get("name") == name for p in profiles.list_profiles()):
            return f"Profile '{name}' already exists. Cancelled."

        base_url = self._ask_free_form("Base URL (e.g. https://api.openai.com/v1)").strip()
        if not base_url or base_url == "[cancelled]":
            return "Cancelled."

        api_key = self._ask_free_form("API key (leave blank if none)").strip()
        if api_key == "[cancelled]":
            return "Cancelled."

        model = self._ask_free_form("Default model name").strip()
        if not model or model == "[cancelled]":
            return "Cancelled."

        self.handle_provider_save({"name": name, "base_url": base_url, "api_key": api_key, "model": model})
        return f"Created and switched to profile '{name}'  (model: {model})"

    def _wait_for_question_response(self) -> str:
        """Block the query thread until the TUI returns an answer."""
        while True:
            if self._cancel:
                return "[cancelled]"
            try:
                msg = self._question_queue.get(timeout=0.1)
                params = msg.get("params", {})
                answers = params.get("answers")
                if isinstance(answers, dict):
                    return "\n".join(str(v) for v in answers.values())
                return params.get("answer", "")
            except queue.Empty:
                continue

    def handle_question_response(self, params: dict[str, Any]) -> None:
        """Enqueue a ``question_response`` so the blocked query thread can wake."""
        self._question_queue.put({"params": params})

    def handle_cancel(self) -> None:
        """Request cancellation of the in-flight query and cancel running sub-agents."""
        self._cancel = True
        try:
            from ..tools.claude_tools import _get_agent_manager

            _get_agent_manager().cancel_all()
        except Exception as exc:
            logger.warning("Failed to cancel sub-agents: %s", exc)

    def handle_cancel_all(self) -> None:
        """Cancel the active query plus every running sub-agent, surfacing the count."""
        self._cancel = True
        try:
            from ..tools.claude_tools import _get_agent_manager

            mgr = _get_agent_manager()
            n = mgr.cancel_all()
            if n:
                self._emit("slash_result", {"output": f"Cancelled {n} running sub-agent(s)."})
        except Exception as exc:
            logger.warning("Failed to cancel sub-agents: %s", exc)

    def handle_provider_list(self) -> None:
        """Emit ``provider_list`` carrying every stored profile."""
        plist = profiles.list_profiles()
        self._emit("provider_list", {"profiles": plist})

    def handle_fetch_models(self, params: dict[str, Any]) -> None:
        """Emit ``models_list`` for ``params["base_url"]`` using :func:`profiles.fetch_models`."""
        base_url = params.get("base_url", "")
        api_key = params.get("api_key", "")
        if not base_url:
            self._emit_error("base_url is required for fetch_models")
            return
        try:
            models = profiles.fetch_models(base_url, api_key)
        except Exception as exc:
            self._emit_error(f"Failed to fetch models: {exc}")
            return
        self._emit("models_list", {"models": models, "base_url": base_url})

    def _switch_model(self, model: str, *, persist_active_profile: bool = True) -> None:
        """Update the runtime model, optionally persist it on the active profile, refresh init."""
        self.config["model"] = model
        if persist_active_profile:
            profiles.update_active_model(model)
        set_global_config(self.config)
        self._emit("model_changed", {"model": model, "provider": detect_provider(model)})
        if self._wire_mode and self._initialized:
            self._emit_wire_init_done()
            self._emit_wire_status()

    def _auto_switch_stale_model(self, available: list[str]) -> str:
        """If the provider returns exactly one model and ours isn't in the list, switch to it.

        Returns the newly-active model id, or empty string when no switch occurred.
        """
        current = self.config.get("model", "")
        if len(available) != 1:
            return ""
        model = available[0]
        if current == model:
            return ""
        if current and current in available:
            return ""
        self._switch_model(model)
        return model

    def handle_provider_save(self, params: dict[str, Any]) -> None:
        """Persist a profile from ``params`` and activate it; emits ``provider_saved`` on success."""
        name = params.get("name", "")
        base_url = params.get("base_url", "")
        api_key = params.get("api_key", "")
        model = params.get("model", "")
        if not name or not base_url or not model:
            self._emit_error("name, base_url, and model are required")
            return

        provider = params.get("provider", "")
        profile = profiles.save_profile(
            name=name,
            base_url=base_url,
            api_key=api_key,
            model=model,
            provider=provider,
            set_active=True,
        )

        self.config["model"] = model
        self.config["base_url"] = base_url
        if api_key:
            self.config["api_key"] = api_key
        set_global_config(self.config)
        self._emit("model_changed", {"model": model, "provider": detect_provider(model)})
        if self._wire_mode and self._initialized:
            self._emit_wire_init_done()
            self._emit_wire_status()

        self._emit(
            "provider_saved",
            {
                "profile": profile,
                "message": f"Profile '{name}' saved and activated. Model: {model}",
            },
        )

    def handle_provider_select(self, params: dict[str, Any]) -> None:
        """Activate the profile named ``params["name"]`` and re-publish runtime config."""
        name = params.get("name", "")
        if not name:
            self._emit_error("Profile name is required")
            return
        if not profiles.set_active(name):
            self._emit_error(f"Profile '{name}' not found")
            return
        profile = profiles.get_active_profile()
        if profile:
            self.config["model"] = profile["model"]
            self.config["base_url"] = profile.get("base_url", "")
            if profile.get("api_key"):
                self.config["api_key"] = profile["api_key"]
            for k, v in profile.get("sampling", {}).items():
                self.config[k] = v
            set_global_config(self.config)
            self._emit(
                "provider_saved",
                {
                    "profile": profile,
                    "message": f"Switched to profile '{name}'. Model: {profile['model']}",
                },
            )

    def handle_provider_delete(self, params: dict[str, Any]) -> None:
        """Delete the named provider profile via :func:`profiles.delete_profile`."""
        name = params.get("name", "")
        if profiles.delete_profile(name):
            self._emit("slash_result", {"output": f"Profile '{name}' deleted."})
        else:
            self._emit_error(f"Profile '{name}' not found")

    def handle_slash(self, params: dict[str, Any]) -> None:
        """Dispatch ``params["command"]`` to the matching slash handler."""
        command = params.get("command", "").strip()
        if not command.startswith("/"):
            self._emit_error(f"Not a slash command: {command}")
            return

        parts = command[1:].split(None, 1)
        cmd = parts[0].lower() if parts else ""
        args = parts[1] if len(parts) > 1 else ""

        self._pending_skill_name = ""

        if self._wire_mode:
            if cmd in ("btw", "plan", "steer"):
                self._run_wire_slash(cmd, args)
                return

        if cmd == "skill":
            self._handle_skill_invoke(args)
            return

        output = self._run_slash(cmd, args)
        if output:
            self._emit("slash_result", {"output": output})

    def _run_wire_slash(self, cmd: str, args: str) -> None:
        """Execute one of the wire-mode-only slash commands (``btw``/``plan``/``steer``)."""
        if cmd == "btw":
            self._emit_wire_event("btw_begin", {})
            self._emit_wire_event("steer_input", {"content": args})

            self._pending_skill_name = ""
            self.handle_query({"text": args})
            self._emit_wire_event("btw_end", {})
        elif cmd == "plan":
            self._emit_wire_event("plan_display", {"content": "", "file_path": None})

            result = self._handle_plan(args)
            self._emit_wire_event(
                "plan_display",
                {"content": result, "file_path": None},
            )
        elif cmd == "steer":
            self._emit_wire_event("steer_input", {"content": args})

            self.state.messages.append({"role": "user", "content": args})

    def _handle_sampling(self, args: str) -> str:
        """Run ``/sampling``: view / set ``<param> <value>`` / ``reset`` / ``save`` to the active profile."""
        valid = profiles.SAMPLING_PARAMS

        if not args.strip():
            lines = ["Sampling parameters (current session):"]
            for k in sorted(valid):
                current_val = self.config.get(k, None)
                if current_val is not None:
                    lines.append(f"  {k}: {current_val}")
                else:
                    lines.append(f"  {k}: (default)")
            lines.append("")
            lines.append("Usage: /sampling <param> <value>")
            lines.append("       /sampling reset")
            lines.append("       /sampling save  (persist to active profile)")
            return "\n".join(lines)

        parts = args.strip().split(None, 1)
        subcmd = parts[0].lower()

        if subcmd == "reset":
            for k in valid:
                self.config.pop(k, None)
            set_global_config(self.config)
            return "Sampling parameters reset to defaults."

        if subcmd == "save":
            profile = profiles.get_active_profile()
            if not profile:
                return "No active profile. Run /provider first."
            sampling = {}
            for k in valid:
                if k in self.config:
                    sampling[k] = self.config[k]
            profiles.update_sampling(profile["name"], sampling)
            return f"Sampling parameters saved to profile '{profile['name']}'."

        if len(parts) != 2:
            return f"Usage: /sampling <param> <value>\nValid params: {', '.join(sorted(valid))}"

        param = subcmd
        val_str = parts[1]

        if param not in valid:
            return f"Unknown param: {param}\nValid: {', '.join(sorted(valid))}"

        try:
            if param in ("max_tokens", "top_k"):
                val: int | float = int(val_str)
            else:
                val = float(val_str)
        except ValueError:
            return f"Invalid value: {val_str}"

        self.config[param] = val
        set_global_config(self.config)
        return f"{param} = {val}"

    def _handle_compact(self) -> str:
        """Run ``/compact``: summarise older messages now via an LLM call."""
        messages = self.state.messages
        if len(messages) < 4:
            return "Nothing to compact (fewer than 4 messages)."

        model = self.config.get("model", "")
        if not model:
            return "No model configured. Run /provider first."

        system_msgs = [m for m in messages if m.get("role") == "system"]
        conv_msgs = [m for m in messages if m.get("role") != "system"]

        if len(conv_msgs) < 3:
            return "Nothing to compact."

        preserve_recent = 2
        older = conv_msgs[:-preserve_recent]
        recent = conv_msgs[-preserve_recent:]

        conv_text = []
        for msg in older:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            if isinstance(content, list):
                parts = []
                for p in content:
                    if isinstance(p, dict):
                        parts.append(p.get("text", str(p)))
                    else:
                        parts.append(str(p))
                content = "\n".join(parts)
            if len(content) > 500:
                content = content[:500] + "..."
            conv_text.append(f"[{role}]: {content}")

        conversation = "\n\n".join(conv_text)

        try:
            from openai import OpenAI

            from ..llms.registry import PROVIDERS, get_api_key

            provider_name = detect_provider(model)
            api_key = self.config.get("api_key") or get_api_key(provider_name, self.config)
            prov = PROVIDERS.get(provider_name, PROVIDERS.get("openai"))
            base_url = (
                self.config.get("base_url")
                or self.config.get("custom_base_url")
                or (prov.base_url if prov else None)
                or "https://api.openai.com/v1"
            )
            client = OpenAI(api_key=api_key or "dummy", base_url=base_url)

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a conversation summarizer. Summarize the following conversation "
                            "into a concise summary that preserves all key information: decisions made, "
                            "files discussed, code changes, tool results, and any important context. "
                            "Be factual and specific. Output only the summary, no preamble."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Summarize this conversation ({len(older)} messages):\n\n{conversation}",
                    },
                ],
                max_tokens=8192,
                temperature=0.2,
            )

            summary = response.choices[0].message.content or ""
            if not summary.strip():
                return "Compaction failed: LLM returned empty summary."

        except Exception as exc:
            return f"Compaction failed: {exc}"

        original_count = len(self.state.messages)
        self.state.messages = [
            *system_msgs,
            {
                "role": "user",
                "content": f"[Previous conversation summary — {len(older)} messages compacted]\n\n{summary}",
            },
            *recent,
        ]
        new_count = len(self.state.messages)

        return (
            f"Compacted {original_count} messages → {new_count} messages.\n"
            f"Summarized {len(older)} older messages, kept {len(recent)} recent + {len(system_msgs)} system."
        )

    def _handle_plan(self, args: str) -> str:
        """Run ``/plan`` by invoking :class:`PlanTool` with ``args`` as the objective."""
        objective = args.strip()
        if not objective:
            return "Usage: /plan <objective>\n\nExample: /plan refactor the auth module into separate files"

        from ..tools.claude_tools import PlanTool

        return PlanTool.static_call(objective=objective, execute=True)

    def _handle_agents_list(self) -> str:
        """Render the ``/agents`` output: registered definitions, load errors, and running sub-agents."""
        from ..agents.definitions import list_agent_definition_load_errors, list_agent_definitions
        from ..tools.claude_tools import _get_agent_manager

        defs = list_agent_definitions()
        lines = [f"Agent types ({len(defs)}):"]
        for d in defs:
            source_tag = f" [{d.source}]" if d.source != "built-in" else ""
            lines.append(f"  {d.name}{source_tag} — {d.description}")
        errors = list_agent_definition_load_errors()
        if errors:
            lines.append("\nAgent spec errors:")
            for error in errors:
                lines.append(f"  {error}")

        mgr = _get_agent_manager()
        tasks = mgr.list_tasks()
        if tasks:
            lines.append(f"\nRunning agents ({len(tasks)}):")
            for t in tasks:
                agent_type = f" ({t.agent_def_name})" if t.agent_def_name else ""
                lines.append(f"  {t.name}{agent_type} [{t.status}] — {t.prompt[:60]}")
        else:
            lines.append("\nNo running agents.")

        return "\n".join(lines)

    def _handle_skills_list(self) -> str:
        """Re-scan the skill directories and render the ``/skills`` output."""
        import xerxes as _xerxes_pkg

        _bundled = Path(_xerxes_pkg.__file__).parent / "skills"
        discover_dirs = [str(self._skills_dir), str(Path.cwd() / "skills")]
        if _bundled.is_dir():
            discover_dirs.insert(0, str(_bundled))
        self._skill_registry.discover(*discover_dirs)
        skills = self._skill_registry.get_all()
        if not skills:
            return f"No skills found.\n  Skills directory: {self._skills_dir}\n  Create one with /skill-create"
        lines = [f"Skills ({len(skills)}):"]
        for s in skills:
            tags = f" [{', '.join(s.metadata.tags)}]" if s.metadata.tags else ""
            lines.append(f"  {s.name}{tags} — {s.metadata.description or 'No description'}")
        lines.append("\nUse /skill <name> to invoke a skill")
        return "\n".join(lines)

    def _handle_skill_invoke(self, args: str) -> None:
        """Activate and run a skill from ``/skill <name>[:<args>]``.

        Validates the skill exists, matches the current platform, then injects
        the skill's prompt section into the conversation and dispatches a
        query against a filtered toolset that excludes ``SkillTool``.
        """
        name = args.strip()
        if not name:
            self._emit("slash_result", {"output": "Usage: /skill <name>\nUse /skills to list available skills."})
            return

        skill_args = ""
        if ":" in name:
            name, skill_args = name.split(":", 1)
            name = name.strip()
            skill_args = skill_args.strip()

        activate_skill(name)

        skill = self._skill_registry.get(name)
        if not skill:
            matches = self._skill_registry.search(name)
            if matches:
                suggestions = ", ".join(s.name for s in matches[:5])
                self._emit("slash_result", {"output": f"Skill '{name}' not found. Did you mean: {suggestions}"})
            else:
                self._emit(
                    "slash_result", {"output": f"Skill '{name}' not found. Use /skills to list available skills."}
                )
            return

        from xerxes.extensions.skills import skill_matches_platform

        if not skill_matches_platform(skill):
            self._emit(
                "slash_result",
                {"output": f"Skill '{name}' is not compatible with this platform ({__import__('sys').platform})."},
            )
            return

        from xerxes.extensions.skills import inject_skill_config

        prompt_section = skill.to_prompt_section()
        config_block = inject_skill_config(skill)
        skill_message = f"[Skill '{name}' activated]{config_block}\n\n{prompt_section}"

        self.state.messages.append(
            {
                "role": "user",
                "content": skill_message,
            }
        )

        self._emit("slash_result", {"output": f"Running skill '{name}'..."})

        trigger = skill_args if skill_args else f"Execute the '{name}' skill now."
        filtered_schemas = [s for s in self.tool_schemas if s.get("name") != "SkillTool"]
        self.handle_query({"text": trigger}, override_tool_schemas=filtered_schemas)

    def _handle_skill_create(self, args: str) -> str:
        """Stage ``/skill-create <name>``: the next user message becomes the description."""
        name = args.strip()
        if not name:
            return (
                "Usage: /skill-create <name>\n"
                "  Example: /skill-create code-review\n\n"
                "After entering the name, describe what the skill should do\n"
                "and the SKILL.md will be auto-generated."
            )

        if not all(c.isalnum() or c in "-_" for c in name):
            return f"Invalid skill name '{name}'. Use only letters, numbers, hyphens, and underscores."

        skill_dir = self._skills_dir / name
        if skill_dir.exists():
            return f"Skill '{name}' already exists at {skill_dir}"

        self._pending_skill_name = name
        return f"Creating skill '{name}'. Describe what this skill should do:"

    def _generate_skill(self, name: str, description: str) -> str:
        """Ask the active model to author ``SKILL.md`` for ``name``; fall back to the static template."""
        model = self.config.get("model", "")
        if not model:
            return self._create_skill_template(name, description)

        try:
            from openai import OpenAI

            from ..llms.registry import PROVIDERS, get_api_key

            provider_name = detect_provider(model)
            api_key = self.config.get("api_key") or get_api_key(provider_name, self.config)
            prov = PROVIDERS.get(provider_name, PROVIDERS.get("openai"))
            base_url = (
                self.config.get("base_url")
                or self.config.get("custom_base_url")
                or (prov.base_url if prov else None)
                or "https://api.openai.com/v1"
            )
            client = OpenAI(api_key=api_key or "dummy", base_url=base_url)

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You generate SKILL.md files for the Xerxes agent framework. "
                            "A skill is a reusable set of instructions that an agent follows "
                            "when the skill is invoked via `/skill <name>`.\n\n"
                            "Output format (YAML frontmatter + markdown body):\n"
                            "```\n"
                            "---\n"
                            "name: skill-name\n"
                            "description: One-line description\n"
                            'version: "1.0"\n'
                            "tags: [tag1, tag2]\n"
                            "---\n\n"
                            "# Skill Title\n\n"
                            "Detailed step-by-step instructions for the agent...\n"
                            "```\n\n"
                            "Write clear, actionable instructions. Be specific about what "
                            "tools to use, what to check, and what format to output. "
                            "Output ONLY the SKILL.md content, nothing else."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Create a skill named '{name}' that does the following:\n\n{description}",
                    },
                ],
                max_tokens=2048,
                temperature=0.3,
            )

            content = response.choices[0].message.content or ""

            if content.startswith("```"):
                lines = content.split("\n")
                if lines[0].strip().startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                content = "\n".join(lines)

            if not content.strip():
                return self._create_skill_template(name, description)

        except Exception as exc:
            return self._create_skill_template(name, description, error=str(exc))

        skill_dir = self._skills_dir / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text(content, encoding="utf-8")

        self._skill_registry.discover(str(self._skills_dir))
        self._emit("skills_updated", {"skills": sorted(self._skill_registry.skill_names)})

        return f"Skill '{name}' generated and saved to {skill_dir}/SKILL.md\nUse /skill {name} to invoke it."

    def _create_skill_template(self, name: str, description: str, error: str = "") -> str:
        """Write a minimal ``SKILL.md`` scaffold when LLM generation is unavailable or fails."""
        title = name.replace("-", " ").replace("_", " ").title()
        skill_dir = self._skills_dir / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text(
            f"---\n"
            f"name: {name}\n"
            f"description: {description[:80]}\n"
            f'version: "1.0"\n'
            f"tags: []\n"
            f"---\n\n"
            f"# {title}\n\n"
            f"{description}\n",
            encoding="utf-8",
        )
        self._skill_registry.discover(str(self._skills_dir))
        self._emit("skills_updated", {"skills": sorted(self._skill_registry.skill_names)})
        err_note = f"\n(LLM generation failed: {error}. Created template instead.)" if error else ""
        return f"Skill '{name}' created at {skill_dir}/SKILL.md{err_note}\nUse /skill {name} to invoke it."

    def _run_slash(self, cmd: str, args: str) -> str:
        """Dispatch a leading-slash command to the matching built-in handler.

        Returns the rendered output text, or an unknown-command message when
        ``cmd`` is neither a built-in nor a registered skill.
        """
        if cmd in ("help", "h"):
            return (
                "Commands:\n"
                "  /provider          Setup or switch provider profile\n"
                "  /sampling          View or set sampling parameters\n"
                "  /compact           Summarize conversation to free context\n"
                "  /plan OBJECTIVE    Plan and execute a multi-step task\n"
                "  /agents            List agent types and running agents\n"
                "  /skills            List available skills\n"
                "  /skill NAME        Invoke a skill by name\n"
                "  /skill-create      Create a new skill\n"
                "  /model NAME        Switch model\n"
                "  /cost              Show cost summary\n"
                "  /context           Show context info\n"
                "  /clear             Clear conversation\n"
                "  /tools             List available tools\n"
                "  /thinking          Toggle thinking display\n"
                "  /verbose           Toggle verbose mode\n"
                "  /debug             Toggle debug mode\n"
                "  /permissions       Cycle permission mode\n"
                "  /yolo              Toggle accept-all permission mode\n"
                "  /config            Show config\n"
                "  /history           Show message count\n"
                "  /exit              Exit"
            )

        if cmd == "model":
            if args:
                self._switch_model(args)
                return f"Model set to: {args}"

            current = self.config.get("model", "(none)")
            base_url = self.config.get("base_url", "")
            api_key = self.config.get("api_key", "")
            lines = [f"Current model: {current}"]
            if base_url:
                try:
                    available = profiles.fetch_models(base_url, api_key)
                except Exception as exc:
                    lines.append(f"Could not fetch models from {base_url}/models: {exc}")
                    return "\n".join(lines)
                if available:
                    switched = self._auto_switch_stale_model(available)
                    if switched:
                        previous = current
                        current = switched
                        lines[0] = f"Current model: {current}"
                        lines.append(f"Switched from unavailable model '{previous}' to '{current}'.")
                    lines.append(f"\nAvailable models ({len(available)}):")
                    for m in available:
                        marker = " (active)" if m == current else ""
                        lines.append(f"  {m}{marker}")
                    lines.append("\nUse /model <name> to switch")
                else:
                    lines.append(f"No models returned from {base_url}/models")
            return "\n".join(lines)

        if cmd == "cost":
            return self.cost_tracker.summary()

        if cmd == "history":
            return f"{len(self.state.messages)} messages, {self.state.turn_count} turns"

        if cmd == "verbose":
            self.config["verbose"] = not self.config.get("verbose", False)
            return f"Verbose: {self.config['verbose']}"

        if cmd == "thinking":
            self.config["thinking"] = not self.config.get("thinking", False)
            return f"Thinking: {self.config['thinking']}"

        if cmd == "sampling":
            return self._handle_sampling(args)

        if cmd == "compact":
            return self._handle_compact()

        if cmd == "skills":
            return self._handle_skills_list()

        if cmd == "skill-create":
            return self._handle_skill_create(args)

        if cmd == "debug":
            self.config["debug"] = not self.config.get("debug", False)
            return f"Debug: {self.config['debug']}"

        if cmd == "clear":
            self.state.messages.clear()
            self.state.thinking_content.clear()
            self.state.tool_executions.clear()
            self.state.turn_count = 0
            return "Conversation cleared."

        if cmd == "context":
            model = self.config.get("model", "")
            provider = detect_provider(model)
            cost = calc_cost(model, self.state.total_input_tokens, self.state.total_output_tokens)
            return (
                f"CWD: {os.getcwd()}\n"
                f"Model: {model}\n"
                f"Provider: {provider}\n"
                f"Turns: {self.state.turn_count}\n"
                f"Messages: {len(self.state.messages)}\n"
                f"Tokens: {self.state.total_input_tokens} in / {self.state.total_output_tokens} out\n"
                f"Cost: ${cost:.4f}"
            )

        if cmd == "config":
            lines = [f"  {k}: {v}" for k, v in sorted(self.config.items()) if not k.startswith("_")]
            return "\n".join(lines) if lines else "(empty config)"

        if cmd == "permissions":
            modes = ["auto", "accept-all", "manual"]
            current = self.config.get("permission_mode", "auto")
            idx = modes.index(current) if current in modes else 0
            new_mode = modes[(idx + 1) % len(modes)]
            self.config["permission_mode"] = new_mode
            set_global_config(self.config)
            return f"Permission mode: {new_mode}"

        if cmd == "yolo":
            current = self.config.get("permission_mode", "auto")
            if current == "accept-all":
                self.config["permission_mode"] = "auto"
            else:
                self.config["permission_mode"] = "accept-all"
            set_global_config(self.config)
            return f"YOLO mode {'OFF (auto)' if current == 'accept-all' else 'ON (accept-all)'}"

        if cmd == "tools":
            registry = populate_registry()
            lines = []
            for entry in registry.list_tools():
                safe = " [safe]" if entry.safe else ""
                lines.append(f"  {entry.name}{safe} -- {entry.description[:60]}")
            lines.append(f"  ({registry.tool_count} total)")
            return "\n".join(lines)

        if cmd == "plan":
            return self._handle_plan(args)

        if cmd == "agents":
            return self._handle_agents_list()

        if cmd == "provider":
            plist = profiles.list_profiles()
            active = profiles.get_active_profile()
            active_name = active.get("name") if active else ""
            if not plist:
                return (
                    "No provider profiles configured.\n"
                    "Add one with the JSON-RPC `provider_save` method, or set\n"
                    "the env vars: XERXES_BASE_URL, XERXES_API_KEY, XERXES_MODEL."
                )

            if args:
                self.handle_provider_select({"name": args.strip()})
                return ""

            if self._wire_mode:
                NEW = "+ Create new profile…"
                options = [
                    {
                        "label": p.get("name", ""),
                        "description": (
                            f"{p.get('model', '?')} @ {p.get('base_url', '')}"
                            + ("  (active)" if p.get("name") == active_name else "")
                        ),
                    }
                    for p in plist
                ]
                options.append({"label": NEW, "description": "Add a new provider profile"})
                self._emit_wire_question_request(
                    [
                        {
                            "id": "provider",
                            "question": "Pick a provider profile",
                            "options": options,
                            "allow_free_form": False,
                        }
                    ]
                )
                answer = self._wait_for_question_response()
                if not answer or answer == "[cancelled]":
                    return "Cancelled."
                if answer == NEW:
                    return self._provider_create_interactive()
                self.handle_provider_select({"name": answer})
                profile = profiles.get_active_profile()
                if profile and profile.get("name") == answer:
                    return f"Switched to '{answer}'  (model: {profile.get('model', '?')})"
                return f"Could not switch to '{answer}'."

            lines = ["Provider profiles:"]
            for p in plist:
                marker = "*" if p.get("name") == active_name else " "
                lines.append(f"  {marker} {p.get('name'):20s}  {p.get('model', '?')}  ({p.get('base_url', '')})")
            lines.append("\n* = active. Pass a profile name to switch: /provider NAME")
            return "\n".join(lines)

        if cmd in ("exit", "quit", "q"):
            self._emit("exit", {})
            sys.exit(0)

        skill = self._skill_registry.get(cmd)
        if skill:
            full_args = f"{cmd}:{args}" if args else cmd
            self._handle_skill_invoke(full_args)
            return ""
        return f"Unknown command: /{cmd} (type /help)"

    def _parse_json_messages(self, line: str) -> list[dict[str, Any]]:
        """Parse one or more JSON objects from one stdin line (fall back to whitespace-split)."""
        try:
            return [json.loads(line)]
        except json.JSONDecodeError:
            pass

        messages = []
        for token in line.split():
            if not token.strip():
                continue
            try:
                messages.append(json.loads(token))
            except json.JSONDecodeError:
                self._emit_error(f"Invalid JSON: {token[:100]}")
        return messages

    def run(self) -> None:
        """Read JSON-RPC requests from stdin until shutdown, dispatching each method."""
        for raw_line in sys.stdin:
            line = raw_line.strip()
            if not line or not self._running:
                continue

            messages = self._parse_json_messages(line)
            if not messages:
                continue

            for msg in messages:
                method = msg.get("method", "")
                params = msg.get("params", {})

                try:
                    if method in ("init", "initialize"):
                        self.handle_init(params)
                    elif method in ("query", "prompt"):
                        if self._query_thread is not None and self._query_thread.is_alive():
                            self._emit_error("A query is already running. Wait or send cancel.")
                        else:

                            def _run_query(_params: dict[str, Any]) -> None:
                                """Worker-thread body: run :meth:`handle_query` and surface any error."""
                                try:
                                    if "user_input" in _params and "text" not in _params:
                                        _params = {**_params, "text": _params.get("user_input", "")}
                                    self.handle_query(_params)
                                except Exception as exc:
                                    self._emit_error(f"{type(exc).__name__}: {exc}")

                                    if self._wire_mode:
                                        self._emit_wire_turn_end()
                                        self._emit_wire_status()
                                    self._emit("query_done", {})

                            self._query_thread = threading.Thread(target=_run_query, args=(params,), daemon=True)
                            self._query_thread.start()
                    elif method == "permission_response":
                        self._permission_queue.put(msg)
                    elif method == "question_response":
                        self.handle_question_response(params)
                    elif method == "steer":
                        content = params.get("user_input", params.get("content", ""))
                        self._run_wire_slash("steer", str(content))
                    elif method == "set_plan_mode":
                        enabled = bool(params.get("enabled", params.get("plan_mode", False)))
                        self.config["plan_mode"] = enabled
                        self.config["mode"] = _normalize_interaction_mode(
                            params.get("mode", self.config.get("mode")), plan_mode=enabled
                        )
                        set_global_config(self.config)
                        if self._wire_mode:
                            self._emit_wire_status()
                    elif method == "set_mode":
                        mode = _normalize_interaction_mode(params.get("mode"), plan_mode=False)
                        self.config["mode"] = mode
                        self.config["plan_mode"] = mode == "plan"
                        set_global_config(self.config)
                        if self._wire_mode:
                            self._emit_wire_status()
                    elif method == "replay":
                        if self._wire_mode:
                            self._emit_wire_notification(
                                notification_id=str(uuid.uuid4()),
                                category="session",
                                type_="replay_unavailable",
                                severity="warning",
                                title="Replay unavailable",
                                body="This Xerxes bridge does not persist Kimi wire replay records yet.",
                            )
                    elif method == "cancel":
                        self.handle_cancel()
                    elif method == "cancel_all":
                        self.handle_cancel_all()
                    elif method == "slash":
                        if self._query_thread is not None and self._query_thread.is_alive():
                            self._emit_error("A query is already running. Wait or send cancel.")
                        else:

                            def _run_slash(_params: dict[str, Any]) -> None:
                                """Worker-thread body: run :meth:`handle_slash` and surface any error."""
                                try:
                                    self.handle_slash(_params)
                                except Exception as exc:
                                    self._emit_error(f"{type(exc).__name__}: {exc}")
                                    if self._wire_mode:
                                        self._emit_wire_status()
                                    self._emit("query_done", {})

                            self._query_thread = threading.Thread(target=_run_slash, args=(params,), daemon=True)
                            self._query_thread.start()
                    elif method == "provider_list":
                        self.handle_provider_list()
                    elif method == "fetch_models":
                        self.handle_fetch_models(params)
                    elif method == "provider_save":
                        self.handle_provider_save(params)
                    elif method == "provider_select":
                        self.handle_provider_select(params)
                    elif method == "provider_delete":
                        self.handle_provider_delete(params)
                    elif method == "shutdown":
                        self._running = False
                    else:
                        self._emit_error(f"Unknown method: {method}")
                except Exception as exc:
                    self._emit_error(f"{type(exc).__name__}: {exc}")


def main() -> None:
    """Console-script entry point: parse flags, build :class:`BridgeServer`, run it on stdin/stdout."""
    import signal

    signal.signal(signal.SIGINT, signal.SIG_IGN)

    parser = argparse.ArgumentParser(description="Xerxes bridge server (JSON-RPC over stdio)")
    parser.add_argument("-m", "--model", default=None, help="Model name")
    parser.add_argument("--base-url", default=None, help="API base URL")
    parser.add_argument("--api-key", default=None, help="API key")
    parser.add_argument(
        "--wire",
        action="store_true",
        help=("Use Kimi-Code-compatible wire protocol (JSON-RPC 2.0 NDJSON) instead of the legacy flat-event protocol."),
    )
    args = parser.parse_args()

    server = BridgeServer(wire_mode=args.wire)

    if args.model:
        init_params: dict[str, Any] = {"model": args.model}
        if args.base_url:
            init_params["base_url"] = args.base_url
        if args.api_key:
            init_params["api_key"] = args.api_key
        server.handle_init(init_params)

    server.run()


if __name__ == "__main__":
    main()
