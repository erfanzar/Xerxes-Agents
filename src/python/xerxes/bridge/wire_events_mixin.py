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
"""Wire-format event emission for the bridge server.

Extracted from bridge/server.py as a mixin.
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any

from ..llms.registry import get_context_limit
from ..streaming.wire_events import to_kimi_event_name


class WireEventMixin:
    """Methods that emit structured wire events to the connected client."""

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

    def _emit_wire_tool_args_part(self, arguments_part: str) -> None:
        """Emit a streaming ``tool_call_part`` delta with an argument fragment."""
        self._emit_wire_event("tool_call_part", {"arguments_part": arguments_part})

    def _emit_wire_turn_end(self) -> None:
        """Mark the end of the current turn on the wire."""
        self._emit_wire_event("turn_end", {})

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

    def _emit_wire_compaction_begin(self) -> None:
        """Signal the TUI to render its compaction-in-progress indicator."""
        self._emit_wire_event("compaction_begin", {})

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

    def _emit_wire_step_begin(self, n: int) -> None:
        """Mark the start of step ``n`` and remember it for subsequent events."""
        self._emit_wire_event("step_begin", {"n": n})
        self._step_count = n

    def _emit_wire_turn_begin(self, user_input: str) -> None:
        """Mark the start of a new turn carrying ``user_input`` as a text part."""
        self._emit_wire_event("turn_begin", {"user_input": [{"type": "text", "text": user_input}]})
        self._step_count = 0

    def _emit_wire_compaction_end(self) -> None:
        """Signal the TUI to clear the compaction indicator."""
        self._emit_wire_event("compaction_end", {})

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

    def _emit_wire_think(self, think: str) -> None:
        """Emit one ``think_part`` event carrying a thinking-stream chunk."""
        self._emit_wire_event("think_part", {"think": think})

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
