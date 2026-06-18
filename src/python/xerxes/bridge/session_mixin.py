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
"""Session persistence and history replay.

Extracted from bridge/server.py as a mixin.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime
from typing import Any


class SessionMixin:
    """Session save/load, message sanitization, and history replay."""

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
