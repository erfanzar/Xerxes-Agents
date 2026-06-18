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
"""Bridge the ACP server surface to Xerxes's real streaming agent loop.

:class:`AcpAgentRunner` bootstraps a :class:`RuntimeManager` (provider,
system prompt, tools, skills — exactly what the daemon uses) and turns an
ACP ``prompt`` into a real turn through :func:`xerxes.streaming.loop.run`.
Each :class:`AcpSession` keeps its own :class:`AgentState`, so multi-turn
conversations persist per session. Stream events are converted to ACP wire
events and pushed through the caller-supplied ``emit`` callback; tool
approvals route through the server's :class:`AcpPermissionBoard`."""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Callable
from typing import Any

from ..streaming.events import AgentState, PermissionRequest, TurnDone
from ..streaming.loop import run as run_loop
from .events import to_acp_event
from .permissions import AcpPermissionBoard, route_permission
from .session import AcpSession

logger = logging.getLogger(__name__)

EmitFn = Callable[[dict[str, Any]], None]


class AcpAgentRunner:
    """Drive real agent turns for the ACP server.

    Holds one bootstrapped :class:`RuntimeManager` for the process and a
    per-session :class:`AgentState`. ``permission_board`` is wired by the
    server so interactive approvals can block until the client responds."""

    def __init__(self, *, default_permission_mode: str = "auto") -> None:
        """Bootstrap the runtime once (provider/tools/skills) from the active profile."""
        from ..daemon.config import load_config
        from ..daemon.runtime import RuntimeManager

        self._runtime = RuntimeManager(config=load_config())
        self._runtime.reload()
        self._states: dict[str, AgentState] = {}
        self._default_permission_mode = default_permission_mode
        self.permission_board: AcpPermissionBoard | None = None
        # Interactive ``AskUserQuestion`` routing: a question board + per-thread
        # turn context so the (global, blocking) tool callback can find the
        # current session/emit and surface the question to the ACP client.
        self._questions: dict[str, dict[str, Any]] = {}
        self._q_lock = threading.Lock()
        self._ctx = threading.local()

    # ----- list providers (wired into AcpServer) ---------------------------

    def list_tools(self) -> list[dict[str, Any]]:
        """Return the live OpenAI-shaped tool schemas."""
        return list(self._runtime.tool_schemas)

    def list_models(self) -> list[dict[str, Any]]:
        """Return available model ids (provider catalogue, else the active model)."""
        from ..bridge import profiles

        cfg = self._runtime.runtime_config
        base_url = str(cfg.get("base_url", ""))
        models: list[str] = []
        if base_url:
            try:
                models = profiles.fetch_models(base_url, str(cfg.get("api_key", "")))
            except Exception as exc:
                logger.debug("fetch_models failed: %s", exc)
        if not models and self._runtime.model:
            models = [self._runtime.model]
        return [{"id": m, "name": m} for m in models]

    @property
    def active_model(self) -> str:
        """The runtime's active model id."""
        return self._runtime.model

    # ----- prompt execution -------------------------------------------------

    def run_prompt(self, *, session: AcpSession, text: str, emit: EmitFn | None = None, **_: Any) -> dict[str, Any]:
        """Run one turn for ``session`` and stream ACP events via ``emit``.

        Returns a summary dict (token counts / stop reason) once the turn
        ends. ``emit`` receives already-wire-shaped ACP event dicts; when
        omitted (pure-logic use) events are simply not streamed."""
        emit = emit or (lambda _event: None)
        state = self._states.setdefault(session.session_id, AgentState(messages=[]))

        config = dict(self._runtime.runtime_config)
        config["permission_mode"] = session.metadata.get("permission_mode", self._default_permission_mode)
        if session.model_override:
            config["model"] = session.model_override

        # Expose this turn's session/emit to the (global, blocking) AskUserQuestion
        # callback running on this same thread during tool execution.
        self._ctx.session = session
        self._ctx.emit = emit
        summary: dict[str, Any] = {"ok": True, "cancelled": False}
        try:
            for event in run_loop(
                user_message=text,
                state=state,
                config=config,
                system_prompt=self._runtime.system_prompt,
                tool_executor=self._runtime.tool_executor,
                tool_schemas=self._runtime.tool_schemas,
                cancel_check=lambda: session.cancelled,
            ):
                if isinstance(event, PermissionRequest):
                    event.granted = self._resolve_permission(session, event, emit)
                    # Still surface the request to the client for visibility.
                    continue
                emit(to_acp_event(event).to_wire())
                if isinstance(event, TurnDone):
                    summary.update(
                        {
                            "input_tokens": event.input_tokens,
                            "output_tokens": event.output_tokens,
                            "tool_calls_count": event.tool_calls_count,
                            "model": event.model,
                        }
                    )
        except Exception as exc:
            logger.warning("ACP prompt failed for session %s: %s", session.session_id, exc)
            return {"ok": False, "error": str(exc)}
        finally:
            self._ctx.session = None
            self._ctx.emit = None
        summary["cancelled"] = session.cancelled
        return summary

    def _resolve_permission(self, session: AcpSession, event: PermissionRequest, emit: EmitFn) -> bool:
        """Surface a permission request to the client and block until decided.

        Auto-denies on cancellation. Without a wired board (pure-logic mode)
        the request is auto-denied so the turn can't hang."""
        board = self.permission_board
        req = route_permission(
            session_id=session.session_id,
            tool_name=event.tool_name,
            description=event.description,
            inputs=event.inputs,
        )
        emit(
            {
                "kind": "permission_request",
                "permission_id": req.id,
                "session_id": session.session_id,
                "tool_name": event.tool_name,
                "description": event.description,
                "inputs": event.inputs,
            }
        )
        if board is None:
            return False
        waiter = board.submit(req)
        while not waiter.wait(timeout=0.2):
            if session.cancelled:
                board.drop(req.id)
                return False
        allowed = req.allowed
        board.drop(req.id)
        return bool(allowed and not session.cancelled)

    # ----- interactive AskUserQuestion (routed to the ACP client) ----------

    def ask_user_question(self, question: str) -> str:
        """Blocking callback for ``AskUserQuestionTool`` — ask the ACP client.

        Wired via ``set_ask_user_question_callback``. Runs on the prompt worker
        thread during tool execution, so it reads this turn's session/emit from
        the thread-local context, surfaces an ``input_request`` to the client,
        and blocks until ``respond_question`` resolves it (or the session is
        cancelled). Raises if no turn is active."""
        session = getattr(self._ctx, "session", None)
        emit = getattr(self._ctx, "emit", None)
        if session is None or emit is None:
            raise RuntimeError("AskUserQuestion called outside an active ACP turn")

        qid = uuid.uuid4().hex
        event = threading.Event()
        with self._q_lock:
            self._questions[qid] = {"event": event, "answer": "", "question": question, "session_id": session.session_id}
        emit(
            {
                "kind": "input_request",
                "input_id": qid,
                "session_id": session.session_id,
                "question": question,
            }
        )
        while not event.wait(timeout=0.2):
            if session.cancelled:
                with self._q_lock:
                    self._questions.pop(qid, None)
                return ""
        with self._q_lock:
            entry = self._questions.pop(qid, None)
        return str(entry["answer"]) if entry else ""

    def respond_question(self, input_id: str, answer: str) -> dict[str, Any]:
        """Resolve a pending ``input_request`` with the client's ``answer``."""
        with self._q_lock:
            entry = self._questions.get(input_id)
            if entry is None or entry["event"].is_set():
                return {"ok": False}
            entry["answer"] = answer
            entry["event"].set()
        return {"ok": True}

    def pending_questions(self) -> list[dict[str, Any]]:
        """Snapshot of input requests awaiting a client answer."""
        with self._q_lock:
            return [
                {"input_id": qid, "session_id": e["session_id"], "question": e["question"]}
                for qid, e in self._questions.items()
                if not e["event"].is_set()
            ]

    def reset_session(self, session_id: str) -> None:
        """Drop the cached conversation state for ``session_id``."""
        self._states.pop(session_id, None)


__all__ = ["AcpAgentRunner"]
