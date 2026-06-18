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
"""Self-contained stdio JSON-RPC 2.0 transport for the ACP server.

Newline-delimited JSON (one object per line) over stdin/stdout — no third-party
dependency required, so ``xerxes-acp`` always works. Requests dispatch to
:class:`AcpServer` methods; ``prompt`` runs on a worker thread so ``cancel`` and
``respond_permission`` can be processed *while a prompt streams*. Streamed agent
events are delivered as ``session/update`` notifications carrying the ACP wire
event; the prompt's final summary is returned as the JSON-RPC response.

Method aliases accept both the ``snake_case`` core names and ``slash/`` forms.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
from typing import IO, Any

from .runner import AcpAgentRunner
from .server import AcpServer

logger = logging.getLogger(__name__)

# JSON-RPC 2.0 error codes.
_PARSE_ERROR = -32700
_INVALID_REQUEST = -32600
_METHOD_NOT_FOUND = -32601
_INTERNAL_ERROR = -32603


class StdioJsonRpcServer:
    """Drive an :class:`AcpServer` over newline-delimited JSON-RPC on stdio."""

    def __init__(
        self,
        server: AcpServer,
        runner: AcpAgentRunner | None = None,
        *,
        stdin: IO[str] | None = None,
        stdout: IO[str] | None = None,
    ) -> None:
        """Bind the server/runner and the stdio streams (defaults to real stdio)."""
        self._server = server
        self._runner = runner
        self._stdin = stdin or sys.stdin
        self._stdout = stdout or sys.stdout
        self._write_lock = threading.Lock()
        self._workers: set[threading.Thread] = set()
        self._running = True

    # ----- wire helpers -----------------------------------------------------

    def _send(self, obj: dict[str, Any]) -> None:
        """Serialise ``obj`` as one JSON line to stdout (thread-safe)."""
        line = json.dumps(obj, default=str)
        with self._write_lock:
            self._stdout.write(line + "\n")
            self._stdout.flush()

    def _result(self, req_id: Any, result: Any) -> None:
        self._send({"jsonrpc": "2.0", "id": req_id, "result": result})

    def _error(self, req_id: Any, code: int, message: str) -> None:
        self._send({"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}})

    def _notify(self, method: str, params: dict[str, Any]) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    # ----- main loop --------------------------------------------------------

    def serve_forever(self) -> None:
        """Read JSON-RPC lines until EOF, dispatching each request."""
        for raw in self._stdin:
            if not self._running:
                break
            line = raw.strip()
            if not line:
                continue
            self._handle_line(line)
        self._running = False
        for worker in list(self._workers):
            worker.join(timeout=5.0)

    def _handle_line(self, line: str) -> None:
        """Parse and dispatch a single JSON-RPC line; never raises."""
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as exc:
            self._error(None, _PARSE_ERROR, f"parse error: {exc}")
            return
        if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0" or "method" not in msg:
            self._error(msg.get("id") if isinstance(msg, dict) else None, _INVALID_REQUEST, "invalid request")
            return
        req_id = msg.get("id")
        method = str(msg.get("method", ""))
        params = msg.get("params") or {}
        if not isinstance(params, dict):
            self._error(req_id, _INVALID_REQUEST, "params must be an object")
            return
        try:
            self._dispatch(req_id, method, params)
        except Exception as exc:  # noqa: BLE001 — one bad call must not kill the server
            logger.warning("ACP dispatch error for %s: %s", method, exc)
            if req_id is not None:
                self._error(req_id, _INTERNAL_ERROR, f"{type(exc).__name__}: {exc}")

    def _dispatch(self, req_id: Any, method: str, params: dict[str, Any]) -> None:
        """Route ``method`` to the matching server call (snake_case or slash form)."""
        srv = self._server
        m = method.replace("/", "_").lower()

        if m in ("initialize",):
            self._result(req_id, srv.initialize(params.get("client_info")))
        elif m in ("list_tools", "tools_list"):
            self._result(req_id, srv.list_tools())
        elif m in ("list_models", "models_list"):
            self._result(req_id, srv.list_models())
        elif m in ("open_session", "session_new", "session_open"):
            self._result(
                req_id,
                srv.open_session(str(params.get("cwd", "")), model=params.get("model"), title=str(params.get("title", ""))),
            )
        elif m in ("list_sessions", "session_list"):
            self._result(req_id, srv.list_sessions())
        elif m in ("set_model", "session_set_model"):
            self._result(req_id, srv.set_model(str(params.get("session_id", "")), params.get("model")))
        elif m in ("cancel", "session_cancel"):
            self._result(req_id, srv.cancel(str(params.get("session_id", ""))))
        elif m in ("close_session", "session_close"):
            self._result(req_id, srv.close_session(str(params.get("session_id", ""))))
        elif m in ("respond_permission", "permission_respond"):
            self._result(req_id, srv.respond_permission(str(params.get("permission_id", "")), bool(params.get("allow", False))))
        elif m in ("pending_permissions", "permission_pending"):
            self._result(req_id, srv.pending_permissions())
        elif m in ("respond_question", "question_respond", "respond_input", "input_respond"):
            if self._runner is None:
                self._result(req_id, {"ok": False})
            else:
                self._result(req_id, self._runner.respond_question(str(params.get("input_id", "")), str(params.get("answer", ""))))
        elif m in ("pending_questions", "question_pending", "pending_inputs"):
            self._result(req_id, self._runner.pending_questions() if self._runner else [])
        elif m in ("prompt", "session_prompt"):
            self._start_prompt(req_id, params)
        elif m in ("shutdown", "exit"):
            self._running = False
            self._result(req_id, {"ok": True})
        else:
            if req_id is not None:
                self._error(req_id, _METHOD_NOT_FOUND, f"unknown method: {method}")

    # ----- prompt (runs on a worker thread) ---------------------------------

    def _start_prompt(self, req_id: Any, params: dict[str, Any]) -> None:
        """Validate the session, then stream the prompt on a worker thread."""
        session_id = str(params.get("session_id", ""))
        text = str(params.get("text", params.get("prompt", "")))
        session = self._server.sessions.get(session_id)
        if session is None:
            self._error(req_id, _INVALID_REQUEST, f"unknown session: {session_id}")
            return
        if self._runner is None:
            self._error(req_id, _INTERNAL_ERROR, "no agent runner wired")
            return

        def emit(event: dict[str, Any]) -> None:
            self._notify("session/update", {"session_id": session_id, "request_id": req_id, "event": event})

        def work() -> None:
            try:
                summary = self._runner.run_prompt(session=session, text=text, emit=emit)
                self._result(req_id, summary)
            except Exception as exc:  # noqa: BLE001
                logger.warning("ACP prompt worker failed: %s", exc)
                self._error(req_id, _INTERNAL_ERROR, f"{type(exc).__name__}: {exc}")
            finally:
                self._workers.discard(threading.current_thread())

        worker = threading.Thread(target=work, name=f"acp-prompt-{session_id[:8]}", daemon=True)
        self._workers.add(worker)
        worker.start()


__all__ = ["StdioJsonRpcServer"]
