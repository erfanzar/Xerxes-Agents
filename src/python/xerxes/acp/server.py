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
"""Transport-agnostic ACP server core.

Exposes the small set of RPC methods Xerxes implements for ACP
clients: session/prompt lifecycle, model switching, cancellation,
approval responses, tool listing. The transport (stdio JSON-RPC or
HTTP) lives elsewhere; this class is pure logic so it's testable
without sockets."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .events import AcpEvent, to_acp_event
from .permissions import AcpPermissionBoard, route_permission
from .session import AcpSessionStore

logger = logging.getLogger(__name__)

PromptHandler = Callable[..., Any]
ToolListProvider = Callable[[], list[dict[str, Any]]]
ModelListProvider = Callable[[], list[dict[str, Any]]]


@dataclass
class ServerCapabilities:
    """Capability advertisement returned in the ``initialize`` response.

    Attributes:
        protocol_version: ACP protocol version Xerxes implements.
        streaming: whether incremental events are emitted during a prompt.
        tools: whether ``list_tools`` returns a non-empty catalog.
        permissions: whether ``request_permission``/``respond_permission`` work.
        fork: whether session forking is supported.
    """

    protocol_version: str = "0.9"
    streaming: bool = True
    tools: bool = True
    permissions: bool = True
    fork: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serialise to the plain JSON-RPC capability dict shape."""
        return {
            "protocol_version": self.protocol_version,
            "streaming": self.streaming,
            "tools": self.tools,
            "permissions": self.permissions,
            "fork": self.fork,
        }


class AcpServer:
    """In-process ACP server logic.

    Construct with:
        * a ``prompt_handler`` callable invoked for ``prompt`` RPCs.
        * a ``tool_list_provider`` returning OpenAI-shaped tool schemas.
        * a ``model_list_provider`` returning known model identifiers.

    The server owns an ``AcpSessionStore`` and an ``AcpPermissionBoard``.
    Transport adapters dispatch JSON-RPC method names to the
    corresponding methods below; each returns plain JSON."""

    def __init__(
        self,
        *,
        prompt_handler: PromptHandler,
        tool_list_provider: ToolListProvider | None = None,
        model_list_provider: ModelListProvider | None = None,
        capabilities: ServerCapabilities | None = None,
    ) -> None:
        """Wire the prompt handler, list providers, and capability advert."""
        self.sessions = AcpSessionStore()
        self.permissions = AcpPermissionBoard()
        self._prompt = prompt_handler
        self._tools = tool_list_provider or (lambda: [])
        self._models = model_list_provider or (lambda: [])
        self.capabilities = capabilities or ServerCapabilities()

    # ----- core RPCs --------------------------------------------------------

    def initialize(self, client_info: dict[str, Any] | None = None) -> dict[str, Any]:
        """Handle the ACP ``initialize`` handshake.

        ``client_info`` is currently informational; capabilities are static
        per-process so we just echo them back."""
        return {
            "server_name": "xerxes",
            "capabilities": self.capabilities.to_dict(),
        }

    def list_tools(self) -> list[dict[str, Any]]:
        """Return the registered tool catalog as a fresh list (snapshot)."""
        return list(self._tools())

    def list_models(self) -> list[dict[str, Any]]:
        """Return the available models as advertised by the provider."""
        return list(self._models())

    def open_session(
        self,
        cwd: str,
        *,
        model: str | None = None,
        title: str = "",
    ) -> dict[str, Any]:
        """Create a new session bound to ``cwd`` and return its identifiers."""
        sess = self.sessions.create(cwd, model=model, title=title)
        return {"session_id": sess.session_id, "cwd": sess.cwd, "model": sess.model_override}

    def list_sessions(self) -> list[dict[str, Any]]:
        """Summarise every live session for the client UI."""
        return [
            {
                "session_id": s.session_id,
                "cwd": s.cwd,
                "model": s.model_override,
                "title": s.title,
                "cancelled": s.cancelled,
            }
            for s in self.sessions.list()
        ]

    def set_model(self, session_id: str, model: str | None) -> dict[str, Any]:
        """Update the model override on ``session_id``; ``{ok: false}`` if missing."""
        ok = self.sessions.set_model(session_id, model)
        return {"ok": ok}

    def cancel(self, session_id: str) -> dict[str, Any]:
        """Mark ``session_id`` as cancelled so the streaming loop bails out."""
        ok = self.sessions.cancel(session_id)
        return {"ok": ok}

    def close_session(self, session_id: str) -> dict[str, Any]:
        """Forget ``session_id`` entirely; subsequent RPCs will error."""
        ok = self.sessions.drop(session_id)
        return {"ok": ok}

    # ----- prompt + streaming ----------------------------------------------

    def prompt(self, session_id: str, text: str, **kwargs: Any) -> Any:
        """Dispatch a prompt to the wired ``prompt_handler``.

        Returns ``{"error": ...}`` if the session is unknown; otherwise the
        handler's return value is passed through unchanged (typically an
        async iterator or sentinel acknowledging streaming has started)."""
        sess = self.sessions.get(session_id)
        if sess is None:
            return {"error": f"unknown session: {session_id}"}
        return self._prompt(session=sess, text=text, **kwargs)

    def stream_event(self, event: Any) -> dict[str, Any]:
        """Wrap an internal StreamEvent for transport."""
        acp_event = event if isinstance(event, AcpEvent) else to_acp_event(event)
        return acp_event.to_wire()

    # ----- approvals --------------------------------------------------------

    def request_permission(
        self,
        *,
        session_id: str,
        tool_name: str,
        description: str,
        inputs: dict[str, Any],
    ) -> dict[str, Any]:
        """Surface a permission request to the client; return its id for polling."""
        req = route_permission(session_id=session_id, tool_name=tool_name, description=description, inputs=inputs)
        self.permissions.submit(req)
        return {"permission_id": req.id}

    def respond_permission(self, permission_id: str, allow: bool) -> dict[str, Any]:
        """Record the client's decision for ``permission_id``."""
        ok = self.permissions.resolve(permission_id, allow)
        return {"ok": ok}

    def pending_permissions(self) -> list[dict[str, Any]]:
        """Snapshot every request still awaiting a client decision."""
        return [
            {
                "id": p.id,
                "session_id": p.session_id,
                "tool_name": p.tool_name,
                "description": p.description,
                "inputs": p.inputs,
            }
            for p in self.permissions.snapshot_pending()
        ]


def build_server(*, default_permission_mode: str = "accept-all") -> tuple[AcpServer, Any]:
    """Bootstrap the agent runner and an :class:`AcpServer` wired to it.

    Returns ``(server, runner)``. The runner bootstraps a full
    :class:`RuntimeManager` (provider/tools/skills) from the active profile,
    so this raises if no provider is configured."""
    from ..tools.claude_tools import set_ask_user_question_callback
    from .runner import AcpAgentRunner

    runner = AcpAgentRunner(default_permission_mode=default_permission_mode)
    server = AcpServer(
        prompt_handler=runner.run_prompt,
        tool_list_provider=runner.list_tools,
        model_list_provider=runner.list_models,
    )
    runner.permission_board = server.permissions
    # Route the blocking AskUserQuestion tool through the ACP client too, so ALL
    # interactive prompts (approvals AND questions) are handled by the editor.
    set_ask_user_question_callback(runner.ask_user_question)
    return server, runner


def main(argv: list[str] | None = None) -> None:
    """Console entry point for ``xerxes-acp`` — a stdio JSON-RPC ACP server.

    ``xerxes-acp`` runs the server on stdin/stdout. ``--write-registry`` writes
    the ACP discovery manifest (``agent.json``) for IDE clients and exits."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="xerxes-acp",
        description="Xerxes Agent Client Protocol (ACP) server over stdio JSON-RPC.",
    )
    parser.add_argument(
        "--write-registry",
        action="store_true",
        help="Write the ACP discovery manifest (agent.json) to the registry dir and exit.",
    )
    parser.add_argument(
        "--permission-mode",
        default="accept-all",
        choices=("accept-all", "auto", "manual"),
        help=(
            "Default tool-approval mode for ACP sessions (default: accept-all). "
            "'auto' routes risky ops to the client, 'manual' routes every tool, "
            "'accept-all' bypasses approval."
        ),
    )
    args = parser.parse_args(argv)

    if args.write_registry:
        from .registry import write_registry_file

        path = write_registry_file()
        # stdout is safe here (not in server/protocol mode).
        print(f"Wrote ACP registry manifest: {path}")
        return

    # NOTE: in server mode stdout is the JSON-RPC channel — never print to it.
    from .transport import StdioJsonRpcServer

    try:
        server, runner = build_server(default_permission_mode=args.permission_mode)
    except Exception as exc:
        raise SystemExit(
            f"xerxes-acp: failed to initialise runtime: {exc}\n"
            "Ensure a provider profile is configured (run `xerxes` and use /provider)."
        ) from exc

    StdioJsonRpcServer(server, runner).serve_forever()


__all__ = ["AcpServer", "ServerCapabilities", "build_server", "main"]
