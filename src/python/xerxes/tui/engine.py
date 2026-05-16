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
"""TUI-side client for the Xerxes daemon socket.

Defines :class:`BridgeClient`, which speaks JSON-RPC over the Unix domain
socket published by the daemon. Spawns the daemon subprocess when none
is reachable, sends JSON-RPC requests / notifications, demultiplexes
inbound events into an :class:`asyncio.Queue` for the TUI, and tracks
correlated request futures."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from ..daemon.config import load_config
from ..streaming.wire_events import (
    WireEvent,
    event_from_dict,
)


class BridgeClient:
    """Asyncio-friendly JSON-RPC client for the Xerxes daemon socket.

    The TUI uses one instance per session. ``spawn`` connects to (and
    if needed, launches) the daemon; events flow back via :meth:`events`
    as :class:`WireEvent` instances. Concurrent ``await`` callers are
    safe — outbound writes serialize through ``_write_lock`` and
    inbound bytes are demultiplexed by a daemon thread that re-enters
    the asyncio loop with ``call_soon_threadsafe``."""

    def __init__(
        self,
        python_executable: str | None = None,
    ) -> None:
        """Stash configuration; no daemon contact happens until :meth:`spawn`.

        Args:
            python_executable: Interpreter used to launch the daemon
                module if none is running. Defaults to ``sys.executable``.
        """
        self._python = python_executable or sys.executable
        self._proc: subprocess.Popen[bytes, bytes] | None = None
        self._sock: socket.socket | None = None
        self._socket_reader: Any = None
        self._socket_writer: Any = None
        self._write_lock = asyncio.Lock()
        self._event_queue: asyncio.Queue[WireEvent] = asyncio.Queue()
        self._read_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._stderr_lines: deque[str] = deque(maxlen=200)
        self._stderr_lock = threading.Lock()
        self._running = False
        self._pending_requests: dict[str, asyncio.Future[WireEvent]] = {}
        self._model: str = ""
        self._initialized = False
        self._loop: asyncio.AbstractEventLoop | None = None

    def spawn(self) -> None:
        """Connect to the daemon, launching one if needed, and start reader threads.

        Stops a stale daemon (older protocol) before launching a fresh
        one. Raises ``RuntimeError`` if the new daemon doesn't become
        reachable within ten seconds. Idempotent once connected."""
        if self._sock is not None:
            return

        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = asyncio.get_event_loop()

        config = load_config()
        socket_path = Path(config.socket_path).expanduser()
        required_protocol = 35
        if self._daemon_protocol(socket_path) < required_protocol:
            self._stop_stale_daemon(Path(config.pid_file).expanduser(), socket_path)
        if not self._connect_socket(socket_path):
            self._proc = subprocess.Popen(
                [self._python, "-m", "xerxes.daemon"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                bufsize=0,
                start_new_session=True,
            )
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline and self._daemon_protocol(socket_path) < required_protocol:
                if self._proc.poll() is not None:
                    break
                time.sleep(0.1)
            self._connect_socket(socket_path)
            if self._sock is None:
                raise RuntimeError("Xerxes daemon did not become ready on the Unix socket")

        self._running = True

        self._read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._read_thread.start()
        if self._proc and self._proc.stderr:
            self._stderr_thread = threading.Thread(target=self._read_stderr_loop, daemon=True)
            self._stderr_thread.start()

    def close(self) -> None:
        """Tear down the socket and reader handles; safe to call multiple times."""
        self._running = False
        for handle in (self._socket_reader, self._socket_writer):
            try:
                if handle:
                    handle.close()
            except Exception:
                pass
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
        self._sock = None
        self._socket_reader = None
        self._socket_writer = None

    def stderr_tail(self) -> list[str]:
        """Snapshot the last ~200 stderr lines captured from the daemon subprocess."""
        with self._stderr_lock:
            return list(self._stderr_lines)

    async def initialize(
        self,
        model: str = "",
        base_url: str = "",
        api_key: str = "",
        permission_mode: str = "auto",
        resume_session_id: str = "",
    ) -> None:
        """Send the ``initialize`` JSON-RPC and wait for the daemon ack.

        Must run before any :meth:`query`. Pass ``resume_session_id`` to
        rehydrate an existing session instead of starting fresh."""
        self._model = model
        await self._send_jsonrpc(
            method="initialize",
            params={
                "model": model,
                "base_url": base_url,
                "api_key": api_key,
                "permission_mode": permission_mode,
                "resume_session_id": resume_session_id,
            },
        )
        self._initialized = True

    async def query(self, user_input: str, plan_mode: bool = False, mode: str = "code") -> None:
        """Submit ``user_input`` as a new turn; raises if the daemon rejects it.

        Args:
            user_input: Raw user text.
            plan_mode: Run the turn in read-only plan mode.
            mode: Interaction mode (``"code"``, ``"researcher"``, ``"plan"``).

        Raises:
            RuntimeError: If :meth:`initialize` has not run yet, or if
                the daemon returns a non-OK response within ten seconds.
        """
        if not self._initialized:
            raise RuntimeError("Call initialize() before query()")
        result = await self._request_jsonrpc(
            method="prompt",
            params={
                "user_input": user_input,
                "plan_mode": plan_mode,
                "mode": mode,
            },
            timeout=10.0,
        )
        if not result.raw.get("ok", False):
            raise RuntimeError(str(result.raw.get("error", "Daemon rejected prompt")))

    async def cancel(self) -> None:
        """Ask the daemon to abort the currently running turn."""
        await self._send_jsonrpc(method="cancel", params={})

    async def cancel_all(self) -> None:
        """Ask the daemon to abort every running turn in this session."""
        await self._send_jsonrpc(method="cancel_all", params={})

    async def permission_response(
        self,
        request_id: str,
        response: str,
        feedback: str | None = None,
    ) -> None:
        """Deliver the user's verdict for a pending permission/approval request.

        Args:
            request_id: Daemon-issued request id from the matching event.
            response: One of ``"approve"``, ``"approve_for_session"``,
                ``"approve_always"``, ``"reject"``.
            feedback: Optional free-text reason routed back to the agent.
        """
        await self._send_jsonrpc(
            method="permission_response",
            params={
                "request_id": request_id,
                "response": response,
                "feedback": feedback,
            },
        )

    async def question_response(
        self,
        request_id: str,
        answers: dict[str, str],
    ) -> None:
        """Submit ``answers`` (question-id → text) for a pending clarification."""
        await self._send_jsonrpc(
            method="question_response",
            params={
                "request_id": request_id,
                "answers": answers,
            },
        )

    async def steer(self, content: str) -> None:
        """Inject mid-turn guidance (``/btw`` / ``/steer``) into the active turn."""
        await self._send_jsonrpc(method="steer", params={"content": content})

    async def fetch_models(self, base_url: str, api_key: str = "") -> list[dict[str, Any]]:
        """Ask the daemon to list available models for ``base_url``.

        Awaits the correlated daemon response and returns its ``models``
        array verbatim. The pending future is always reaped, even on
        cancellation, to prevent dict-growth leaks across the session."""
        future: asyncio.Future[WireEvent] = asyncio.get_running_loop().create_future()
        req_id = f"fetch_models_{id(future)}"
        self._pending_requests[req_id] = future
        try:
            await self._send_jsonrpc(
                method="fetch_models",
                params={"base_url": base_url, "api_key": api_key},
                req_id=req_id,
            )
            result = await future
        finally:
            # Always reap the registration; otherwise a cancelled/timed-out
            # caller leaves a dangling future in _pending_requests that grows
            # the dict over the session lifetime.
            self._pending_requests.pop(req_id, None)

        return result.raw.get("models", []) if hasattr(result, "raw") else []

    async def provider_save(
        self,
        name: str,
        base_url: str,
        api_key: str,
        model: str,
        provider: str = "",
    ) -> None:
        """Persist a provider profile on the daemon side (config + secrets store)."""
        await self._send_jsonrpc(
            method="provider_save",
            params={
                "name": name,
                "base_url": base_url,
                "api_key": api_key,
                "model": model,
                "provider": provider,
            },
        )

    async def provider_list(self) -> list[dict[str, Any]]:
        """Return saved provider profiles from the daemon."""
        future = asyncio.get_running_loop().create_future()
        req_id = f"provider_list_{id(future)}"
        self._pending_requests[req_id] = future
        try:
            await self._send_jsonrpc(method="provider_list", params={}, req_id=req_id)
            result = await future
        finally:
            self._pending_requests.pop(req_id, None)
        return result.raw.get("profiles", []) if hasattr(result, "raw") else []

    async def provider_select(self, name: str) -> None:
        """Mark provider profile ``name`` as the active one."""
        await self._send_jsonrpc(method="provider_select", params={"name": name})

    async def provider_delete(self, name: str) -> None:
        """Delete the saved provider profile named ``name``."""
        await self._send_jsonrpc(method="provider_delete", params={"name": name})

    async def shutdown(self) -> None:
        """Cancel every running turn before the TUI disconnects."""
        await self._send_jsonrpc(method="cancel_all", params={})

    async def events(self) -> AsyncIterator[WireEvent]:
        """Yield :class:`WireEvent` instances as the daemon pushes them.

        Polls the internal queue with a 0.5 s timeout so the loop exits
        promptly when :meth:`close` flips ``_running`` to false."""
        while self._running:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=0.5)
                yield event
            except TimeoutError:
                continue

    async def _send_jsonrpc(
        self,
        method: str,
        params: dict[str, Any],
        req_id: str | None = None,
    ) -> None:
        """Encode and write one JSON-RPC frame to the daemon socket.

        Pass ``req_id`` for request/response style calls; omit it for
        notifications. Raises ``RuntimeError`` when the socket is missing
        or the peer has closed it."""
        if self._socket_writer is None:
            raise RuntimeError("Xerxes daemon socket not connected. Call spawn() first.")

        message: dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        if req_id:
            message["id"] = req_id

        line = json.dumps(message, ensure_ascii=False, default=str)
        async with self._write_lock:
            try:
                self._socket_writer.write((line + "\n").encode("utf-8"))
                self._socket_writer.flush()
            except BrokenPipeError as exc:
                raise RuntimeError("Xerxes daemon socket closed") from exc

    async def _request_jsonrpc(
        self,
        method: str,
        params: dict[str, Any],
        *,
        timeout: float = 10.0,
    ) -> WireEvent:
        """Send a JSON-RPC request and await the correlated response.

        Raises ``RuntimeError`` if the daemon doesn't reply within
        ``timeout`` seconds; either way the pending future is reaped."""
        future = asyncio.get_running_loop().create_future()
        req_id = f"{method}_{id(future)}"
        self._pending_requests[req_id] = future
        try:
            await self._send_jsonrpc(method=method, params=params, req_id=req_id)
            return await asyncio.wait_for(future, timeout=timeout)
        except TimeoutError as exc:
            raise RuntimeError(f"Daemon did not acknowledge {method}") from exc
        finally:
            # Reap on every exit (success, timeout, cancellation). Previously
            # only TimeoutError was cleaned up; a cancellation between the
            # send and the wait left a dangling future entry.
            self._pending_requests.pop(req_id, None)

    def _read_loop(self) -> None:
        """Daemon thread: pull newline-delimited JSON from the socket forever.

        Each parsed line is handed to :meth:`_handle_inbound_line` which
        either pushes a :class:`WireEvent` onto the asyncio queue or
        completes a pending request future."""
        assert self._socket_reader is not None
        reader = self._socket_reader
        while self._running:
            try:
                raw = reader.readline()
            except Exception:
                break
            if not raw:
                break
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            self._handle_inbound_line(line)

    def _read_stderr_loop(self) -> None:
        """Daemon thread: drain daemon stderr into the rolling ring buffer."""
        assert self._proc is not None and self._proc.stderr is not None
        reader = self._proc.stderr

        while self._running:
            try:
                line = reader.readline()
            except Exception:
                break
            if not line:
                break
            text = line.decode("utf-8", errors="replace").rstrip()
            if not text:
                continue
            with self._stderr_lock:
                self._stderr_lines.append(text)

    def _connect_socket(self, socket_path: Path) -> bool:
        """Attempt one AF_UNIX connect and stash readable/writable handles.

        Returns ``True`` on success; ``False`` (with sock closed) when
        the daemon isn't listening yet."""
        sock: socket.socket | None = None
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(str(socket_path))
        except OSError:
            try:
                if sock:
                    sock.close()
            except Exception:
                pass
            return False
        self._sock = sock
        self._socket_reader = sock.makefile("rb", buffering=0)
        self._socket_writer = sock.makefile("wb", buffering=0)
        return True

    def _daemon_protocol(self, socket_path: Path) -> int:
        """Probe the daemon health endpoint and return its protocol version.

        Returns ``0`` when no daemon is listening or the response is
        malformed — the caller treats that as ``stale`` and respawns."""
        sock: socket.socket | None = None
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(1.5)
            sock.connect(str(socket_path))
            msg = {"jsonrpc": "2.0", "id": "health", "method": "runtime.status", "params": {}}
            sock.sendall((json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8"))
            buffer = b""
            deadline = time.monotonic() + 1.5
            while time.monotonic() < deadline:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                buffer += chunk
                while b"\n" in buffer:
                    raw, buffer = buffer.split(b"\n", 1)
                    if not raw.strip():
                        continue
                    data = json.loads(raw.decode("utf-8", errors="replace"))
                    if data.get("id") == "health":
                        result = data.get("result", {}) or {}
                        return int(result.get("daemon_protocol", 0) or 0)
        except Exception:
            return 0
        finally:
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass
        return 0

    @staticmethod
    def _stop_stale_daemon(pid_file: Path, socket_path: Path) -> None:
        """SIGTERM the recorded pid, wait briefly, then unlink the socket file."""
        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip())
        except Exception:
            pid = 0
        if pid > 0:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except OSError:
                pass
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    break
                except OSError:
                    break
                time.sleep(0.05)
        socket_path.unlink(missing_ok=True)

    def _handle_inbound_line(self, line: str) -> None:
        """Demultiplex one inbound JSON-RPC frame into an event or request future."""
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            return

        method = msg.get("method", "")
        msg_id = msg.get("id")

        if method in {"event", "request"}:
            payload = msg.get("params", {})
            event_type = payload.get("type", "")
            event_data = payload.get("payload", {}) or {}
            event_data["type"] = event_type
            try:
                wire_event = event_from_dict(event_data)
                if self._loop is not None:
                    self._loop.call_soon_threadsafe(self._event_queue.put_nowait, wire_event)
            except Exception:
                pass
            return

        if msg_id and str(msg_id) in self._pending_requests:
            future = self._pending_requests.pop(str(msg_id))
            result = msg.get("result", {})
            from ..streaming.wire_events import GenericWireEvent

            wire_event = GenericWireEvent(raw=result)
            if self._loop is not None:
                self._loop.call_soon_threadsafe(lambda f, ev: f.set_result(ev), future, wire_event)

    def __enter__(self) -> BridgeClient:
        """Spawn the daemon on context entry and return ``self``."""
        self.spawn()
        return self

    def __exit__(self, *args: Any) -> None:
        """Close the socket on context exit."""
        self.close()
