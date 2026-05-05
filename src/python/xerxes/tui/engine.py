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
"""Bridge client for communicating with the Xerxes backend subprocess.

This module defines :class:`BridgeClient`, which manages the lifecycle of the
Xerxes bridge process, sends JSON-RPC commands, and yields wire events back to
the TUI.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import threading
from collections.abc import AsyncIterator
from typing import Any

from ..streaming.wire_events import (
    WireEvent,
    event_from_dict,
)


class BridgeClient:
    """Manages a subprocess bridge to the Xerxes backend.

    Spawns the bridge process, sends JSON-RPC messages over stdin, and
    asynchronously reads events from stdout via a background thread.
    """

    def __init__(
        self,
        python_executable: str | None = None,
    ) -> None:
        """Initialize the bridge client without spawning the process.

        Args:
            python_executable (str | None): IN: Path to the Python executable
                used to run the bridge module. OUT: Stored internally; defaults
                to ``sys.executable`` when spawning.
        """
        self._python = python_executable or sys.executable
        self._proc: subprocess.Popen[bytes, bytes] | None = None
        self._write_lock = asyncio.Lock()
        self._event_queue: asyncio.Queue[WireEvent] = asyncio.Queue()
        self._read_thread: threading.Thread | None = None
        self._running = False
        self._pending_requests: dict[str, asyncio.Future[WireEvent]] = {}
        self._model: str = ""
        self._initialized = False
        self._loop: asyncio.AbstractEventLoop | None = None

    def spawn(self) -> None:
        """Spawn the bridge subprocess and start the background reader thread.

        If the process is already running, this method is a no-op.
        """
        if self._proc is not None:
            return

        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = asyncio.get_event_loop()

        self._proc = subprocess.Popen(
            [self._python, "-m", "xerxes.bridge", "--wire"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        self._running = True

        self._read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._read_thread.start()

    def close(self) -> None:
        """Terminate the bridge subprocess and clean up resources."""
        self._running = False
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None

    async def initialize(
        self,
        model: str = "",
        base_url: str = "",
        api_key: str = "",
        permission_mode: str = "auto",
        resume_session_id: str = "",
    ) -> None:
        """Send the ``initialize`` JSON-RPC command to the bridge.

        Args:
            model (str): IN: Model identifier. OUT: Sent to the bridge.
            base_url (str): IN: Base URL for the model provider. OUT: Sent to the bridge.
            api_key (str): IN: API key for authentication. OUT: Sent to the bridge.
            permission_mode (str): IN: Permission mode such as ``"auto"``. OUT: Sent
                to the bridge.
            resume_session_id (str): IN: Session ID to resume, if any. OUT: Sent
                to the bridge.
        """
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

    async def query(self, user_input: str, plan_mode: bool = False) -> None:
        """Send a user prompt to the bridge via the ``prompt`` JSON-RPC method.

        Args:
            user_input (str): IN: Raw user input text. OUT: Sent to the bridge.
            plan_mode (bool): IN: Whether plan mode is enabled. OUT: Sent to the bridge.

        Raises:
            RuntimeError: If :meth:`initialize` has not been called yet.
        """
        if not self._initialized:
            raise RuntimeError("Call initialize() before query()")
        await self._send_jsonrpc(
            method="prompt",
            params={
                "user_input": user_input,
                "plan_mode": plan_mode,
            },
        )

    async def cancel(self) -> None:
        """Send the ``cancel`` JSON-RPC command to abort the current turn."""
        await self._send_jsonrpc(method="cancel", params={})

    async def cancel_all(self) -> None:
        """Send the ``cancel_all`` JSON-RPC command to abort all turns."""
        await self._send_jsonrpc(method="cancel_all", params={})

    async def permission_response(
        self,
        request_id: str,
        response: str,
        feedback: str | None = None,
    ) -> None:
        """Send a permission response for an approval request.

        Args:
            request_id (str): IN: ID of the approval request. OUT: Identifies
                which request is being answered.
            response (str): IN: Response value such as ``"approve"``. OUT: Sent
                to the bridge.
            feedback (str | None): IN: Optional user feedback text. OUT: Sent
                to the bridge if provided.
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
        """Send answers for a question request.

        Args:
            request_id (str): IN: ID of the question request. OUT: Identifies
                which request is being answered.
            answers (dict[str, str]): IN: Mapping from question ID to answer text.
                OUT: Serialized and sent to the bridge.
        """
        await self._send_jsonrpc(
            method="question_response",
            params={
                "request_id": request_id,
                "answers": answers,
            },
        )

    async def steer(self, content: str) -> None:
        """Send mid-turn guidance (steer) to the bridge.

        Args:
            content (str): IN: Steer text content. OUT: Sent to the bridge.
        """
        await self._send_jsonrpc(method="steer", params={"content": content})

    async def fetch_models(self, base_url: str, api_key: str = "") -> list[dict[str, Any]]:
        """Fetch available models from the bridge.

        Args:
            base_url (str): IN: Provider base URL. OUT: Sent to the bridge.
            api_key (str): IN: Optional API key. OUT: Sent to the bridge.

        Returns:
            list[dict[str, Any]]: OUT: List of model metadata dictionaries.
        """
        future: asyncio.Future[WireEvent] = asyncio.get_event_loop().create_future()
        req_id = f"fetch_models_{id(future)}"
        self._pending_requests[req_id] = future
        await self._send_jsonrpc(
            method="fetch_models",
            params={"base_url": base_url, "api_key": api_key},
            req_id=req_id,
        )
        result = await future

        return result.raw.get("models", []) if hasattr(result, "raw") else []

    async def provider_save(
        self,
        name: str,
        base_url: str,
        api_key: str,
        model: str,
        provider: str = "",
    ) -> None:
        """Save a provider profile via the bridge.

        Args:
            name (str): IN: Profile name. OUT: Sent to the bridge.
            base_url (str): IN: Provider base URL. OUT: Sent to the bridge.
            api_key (str): IN: API key. OUT: Sent to the bridge.
            model (str): IN: Model identifier. OUT: Sent to the bridge.
            provider (str): IN: Optional provider type. OUT: Sent to the bridge.
        """
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
        """List saved provider profiles from the bridge.

        Returns:
            list[dict[str, Any]]: OUT: List of profile dictionaries.
        """
        future = asyncio.get_event_loop().create_future()
        req_id = f"provider_list_{id(future)}"
        self._pending_requests[req_id] = future
        await self._send_jsonrpc(method="provider_list", params={}, req_id=req_id)
        result = await future
        return result.raw.get("profiles", []) if hasattr(result, "raw") else []

    async def provider_select(self, name: str) -> None:
        """Select a saved provider profile.

        Args:
            name (str): IN: Profile name to activate. OUT: Sent to the bridge.
        """
        await self._send_jsonrpc(method="provider_select", params={"name": name})

    async def provider_delete(self, name: str) -> None:
        """Delete a saved provider profile.

        Args:
            name (str): IN: Profile name to delete. OUT: Sent to the bridge.
        """
        await self._send_jsonrpc(method="provider_delete", params={"name": name})

    async def shutdown(self) -> None:
        """Send the ``shutdown`` JSON-RPC command to gracefully stop the bridge."""
        await self._send_jsonrpc(method="shutdown", params={})

    async def events(self) -> AsyncIterator[WireEvent]:
        """Yield wire events from the bridge as an async iterator.

        Yields:
            WireEvent: OUT: Incoming events from the bridge event queue.
        """
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
        """Serialize and send a JSON-RPC message to the bridge stdin.

        Args:
            method (str): IN: JSON-RPC method name. OUT: Written to the message.
            params (dict[str, Any]): IN: Method parameters. OUT: Serialized into
                the JSON-RPC payload.
            req_id (str | None): IN: Optional request ID for correlating responses.
                OUT: Included in the message if provided.

        Raises:
            RuntimeError: If the bridge subprocess is not running or its stdin
                pipe is broken.
        """
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("Bridge subprocess not running. Call spawn() first.")

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
                self._proc.stdin.write((line + "\n").encode("utf-8"))
                self._proc.stdin.flush()
            except BrokenPipeError as exc:
                raise RuntimeError("Bridge stdin pipe broken (process exited?)") from exc

    def _read_loop(self) -> None:
        """Background thread loop that reads JSON-RPC messages from bridge stdout.

        Parses lines into events or responses and dispatches them to the
        asyncio event queue or pending request futures.
        """
        assert self._proc is not None and self._proc.stdout is not None
        reader = self._proc.stdout

        buf = ""
        while self._running:
            try:
                chunk = reader.read(4096)
                if not chunk:
                    break
                buf += chunk.decode("utf-8", errors="replace")
            except Exception:
                break

            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue

                method = msg.get("method", "")
                msg_id = msg.get("id")

                if method == "event":
                    payload = msg.get("params", {})
                    event_type = payload.get("type", "")
                    event_data = payload.get("payload", {}) or {}
                    event_data["type"] = event_type
                    try:
                        wire_event = event_from_dict(event_data)
                        if self._loop is None:
                            continue
                        self._loop.call_soon_threadsafe(self._event_queue.put_nowait, wire_event)
                    except Exception:
                        pass

                elif method == "request":
                    payload = msg.get("params", {})
                    event_type = payload.get("type", "")
                    event_data = payload.get("payload", {}) or {}
                    event_data["type"] = event_type
                    try:
                        wire_event = event_from_dict(event_data)
                        if self._loop is None:
                            continue
                        self._loop.call_soon_threadsafe(self._event_queue.put_nowait, wire_event)
                    except Exception:
                        pass

                elif msg_id and str(msg_id) in self._pending_requests:
                    future = self._pending_requests.pop(str(msg_id))
                    result = msg.get("result", {})

                    from ..streaming.wire_events import GenericWireEvent

                    wire_event = GenericWireEvent(raw=result)
                    if self._loop is None:
                        continue
                    self._loop.call_soon_threadsafe(lambda f, ev: f.set_result(ev), future, wire_event)

    def __enter__(self) -> BridgeClient:
        """Enter the runtime context, spawning the bridge process.

        Returns:
            BridgeClient: OUT: Self for use in a ``with`` statement.
        """
        self.spawn()
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit the runtime context, closing the bridge process."""
        self.close()
