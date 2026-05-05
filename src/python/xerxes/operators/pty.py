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
"""Pty module for Xerxes.

Exports:
    - PTYSession
    - PTYSessionManager"""

from __future__ import annotations

import os
import pty
import select
import signal
import subprocess
import time
import typing as tp
import uuid
from dataclasses import dataclass


@dataclass
class PTYSession:
    """Ptysession.

    Attributes:
        session_id (str): session id.
        process (subprocess.Popen[str]): process.
        master_fd (int): master fd.
        command (str): command.
        workdir (str): workdir."""

    session_id: str
    process: subprocess.Popen[str]
    master_fd: int
    command: str
    workdir: str


class PTYSessionManager:
    """Ptysession manager."""

    def __init__(self) -> None:
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access."""

        self._sessions: dict[str, PTYSession] = {}

    def create_session(
        self,
        cmd: str,
        *,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
        login: bool = True,
        yield_time_ms: int = 1000,
        max_output_chars: int = 4000,
    ) -> dict[str, tp.Any]:
        """Create session.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            cmd (str): IN: cmd. OUT: Consumed during execution.
            workdir (str | None, optional): IN: workdir. Defaults to None. OUT: Consumed during execution.
            env (dict[str, str] | None, optional): IN: env. Defaults to None. OUT: Consumed during execution.
            login (bool, optional): IN: login. Defaults to True. OUT: Consumed during execution.
            yield_time_ms (int, optional): IN: yield time ms. Defaults to 1000. OUT: Consumed during execution.
            max_output_chars (int, optional): IN: max output chars. Defaults to 4000. OUT: Consumed during execution.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

        resolved_workdir = os.path.abspath(workdir or os.getcwd())
        master_fd, slave_fd = pty.openpty()
        shell = os.environ.get("SHELL", "/bin/sh")
        shell_args = [shell]
        if login and os.path.basename(shell).endswith("zsh"):
            shell_args.append("-l")
        elif login and os.path.basename(shell).endswith("bash"):
            shell_args.append("-l")
        shell_args.extend(["-c", cmd])

        process = subprocess.Popen(
            shell_args,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            cwd=resolved_workdir,
            env={**os.environ, **(env or {})},
            text=True,
            start_new_session=True,
        )
        os.close(slave_fd)
        os.set_blocking(master_fd, False)

        session_id = f"pty_{uuid.uuid4().hex[:10]}"
        self._sessions[session_id] = PTYSession(
            session_id=session_id,
            process=process,
            master_fd=master_fd,
            command=cmd,
            workdir=resolved_workdir,
        )
        output = self._read_output(session_id, yield_time_ms=yield_time_ms, max_output_chars=max_output_chars)
        return {
            "session_id": session_id,
            "command": cmd,
            "workdir": resolved_workdir,
            "stdout": output,
            "running": process.poll() is None,
            "exit_code": process.poll(),
        }

    def write(
        self,
        session_id: str,
        *,
        chars: str = "",
        close_stdin: bool = False,
        interrupt: bool = False,
        yield_time_ms: int = 1000,
        max_output_chars: int = 4000,
    ) -> dict[str, tp.Any]:
        """Write.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            session_id (str): IN: session id. OUT: Consumed during execution.
            chars (str, optional): IN: chars. Defaults to ''. OUT: Consumed during execution.
            close_stdin (bool, optional): IN: close stdin. Defaults to False. OUT: Consumed during execution.
            interrupt (bool, optional): IN: interrupt. Defaults to False. OUT: Consumed during execution.
            yield_time_ms (int, optional): IN: yield time ms. Defaults to 1000. OUT: Consumed during execution.
            max_output_chars (int, optional): IN: max output chars. Defaults to 4000. OUT: Consumed during execution.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

        session = self._require_session(session_id)
        if interrupt and session.process.poll() is None:
            os.killpg(session.process.pid, signal.SIGINT)
        if chars:
            os.write(session.master_fd, chars.encode())
        if close_stdin:
            try:
                os.write(session.master_fd, b"\x04")
            except OSError:
                pass
        output = self._read_output(session_id, yield_time_ms=yield_time_ms, max_output_chars=max_output_chars)
        return {
            "session_id": session_id,
            "stdout": output,
            "running": session.process.poll() is None,
            "exit_code": session.process.poll(),
        }

    def close(self, session_id: str) -> dict[str, tp.Any]:
        """Close.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            session_id (str): IN: session id. OUT: Consumed during execution.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

        session = self._require_session(session_id)
        if session.process.poll() is None:
            session.process.terminate()
            try:
                session.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                session.process.kill()
        try:
            os.close(session.master_fd)
        except OSError:
            pass
        self._sessions.pop(session_id, None)
        return {"session_id": session_id, "closed": True, "exit_code": session.process.poll()}

    def list_sessions(self) -> list[dict[str, tp.Any]]:
        """List sessions.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            list[dict[str, tp.Any]]: OUT: Result of the operation."""

        return [
            {
                "session_id": sid,
                "command": session.command,
                "workdir": session.workdir,
                "running": session.process.poll() is None,
                "exit_code": session.process.poll(),
            }
            for sid, session in sorted(self._sessions.items())
        ]

    def _read_output(self, session_id: str, *, yield_time_ms: int, max_output_chars: int) -> str:
        """Internal helper to read output.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            session_id (str): IN: session id. OUT: Consumed during execution.
            yield_time_ms (int): IN: yield time ms. OUT: Consumed during execution.
            max_output_chars (int): IN: max output chars. OUT: Consumed during execution.
        Returns:
            str: OUT: Result of the operation."""

        session = self._require_session(session_id)
        deadline = time.time() + max(yield_time_ms, 0) / 1000
        chunks: list[str] = []
        remaining = max_output_chars
        while remaining > 0:
            timeout = max(0.0, deadline - time.time())
            ready, _, _ = select.select([session.master_fd], [], [], timeout)
            if not ready:
                break
            try:
                data = os.read(session.master_fd, min(remaining, 4096))
            except BlockingIOError:
                break
            except OSError:
                break
            if not data:
                break
            text = data.decode(errors="replace")
            chunks.append(text)
            remaining -= len(text)
            if time.time() >= deadline:
                break
        return "".join(chunks)

    def _require_session(self, session_id: str) -> PTYSession:
        """Internal helper to require session.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            session_id (str): IN: session id. OUT: Consumed during execution.
        Returns:
            PTYSession: OUT: Result of the operation."""

        if session_id not in self._sessions:
            raise ValueError(f"PTY session not found: {session_id}")
        return self._sessions[session_id]
