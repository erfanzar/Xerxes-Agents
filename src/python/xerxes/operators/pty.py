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
"""PTY-backed shell sessions for the operator subsystem.

Hosts a long-lived collection of pseudo-terminal subprocesses keyed by
session id so the model can drive interactive shells, REPLs and TUI
commands across multiple tool calls. The manager owns the master fd and
the :class:`subprocess.Popen` handle for each session and exposes
non-blocking, bounded reads that yield after a soft deadline so the
streaming runtime stays responsive.
"""

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
    """Live PTY-backed shell tracked by :class:`PTYSessionManager`.

    Attributes:
        session_id: Stable identifier the model uses to address the session.
        process: The :class:`subprocess.Popen` running the shell command.
        master_fd: Master side of the pseudo-terminal — owned by the manager
            and closed when the session is torn down.
        command: Original command string the session was started with.
        workdir: Absolute working directory the shell was launched in.
    """

    session_id: str
    process: subprocess.Popen[str]
    master_fd: int
    command: str
    workdir: str


class PTYSessionManager:
    """Lifecycle manager for a session's PTY-backed shells.

    Tracks every live :class:`PTYSession` keyed by session id, owns the
    master fds and ensures they're closed on teardown. All reads are
    non-blocking with a configurable soft deadline; output is capped per
    call to avoid flooding the wire.
    """

    def __init__(self) -> None:
        """Start with no active sessions."""

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
        """Spawn a new PTY-backed shell session and return its first output.

        Args:
            cmd: Shell command line executed via ``$SHELL -c``.
            workdir: Working directory; falls back to ``os.getcwd()``.
            env: Extra environment variables merged over ``os.environ``.
            login: Append ``-l`` when the shell is bash or zsh.
            yield_time_ms: Soft deadline before returning partial output.
            max_output_chars: Hard cap on the bytes captured for this call.

        Returns:
            A wire dict with the new ``session_id``, captured ``stdout``,
            and the process ``running`` / ``exit_code`` flags.
        """

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
        """Drive an existing PTY session and return any new output.

        The operations are applied in order: optional ``SIGINT`` to the
        process group, then write ``chars``, then optionally close stdin by
        sending the EOT control byte (``0x04``). Finally a bounded read
        collects whatever the shell produced.

        Args:
            session_id: Target session previously created by
                :meth:`create_session`.
            chars: Text to feed into the PTY (no implicit newline).
            close_stdin: If ``True``, send EOT after writing ``chars``.
            interrupt: If ``True``, send ``SIGINT`` to the process group
                before writing.
            yield_time_ms: Soft deadline for the follow-up read.
            max_output_chars: Hard cap on captured output for this call.
        """

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
        """Terminate a PTY session and release its resources.

        Sends ``SIGTERM`` (escalating to ``SIGKILL`` after 2s), closes the
        master fd, and removes the entry from the manager's table.
        """

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
        """Return a wire-safe snapshot of every active PTY session."""

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
        """Bounded non-blocking drain of the session's master fd."""

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
        """Return the tracked session or raise ``ValueError`` if unknown."""

        if session_id not in self._sessions:
            raise ValueError(f"PTY session not found: {session_id}")
        return self._sessions[session_id]
