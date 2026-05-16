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
"""In-memory registry of background processes spawned by tools.

Tools that fire-and-forget shell commands (long-running servers, watchers,
``run_in_background`` shells) register their ``subprocess.Popen`` here so the
user can later list, poll, kill, or wait on them via slash commands or
dedicated process-management tools. The :func:`get_default_registry` helper
returns a process-wide singleton for tools that aren't dependency-injected.
"""

from __future__ import annotations

import threading
import time
import typing as tp
import uuid
from dataclasses import dataclass, field

if tp.TYPE_CHECKING:
    import subprocess


@dataclass
class ProcessRecord:
    """Snapshot of a registered background process.

    Attributes:
        pid: OS process id at registration time.
        name: Human-readable label (defaults to ``"pid-<n>"``).
        command: Original command line that started the process.
        started_at: Epoch seconds when the process was registered.
        proc_id: Stable registry identifier (12-char hex uuid slice).
        cwd: Working directory the process was launched in, if known.
        metadata: Free-form metadata attached by the tool that registered it.
    """

    pid: int
    name: str
    command: str
    started_at: float
    proc_id: str
    cwd: str | None = None
    metadata: dict[str, tp.Any] = field(default_factory=dict)


class ProcessRegistry:
    """Thread-safe map of ``proc_id`` to ``subprocess.Popen`` plus metadata."""

    def __init__(self) -> None:
        """Initialise an empty registry guarded by an internal lock."""
        self._procs: dict[str, subprocess.Popen[tp.Any]] = {}
        self._records: dict[str, ProcessRecord] = {}
        self._lock = threading.Lock()

    def register(
        self,
        proc: subprocess.Popen[tp.Any],
        *,
        name: str = "",
        command: str = "",
        cwd: str | None = None,
        metadata: dict[str, tp.Any] | None = None,
    ) -> str:
        """Register ``proc`` and return the freshly minted ``proc_id``.

        Args:
            proc: Live ``subprocess.Popen`` to track.
            name: Optional human-readable label; defaults to ``"pid-<pid>"``.
            command: Original command string for display purposes.
            cwd: Working directory the process was launched in.
            metadata: Tool-specific metadata to attach to the record.

        Returns:
            A 12-character hex id used for every subsequent registry lookup.
        """

        proc_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._procs[proc_id] = proc
            self._records[proc_id] = ProcessRecord(
                pid=proc.pid,
                name=name or f"pid-{proc.pid}",
                command=command,
                started_at=time.time(),
                proc_id=proc_id,
                cwd=cwd,
                metadata=dict(metadata or {}),
            )
        return proc_id

    def list(self) -> list[ProcessRecord]:
        """Return records for every process currently in the registry.

        Note:
            Returns *registered* processes, alive or not — call :meth:`poll`
            on a specific id to learn whether it has exited.
        """

        with self._lock:
            return list(self._records.values())

    def get(self, proc_id: str) -> subprocess.Popen[tp.Any] | None:
        """Return the live ``Popen`` handle for ``proc_id``, or ``None``."""
        with self._lock:
            return self._procs.get(proc_id)

    def record(self, proc_id: str) -> ProcessRecord | None:
        """Return the :class:`ProcessRecord` for ``proc_id``, or ``None``."""
        with self._lock:
            return self._records.get(proc_id)

    def poll(self, proc_id: str) -> int | None:
        """Return the exit code, or ``None`` if still running / unknown id.

        Unknown ids also yield ``None``; callers should disambiguate with
        :meth:`record` when that distinction matters.
        """

        proc = self.get(proc_id)
        if proc is None:
            return None
        return proc.poll()

    def wait(self, proc_id: str, timeout: float | None = None) -> int | None:
        """Block up to ``timeout`` seconds for ``proc_id`` to exit.

        Returns the process's exit code or ``None`` on timeout / unknown id.
        """
        proc = self.get(proc_id)
        if proc is None:
            return None
        try:
            return proc.wait(timeout=timeout)
        except Exception:
            return None

    def kill(self, proc_id: str, *, force: bool = False) -> bool:
        """Signal the process to terminate.

        Args:
            proc_id: Registry id of the process.
            force: When ``True``, send ``SIGKILL`` (``Popen.kill``); otherwise
                send ``SIGTERM`` (``Popen.terminate``).

        Returns:
            ``True`` if the process was found and a signal was delivered.
        """

        proc = self.get(proc_id)
        if proc is None:
            return False
        try:
            if force:
                proc.kill()
            else:
                proc.terminate()
            return True
        except ProcessLookupError:
            return False

    def remove(self, proc_id: str) -> bool:
        """Drop a record without signalling the process; return whether it existed."""
        with self._lock:
            present = proc_id in self._procs
            self._procs.pop(proc_id, None)
            self._records.pop(proc_id, None)
            return present

    def clear(self) -> int:
        """Drop every record and return how many entries were removed."""
        with self._lock:
            count = len(self._records)
            self._procs.clear()
            self._records.clear()
            return count


_default_registry: ProcessRegistry | None = None
_default_lock = threading.Lock()


def get_default_registry() -> ProcessRegistry:
    """Return the lazily-created process-global :class:`ProcessRegistry`.

    Tools that aren't dependency-injected can fall back to this singleton;
    test code is encouraged to instantiate its own :class:`ProcessRegistry`
    instead to keep state contained.
    """

    global _default_registry
    with _default_lock:
        if _default_registry is None:
            _default_registry = ProcessRegistry()
        return _default_registry


__all__ = ["ProcessRecord", "ProcessRegistry", "get_default_registry"]
