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
"""Standalone module for Xerxes.

Exports:
    - ReadFile
    - WriteFile
    - ListDir
    - ExecutePythonCode
    - ExecuteShell
    - AppendFile"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from collections.abc import Iterable
from pathlib import Path

from ..types import AgentBaseFn


class ReadFile(AgentBaseFn):
    """Read file.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    def static_call(
        file_path: str,
        max_chars: int | None = 4_096,
        encoding: str = "utf-8",
        errors: str = "ignore",
        **context_variables,
    ) -> str:
        """Static call.

        Args:
            file_path (str): IN: file path. OUT: Consumed during execution.
            max_chars (int | None, optional): IN: max chars. Defaults to 4096. OUT: Consumed during execution.
            encoding (str, optional): IN: encoding. Defaults to 'utf-8'. OUT: Consumed during execution.
            errors (str, optional): IN: errors. Defaults to 'ignore'. OUT: Consumed during execution.
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            str: OUT: Result of the operation."""

        p = Path(file_path).expanduser().resolve()
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"File '{p}' does not exist")

        text = p.read_text(encoding=encoding, errors=errors)
        if max_chars and len(text) > max_chars:
            text = text[:max_chars] + "\n\n…[truncated]…"
        return text


class WriteFile(AgentBaseFn):
    """Write file.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    def static_call(
        file_path: str,
        content: str,
        overwrite: bool = False,
        encoding: str = "utf-8",
        **context_variables,
    ) -> str:
        """Static call.

        Args:
            file_path (str): IN: file path. OUT: Consumed during execution.
            content (str): IN: content. OUT: Consumed during execution.
            overwrite (bool, optional): IN: overwrite. Defaults to False. OUT: Consumed during execution.
            encoding (str, optional): IN: encoding. Defaults to 'utf-8'. OUT: Consumed during execution.
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            str: OUT: Result of the operation."""

        p = Path(file_path).expanduser().resolve()
        if p.exists() and not overwrite:
            raise FileExistsError(f"File '{p}' already exists. Pass overwrite=True to replace it.")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding=encoding)
        return f"✅ Wrote {len(content)} characters to {p}"


class ListDir(AgentBaseFn):
    """List dir.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    def static_call(
        directory_path: str = ".",
        extension_filter: str | None = None,
        **context_variables,
    ) -> list[str]:
        """Static call.

        Args:
            directory_path (str, optional): IN: directory path. Defaults to '.'. OUT: Consumed during execution.
            extension_filter (str | None, optional): IN: extension filter. Defaults to None. OUT: Consumed during execution.
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            list[str]: OUT: Result of the operation."""

        p = Path(directory_path).expanduser().resolve()
        if not p.exists() or not p.is_dir():
            raise FileNotFoundError(f"Directory '{p}' does not exist")

        files: Iterable[Path] = p.iterdir()
        if extension_filter:
            files = [f for f in files if f.name.lower().endswith(extension_filter.lower())]

        entries = []
        for f in files:
            if f.is_dir():
                entries.append(f.name + "/")
            else:
                entries.append(f.name)
        return sorted(entries)


class ExecutePythonCode(AgentBaseFn):
    """Execute python code.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    def static_call(
        code: str,
        timeout: float | None = 10.0,
        **context_variables,
    ) -> dict[str, str]:
        """Static call.

        Args:
            code (str): IN: code. OUT: Consumed during execution.
            timeout (float | None, optional): IN: timeout. Defaults to 10.0. OUT: Consumed during execution.
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            dict[str, str]: OUT: Result of the operation."""

        wrapped = textwrap.dedent(code).strip()

        proc = subprocess.run(
            [sys.executable, "-c", wrapped],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {"stdout": proc.stdout, "stderr": proc.stderr}


class ExecuteShell(AgentBaseFn):
    """Execute shell.

    Inherits from: AgentBaseFn

    The timeout the agent passes is honored as-is. If unset, falls back to
    ``XERXES_SHELL_TIMEOUT`` (env), then ``DEFAULT_TIMEOUT_SECS``. Pass
    ``timeout=0`` to wait forever (no cap) for intentional long-running work.

    Attributes:
        DEFAULT_TIMEOUT_SECS (float): default timeout when none requested."""

    DEFAULT_TIMEOUT_SECS: float = 30.0

    @staticmethod
    def static_call(
        command: str,
        timeout: float | None = None,
        cwd: str | None = None,
        **context_variables,
    ) -> dict[str, str]:
        """Static call.

        Args:
            command (str): IN: command. OUT: Consumed during execution.
            timeout (float | None, optional): IN: Timeout in seconds. ``None``
                uses the default (env override ``XERXES_SHELL_TIMEOUT`` or
                ``DEFAULT_TIMEOUT_SECS``). ``0`` disables the timeout entirely.
                Any positive value is used verbatim.
            cwd (str | None, optional): IN: cwd. Defaults to None. OUT: Consumed during execution.
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            dict[str, str]: OUT: Result of the operation."""

        import os as _os

        effective: float | None
        if timeout is None:
            env_default = _os.environ.get("XERXES_SHELL_TIMEOUT")
            try:
                effective = float(env_default) if env_default else ExecuteShell.DEFAULT_TIMEOUT_SECS
            except (TypeError, ValueError):
                effective = ExecuteShell.DEFAULT_TIMEOUT_SECS
        else:
            try:
                t = float(timeout)
            except (TypeError, ValueError):
                t = ExecuteShell.DEFAULT_TIMEOUT_SECS
            effective = None if t <= 0 else t

        try:
            proc = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=effective,
                cwd=cwd,
            )
            return {
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "returncode": str(proc.returncode),
            }
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            cap = f"{effective:.0f}s" if effective is not None else "no-cap"
            return {
                "stdout": stdout,
                "stderr": (stderr + f"\n[ExecuteShell] command timed out after {cap}").strip(),
                "returncode": "124",
            }


class AppendFile(AgentBaseFn):
    """Append file.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    def static_call(
        file_path: str,
        lines: str,
        encoding: str = "utf-8",
        newline: str = "\n",
        **context_variables,
    ) -> str:
        """Static call.

        Args:
            file_path (str): IN: file path. OUT: Consumed during execution.
            lines (str): IN: lines. OUT: Consumed during execution.
            encoding (str, optional): IN: encoding. Defaults to 'utf-8'. OUT: Consumed during execution.
            newline (str, optional): IN: newline. Defaults to '\n'. OUT: Consumed during execution.
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            str: OUT: Result of the operation."""

        p = Path(file_path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding=encoding) as f:
            f.write(lines + newline)
        return f"✅ Appended {len(lines)} characters to {p}"


__all__ = (
    "AppendFile",
    "ExecutePythonCode",
    "ExecuteShell",
    "ListDir",
    "ReadFile",
    "WriteFile",
)
