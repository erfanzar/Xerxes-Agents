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
"""Standalone file system tools for basic file operations and shell execution.

This module provides simple, standalone file and shell tools that don't require
external dependencies. These tools are always available for basic file operations.

Example:
    >>> from xerxes.tools.standalone import ReadFile, WriteFile, ExecuteShell
    >>> ReadFile.static_call(file_path="config.json", offset=0, limit=400)
    >>> ExecuteShell.static_call(command="ls -la")
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..types import AgentBaseFn

DEFAULT_READ_LINE_LIMIT = 400


class ReadFile(AgentBaseFn):
    """Read the contents of a file from the file system.

    Reads files in line chunks by default. Pass ``limit=-1`` only when the
    entire file is intentionally needed.

    Example:
        >>> ReadFile.static_call(file_path="README.md")
        >>> ReadFile.static_call(file_path="large_file.txt", offset=400, limit=400)
        >>> ReadFile.static_call(file_path="small_file.txt", limit=-1)
    """

    @staticmethod
    def get_schema() -> dict[str, Any]:
        return {
            "name": "ReadFile",
            "description": (
                "Read a file in line chunks. Defaults to the first 400 lines; pass offset to continue. "
                "Use limit=-1 only when the whole file is intentionally required. Use 'file_path' parameter "
                "(not 'path')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to read. Use this parameter name, not 'path'.",
                    },
                    "offset": {
                        "type": "integer",
                        "description": (
                            "Zero-based line offset to start reading from. Defaults to 0. "
                            "Use the next offset reported by the previous result to continue."
                        ),
                        "default": 0,
                    },
                    "limit": {
                        "type": "integer",
                        "description": (
                            "Maximum number of lines to return. Defaults to 400. "
                            "Pass -1 to read the whole file intentionally."
                        ),
                        "default": DEFAULT_READ_LINE_LIMIT,
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": (
                            "Optional legacy character cap for the selected chunk. Pass -1 for no character cap."
                        ),
                    },
                    "encoding": {
                        "type": "string",
                        "description": "Text encoding. Defaults to 'utf-8'.",
                        "default": "utf-8",
                    },
                },
                "required": ["file_path"],
            },
        }

    @staticmethod
    def static_call(
        file_path: str,
        max_chars: int | None = None,
        offset: int | None = 0,
        limit: int | None = DEFAULT_READ_LINE_LIMIT,
        encoding: str = "utf-8",
        errors: str = "ignore",
        **context_variables,
    ) -> str:
        """Read file contents.

        Args:
            file_path: Path to the file to read.
            max_chars: Optional legacy maximum characters for the selected chunk.
                ``-1`` disables the character cap.
            offset: Zero-based line offset for chunked reads. Defaults to 0.
            limit: Maximum lines to return. Defaults to 400. ``-1`` reads the
                whole file intentionally.
            encoding: Text encoding. Defaults to 'utf-8'.
            errors: How to handle encoding errors ('ignore', 'replace', etc.).
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            File contents as string, or error message if file not found.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        p = Path(file_path).expanduser().resolve()
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"File '{p}' does not exist")

        offset = 0 if offset is None else int(offset)
        limit = DEFAULT_READ_LINE_LIMIT if limit is None else int(limit)
        text = p.read_text(encoding=encoding, errors=errors)
        if limit == -1:
            selected = text
            chunk_notice = ""
        else:
            if limit < 1:
                raise ValueError("limit must be a positive integer, or -1 to read the whole file")
            if offset < 0:
                raise ValueError("offset must be >= 0")

            lines = text.splitlines(keepends=True)
            total_lines = len(lines)
            if offset >= total_lines and total_lines:
                return f"[ReadFile] Offset {offset} is past end of file ({total_lines} lines)."

            end_offset = min(offset + limit, total_lines)
            selected = "".join(lines[offset:end_offset])
            if end_offset < total_lines:
                chunk_notice = (
                    f"\n\n[ReadFile] Showing lines {offset + 1}-{end_offset} of {total_lines}. "
                    f"Continue with offset={end_offset}, limit={limit}. "
                    "Use limit=-1 only when the whole file is intentionally required."
                )
            else:
                chunk_notice = ""

        if max_chars is not None and max_chars != -1 and len(selected) > max_chars:
            selected = selected[:max_chars]
            chunk_notice = (chunk_notice + "\n\n" if chunk_notice else "\n\n") + "…[truncated by max_chars]…"
        return selected + chunk_notice


class WriteFile(AgentBaseFn):
    """Write text to a file, creating parent directories as needed.

    Creates files with support for atomic writes and parent directory creation.

    Example:
        >>> WriteFile.static_call(file_path="output.txt", content="Hello, World!")
    """

    @staticmethod
    def static_call(
        file_path: str,
        content: str,
        overwrite: bool = False,
        encoding: str = "utf-8",
        **context_variables,
    ) -> str:
        """Write text to a file.

        Args:
            file_path: Path to the file to write.
            content: Text content to write.
            overwrite: Allow overwriting existing files. Defaults to False.
            encoding: Text encoding. Defaults to 'utf-8'.
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Success message with file path and character count.

        Raises:
            FileExistsError: If file exists and overwrite is False.
        """
        p = Path(file_path).expanduser().resolve()
        if p.exists() and not overwrite:
            raise FileExistsError(f"File '{p}' already exists. Pass overwrite=True to replace it.")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding=encoding)
        return f"✅ Wrote {len(content)} characters to {p}"


class ListDir(AgentBaseFn):
    """List directory contents with optional file extension filtering.

    Simple directory listing tool that returns file and subdirectory names.

    Example:
        >>> ListDir.static_call(directory_path="src")
        >>> ListDir.static_call(directory_path="src", extension_filter=".py")
    """

    @staticmethod
    def static_call(
        directory_path: str = ".",
        extension_filter: str | None = None,
        **context_variables,
    ) -> list[str]:
        """List directory contents.

        Args:
            directory_path: Path to the directory. Defaults to current directory.
            extension_filter: Optional extension to filter by (e.g., ".py").
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Sorted list of file and directory names.

        Raises:
            FileNotFoundError: If the directory does not exist.
        """
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
    """Execute Python code in a subprocess and return captured output."""

    DEFAULT_TIMEOUT_SECS: float = 30.0

    @staticmethod
    def static_call(
        code: str,
        timeout: float | None = None,
        **context_variables,
    ) -> dict[str, str]:
        """Execute Python code with the current interpreter.

        Args:
            code: Python source to execute.
            timeout: Timeout in seconds. Defaults to 30 seconds. Set to 0 for no timeout.
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Dictionary with ``stdout``, ``stderr``, and ``returncode``.
        """
        if timeout is None:
            effective = ExecutePythonCode.DEFAULT_TIMEOUT_SECS
        else:
            try:
                parsed = float(timeout)
            except (TypeError, ValueError):
                parsed = ExecutePythonCode.DEFAULT_TIMEOUT_SECS
            effective = None if parsed <= 0 else parsed

        try:
            proc = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=effective,
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
                "stderr": (stderr + f"\n[ExecutePythonCode] code timed out after {cap}").strip(),
                "returncode": "124",
            }


class ExecuteShell(AgentBaseFn):
    """Execute shell commands in a subprocess.

    Runs shell commands with configurable timeout and working directory.

    Example:
        >>> ExecuteShell.static_call(command="ls -la")
        >>> ExecuteShell.static_call(command="git status", timeout=30)
    """

    DEFAULT_TIMEOUT_SECS: float = 30.0

    @staticmethod
    def static_call(
        command: str,
        timeout: float | None = None,
        cwd: str | None = None,
        **context_variables,
    ) -> dict[str, str]:
        """Execute a shell command.

        Args:
            command: Shell command string to execute.
            timeout: Timeout in seconds. Uses XERXES_SHELL_TIMEOUT env or DEFAULT_TIMEOUT_SECS
                if not specified. Set to 0 for no timeout. Defaults to None.
            cwd: Working directory for command execution.
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Dictionary with 'stdout', 'stderr', and 'returncode'.
        """
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
    """Append content to a file.

    Creates the file if it doesn't exist, appends content otherwise.

    Example:
        >>> AppendFile.static_call(file_path="log.txt", lines="Processing complete")
    """

    @staticmethod
    def static_call(
        file_path: str,
        lines: str,
        encoding: str = "utf-8",
        newline: str = "\n",
        **context_variables,
    ) -> str:
        """Append content to a file.

        Args:
            file_path: Path to the file to append to.
            lines: Text content to append.
            encoding: Text encoding. Defaults to 'utf-8'.
            newline: Newline character. Defaults to '\\n'.
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Success message with file path and character count.
        """
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
