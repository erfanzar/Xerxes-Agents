# Copyright 2025 The EasyDeL/Calute Author @erfanzar (Erfan Zare Chavoshi).
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


"""Standalone tool classes for agent file and system operations.

This module provides a collection of standalone tool classes that extend
AgentBaseFn for common file system and code execution operations.
These tools are designed to be used by agents to interact with the
file system and execute code in a controlled manner.

Available tools:
- ReadFile: Read text files with optional truncation
- WriteFile: Write text to files with directory creation
- ListDir: List directory contents with optional filtering
- ExecutePythonCode: Execute Python code in a subprocess
- ExecuteShell: Execute shell commands
- AppendFile: Append text to files

Example:
    >>> from calute.tools.standalone import ReadFile, WriteFile
    >>> content = ReadFile.static_call("config.yaml")
    >>> WriteFile.static_call("output.txt", "Hello, World!")
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from collections.abc import Iterable
from pathlib import Path

from ..types import AgentBaseFn


class ReadFile(AgentBaseFn):
    """Tool for reading text files.

    Reads the contents of a text file and returns them as a string.
    Supports optional truncation for large files and configurable
    character encoding and error handling.

    Example:
        >>> content = ReadFile.static_call("config.yaml")
        >>> content = ReadFile.static_call("large_file.txt", max_chars=1000)
    """

    @staticmethod
    def static_call(
        file_path: str,
        max_chars: int | None = 4_096,
        encoding: str = "utf-8",
        errors: str = "ignore",
        **context_variables,
    ) -> str:
        """
        Read a text file and return its contents.

        Args:
            file_path (str):
                Absolute or relative path to the file that should be read.
            max_chars (int | None, optional):
                Maximum number of characters to return.
                If ``None`` the file is returned in full.
                Defaults to ``4_096``.
            encoding (str, optional):
                Character encoding used to decode the file.
                Defaults to ``"utf-8"``.
            errors (str, optional):
                Error-handling strategy passed to :pymeth:`Path.read_text`.
                Defaults to ``"ignore"``.
        Returns:
            str: File content (possibly truncated and suffixed with the marker
            ``"\n\n…[truncated]…"``).

        Raises:
            FileNotFoundError: If the supplied path does not exist or is not a
            regular file.
        """
        p = Path(file_path).expanduser().resolve()
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"File '{p}' does not exist")

        text = p.read_text(encoding=encoding, errors=errors)
        if max_chars and len(text) > max_chars:
            text = text[:max_chars] + "\n\n…[truncated]…"
        return text


class WriteFile(AgentBaseFn):
    """Tool for writing text to files.

    Writes text content to a file, automatically creating parent
    directories if they do not exist. Supports optional overwrite
    protection and configurable character encoding.

    Example:
        >>> WriteFile.static_call("output.txt", "Hello, World!")
        >>> WriteFile.static_call("existing.txt", "New content", overwrite=True)
    """

    @staticmethod
    def static_call(
        file_path: str,
        content: str,
        overwrite: bool = False,
        encoding: str = "utf-8",
        **context_variables,
    ) -> str:
        """
        Write text to a file, creating parent directories if necessary.

        Args:
            file_path (str): Destination file path.
            content (str): Text to be written to the file.
            overwrite (bool, optional):
                If ``False`` (default) the call fails when the file already
                exists.  Set to ``True`` to overwrite.
            encoding (str, optional): Text encoding used for writing.

        Returns:
            str: Human-readable status message (✅ emoji included).

        Raises:
            FileExistsError: When the file exists and ``overwrite`` is ``False``.

        """
        p = Path(file_path).expanduser().resolve()
        if p.exists() and not overwrite:
            raise FileExistsError(f"File '{p}' already exists. Pass overwrite=True to replace it.")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding=encoding)
        return f"✅ Wrote {len(content)} characters to {p}"


class ListDir(AgentBaseFn):
    """Tool for listing directory contents.

    Lists files in a directory with optional filtering by file
    extension. Returns a sorted list of file names (directories
    are excluded from results).

    Example:
        >>> files = ListDir.static_call("./src")
        >>> python_files = ListDir.static_call("./src", extension_filter=".py")
    """

    @staticmethod
    def static_call(
        directory_path: str = ".",
        extension_filter: str | None = None,
        **context_variables,
    ) -> list[str]:
        """
        List files in a directory, optionally filtering by extension.

        Args:
            directory_path (str, optional):
                Directory to inspect. Defaults to current working directory
                (``"."``).
            extension_filter (str | None, optional):
                If provided, only return files whose name ends with the given
                extension (case-insensitive). E.g. ``".py"``.

        Returns:
            list[str]: Sorted list of file names (no directory paths included).

        Raises:
            FileNotFoundError: If the provided path does not exist or is not a
            directory.
        """
        p = Path(directory_path).expanduser().resolve()
        if not p.exists() or not p.is_dir():
            raise FileNotFoundError(f"Directory '{p}' does not exist")

        files: Iterable[Path] = p.iterdir()
        if extension_filter:
            files = [f for f in files if f.name.lower().endswith(extension_filter.lower())]

        return sorted(f.name for f in files if f.is_file())


class ExecutePythonCode(AgentBaseFn):
    """Tool for executing Python code in a subprocess.

    Executes arbitrary Python code in a separate subprocess with
    optional timeout protection. Captures stdout and stderr for
    inspection of execution results.

    Warning:
        The executed code runs with full system privileges. Use only
        in trusted environments or inside a sandbox (Docker, firejail, etc.).

    Example:
        >>> result = ExecutePythonCode.static_call("print('Hello')")
        >>> print(result["stdout"])  # "Hello\\n"
    """

    @staticmethod
    def static_call(
        code: str,
        timeout: float | None = 10.0,
        **context_variables,
    ) -> dict[str, str]:
        """
        Execute arbitrary Python code in a separate subprocess.

        SECURITY WARNING:
            The provided snippet runs with the same privileges as the caller and
            therefore **has full access to the machine**.
            Use only in trusted environments or inside a sandbox (Docker,
            `firejail`, etc.).

        Args:
            code (str): Python source code to be executed.
            timeout (float | None, optional):
                Maximum wall-clock time in seconds before the subprocess is
                terminated. ``None`` disables the limit. Defaults to ``10.0``.

        Returns:
            dict[str, str]:
                A mapping containing the captured standard streams:
                ``{"stdout": "<captured>", "stderr": "<captured>"}``.

        Raises:
            subprocess.TimeoutExpired: If execution exceeds ``timeout``.
            Exception: Any exception raised by the executed code will appear in
            ``stderr`` but will **not** be raised in the parent process.

        """
        wrapped = textwrap.dedent(code).strip()

        proc = subprocess.run(
            [sys.executable, "-c", wrapped],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {"stdout": proc.stdout, "stderr": proc.stderr}


class ExecuteShell(AgentBaseFn):
    """Tool for executing shell commands.

    Executes shell commands with optional timeout and working
    directory configuration. Captures stdout and stderr for
    inspection of command results.

    Warning:
        Shell commands run with full system privileges. Use only
        in trusted environments or with proper input sanitization.

    Example:
        >>> result = ExecuteShell.static_call("ls -la")
        >>> result = ExecuteShell.static_call("pwd", cwd="/tmp")
    """

    @staticmethod
    def static_call(
        command: str,
        timeout: float | None = 10.0,
        cwd: str | None = None,
        **context_variables,
    ) -> dict[str, str]:
        """
        Execute a shell command.

        Args:
            command (str): The exact command string passed to the system shell.
            timeout (float | None, optional):
                Maximum execution time in seconds. Defaults to ``10.0``.
            cwd (str | None, optional):
                Working directory for the command. ``None`` (default) means
                current directory.

        Returns:
            dict[str, str]: ``{"stdout": ..., "stderr": ...}``

        Raises:
            subprocess.TimeoutExpired: If the command times out.
            FileNotFoundError: When ``cwd`` does not exist.
        """
        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        return {"stdout": proc.stdout, "stderr": proc.stderr}


class AppendFile(AgentBaseFn):
    """Tool for appending text to files.

    Appends text content to the end of a file, creating the file
    and parent directories if they do not exist. Supports configurable
    encoding and newline characters.

    Example:
        >>> AppendFile.static_call("log.txt", "New log entry")
        >>> AppendFile.static_call("data.csv", "row1,row2,row3", newline="\\r\\n")
    """

    @staticmethod
    def static_call(
        file_path: str,
        lines: str,
        encoding: str = "utf-8",
        newline: str = "\n",
        **context_variables,
    ) -> str:
        """
        Append one or more lines to a text file.

        The file is created if it does not yet exist.

        Args:
            file_path (str): Destination file.
            lines (str): Text to append (no line breaks are added automatically).
            encoding (str, optional): Encoding used when opening the file.
            newline (str, optional):
                Character(s) to append after ``lines``. Defaults to ``"\\n"``.

        Returns:
            str: Status message specifying how many characters were appended.

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
