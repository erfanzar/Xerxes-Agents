# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# distributed under the License is distributed on an "AS IS" BASIS,
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
    >>> from xerxes.tools.standalone import ReadFile, WriteFile
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
        Read a text file from disk and return decoded text content.

        Use this tool when you need to inspect a local file before deciding
        what to do next, for example reading a README, source file, config, or
        report that already exists in the workspace. The tool reads the file as
        text only; it does not parse structured formats for you and it is not
        suitable for binary files.

        Args:
            file_path (str):
                Absolute or relative path to the file that should be read. The
                path is expanded and resolved before opening the file, so
                ``~/notes.txt`` and ``./src/app.py`` are both valid forms.
            max_chars (int | None, optional):
                Maximum number of characters to return. If the file is longer,
                the returned text is truncated and a visible marker is appended
                so the caller can tell the result is incomplete. Set this to
                ``None`` to return the full file.
            encoding (str, optional):
                Character encoding used to decode the file, such as
                ``"utf-8"`` or ``"latin-1"``.
            errors (str, optional):
                Error-handling strategy passed to :pymeth:`Path.read_text`.
                ``"ignore"`` drops undecodable bytes, while ``"replace"`` keeps
                the read going and inserts replacement characters.
        Returns:
            str: The decoded file content. If truncation happens, the text is
            suffixed with ``"\n\n…[truncated]…"`` so the model can request a
            more targeted follow-up read if needed.

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

        Use this tool when you need to create a new text file or replace an
        existing one with generated content such as a report, config, patch
        notes, or source code. This tool writes the entire file in one shot; if
        you want to add to an existing file without replacing it, use the append
        tool instead.

        Args:
            file_path (str):
                Destination file path. Parent directories are created
                automatically if they do not already exist.
            content (str):
                Full text content to write. Whatever you pass becomes the entire
                file contents after the write completes.
            overwrite (bool, optional):
                If ``False`` (default) the call fails when the file already
                exists. Set to ``True`` to replace an existing file.
            encoding (str, optional):
                Text encoding used for writing.

        Returns:
            str: Human-readable status message that includes the final resolved
            path and the number of characters written.

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

    Lists files and directories with optional filtering by file
    extension. Directories are shown with a trailing ``/``.

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

        Use this tool to discover what files are available before reading or
        editing them. The result is intentionally shallow: it lists only the
        immediate files in the target directory, not nested files in
        subdirectories, and it excludes directories from the returned list.

        Args:
            directory_path (str, optional):
                Directory to inspect. Defaults to current working directory
                (``"."``).
            extension_filter (str | None, optional):
                If provided, only return files whose name ends with the given
                extension (case-insensitive), for example ``".py"`` or
                ``".md"``.

        Returns:
            list[str]: Sorted list of file and directory names. Directories
            have a trailing ``/`` (e.g. ``"src/"``). File entries are base
            names such as ``"README.md"``.

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

        entries = []
        for f in files:
            if f.is_dir():
                entries.append(f.name + "/")
            else:
                entries.append(f.name)
        return sorted(entries)


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
        >>> print(result["stdout"])
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

        This is best for quick one-shot computations, data transformations,
        environment inspection, or script-like tasks where you want a fresh
        Python interpreter. State does not persist between calls. If you need an
        interactive session that survives across multiple writes, use the PTY
        operator tools instead.

        Args:
            code (str):
                Python source code to execute. The code is dedented and passed
                to ``python -c``.
            timeout (float | None, optional):
                Maximum wall-clock time in seconds before the subprocess is
                terminated. ``None`` disables the limit. Defaults to ``10.0``.

        Returns:
            dict[str, str]:
                A mapping containing the captured standard streams:
                ``{"stdout": "<captured>", "stderr": "<captured>"}``. Normal
                prints go to ``stdout``; tracebacks and interpreter errors show
                up in ``stderr``.

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
        Execute a one-shot shell command and capture its output.

        Use this tool for short, non-interactive commands such as ``ls``,
        ``pwd``, ``git status``, or a single build/test command. The command is
        run through the system shell with ``shell=True`` and does not preserve
        interactive state between calls. If you need to start a long-running
        terminal session and send more input later, use ``exec_command`` and
        ``write_stdin`` instead.

        Args:
            command (str):
                The exact command string passed to the system shell. Shell
                features such as pipes, redirects, ``&&``, and variable
                expansion are available because the command is executed by the
                shell.
            timeout (float | None, optional):
                Maximum execution time in seconds. If the command runs longer,
                the subprocess is terminated and ``TimeoutExpired`` is raised.
            cwd (str | None, optional):
                Working directory for the command. ``None`` (default) means
                the current directory.

        Returns:
            dict[str, str]: ``{"stdout": ..., "stderr": ...}``. A non-zero exit
            code does not raise automatically; you should inspect ``stderr`` and
            the command output to decide what happened.

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

        Use this tool for logs, incremental reports, notes, or any workflow
        where you want to keep existing content and add new text at the end. The
        file is created if it does not yet exist.

        Args:
            file_path (str):
                Destination file path. Parent directories are created if
                required.
            lines (str):
                Text to append. The tool does not try to inspect existing file
                structure; it simply writes the supplied text followed by the
                configured newline string.
            encoding (str, optional):
                Encoding used when opening the file.
            newline (str, optional):
                Character(s) appended after ``lines``. Defaults to ``"\\n"``,
                but can be set to ``""`` if you do not want an automatic line
                ending.

        Returns:
            str: Status message specifying how many characters were appended and
            where they were written.

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
