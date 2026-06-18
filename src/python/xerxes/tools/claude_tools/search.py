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
"""Code search tools: glob, grep, and LSP integration."""

from __future__ import annotations

import subprocess
from pathlib import Path

from ...types import AgentBaseFn


def _has_ripgrep() -> bool:
    """Check if ripgrep is available on the system.

    Returns:
        True if ripgrep (rg) is installed and accessible.
    """
    try:
        subprocess.run(["rg", "--version"], capture_output=True, check=True)
        return True
    except Exception:
        return False


class GlobTool(AgentBaseFn):
    """Find files matching a glob pattern.

    Recursively or non-recursively searches for files matching the pattern.

    Example:
        >>> GlobTool.static_call(pattern="**/*.py")
    """

    @staticmethod
    def static_call(
        pattern: str,
        path: str | None = None,
        **context_variables,
    ) -> str:
        """Find files matching a glob pattern.

        Args:
            pattern: Glob pattern (e.g., "*.py", "**/*.txt").
            path: Directory to search in. Defaults to current working directory.
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            List of matching file paths, or message if no matches found.
        """
        base = Path(path).expanduser().resolve() if path else Path.cwd()
        try:
            matches = sorted(base.glob(pattern))
            if not matches:
                return "No files matched."
            paths = [str(m) for m in matches[:500]]
            result = "\n".join(paths)
            if len(matches) > 500:
                result += f"\n... ({len(matches) - 500} more matches)"
            return result
        except Exception as e:
            return f"Error: {e}"


class GrepTool(AgentBaseFn):
    """Search file contents using pattern matching.

    Supports both ripgrep (preferred) and standard grep with various output formats.

    Example:
        >>> GrepTool.static_call(pattern="TODO", case_insensitive=True)
    """

    @staticmethod
    def static_call(
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        output_mode: str = "files_with_matches",
        case_insensitive: bool = False,
        context: int = 0,
        **context_variables,
    ) -> str:
        """Search file contents for a pattern.

        Args:
            pattern: Regex or literal string to search for.
            path: Directory or file to search. Defaults to current directory.
            glob: Filter files by glob pattern (e.g., "*.py").
            output_mode: Output format. Options: 'files_with_matches' (default),
                'count', 'content' (line numbers).
            case_insensitive: Make search case-insensitive.
            context: Number of lines of context around matches.
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Matching lines or file list, depending on output_mode.
        """
        use_rg = _has_ripgrep()
        cmd: list[str] = ["rg" if use_rg else "grep", "--no-heading"]

        if case_insensitive:
            cmd.append("-i")
        if output_mode == "files_with_matches":
            cmd.append("-l")
        elif output_mode == "count":
            cmd.append("-c")
        else:
            cmd.append("-n")
            if context:
                cmd.extend(["-C", str(context)])

        if glob:
            if use_rg:
                cmd.extend(["--glob", glob])
            else:
                cmd.extend(["--include", glob])

        if use_rg:
            cmd.append("--no-ignore-vcs")
        else:
            cmd.append("-r")

        cmd.append(pattern)
        cmd.append(path or str(Path.cwd()))

        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            out = r.stdout.strip()
            if not out:
                return "No matches found."
            return out[:20000]
        except FileNotFoundError:
            return "Error: neither rg nor grep found on PATH."
        except subprocess.TimeoutExpired:
            return "Error: search timed out after 30s."
        except Exception as e:
            return f"Error: {e}"


class LSPTool(AgentBaseFn):
    """Interface for Language Server Protocol operations.

    Provides code navigation, diagnostics, and refactoring via LSP.

    Example:
        >>> LSPTool.static_call(action="definition", file_path="main.py", line=10)
    """

    @staticmethod
    def static_call(
        action: str,
        file_path: str = "",
        line: int = 0,
        character: int = 0,
        **context_variables,
    ) -> str:
        """Execute an LSP action.

        Args:
            action: LSP action (e.g., 'definition', 'references', 'hover').
            file_path: File to operate on.
            line: Line number (1-indexed).
            character: Character position.
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Informational message (LSP requires active server in TUI).
        """
        return (
            f"[LSP:{action}] file={file_path} line={line} char={character}\n"
            "LSP tool requires an active language server. In the TUI, this is "
            "handled by the IDE integration layer. Use Grep/Glob for code search instead."
        )
