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
"""File editing tools: search/replace and whole-file replacement."""

from __future__ import annotations

import json
from pathlib import Path

from ...types import AgentBaseFn
from ._common import _closest_match_hint, _fuzzy_whitespace_replace, _unified_diff


class FileEditTool(AgentBaseFn):
    """Edit a file by replacing text or overwriting the whole file.

    Supports two edit modes:
    - ``search_replace`` (default): find ``old_string`` and replace with
      ``new_string``. Includes whitespace-normalized fuzzy matching as a
      fallback before failing on no-match.
    - ``whole_file``: ignore ``old_string`` and replace the entire file
      content with ``new_string``. Use for small files or when the
      change spans most of the file.

    Example (search/replace):
        >>> FileEditTool.static_call(
        ...     file_path="config.py",
        ...     old_string="DEBUG = True",
        ...     new_string="DEBUG = False"
        ... )

    Example (whole file):
        >>> FileEditTool.static_call(
        ...     file_path="small.py",
        ...     new_string="print('hello')",
        ...     edit_mode="whole_file"
        ... )
    """

    @staticmethod
    def static_call(
        file_path: str,
        old_string: str = "",
        new_string: str = "",
        replace_all: bool = False,
        edit_mode: str = "search_replace",
        **context_variables,
    ) -> str:
        """Edit a file by replacing text or overwriting the whole file.

        Args:
            file_path: Path to the file to edit.
            old_string: Text to find and replace. Required for
                ``search_replace`` mode; ignored for ``whole_file``.
            new_string: Replacement text (or full new content for
                ``whole_file`` mode).
            replace_all: If True, replace all occurrences. Defaults to False.
            edit_mode: ``"search_replace"`` (default) or ``"whole_file"``.
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Success message with diff summary, or error message.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If old_string is not found or matches new_string.
        """
        p = Path(file_path).expanduser().resolve()
        if not p.exists():
            return f"Error: file not found: {file_path}"

        content = p.read_text(errors="replace")

        if edit_mode == "whole_file":
            if not new_string:
                return "Error: new_string is empty — cannot write empty file in whole_file mode."
            old_content = content
            new_content = new_string
            p.write_text(new_content)
            diff = _unified_diff(old_content, new_content, p.name)
            return f"Replaced entire file {p.name}:\n\n{diff}"

        # --- search_replace mode ---

        count = content.count(old_string)

        if count == 0:
            # Whitespace-normalized fallback: try matching after collapsing
            # runs of whitespace in both the file and the search string. This
            # handles common failures where the model's old_string has
            # slightly different indentation or line-ending spacing.
            fuzzy_result = _fuzzy_whitespace_replace(content, old_string, new_string)
            if fuzzy_result is not None:
                old_content = content
                new_content = fuzzy_result
                p.write_text(new_content)
                diff = _unified_diff(old_content, new_content, p.name)
                return f"Applied 1 fuzzy replacement(s) to {p.name}:\n\n{diff}"
            return "Error: old_string not found in file.\n" + _closest_match_hint(content, old_string)

        if count > 1 and not replace_all:
            return (
                f"Error: old_string appears {count} times. "
                "Provide more surrounding context to make it unique, or set replace_all=true."
            )

        if old_string == new_string:
            return "Error: old_string and new_string are identical."

        old_content = content
        new_content = (
            content.replace(old_string, new_string) if replace_all else content.replace(old_string, new_string, 1)
        )

        p.write_text(new_content)
        diff = _unified_diff(old_content, new_content, p.name)
        replacements = count if replace_all else 1
        return f"Applied {replacements} replacement(s) to {p.name}:\n\n{diff}"


class NotebookEditTool(AgentBaseFn):
    """Edit a Jupyter notebook cell.

    Example:
        >>> NotebookEditTool.static_call(
        ...     notebook_path="analysis.ipynb",
        ...     cell_index=2,
        ...     new_source="print('Updated!')"
        ... )
    """

    @staticmethod
    def static_call(
        notebook_path: str,
        cell_index: int,
        new_source: str,
        cell_type: str = "code",
        **context_variables,
    ) -> str:
        """Edit a notebook cell.

        Args:
            notebook_path: Path to the .ipynb file.
            cell_index: Zero-based index of the cell to edit.
            new_source: New content for the cell.
            cell_type: Cell type ('code' or 'markdown').
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Success message or error.
        """
        p = Path(notebook_path).expanduser().resolve()
        if not p.exists():
            return f"Error: notebook not found: {notebook_path}"

        try:
            nb = json.loads(p.read_text())
            cells = nb.get("cells", [])
            if cell_index < 0 or cell_index >= len(cells):
                return f"Error: cell_index {cell_index} out of range (0-{len(cells) - 1})."

            cells[cell_index]["source"] = new_source.splitlines(keepends=True)
            cells[cell_index]["cell_type"] = cell_type
            p.write_text(json.dumps(nb, indent=1) + "\n")
            return f"Updated cell {cell_index} in {p.name} ({cell_type}, {len(new_source)} chars)."
        except json.JSONDecodeError:
            return "Error: invalid notebook format."
        except Exception as e:
            return f"Error: {e}"
