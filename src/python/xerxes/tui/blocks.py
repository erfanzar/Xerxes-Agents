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
"""Content block renderers for the Xerxes TUI.

This module defines various block types used to accumulate and format
streaming content, thinking traces, tool calls, notifications, and
interactive panels such as approval and question requests.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, ClassVar

from prompt_toolkit.formatted_text import AnyFormattedText

from .console import severity_color, severity_icon

try:
    import markdown_it

    _MD_PARSER = markdown_it.__dict__.get("MarkdownIt", markdown_it.MarkdownIt())()
except Exception:

    class _DummyParser:
        """Dummy parser."""

        def parse(self, text: str) -> list[Any]:
            """Parse.

            Args:
                self: IN: The instance. OUT: Used for attribute access.
                text (str): IN: text. OUT: Consumed during execution.
            Returns:
                list[Any]: OUT: Result of the operation."""
            return []

    _MD_PARSER = _DummyParser()

_BLOCK_PATTERN = re.compile(
    r"^(?:(#{1,6}\s)|"
    r"(\`\`\`)|"
    r"(```)|"
    r"(\>)\s|"
    r"(\*\s)|"
    r"(\d+\.\s)|"
    r"(\-\s)|"
    r"(\+\s)|"
    r"(---)|"
    r"(\*\*\*)|"
    r"(___))",
    re.MULTILINE,
)

_CODE_FENCE_PATTERN = re.compile(r"```(\w*)")


def _is_block_boundary(text: str) -> bool:
    """Check whether the end of ``text`` looks like a block boundary.

    A block boundary is detected by a double newline or a line that matches
    common Markdown block-start patterns.

    Args:
        text (str): IN: Accumulated text to inspect. OUT: Checked for boundary
            markers at its end.

    Returns:
        bool: OUT: ``True`` if the text ends at a block boundary.
    """
    if not text:
        return False

    if text.endswith("\n\n"):
        return True

    return bool(_BLOCK_PATTERN.match(text.rstrip("\n").split("\n")[-1]))


def _extract_key_argument(arguments: str | dict[str, Any]) -> str:
    """Extract a short representative argument string from tool arguments.

    Tries to parse JSON if needed, then looks for common keys such as
    ``"path"``, ``"file"``, ``"query"``, etc.

    Args:
        arguments (str | dict[str, Any]): IN: Raw tool arguments, either a JSON
            string or a dictionary. OUT: Parsed and inspected for a key value.

    Returns:
        str: OUT: Truncated representative argument value, or empty string.
    """
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except Exception:
            return arguments[:60] if arguments else ""

    if not isinstance(arguments, dict):
        return str(arguments)[:60]

    for key in ("path", "file", "query", "url", "name", "code", "command", "target"):
        if arguments.get(key):
            val = arguments[key]
            return str(val)[:80]

    for v in arguments.values():
        if v:
            return str(v)[:80]
    return ""


class _ToolCallLexer:
    """Incremental JSON lexer for streaming tool-call arguments.

    Buffers incoming chunks and attempts to parse a complete JSON object.
    """

    def __init__(self) -> None:
        """Initialize the lexer with an empty buffer."""
        self._buffer = ""
        self._complete = False
        self._arguments: dict[str, Any] | None = None

    def append(self, chunk: str) -> None:
        """Append a text chunk and attempt to parse complete JSON.

        Args:
            chunk (str): IN: Next fragment of JSON text. OUT: Added to the
                internal buffer and parsed.
        """
        if self._complete:
            return
        self._buffer += chunk
        try:
            self._arguments = json.loads(self._buffer)
            self._complete = True
        except json.JSONDecodeError:
            pass

    @property
    def arguments(self) -> dict[str, Any] | None:
        """Return the parsed arguments dictionary if complete.

        Returns:
            dict[str, Any] | None: OUT: Parsed arguments or ``None``.
        """
        return self._arguments

    @property
    def is_complete(self) -> bool:
        """Return whether the JSON has been fully parsed.

        Returns:
            bool: OUT: ``True`` if parsing succeeded.
        """
        return self._complete


class _ContentBlock:
    """Accumulates streaming text content and splits it at block boundaries.

    Attributes:
        MAX_LIVE_LINES (int): Maximum number of live lines to keep.
    """

    MAX_LIVE_LINES = 40

    def __init__(self, block_id: str) -> None:
        """Initialize a content block.

        Args:
            block_id (str): IN: Unique identifier for this block. OUT: Stored
                as an instance attribute.
        """
        self.block_id = block_id
        self._raw = ""
        self._committed = False
        self._started_at = time.monotonic()
        self._token_count = 0
        self._subagent_text_len = 0

    def append(self, text: str) -> list[str]:
        """Append text and return any committed paragraph segments.

        Args:
            text (str): IN: Incoming text chunk. OUT: Appended to the block
                buffer and split at block boundaries.

        Returns:
            list[str]: OUT: List of committed paragraph strings, if any.
        """
        self._raw += text
        self._token_count += len(text.split())

        if not _is_block_boundary(self._raw):
            return []

        parts = self._raw.split("\n\n")
        if len(parts) <= 1:
            return []

        committed = "\n\n".join(parts[:-1])
        self._raw = parts[-1] if parts[-1].strip() else ""
        return [committed]

    @property
    def raw_text(self) -> str:
        """Return the current raw buffer text.

        Returns:
            str: OUT: Uncommitted raw text.
        """
        return self._raw

    @property
    def committed_text(self) -> str:
        """Return the current committed text.

        Returns:
            str: OUT: Same as :attr:`raw_text` for this implementation.
        """
        return self._raw

    def finalize(self) -> None:
        """Mark the block as finalized."""
        self._committed = True

    @property
    def is_finalized(self) -> bool:
        """Return whether the block has been finalized.

        Returns:
            bool: OUT: ``True`` if finalized.
        """
        return self._committed

    def compose(self, *, running: bool = False) -> AnyFormattedText:
        """Render the block into formatted text for display.

        Args:
            running (bool): IN: Whether the block is actively streaming. OUT:
                Controls spinner display in the rendered output.

        Returns:
            AnyFormattedText: OUT: ANSI-formatted text representation.
        """
        from prompt_toolkit.formatted_text import ANSI

        if not self._raw.strip():
            return ANSI("")

        elapsed = int(time.monotonic() - self._started_at)
        spinner = "⠋" if running else " "
        tail = self._raw.strip()
        return ANSI(f"[dim]{spinner} {tail}  [dim]{elapsed}s · {self._token_count} tokens[/dim]")


@dataclass
class _ThinkingBlock:
    """Accumulates streaming thinking traces with an animated spinner.

    Attributes:
        block_id (str): Unique block identifier.
    """

    block_id: str
    _raw: str = ""
    _started_at: float = field(default_factory=lambda: time.monotonic(), repr=False)
    _frame: int = 0
    _frames: tuple[str, ...] = (".  ", ".. ", "...", " ..", "  .", "   ")
    _committed: bool = field(default=False, repr=False)

    def append(self, text: str) -> list[str]:
        """Append thinking text and return committed paragraph segments.

        Args:
            text (str): IN: Incoming thinking chunk. OUT: Appended to the buffer
                and split at block boundaries.

        Returns:
            list[str]: OUT: List of committed paragraph strings, if any.
        """
        self._raw += text

        if "\n\n" not in text and not _is_block_boundary(self._raw):
            return []
        parts = self._raw.split("\n\n")
        committed = "\n\n".join(parts[:-1])
        self._raw = parts[-1] if parts[-1].strip() else ""
        return [committed]

    def finalize(self) -> None:
        """Mark the thinking block as finalized."""
        self._committed = True

    @property
    def raw_text(self) -> str:
        """Return the current raw thinking text.

        Returns:
            str: OUT: Uncommitted raw text.
        """
        return self._raw

    def compose(self, *, running: bool = False) -> AnyFormattedText:
        """Render the thinking block into formatted text.

        Args:
            running (bool): IN: Whether the thinking stream is active. OUT:
                Advances the spinner animation frame.

        Returns:
            AnyFormattedText: OUT: ANSI-formatted thinking display.
        """
        from prompt_toolkit.formatted_text import ANSI

        self._frame = (self._frame + 1) % len(self._frames)
        frame = self._frames[self._frame]
        elapsed = int(time.monotonic() - self._started_at)
        content = self._raw.strip() if self._raw.strip() else "[thinking...]"
        return ANSI(
            f"[dim i]Thinking {frame}  {elapsed}s · {len(self._raw.split())} tokens[/dim i]\n[dim i]{content}[/dim i]"
        )


@dataclass
class _SubToolCall:
    """Represents a single sub-tool call within a parent tool call.

    Attributes:
        tool_call_id (str): Unique sub-tool call ID.
        name (str): Tool name.
        key_arg (str): Short representative argument.
        status (str): Current status (``"running"``, ``"done"``, or ``"error"``).
        result (str): Tool result string.
        duration_ms (float): Execution duration in milliseconds.
    """

    tool_call_id: str
    name: str
    key_arg: str
    status: str = "running"
    result: str = ""
    duration_ms: float = 0.0

    def finish(self, result: str, duration_ms: float) -> None:
        """Mark the sub-tool call as finished.

        Args:
            result (str): IN: Result value from the tool. OUT: Stored internally.
            duration_ms (float): IN: Execution duration. OUT: Stored internally.
        """
        self.result = result
        self.duration_ms = duration_ms
        self.status = "done"

    def compose(self) -> str:
        """Render the sub-tool call into an ANSI-escaped string.

        Returns:
            str: OUT: Formatted sub-tool call line with icon and timing.
        """
        from .console import _prompt_text_to_ansi

        icon = {"running": "◐", "done": "✓", "error": "✗"}.get(self.status, "·")
        if self.status == "done":
            markup = f"  {icon} [cyan]{self.name}[/cyan] ({self.key_arg}) — {self.duration_ms:.0f}ms"
        else:
            markup = f"  {icon} [cyan]{self.name}[/cyan] ({self.key_arg})"
        return _prompt_text_to_ansi(markup)


MAX_SUBAGENT_TOOL_CALLS = 4


class _ToolCallBlock:
    """Accumulates and renders a single tool call and its sub-tool calls.

    Attributes:
        MAX_RESULT_LINES (int): Maximum result lines to display.
    """

    MAX_RESULT_LINES = 5

    def __init__(self, block_id: str, tool_call_id: str, name: str, arguments: str | None) -> None:
        """Initialize a tool call block.

        Args:
            block_id (str): IN: Unique block identifier. OUT: Stored internally.
            tool_call_id (str): IN: Tool call ID. OUT: Stored internally.
            name (str): IN: Tool name. OUT: Stored internally.
            arguments (str | None): IN: Raw tool arguments JSON string. OUT:
                Stored for incremental parsing.
        """
        self.block_id = block_id
        self.tool_call_id = tool_call_id
        self.name = name
        self._started_at = time.monotonic()
        self._status: str = "running"
        self._result: str = ""
        self._duration_ms: float = 0.0
        self._arguments: str | None = arguments
        self._args_lexer = _ToolCallLexer()
        self._sub_tool_calls: list[_SubToolCall] = []
        self._sub_tool_calls_shown = 0

    @property
    def key_arg(self) -> str:
        """Return a short representative argument for this tool call.

        Returns:
            str: OUT: Extracted key argument or truncated raw arguments.
        """
        if self._arguments:
            try:
                args = json.loads(self._arguments) if isinstance(self._arguments, str) else self._arguments
                return _extract_key_argument(args)
            except Exception:
                return self._arguments[:60]
        if self._args_lexer.arguments:
            return _extract_key_argument(self._args_lexer.arguments)
        return ""

    def append_args_part(self, chunk: str) -> None:
        """Append a fragment of tool arguments.

        Args:
            chunk (str): IN: Partial JSON string. OUT: Fed into the incremental
                lexer and appended to the raw arguments.
        """
        self._arguments = (self._arguments or "") + chunk
        self._args_lexer.append(chunk)

    def set_result(self, result: str, duration_ms: float) -> None:
        """Set the tool result and mark the call as done.

        Args:
            result (str): IN: Tool result string. OUT: Stored internally.
            duration_ms (float): IN: Execution duration. OUT: Stored internally.
        """
        self._result = result
        self._duration_ms = duration_ms
        self._status = "done"

    def append_sub_tool_call(self, tool_call_id: str, name: str, key_arg: str) -> _SubToolCall:
        """Register a new sub-tool call.

        Args:
            tool_call_id (str): IN: Sub-tool call ID. OUT: Stored in the new
                sub-tool call instance.
            name (str): IN: Sub-tool name. OUT: Stored in the new instance.
            key_arg (str): IN: Short representative argument. OUT: Stored in the
                new instance.

        Returns:
            _SubToolCall: OUT: The created sub-tool call object.
        """
        sub = _SubToolCall(tool_call_id=tool_call_id, name=name, key_arg=key_arg)
        self._sub_tool_calls.append(sub)
        return sub

    def finish_sub_tool_call(self, tool_call_id: str, result: str, duration_ms: float) -> None:
        """Mark a sub-tool call as finished by ID.

        Args:
            tool_call_id (str): IN: ID of the sub-tool call to finish. OUT: Used
                to look up the matching sub-tool call.
            result (str): IN: Result string. OUT: Passed to the sub-tool call.
            duration_ms (float): IN: Execution duration. OUT: Passed to the sub-tool call.
        """
        for sub in self._sub_tool_calls:
            if sub.tool_call_id == tool_call_id:
                sub.finish(result, duration_ms)
                break

    def set_subagent_metadata(self, agent_id: str, subagent_type: str) -> None:
        """Store metadata identifying this tool call as a subagent invocation.

        Args:
            agent_id (str): IN: Subagent identifier. OUT: Stored internally.
            subagent_type (str): IN: Subagent type. OUT: Stored internally.
        """
        self._subagent_id = agent_id
        self._subagent_type = subagent_type

    def compose(self) -> str:
        """Render the tool call block into an ANSI-escaped string.

        Returns:
            str: OUT: Formatted tool call display with status, timing, result,
                and sub-tool calls.
        """
        from .console import _prompt_text_to_ansi

        elapsed_s = int(time.monotonic() - self._started_at)
        duration = f"{elapsed_s}s" if self._status == "running" else f"{self._duration_ms:.0f}ms"
        icon = {"running": "◐", "done": "✓", "error": "✗"}.get(self._status, "·")
        status_word = "Using" if self._status == "running" else "Used"

        lines = [f"{icon} [bold cyan]{status_word} {self.name}[/bold cyan] ([dim]{self.key_arg}[/dim]) — {duration}"]

        sub_lines: list[str] = []
        for sub in self._sub_tool_calls[:MAX_SUBAGENT_TOOL_CALLS]:
            sub_lines.append(sub.compose())

        overflow = len(self._sub_tool_calls) - MAX_SUBAGENT_TOOL_CALLS
        if overflow > 0:
            lines.append(f"  [dim]... and {overflow} more sub-agent tool calls[/dim]")

        if self._status == "done" and self._result:
            result_lines = self._result.strip().split("\n")[: self.MAX_RESULT_LINES]
            result_text = "\n".join(result_lines)
            if len(result_lines) == self.MAX_RESULT_LINES:
                result_text += "\n  [dim](truncated)[/dim]"
            lines.append(f"  [dim]{result_text}[/dim]")

        rendered = _prompt_text_to_ansi("\n".join(lines))
        if sub_lines:
            rendered = (
                rendered.split("\n", 1)[0]
                + "\n"
                + "\n".join(sub_lines)
                + (("\n" + rendered.split("\n", 1)[1]) if "\n" in rendered else "")
            )
        return rendered


class _NotificationBlock:
    """Renders a single notification message.

    Attributes:
        MAX_BODY_LINES (int): Maximum body lines to display.
        PROSE_CATEGORIES (tuple[str, ...]): Categories rendered as prose.
    """

    MAX_BODY_LINES = 2

    def __init__(self, notification_id: str, category: str, severity: str, title: str, body: str) -> None:
        """Initialize a notification block.

        Args:
            notification_id (str): IN: Unique notification ID. OUT: Stored internally.
            category (str): IN: Notification category. OUT: Stored internally.
            severity (str): IN: Severity level. OUT: Stored internally.
            title (str): IN: Notification title. OUT: Stored internally.
            body (str): IN: Notification body text. OUT: Stored internally.
        """
        self.notification_id = notification_id
        self.category = category
        self.severity = severity
        self.title = title
        self.body = body

    PROSE_CATEGORIES = ("slash", "history")

    def compose(self) -> str:
        """Render the notification into an ANSI-escaped string.

        Returns:
            str: OUT: Formatted notification with icon, color, and optional
                Markdown rendering.
        """
        from .console import _prompt_text_to_ansi, markdown_to_ansi

        color = severity_color(self.severity)
        icon = severity_icon(self.severity)
        all_lines = self.body.strip().split("\n")

        if self.category in self.PROSE_CATEGORIES and not self.body.lstrip().startswith("✨"):
            try:
                body_text = markdown_to_ansi(self.body)
                return f"\x1b[38;5;{_severity_ansi(self.severity)}m{icon}\x1b[39m {body_text}"
            except Exception:
                pass

        if self.category == "slash" or not self.title:
            body_text = "\n".join(all_lines)
            markup = f"[{color}]{icon}[/{color}] {body_text}"
        else:
            body_text = "\n".join(all_lines[: self.MAX_BODY_LINES])
            if len(all_lines) > self.MAX_BODY_LINES:
                body_text += " [dim](truncated)[/dim]"
            markup = f"[{color}]{icon} [bold]{self.title}[/bold][/{color}]\n[dim]{body_text}[/dim]"
        return _prompt_text_to_ansi(markup)


def _severity_ansi(sev: str) -> str:
    """Return an ANSI 256-color index for a severity level.

    Args:
        sev (str): IN: Severity string such as ``"info"`` or ``"error"``. OUT:
            Looked up in the severity-to-color mapping.

    Returns:
        str: OUT: ANSI color index string, defaulting to ``"245"``.
    """
    return {"info": "51", "warning": "214", "error": "196"}.get(sev, "245")


class ApprovalRequestPanel:
    """Interactive panel for rendering and navigating an approval request.

    Attributes:
        SELECTIONS (list[str]): Available response options.
        LABELS (dict[str, str]): Human-readable labels for each option.
    """

    SELECTIONS: ClassVar[list[str]] = ["approve", "approve_for_session", "reject"]
    LABELS: ClassVar[dict[str, str]] = {
        "approve": "ENTER — approve",
        "approve_for_session": "A — approve for session",
        "reject": "R — reject",
    }

    def __init__(
        self,
        request_id: str,
        tool_call_id: str,
        action: str,
        description: str,
        diff: str | None = None,
    ) -> None:
        """Initialize an approval request panel.

        Args:
            request_id (str): IN: Approval request ID. OUT: Stored internally.
            tool_call_id (str): IN: Associated tool call ID. OUT: Stored internally.
            action (str): IN: Action description. OUT: Stored internally.
            description (str): IN: Detailed description. OUT: Stored internally.
            diff (str | None): IN: Optional diff text. OUT: Stored internally.
        """
        self.request_id = request_id
        self.tool_call_id = tool_call_id
        self.action = action
        self.description = description
        self.diff = diff
        self._selected = 0
        self._expanded = False

    @property
    def selected_response(self) -> str:
        """Return the currently selected response value.

        Returns:
            str: OUT: One of :attr:`SELECTIONS`.
        """
        return self.SELECTIONS[self._selected]

    def move_cursor_up(self) -> None:
        """Move the selection cursor up (wraps around)."""
        self._selected = (self._selected - 1) % len(self.SELECTIONS)

    def move_cursor_down(self) -> None:
        """Move the selection cursor down (wraps around)."""
        self._selected = (self._selected + 1) % len(self.SELECTIONS)

    def toggle_expand(self) -> None:
        """Toggle expansion of the diff view, if a diff is present."""
        if self.diff:
            self._expanded = not self._expanded

    def compose(self) -> str:
        """Render the approval panel into an ANSI-escaped string.

        Returns:
            str: OUT: Formatted approval panel with options and optional diff.
        """
        from .console import _prompt_text_to_ansi

        lines = [
            "[yellow bold]? Permission required[/yellow bold]",
            "",
            f"  [bold]{self.action}[/bold]",
            f"  [dim]{self.description}[/dim]",
            "",
        ]

        for i, response in enumerate(self.SELECTIONS):
            marker = "▶" if i == self._selected else " "
            label = self.LABELS[response]
            color = "green" if response == "approve" else ("cyan" if response == "approve_for_session" else "red")
            lines.append(f"  {marker} [{color}]{label}[/{color}]")

        if self.diff and self._expanded:
            lines.append("")
            lines.append("[dim]--- diff (Ctrl+E to collapse) ---[/dim]")
            lines.append(self.diff)

        return _prompt_text_to_ansi("\n".join(lines))


class QuestionRequestPanel:
    """Interactive panel for rendering and answering question requests.

    Tracks the current question index, selected option, and free-text mode.
    """

    def __init__(
        self,
        request_id: str,
        questions: list[dict[str, Any]],
    ) -> None:
        """Initialize a question request panel.

        Args:
            request_id (str): IN: Request ID. OUT: Stored internally.
            questions (list[dict[str, Any]]): IN: List of question dictionaries.
                Each dictionary may contain:

                - ``"id"`` (str): Question identifier.
                - ``"question"`` (str): Question text.
                - ``"options"`` (list[str]): Available answer options.
                - ``"allow_free_form"`` (bool): Whether free-text input is allowed.

                OUT: Stored internally for sequential answering.
        """
        self.request_id = request_id
        self.questions = questions
        self._question_index = 0
        self._option_index = 0
        self._free_text_mode = False
        self._free_text = ""
        self._answers: dict[str, str] = {}

    @property
    def current_question(self) -> dict[str, Any]:
        """Return the current question dictionary.

        Returns:
            dict[str, Any]: OUT: Current question being answered.
        """
        return self.questions[self._question_index]

    @property
    def current_options(self) -> list[str]:
        """Return the options for the current question.

        Returns:
            list[str]: OUT: List of option strings.
        """
        return self.current_question.get("options", [])

    @property
    def selected_label(self) -> str:
        """Return the label of the currently selected answer.

        Returns:
            str: OUT: Selected option label, free text, or placeholder.
        """
        opts = self.current_options
        if self._free_text_mode:
            return self._free_text or "(type your answer)"
        if opts and self._option_index < len(opts):
            return opts[self._option_index]
        return ""

    def move_up(self) -> None:
        """Move the option cursor up (wraps around)."""
        if self._free_text_mode:
            return
        self._option_index = (self._option_index - 1) % max(1, len(self.current_options))

    def move_down(self) -> None:
        """Move the option cursor down (wraps around)."""
        if self._free_text_mode:
            return
        self._option_index = (self._option_index + 1) % max(1, len(self.current_options))

    def select_other(self) -> None:
        """Switch to free-text input mode."""
        self._free_text_mode = True

    def append_free_text(self, char: str) -> None:
        """Append a character to the free-text answer.

        Args:
            char (str): IN: Character to append. OUT: Added to the free-text buffer.
        """
        if self._free_text_mode:
            self._free_text += char

    def backspace_free_text(self) -> None:
        """Remove the last character from the free-text answer."""
        if self._free_text_mode and self._free_text:
            self._free_text = self._free_text[:-1]

    def confirm_current(self) -> str | None:
        """Confirm the current answer and advance to the next question.

        Returns:
            str | None: OUT: The question ID if all questions are answered,
                otherwise ``None``.
        """
        question_id = self.current_question.get("id", f"q_{self._question_index}")
        if self._free_text_mode:
            answer = self._free_text
        else:
            opts = self.current_options
            answer = opts[self._option_index] if opts and self._option_index < len(opts) else ""

        self._answers[question_id] = answer

        if self._question_index < len(self.questions) - 1:
            self._question_index += 1
            self._option_index = 0
            self._free_text_mode = False
            self._free_text = ""
            return None
        return question_id

    @property
    def is_complete(self) -> bool:
        """Return whether all questions have been answered.

        Returns:
            bool: OUT: ``True`` if the number of answers equals the number of questions.
        """
        return len(self._answers) == len(self.questions)

    def compose(self) -> str:
        """Render the question panel into an ANSI-escaped string.

        Returns:
            str: OUT: Formatted question panel with options and instructions.
        """
        from .console import _prompt_text_to_ansi

        lines = [
            f"[bold cyan]? Question ({self._question_index + 1}/{len(self.questions)})[/bold cyan]",
            "",
            f"  [bold]{self.current_question.get('question', '')}[/bold]",
            "",
        ]

        for i, option in enumerate(self.current_options):
            marker = "▶" if i == self._option_index else " "
            lines.append(f"  {i + 1}. {marker} {option}")

        if self.current_question.get("allow_free_form"):
            other_marker = "▶" if self._free_text_mode else " "
            lines.append(f"     {other_marker} [dim]Other (type your answer)[/dim]")
            if self._free_text_mode:
                lines.append(f"       [green]> {self._free_text}_[/green]")

        lines.append("")
        lines.append("[dim]Type a number to pick, free text for custom, or /cancel to abort.[/dim]")

        return _prompt_text_to_ansi("\n".join(lines))
