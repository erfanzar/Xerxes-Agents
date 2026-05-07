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
"""Prompt toolkit-based terminal UI components for Xerxes.

This module defines the interactive prompt application using
``prompt_toolkit``, including slash-command completion, status/footer
renderers, and the :class:`PersistentPrompt` orchestrator.
"""

from __future__ import annotations

import asyncio
import os
import re
from collections.abc import Callable, Coroutine
from typing import Any

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.data_structures import Point
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import ANSI, AnyFormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout.containers import Float, FloatContainer, HSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType

_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

SLASH_COMMANDS: list[tuple[str, str]] = [
    ("/help", "Show available commands"),
    ("/provider", "Setup or switch provider profile"),
    ("/model", "Switch model"),
    ("/sampling", "View or set sampling parameters"),
    ("/compact", "Summarize conversation to free context"),
    ("/plan", "Toggle plan mode (or /plan OBJECTIVE)"),
    ("/agents", "List agent types and running agents"),
    ("/skills", "List available skills"),
    ("/skill", "Invoke a skill by name"),
    ("/skill-create", "Create a new skill"),
    ("/cost", "Show cost summary"),
    ("/context", "Show context info"),
    ("/clear", "Clear conversation"),
    ("/tools", "List available tools"),
    ("/thinking", "Toggle thinking display"),
    ("/verbose", "Toggle verbose mode"),
    ("/debug", "Toggle debug mode"),
    ("/permissions", "Cycle permission mode"),
    ("/yolo", "Toggle accept-all permission mode"),
    ("/config", "Show config"),
    ("/history", "Show message count"),
    ("/btw", "Send a side question while running"),
    ("/steer", "Inject mid-turn guidance"),
    ("/cancel", "Cancel the current turn"),
    ("/cancel-all", "Cancel all running turns"),
    ("/exit", "Exit Xerxes"),
]


class SlashCompleter(Completer):
    """Completer for slash commands and registered skills.

    Provides tab-completion for entries in :data:`SLASH_COMMANDS` and any
    dynamically registered skill names.
    """

    def __init__(self, commands: list[tuple[str, str]] = SLASH_COMMANDS) -> None:
        """Initialize the completer.

        Args:
            commands (list[tuple[str, str]]): IN: List of ``(command, description)``
                tuples. OUT: Stored for completion matching.
        """
        self._commands = commands
        self._skills: list[str] = []

    def set_skills(self, skills: list[str]) -> None:
        """Update the list of skill names used for completion.

        Args:
            skills (list[str]): IN: Skill names. OUT: Deduplicated and sorted
                internally.
        """
        self._skills = sorted(set(skills))

    def get_completions(self, document: Document, complete_event: CompleteEvent):
        """Yield completions matching the current document text.

        Args:
            document (Document): IN: Current prompt_toolkit document. OUT: Used
                to extract text before the cursor for prefix matching.
            complete_event (CompleteEvent): IN: Completion trigger event. OUT:
                Ignored in this implementation.

        Yields:
            Completion: OUT: Matching slash commands or skills.
        """
        text = document.text_before_cursor

        if not text.startswith("/"):
            return

        if " " in text:
            return
        prefix = text[1:].lower()
        for name, desc in self._commands:
            stem = name[1:].lower()
            if prefix and prefix not in stem:
                continue
            yield Completion(
                name,
                start_position=-len(text),
                display=name,
                display_meta=desc,
            )
        for skill in self._skills:
            stem = skill.lower()
            if prefix and prefix not in stem:
                continue
            yield Completion(
                f"/{skill}",
                start_position=-len(text),
                display=f"/{skill}",
                display_meta="skill",
            )


class StatusRenderer:
    """Renders the main scrolling status area above the input line.

    Accumulates content lines, streaming text, thinking text, active tools,
    and spinner state into a single markup string.
    """

    SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

    def __init__(self) -> None:
        """Initialize the status renderer with empty state."""
        self._content_lines: list[str] = []
        self._running = False
        self._queue_count = 0
        self._plan_mode = False
        self._token_info = ""
        self._cost_info = ""

        self._active_panel: str = ""

        self._streaming_text: str = ""

        self._thinking_text: str = ""

        self._active_tools: dict[str, Callable[[], str]] = {}

        self._subagent_previews: dict[str, str] = {}

        self._spinner_frame: int = 0
        self._spinner_started_at: float = 0.0
        self._spinner_label: str = "Working"

    def set_running(self, running: bool) -> None:
        """Update the running state and reset spinner timing.

        Args:
            running (bool): IN: Whether a turn is currently running. OUT: Updates
                internal state and resets spinner timing when transitioning to running.
        """
        import time

        was_running = self._running
        self._running = running
        if running and not was_running:
            self._spinner_started_at = time.monotonic()
            self._spinner_frame = 0

    def reset_spinner_timer(self) -> None:
        """Restart the spinner elapsed counter from zero.

        Called whenever a new tool call begins or the active label changes so
        the spinner shows the current step's runtime, not the whole turn's.
        """
        import time

        self._spinner_started_at = time.monotonic()

    def set_spinner_label(self, label: str) -> None:
        """Set the descriptive label shown next to the spinner.

        Args:
            label (str): IN: Spinner label text. OUT: Stored internally.
        """
        new_label = label or "Working"
        if new_label != self._spinner_label:
            self.reset_spinner_timer()
        self._spinner_label = new_label

    def set_queue_count(self, count: int) -> None:
        """Set the number of queued user inputs.

        Args:
            count (int): IN: Queue count. OUT: Clamped to non-negative and stored.
        """
        self._queue_count = max(0, count)

    def set_plan_mode(self, plan_mode: bool) -> None:
        """Toggle plan mode display.

        Args:
            plan_mode (bool): IN: Whether plan mode is active. OUT: Stored internally.
        """
        self._plan_mode = plan_mode

    def set_stats(self, tokens: str = "", cost: str = "") -> None:
        """Set token and cost statistics strings.

        Args:
            tokens (str): IN: Token count info. OUT: Stored internally.
            cost (str): IN: Cost info. OUT: Stored internally.
        """
        self._token_info = tokens
        self._cost_info = cost

    def append_line(self, line: str) -> None:
        """Append a line to the content history.

        Args:
            line (str): IN: Line to append. OUT: Added to content lines.
        """
        self._content_lines.append(line)

    def clear_content(self) -> None:
        """Clear all accumulated content lines."""
        self._content_lines.clear()

    def set_active_panel(self, text: str) -> None:
        """Set the active panel text (e.g., question panel).

        Args:
            text (str): IN: Panel markup text. OUT: Stored internally.
        """
        self._active_panel = text or ""

    def clear_active_panel(self) -> None:
        """Clear the active panel text."""
        self._active_panel = ""

    def append_streaming(self, text: str) -> None:
        """Append text to the live streaming buffer.

        Args:
            text (str): IN: Streaming chunk. OUT: Added to the streaming buffer.
        """
        self._streaming_text += text

    def commit_streaming(self) -> None:
        """Move the streaming buffer into committed content lines."""
        if self._streaming_text:
            self._content_lines.append(self._streaming_text)
            self._streaming_text = ""

    def clear_streaming(self) -> None:
        """Discard the streaming buffer without committing."""
        self._streaming_text = ""

    def append_thinking(self, text: str) -> None:
        """Append text to the live thinking buffer.

        Args:
            text (str): IN: Thinking chunk. OUT: Added to the thinking buffer.
        """
        self._thinking_text += text

    def clear_thinking(self) -> None:
        """Clear the thinking buffer."""
        self._thinking_text = ""

    def set_active_tool(self, tool_call_id: str, render_fn: Callable[[], str]) -> None:
        """Register a render function for an active tool call.

        Args:
            tool_call_id (str): IN: Tool call identifier. OUT: Used as the key
                in the active tools mapping.
            render_fn (Callable[[], str]): IN: Callable returning the tool display
                string. OUT: Stored for rendering on each status update.
        """
        self._active_tools[tool_call_id] = render_fn

    def pop_active_tool(self, tool_call_id: str) -> str:
        """Remove and render an active tool call.

        Args:
            tool_call_id (str): IN: Tool call identifier. OUT: Looked up and
                removed from the active tools mapping.

        Returns:
            str: OUT: Rendered tool text, or empty string if not found.
        """
        render = self._active_tools.pop(tool_call_id, None)
        return render() if render is not None else ""

    def clear_active_tools(self) -> None:
        """Remove all active tool render functions."""
        self._active_tools.clear()

    def set_subagent_preview(self, task_id: str, label: str, text: str) -> None:
        """Set or update the live preview line for a sub-agent.

        Args:
            task_id (str): IN: Sub-agent task identifier (key for the preview).
            label (str): IN: Display label (e.g., ``"telegram-planner#abc"``).
            text (str): IN: Latest preview text. Replaces any prior content for
                the same ``task_id`` — never appended.
        """
        if not task_id:
            return
        self._subagent_previews[task_id] = f"{label}: {text}" if text else label

    def clear_subagent_preview(self, task_id: str) -> None:
        """Remove the live preview line for a finished sub-agent.

        Args:
            task_id (str): IN: Sub-agent task identifier whose preview to drop.
        """
        self._subagent_previews.pop(task_id, None)

    def clear_subagent_previews(self) -> None:
        """Remove all live sub-agent preview lines."""
        self._subagent_previews.clear()

    def _terminal_columns(self) -> int:
        """Return the current terminal column count.

        Returns:
            int: OUT: Terminal columns, defaulting to 80 if unavailable.
        """
        columns = 80
        try:
            from prompt_toolkit.application.current import get_app

            columns = get_app().output.get_size().columns
        except Exception:
            pass
        return columns

    def _terminal_rows(self) -> int:
        """Return the current terminal row count.

        Returns:
            int: OUT: Terminal rows, defaulting to 24 if unavailable.
        """
        try:
            from prompt_toolkit.application.current import get_app

            return get_app().output.get_size().rows
        except Exception:
            return 24

    def _markup(self) -> str:
        """Build the full status markup string.

        Returns:
            str: OUT: Concatenated markup of content, thinking, tools, streaming,
                panels, spinner, and border.
        """
        parts: list[str] = []

        budget = max(20, self._terminal_rows() * 2)
        for line in self._content_lines[-budget:]:
            parts.append(line)
            if line and not line.endswith("\n"):
                parts.append("\n")

        if self._thinking_text:
            parts.append(f"\x1b[2;3m✻ {self._thinking_text}\x1b[0m")
            if not self._thinking_text.endswith("\n"):
                parts.append("\n")

        for render_fn in self._active_tools.values():
            try:
                tool_text = render_fn()
            except Exception:
                continue
            parts.append(tool_text)
            if not tool_text.endswith("\n"):
                parts.append("\n")

        if self._subagent_previews:
            spin_frame = self.SPINNER_FRAMES[self._spinner_frame % len(self.SPINNER_FRAMES)]
            for preview in self._subagent_previews.values():
                parts.append(f"\x1b[36m{spin_frame}\x1b[0m \x1b[2;36m↳ {preview}\x1b[0m\n")

        if self._streaming_text:
            parts.append(self._streaming_text)
            if not self._streaming_text.endswith("\n"):
                parts.append("\n")

        if self._active_panel:
            parts.append(self._active_panel)
            if not self._active_panel.endswith("\n"):
                parts.append("\n")

        columns = self._terminal_columns()

        if self._running:
            import time

            self._spinner_frame = (self._spinner_frame + 1) % len(self.SPINNER_FRAMES)
            frame = self.SPINNER_FRAMES[self._spinner_frame]
            elapsed = int(time.monotonic() - self._spinner_started_at) if self._spinner_started_at else 0
            queued = f"  ·  {self._queue_count} queued" if self._queue_count else ""
            spinner_line = (
                f"\x1b[36m{frame}\x1b[0m \x1b[1m{self._spinner_label}…\x1b[0m  "
                f"\x1b[2m{elapsed}s{queued}  ·  esc to interrupt\x1b[0m"
            )
            parts.append(spinner_line + "\n")

        title_parts = ["input"]
        if self._plan_mode:
            title_parts.append("plan")
        if self._queue_count:
            title_parts.append(f"{self._queue_count} queued")
        title = f" {' · '.join(title_parts)} "
        dash = "╌" if self._plan_mode else "─"
        border = f"{dash}{dash}{title}{dash * max(0, columns - len(title) - 2)}"
        parts.append(f"\x1b[2m{border}\x1b[0m\n")

        return "".join(parts)

    def line_count(self) -> int:
        """Count the number of visible lines in the current markup.

        Returns:
            int: OUT: Line count after stripping ANSI codes.
        """
        markup = self._markup()
        if not markup:
            return 0
        plain = _ANSI_RE.sub("", markup)
        return plain.count("\n") + (0 if plain.endswith("\n") else 1)

    def __call__(self) -> AnyFormattedText:
        """Return the status markup as formatted text.

        Returns:
            AnyFormattedText: OUT: ANSI-formatted status text for prompt_toolkit.
        """
        return ANSI(self._markup())


class FooterRenderer:
    """Renders the footer bar with session, model, and context info."""

    def __init__(self) -> None:
        """Initialize the footer with default values."""
        self._agent_name = "agent"
        self._model = ""
        self._cwd = ""
        self._branch = ""
        self._running = False
        self._context_used = 0
        self._context_max = 0
        self._tip = "ctrl-j: newline"

    def line_count(self) -> int:
        """Count the visible lines in the footer markup.

        Returns:
            int: OUT: Number of visible footer lines.
        """
        markup = self._markup()
        plain = _ANSI_RE.sub("", markup)
        return max(1, plain.count("\n"))

    def set_session(self, agent_name: str, model: str, cwd: str, branch: str) -> None:
        """Update session metadata displayed in the footer.

        Args:
            agent_name (str): IN: Agent name. OUT: Stored if non-empty.
            model (str): IN: Model identifier. OUT: Stored if non-empty.
            cwd (str): IN: Current working directory. OUT: Stored shortened if non-empty.
            branch (str): IN: Git branch name. OUT: Stored if non-empty.
        """
        if agent_name:
            self._agent_name = agent_name
        if model:
            self._model = model
        if cwd:
            self._cwd = _shorten_home(cwd)
        if branch:
            self._branch = branch

    def set_running(self, running: bool) -> None:
        """Update the running indicator in the footer.

        Args:
            running (bool): IN: Whether a turn is running. OUT: Stored internally.
        """
        self._running = running

    def set_context(self, used: int, max_: int) -> None:
        """Update context usage numbers in the footer.

        Args:
            used (int): IN: Used context tokens. OUT: Clamped to non-negative.
            max_ (int): IN: Maximum context tokens. OUT: Clamped to non-negative.
        """
        self._context_used = max(0, used)
        self._context_max = max(0, max_)

    def _terminal_columns(self) -> int:
        """Return the current terminal column count.

        Returns:
            int: OUT: Terminal columns, defaulting to 80 if unavailable.
        """
        try:
            from prompt_toolkit.application.current import get_app

            return get_app().output.get_size().columns
        except Exception:
            return 80

    @staticmethod
    def _format_tokens(n: int) -> str:
        """Format a token count with k/M suffixes.

        Args:
            n (int): IN: Raw token count. OUT: Formatted to a human-readable string.

        Returns:
            str: OUT: Formatted token count.
        """
        if n >= 1_000_000:
            return f"{n / 1_000_000:.1f}M"
        if n >= 1_000:
            return f"{n / 1_000:.1f}k"
        return str(n)

    @staticmethod
    def _truncate_path(path: str, budget: int) -> str:
        """Truncate a path to fit within a character budget.

        Args:
            path (str): IN: Full path string. OUT: Truncated with a leading ellipsis
                if it exceeds the budget.
            budget (int): IN: Maximum character length. OUT: Used to calculate
                truncation.

        Returns:
            str: OUT: Truncated path string.
        """
        if len(path) <= budget:
            return path
        return "…" + path[-(budget - 1) :]

    def _markup(self) -> str:
        """Build the footer markup string.

        Returns:
            str: OUT: Formatted footer with agent, model, cwd, branch, and context.
        """
        spinner = "●" if self._running else "○"
        model = self._model or "—"
        agent = f"agent ({model} {spinner})"
        columns = self._terminal_columns()

        cwd_max = max(20, columns // 3)
        cwd_short = self._truncate_path(self._cwd, cwd_max) if self._cwd else ""
        left_segments = [agent]
        if cwd_short:
            left_segments.append(cwd_short)
        if self._branch:
            left_segments.append(self._branch)
        left_segments.append(self._tip)
        left = "  ".join(left_segments)

        if self._context_max > 0:
            pct = (self._context_used / self._context_max) * 100
            ctx_text = (
                f"context: {pct:.1f}% "
                f"({self._format_tokens(self._context_used)}/"
                f"{self._format_tokens(self._context_max)})"
            )
        else:
            ctx_text = "context: 0.0% (0/0)"

        plain_left = _ANSI_RE.sub("", left)
        plain_right = _ANSI_RE.sub("", ctx_text)

        rule = "\x1b[2m" + ("─" * columns) + "\x1b[0m"

        if len(plain_left) + len(plain_right) + 2 <= columns:
            gap = max(1, columns - len(plain_left) - len(plain_right))
            row = f"\x1b[2m{left}{' ' * gap}{ctx_text}\x1b[0m"
            return rule + "\n" + row + "\n"

        right_pad = max(0, columns - len(plain_right))
        line1 = f"\x1b[2m{left}\x1b[0m"
        line2 = f"{' ' * right_pad}\x1b[2m{ctx_text}\x1b[0m"
        return rule + "\n" + line1 + "\n" + line2 + "\n"

    def __call__(self) -> AnyFormattedText:
        """Return the footer markup as formatted text.

        Returns:
            AnyFormattedText: OUT: ANSI-formatted footer text for prompt_toolkit.
        """
        return ANSI(self._markup())


def _shorten_home(path: str) -> str:
    """Replace the user's home directory prefix with ``~``.

    Args:
        path (str): IN: Absolute or relative path. OUT: Shortened if it starts
            with the home directory.

    Returns:
        str: OUT: Path with home directory replaced by ``~``.
    """
    import os

    home = os.path.expanduser("~")
    if path == home:
        return "~"
    if path.startswith(home + os.sep):
        return "~" + path[len(home) :]
    return path


class PersistentPrompt:
    """Persistent terminal prompt built on ``prompt_toolkit``.

    Manages the full-screen layout including the scrolling status area,
    input buffer with slash-command completion, footer, key bindings, and
    an input queue for consuming submitted text.
    """

    def __init__(
        self,
        on_slash: Callable[[str], Coroutine[Any, Any, None]] | None = None,
        on_submit: Callable[[str], Coroutine[Any, Any, None]] | None = None,
    ) -> None:
        """Initialize the prompt application and layout.

        Args:
            on_slash (Callable[[str], Coroutine[Any, Any, None]] | None): IN:
                Async callback invoked for slash commands. OUT: Stored internally.
            on_submit (Callable[[str], Coroutine[Any, Any, None]] | None): IN:
                Async callback invoked on normal text submission. OUT: Stored internally.
        """
        self._status = StatusRenderer()
        self._footer = FooterRenderer()
        self._input_queue: asyncio.Queue[str] = asyncio.Queue()
        self._running = False
        self._exit_armed = False

        self._active_question: Any = None
        self._active_approval: Any = None

        self._scroll_y: int | None = None
        self._kb = self._build_key_bindings()
        self._on_slash = on_slash
        self._on_submit = on_submit

        self._completer = SlashCompleter()
        self._input_buffer = Buffer(
            multiline=False,
            accept_handler=self._accept_handler,
            completer=self._completer,
            complete_while_typing=True,
        )

        self._status_control = FormattedTextControl(
            text=self._status,
            focusable=False,
            show_cursor=False,
            get_cursor_position=self._status_cursor_position,
        )

        self._status_window = Window(
            content=self._status_control,
            height=Dimension(weight=1),
            wrap_lines=True,
        )

        self._status_control.mouse_handler = self._on_status_mouse

        self._buffer_control = BufferControl(
            buffer=self._input_buffer,
            focusable=True,
        )

        self._buffer_window = Window(
            content=self._buffer_control,
            height=1,
            get_line_prefix=self._input_prefix,
        )

        self._footer_control = FormattedTextControl(
            text=self._footer,
            focusable=False,
            show_cursor=False,
        )

        self._footer_window = Window(
            content=self._footer_control,
            height=lambda: Dimension.exact(self._footer.line_count()),
            dont_extend_height=True,
        )

        self._root_container = FloatContainer(
            content=HSplit([self._status_window, self._buffer_window, self._footer_window]),
            floats=[
                Float(
                    xcursor=True,
                    ycursor=True,
                    content=CompletionsMenu(max_height=10, scroll_offset=1),
                ),
            ],
        )

        self._layout = Layout(
            container=self._root_container,
            focused_element=self._buffer_control,
        )

        self._app: Application[None] | None = None

    def _status_cursor_position(self) -> Point:
        """Compute the cursor position within the status control.

        Returns:
            Point: OUT: Cursor coordinates for prompt_toolkit.
        """
        plain = _ANSI_RE.sub("", self._status._markup())
        lines = plain.split("\n")
        last = max(0, len(lines) - 1)
        if self._scroll_y is None or self._scroll_y >= last:
            return Point(x=len(lines[-1]) if lines else 0, y=last)
        y = max(0, self._scroll_y)
        return Point(x=0, y=y)

    def _status_total_lines(self) -> int:
        """Return the total number of lines in the status markup.

        Returns:
            int: OUT: Total visible lines.
        """
        return max(0, len(_ANSI_RE.sub("", self._status._markup()).split("\n")) - 1)

    def _scroll_by(self, delta: int) -> None:
        """Scroll the status view by a delta.

        Args:
            delta (int): IN: Number of lines to scroll (negative for up). OUT:
                Applied to the scroll offset.
        """
        total = self._status_total_lines()
        current = self._scroll_y if self._scroll_y is not None else total
        new = current + delta
        if new >= total:
            self._scroll_y = None
        else:
            self._scroll_y = max(0, new)
        self._invalidate()

    def _on_status_mouse(self, mouse_event: MouseEvent) -> Any:
        """Handle mouse scroll events on the status area.

        Args:
            mouse_event (MouseEvent): IN: Mouse event from prompt_toolkit. OUT:
                Used to determine scroll direction.

        Returns:
            Any: OUT: ``None`` if handled, or ``NotImplemented`` otherwise.
        """
        if mouse_event.event_type == MouseEventType.SCROLL_UP:
            self._scroll_by(-1)
            return None
        if mouse_event.event_type == MouseEventType.SCROLL_DOWN:
            self._scroll_by(1)
            return None
        return NotImplemented

    def _input_prefix(self, line_number: int, wrap_count: int) -> AnyFormattedText:
        """Return the input prompt prefix character.

        Args:
            line_number (int): IN: Line number in the buffer. OUT: Used to show
                the prefix only on the first line.
            wrap_count (int): IN: Wrap count for wrapped lines. OUT: Used to show
                the prefix only on unwrapped first lines.

        Returns:
            AnyFormattedText: OUT: ANSI-formatted prompt prefix, or empty string.
        """
        if line_number == 0 and wrap_count == 0:
            color = "2" if self._running else "1;32"
            return ANSI(f"\x1b[{color}m›\x1b[0m ")  # noqa: RUF001
        return ""

    SELECT_SENTINEL = "\x00__select_active_question__\x00"
    APPROVAL_SENTINEL = "\x00__select_active_approval__\x00"

    def set_active_approval(self, panel: Any) -> None:
        """Display an approval panel in the status area.

        Args:
            panel (Any): IN: Approval panel object with a ``compose`` method. OUT:
                Rendered and set as the active panel.
        """
        self._active_approval = panel
        self._status.set_active_panel(panel.compose() if panel else "")
        self._invalidate()

    def clear_active_approval(self) -> None:
        """Remove the active approval panel from the status area."""
        self._active_approval = None
        self._status.clear_active_panel()
        self._invalidate()

    def refresh_active_approval(self) -> None:
        """Re-render the active approval panel."""
        if self._active_approval is not None:
            self._status.set_active_panel(self._active_approval.compose())
            self._invalidate()

    def set_active_question(self, panel: Any) -> None:
        """Display a question panel in the status area.

        Args:
            panel (Any): IN: Question panel object with a ``compose`` method. OUT:
                Rendered and set as the active panel.
        """
        self._active_question = panel
        self._status.set_active_panel(panel.compose() if panel else "")
        self._invalidate()

    def clear_active_question(self) -> None:
        """Remove the active question panel from the status area."""
        self._active_question = None
        self._status.clear_active_panel()
        self._invalidate()

    def refresh_active_question(self) -> None:
        """Re-render the active question panel."""
        if self._active_question is not None:
            self._status.set_active_panel(self._active_question.compose())
            self._invalidate()

    def _build_key_bindings(self) -> KeyBindings:
        """Build and return the key bindings for the prompt application.

        Returns:
            KeyBindings: OUT: Configured key bindings for scrolling, completion,
                submission, and exit handling.
        """
        kb = KeyBindings()

        def _queue_interrupt() -> None:
            self._input_queue.put_nowait("/interrupt")
            self._exit_armed = False

        @kb.add(Keys.PageUp, eager=True)
        def _pgup(event: KeyPressEvent) -> None:
            """Internal helper to pgup.

            Args:
                event (KeyPressEvent): IN: event. OUT: Consumed during execution."""
            self._scroll_by(-10)

        @kb.add(Keys.PageDown, eager=True)
        def _pgdown(event: KeyPressEvent) -> None:
            """Internal helper to pgdown.

            Args:
                event (KeyPressEvent): IN: event. OUT: Consumed during execution."""
            self._scroll_by(10)

        @kb.add(Keys.ControlEnd, eager=True)
        def _to_bottom(event: KeyPressEvent) -> None:
            """Internal helper to to bottom.

            Args:
                event (KeyPressEvent): IN: event. OUT: Consumed during execution."""
            self._scroll_y = None
            self._invalidate()

        @kb.add(Keys.ControlHome, eager=True)
        def _to_top(event: KeyPressEvent) -> None:
            """Internal helper to to top.

            Args:
                event (KeyPressEvent): IN: event. OUT: Consumed during execution."""
            self._scroll_y = 0
            self._invalidate()

        @kb.add(Keys.Up)
        def _up(event: KeyPressEvent) -> None:
            """Internal helper to up.

            Args:
                event (KeyPressEvent): IN: event. OUT: Consumed during execution."""
            buffer = event.app.current_buffer

            if buffer.complete_state is not None:
                buffer.complete_previous()
                return
            if self._active_question is not None:
                self._active_question.move_up()
                self.refresh_active_question()
                return
            if self._active_approval is not None:
                self._active_approval.move_cursor_up()
                self.refresh_active_approval()
                return

            buffer.history_backward()

        @kb.add(Keys.Down)
        def _down(event: KeyPressEvent) -> None:
            """Internal helper to down.

            Args:
                event (KeyPressEvent): IN: event. OUT: Consumed during execution."""
            buffer = event.app.current_buffer
            if buffer.complete_state is not None:
                buffer.complete_next()
                return
            if self._active_question is not None:
                self._active_question.move_down()
                self.refresh_active_question()
                return
            if self._active_approval is not None:
                self._active_approval.move_cursor_down()
                self.refresh_active_approval()
                return
            buffer.history_forward()

        @kb.add(Keys.Tab)
        def _tab(event: KeyPressEvent) -> None:
            """Internal helper to tab.

            Args:
                event (KeyPressEvent): IN: event. OUT: Consumed during execution."""
            buffer = event.app.current_buffer
            if buffer.complete_state is not None:
                buffer.complete_next()
            else:
                buffer.start_completion(select_first=True)

        @kb.add(Keys.Enter)
        def _enter(event: KeyPressEvent) -> None:
            """Internal helper to enter.

            Args:
                event (KeyPressEvent): IN: event. OUT: Consumed during execution."""
            buffer = event.app.current_buffer

            if buffer.complete_state is not None and buffer.complete_state.current_completion is not None:
                buffer.apply_completion(buffer.complete_state.current_completion)
                buffer.cancel_completion()

            if self._active_question is not None and not buffer.text.strip():
                self._input_queue.put_nowait(self.SELECT_SENTINEL)
                return
            if self._active_approval is not None and not buffer.text.strip():
                self._input_queue.put_nowait(self.APPROVAL_SENTINEL)
                return
            buffer.validate_and_handle()

        @kb.add(Keys.Escape, eager=True)
        def _esc(event: KeyPressEvent) -> None:
            """Internal helper to esc.

            Args:
                event (KeyPressEvent): IN: event. OUT: Consumed during execution."""
            if self._running:
                _queue_interrupt()
                return
            buffer = event.app.current_buffer
            if buffer.complete_state is not None:
                buffer.cancel_completion()

        @kb.add(Keys.ControlC, eager=True)
        def _ctrl_c(event: KeyPressEvent) -> None:
            """Internal helper to ctrl c.

            Args:
                event (KeyPressEvent): IN: event. OUT: Consumed during execution."""
            buffer = event.app.current_buffer

            if self._running:
                _queue_interrupt()
                return

            if buffer.text:
                buffer.text = ""
                self._exit_armed = False
                return

            if not self._exit_armed:
                self._exit_armed = True
                self._status.append_line("\x1b[2m(press Ctrl+C again or Ctrl+D to exit)\x1b[0m")
                self._invalidate()
                return
            event.app.exit()

        @kb.add("c-c", eager=True)
        def _ctrl_c_alias(event: KeyPressEvent) -> None:
            """Handle terminals that report Ctrl+C as c-c."""
            _ctrl_c(event)

        @kb.add(Keys.ControlD)
        def _ctrl_d(event: KeyPressEvent) -> None:
            """Internal helper to ctrl d.

            Args:
                event (KeyPressEvent): IN: event. OUT: Consumed during execution."""
            buffer = event.app.current_buffer
            if buffer.text:
                return
            event.app.exit()

        return kb

    def _accept_handler(self, buffer: Buffer) -> bool:
        """Handle accepted (submitted) input from the buffer.

        Args:
            buffer (Buffer): IN: prompt_toolkit buffer containing the input. OUT:
                Text is read, normalized, and enqueued.

        Returns:
            bool: OUT: Always ``False`` to prevent default buffer behavior.
        """
        text = buffer.text.strip()
        if not text:
            return False

        if text.startswith("/") and " " not in text:
            stem = text[1:].lower()
            matches = [name for name, _ in SLASH_COMMANDS if name[1:].lower().startswith(stem)]
            if len(matches) == 1:
                text = matches[0]
        buffer.text = ""
        self._exit_armed = False
        self._scroll_y = None
        self._input_queue.put_nowait(text)
        return False

    @property
    def input_queue(self) -> asyncio.Queue[str]:
        """Return the input queue for consuming submitted text.

        Returns:
            asyncio.Queue[str]: OUT: Queue of submitted strings.
        """
        return self._input_queue

    @property
    def is_running(self) -> bool:
        """Return whether a turn is currently running.

        Returns:
            bool: OUT: Current running state.
        """
        return self._running

    def set_running(self, running: bool) -> None:
        """Update the running state and propagate to renderers.

        Args:
            running (bool): IN: New running state. OUT: Propagated to status and
                footer renderers.
        """
        self._running = running
        self._status.set_running(running)
        self._footer.set_running(running)
        if not running:
            self._status.set_spinner_label("Working")
        self._invalidate()

    def set_spinner_label(self, label: str) -> None:
        """Update the spinner label in the status renderer.

        Args:
            label (str): IN: New spinner label. OUT: Passed to status renderer.
        """
        self._status.set_spinner_label(label)
        self._invalidate()

    def reset_spinner_timer(self) -> None:
        """Reset the spinner elapsed counter to zero."""
        self._status.reset_spinner_timer()
        self._invalidate()

    def set_stats(self, tokens: str = "", cost: str = "") -> None:
        """Update token and cost statistics.

        Args:
            tokens (str): IN: Token info string. OUT: Passed to status renderer.
            cost (str): IN: Cost info string. OUT: Passed to status renderer.
        """
        self._status.set_stats(tokens, cost)
        self._invalidate()

    def set_session(self, *, agent_name: str = "", model: str = "", cwd: str = "", branch: str = "") -> None:
        """Update session metadata in the footer.

        Args:
            agent_name (str): IN: Agent name. OUT: Passed to footer renderer.
            model (str): IN: Model identifier. OUT: Passed to footer renderer.
            cwd (str): IN: Working directory. OUT: Passed to footer renderer.
            branch (str): IN: Git branch. OUT: Passed to footer renderer.
        """
        self._footer.set_session(agent_name, model, cwd, branch)
        self._invalidate()

    def set_skills(self, skills: list[str]) -> None:
        """Update the skill list used for slash-command completion.

        Args:
            skills (list[str]): IN: Skill names. OUT: Passed to the completer.
        """
        self._completer.set_skills(skills)

    def set_context(self, used: int, max_: int) -> None:
        """Update context usage in the footer.

        Args:
            used (int): IN: Used tokens. OUT: Passed to footer renderer.
            max_ (int): IN: Max tokens. OUT: Passed to footer renderer.
        """
        self._footer.set_context(used, max_)
        self._invalidate()

    def set_queue_count(self, count: int) -> None:
        """Update the queued input count in the status renderer.

        Args:
            count (int): IN: Queue count. OUT: Passed to status renderer.
        """
        self._status.set_queue_count(count)
        self._invalidate()

    def set_plan_mode(self, plan_mode: bool) -> None:
        """Update plan mode display state.

        Args:
            plan_mode (bool): IN: Whether plan mode is active. OUT: Passed to
                status renderer.
        """
        self._status.set_plan_mode(plan_mode)
        self._invalidate()

    def append_line(self, line: str) -> None:
        """Append a line to the status content.

        Args:
            line (str): IN: Line to append. OUT: Passed to status renderer.
        """
        self._status.append_line(line)
        self._invalidate()

    def clear_content(self) -> None:
        """Clear all status content lines."""
        self._status.clear_content()
        self._invalidate()

    def append_streaming(self, text: str) -> None:
        """Append text to the streaming buffer.

        Args:
            text (str): IN: Streaming chunk. OUT: Passed to status renderer.
        """
        self._status.append_streaming(text)
        self._invalidate()

    def commit_streaming(self) -> None:
        """Commit the streaming buffer to content lines, rendering Markdown."""
        text = self._status._streaming_text
        self._status.clear_streaming()
        if text:
            from .console import markdown_to_ansi

            try:
                rendered = markdown_to_ansi(text)
            except Exception:
                rendered = text
            self._status.append_line(rendered or text)
        self._invalidate()

    def clear_streaming(self) -> None:
        """Discard the streaming buffer without committing."""
        self._status.clear_streaming()
        self._invalidate()

    def append_thinking(self, text: str) -> None:
        """Append text to the thinking buffer.

        Args:
            text (str): IN: Thinking chunk. OUT: Passed to status renderer.
        """
        self._status.append_thinking(text)
        self._invalidate()

    def clear_thinking(self) -> None:
        """Clear the thinking buffer."""
        self._status.clear_thinking()
        self._invalidate()

    def set_active_tool(self, tool_call_id: str, render_fn: Callable[[], str]) -> None:
        """Register an active tool render function.

        Args:
            tool_call_id (str): IN: Tool call identifier. OUT: Passed to status renderer.
            render_fn (Callable[[], str]): IN: Render callable. OUT: Passed to status renderer.
        """
        self._status.set_active_tool(tool_call_id, render_fn)
        self._invalidate()

    def commit_active_tool(self, tool_call_id: str, final_text: str) -> None:
        """Finalize an active tool and append its rendered text.

        Args:
            tool_call_id (str): IN: Tool call identifier. OUT: Passed to status renderer.
            final_text (str): IN: Final rendered text. OUT: Appended to content lines.
        """
        self._status.pop_active_tool(tool_call_id)
        if final_text:
            self._status.append_line(final_text)
        self._invalidate()

    def clear_active_tools(self) -> None:
        """Remove all active tool renderers from the live status area."""
        self._status.clear_active_tools()
        self._invalidate()

    def set_subagent_preview(self, task_id: str, label: str, text: str) -> None:
        """Set or update a transient sub-agent preview line.

        Args:
            task_id (str): IN: Sub-agent task identifier.
            label (str): IN: Display label.
            text (str): IN: Latest preview text (replaces, doesn't append).
        """
        self._status.set_subagent_preview(task_id, label, text)
        self._invalidate()

    def clear_subagent_preview(self, task_id: str) -> None:
        """Remove a sub-agent preview line once the task ends.

        Args:
            task_id (str): IN: Sub-agent task identifier.
        """
        self._status.clear_subagent_preview(task_id)
        self._invalidate()

    def clear_subagent_previews(self) -> None:
        """Remove all transient sub-agent preview lines."""
        self._status.clear_subagent_previews()
        self._invalidate()

    def _invalidate(self) -> None:
        """Trigger a redraw of the prompt application if running."""
        if self._app:
            self._app.invalidate()

    async def run(self) -> Application[None]:
        """Run the prompt application asynchronously.

        Returns:
            Application[None]: OUT: The running prompt_toolkit application instance.
        """
        self._app = Application(
            self._layout,
            key_bindings=self._kb,
            erase_when_done=True,
            mouse_support=os.environ.get("XERXES_MOUSE", "0") == "1",
            full_screen=True,
            refresh_interval=0.1,
        )
        await self._app.run_async(handle_sigint=False)
        return self._app

    def stop(self) -> None:
        """Stop the running prompt application."""
        if self._app and self._app.is_running:
            self._app.exit()

    async def __aenter__(self) -> PersistentPrompt:
        """Enter the async runtime context.

        Returns:
            PersistentPrompt: OUT: Self.
        """
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Exit the async runtime context, stopping the application."""
        self.stop()

    def push_modal(
        self,
        panel: Any,
    ) -> None:
        """Placeholder for modal panel support.

        Args:
            panel (Any): IN: Panel object. OUT: Currently unused.
        """
        pass
