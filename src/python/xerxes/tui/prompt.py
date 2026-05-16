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
"""prompt_toolkit-based prompt application for the Xerxes TUI.

Holds the :class:`PersistentPrompt` orchestrator (input buffer +
multi-region full-screen layout), the :class:`StatusRenderer` /
:class:`FooterRenderer` markup generators, and :class:`SlashCompleter`
which surfaces both built-in slash commands and dynamically-registered
skills. All key bindings live here as well."""

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


def _mode_style(plan_mode: bool, activity_mode: str) -> str:
    """Return the ANSI escape that tints prompt chrome for the current mode.

    Plan mode renders magenta; researcher mode renders cyan; everything
    else falls back to the dim default."""
    if plan_mode:
        return "\x1b[35m"
    if (activity_mode or "").lower() == "researcher":
        return "\x1b[36m"
    return "\x1b[2m"


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
    """Completer for slash commands and registered skills."""

    def __init__(self, commands: list[tuple[str, str]] = SLASH_COMMANDS) -> None:
        """Seed the completer with built-in ``(name, description)`` pairs."""
        self._commands = commands
        self._skills: list[str] = []

    def set_skills(self, skills: list[str]) -> None:
        """Replace the dynamic skill list (deduplicated, sorted)."""
        self._skills = sorted(set(skills))

    def get_completions(self, document: Document, complete_event: CompleteEvent):
        """Yield matching slash commands and skills for the input under the cursor.

        Only fires when the text starts with ``/`` and contains no space —
        post-space we let the underlying command parser handle args."""
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
    """Builds the scrolling status markup shown above the input line.

    Owns the committed content history, in-flight streaming/thinking
    buffers, active tool render callbacks, subagent previews, spinner
    state, and the bottom separator that doubles as a mode indicator.
    Calling the instance returns ``ANSI(markup)`` for prompt_toolkit.
    """

    SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
    THINKING_PREVIEW_LINES = 4
    SUBAGENT_PREVIEW_LINES = 5

    def __init__(self) -> None:
        """Build an empty renderer with the spinner parked and no panels active."""
        self._content_lines: list[str] = []
        self._running = False
        self._queue_count = 0
        self._plan_mode = False
        self._activity_mode = "code"
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
        self._last_render_line_count: int = 1
        self._last_render_last_line_width: int = 0

    def set_running(self, running: bool) -> None:
        """Flip the running indicator; rearms the spinner clock on each off→on edge."""
        import time

        was_running = self._running
        self._running = running
        if running and not was_running:
            self._spinner_started_at = time.monotonic()
            self._spinner_frame = 0

    def reset_spinner_timer(self) -> None:
        """Restart the spinner elapsed clock at 0.

        Call when a new tool starts or the spinner label flips so the
        timer reflects the current step, not the turn."""
        import time

        self._spinner_started_at = time.monotonic()

    def set_spinner_label(self, label: str) -> None:
        """Update the spinner caption; restarts the elapsed clock if it changed."""
        new_label = label or "Working"
        if new_label != self._spinner_label:
            self.reset_spinner_timer()
        self._spinner_label = new_label

    def set_queue_count(self, count: int) -> None:
        """Update the queued-inputs badge (clamped at 0)."""
        self._queue_count = max(0, count)

    def set_plan_mode(self, plan_mode: bool) -> None:
        """Record whether plan mode is currently active."""
        self._plan_mode = plan_mode

    def set_activity_mode(self, mode: str) -> None:
        """Record the activity label shown in the bottom rule when plan mode is off."""
        self._activity_mode = mode or "code"

    def set_stats(self, tokens: str = "", cost: str = "") -> None:
        """Store optional token / cost strings (currently informational only)."""
        self._token_info = tokens
        self._cost_info = cost

    def append_line(self, line: str) -> None:
        """Commit ``line`` to history after stripping leading/trailing blank edges."""
        line = self._strip_blank_edges(line)
        if not line:
            return
        self._content_lines.append(line)

    def clear_content(self) -> None:
        """Drop every committed history line."""
        self._content_lines.clear()

    def set_active_panel(self, text: str) -> None:
        """Pin ``text`` as the active modal panel (approval / question)."""
        self._active_panel = text or ""

    def clear_active_panel(self) -> None:
        """Remove the pinned modal panel."""
        self._active_panel = ""

    def append_streaming(self, text: str) -> None:
        """Append ``text`` to the in-flight assistant response buffer."""
        self._streaming_text += text

    def commit_streaming(self) -> None:
        """Move the streaming buffer onto the committed history and clear it."""
        text = self._strip_blank_edges(self._streaming_text)
        self._streaming_text = ""
        if text:
            self._content_lines.append(text)

    def clear_streaming(self) -> None:
        """Discard the streaming buffer without committing it."""
        self._streaming_text = ""

    def append_thinking(self, text: str) -> None:
        """Append ``text`` to the rolling thinking preview (last ``N`` lines only)."""
        self._thinking_text += text
        self._thinking_text = self._tail_lines(self._thinking_text, self.THINKING_PREVIEW_LINES)

    def clear_thinking(self) -> None:
        """Discard the thinking preview."""
        self._thinking_text = ""

    @staticmethod
    def _tail_lines(text: str, limit: int) -> str:
        """Return only the last ``limit`` non-empty logical lines of ``text``."""
        if limit <= 0 or not text:
            return ""
        text = StatusRenderer._strip_blank_edges(text)
        lines = [line.rstrip() for line in text.splitlines() if line.strip()]
        if len(lines) <= limit:
            return "\n".join(lines)
        return "\n".join(lines[-limit:])

    @staticmethod
    def _strip_blank_edges(text: str) -> str:
        """Trim empty lines around a rendered block without touching its body."""
        if not text:
            return ""
        return re.sub(r"(?:\n[ \t]*)+\Z", "", re.sub(r"\A(?:[ \t]*\n)+", "", text))

    def set_active_tool(self, tool_call_id: str, render_fn: Callable[[], str]) -> None:
        """Register an ongoing tool call; ``render_fn`` is called every redraw."""
        self._active_tools[tool_call_id] = render_fn

    def pop_active_tool(self, tool_call_id: str) -> str:
        """Remove a tool render and return its final markup (``""`` if unknown)."""
        render = self._active_tools.pop(tool_call_id, None)
        return render() if render is not None else ""

    def clear_active_tools(self) -> None:
        """Drop every active tool renderer."""
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
        self._subagent_previews.pop(task_id, None)
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
        """Return the active prompt_toolkit output columns (default 80)."""
        columns = 80
        try:
            from prompt_toolkit.application.current import get_app

            columns = get_app().output.get_size().columns
        except Exception:
            pass
        return columns

    def _terminal_rows(self) -> int:
        """Return the active prompt_toolkit output rows (default 24)."""
        try:
            from prompt_toolkit.application.current import get_app

            return get_app().output.get_size().rows
        except Exception:
            return 24

    def _markup(self) -> str:
        """Compose the full ANSI markup string consumed by ``ANSI(...)``.

        Order: committed history → thinking preview → active tool blocks
        → subagent previews → streaming text → modal panel → spinner row
        → mode-aware separator. Side effect: caches the line count and
        last-line width so :meth:`_status_cursor_position` can position
        the cursor accurately."""
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
            for preview in list(self._subagent_previews.values())[-self.SUBAGENT_PREVIEW_LINES :]:
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
        elif self._activity_mode == "researcher":
            title_parts.append("research")
        if self._queue_count:
            title_parts.append(f"{self._queue_count} queued")
        title = f" {' · '.join(title_parts)} "
        dash = "╌" if self._plan_mode else "─"
        border = f"{dash}{dash}{title}{dash * max(0, columns - len(title) - 2)}"
        border_style = _mode_style(self._plan_mode, self._activity_mode)
        parts.append(f"{border_style}{border}\x1b[0m\n")

        markup = "".join(parts)
        plain = _ANSI_RE.sub("", markup)
        lines = plain.split("\n") if plain else [""]
        self._last_render_line_count = max(1, len(lines))
        self._last_render_last_line_width = len(lines[-1]) if lines else 0
        return markup

    def line_count(self) -> int:
        """Count the printable lines in the current markup (ANSI stripped)."""
        markup = self._markup()
        if not markup:
            return 0
        plain = _ANSI_RE.sub("", markup)
        return plain.count("\n") + (0 if plain.endswith("\n") else 1)

    def __call__(self) -> AnyFormattedText:
        """Return ``ANSI(self._markup())`` so this object plugs into prompt_toolkit."""
        return ANSI(self._markup())


class FooterRenderer:
    """Builds the two-line footer (rule + status row) below the input.

    Tracks session metadata (agent, model, cwd, branch), context
    utilization, and the currently active mode. Calling the instance
    returns ANSI markup for prompt_toolkit."""

    def __init__(self) -> None:
        """Set neutral defaults; no terminal access happens here."""
        self._agent_name = "agent"
        self._model = ""
        self._cwd = ""
        self._branch = ""
        self._running = False
        self._plan_mode = False
        self._activity_mode = "code"
        self._context_used = 0
        self._context_max = 0
        self._tip = "shift-tab: mode  ctrl-j: newline"

    def line_count(self) -> int:
        """Return the visible footer line count (ANSI stripped, min 1)."""
        markup = self._markup()
        plain = _ANSI_RE.sub("", markup)
        return max(1, plain.count("\n"))

    def set_session(self, agent_name: str, model: str, cwd: str, branch: str) -> None:
        """Update session metadata; empty strings are ignored so callers can do partial updates."""
        if agent_name:
            self._agent_name = agent_name
        if model:
            self._model = model
        if cwd:
            self._cwd = _shorten_home(cwd)
        if branch:
            self._branch = branch

    def set_running(self, running: bool) -> None:
        """Flip the activity dot (●/○) shown beside the model name."""
        self._running = running

    def set_plan_mode(self, plan_mode: bool) -> None:
        """Record plan mode for use by :meth:`_markup`."""
        self._plan_mode = plan_mode

    def set_activity_mode(self, mode: str) -> None:
        """Record the non-plan activity label (``"code"`` / ``"researcher"`` / ...)."""
        self._activity_mode = mode or "code"

    def set_context(self, used: int, max_: int) -> None:
        """Record used / max context tokens (clamped at 0)."""
        self._context_used = max(0, used)
        self._context_max = max(0, max_)

    def _terminal_columns(self) -> int:
        """Return the live prompt_toolkit columns (default 80)."""
        try:
            from prompt_toolkit.application.current import get_app

            return get_app().output.get_size().columns
        except Exception:
            return 80

    @staticmethod
    def _format_tokens(n: int) -> str:
        """Compact-format ``n`` with ``k`` / ``M`` suffixes for footer display."""
        if n >= 1_000_000:
            return f"{n / 1_000_000:.1f}M"
        if n >= 1_000:
            return f"{n / 1_000:.1f}k"
        return str(n)

    @staticmethod
    def _truncate_path(path: str, budget: int) -> str:
        """Trim ``path`` to ``budget`` chars from the right, prefixed with ``…``."""
        if len(path) <= budget:
            return path
        return "…" + path[-(budget - 1) :]

    def _markup(self) -> str:
        """Build the two-line footer markup (separator rule + status row).

        Lays out left segments (agent/cwd/branch/mode/tip) against a
        right-aligned context badge, wrapping to a second line only
        when both halves don't fit on one row."""
        spinner = "●" if self._running else "○"
        model = self._model or "—"
        agent = f"agent ({model} {spinner})"
        mode = f"mode: {'plan' if self._plan_mode else self._activity_mode}"
        columns = self._terminal_columns()

        cwd_max = max(20, columns // 3)
        cwd_short = self._truncate_path(self._cwd, cwd_max) if self._cwd else ""
        left_segments = [agent]
        if cwd_short:
            left_segments.append(cwd_short)
        if self._branch:
            left_segments.append(self._branch)
        left_segments.append(mode)
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

        style = _mode_style(self._plan_mode, self._activity_mode)
        rule = style + ("─" * columns) + "\x1b[0m"

        if len(plain_left) + len(plain_right) + 2 <= columns:
            gap = max(1, columns - len(plain_left) - len(plain_right))
            row = f"{style}{left}{' ' * gap}{ctx_text}\x1b[0m"
            return rule + "\n" + row + "\n"

        right_pad = max(0, columns - len(plain_right))
        line1 = f"{style}{left}\x1b[0m"
        line2 = f"{' ' * right_pad}{style}{ctx_text}\x1b[0m"
        return rule + "\n" + line1 + "\n" + line2 + "\n"

    def __call__(self) -> AnyFormattedText:
        """Return ``ANSI(self._markup())`` for prompt_toolkit consumption."""
        return ANSI(self._markup())


def _shorten_home(path: str) -> str:
    """Collapse ``$HOME`` prefix in ``path`` to ``~``."""
    import os

    home = os.path.expanduser("~")
    if path == home:
        return "~"
    if path.startswith(home + os.sep):
        return "~" + path[len(home) :]
    return path


class PersistentPrompt:
    """Full-screen prompt_toolkit application that drives the Xerxes input loop.

    The layout has three regions: a scrolling :class:`StatusRenderer`
    area on top, a one-line input buffer with slash + skill completion
    in the middle, and a :class:`FooterRenderer` at the bottom.

    Submitted text is enqueued on :attr:`input_queue`; the consuming
    coroutine (typically :meth:`XerxesTUI._process_prompt_input`) pulls
    it back out. ``on_submit`` / ``on_slash`` are reserved for future
    callback-driven embedders and are not invoked by the default path."""

    def __init__(
        self,
        on_slash: Callable[[str], Coroutine[Any, Any, None]] | None = None,
        on_submit: Callable[[str], Coroutine[Any, Any, None]] | None = None,
    ) -> None:
        """Construct renderers, the input buffer, the layout, and key bindings."""
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
        """Position the (hidden) status cursor for prompt_toolkit scroll math."""
        last = max(0, self._status._last_render_line_count - 1)
        if self._scroll_y is None or self._scroll_y >= last:
            return Point(x=self._status._last_render_last_line_width, y=last)
        y = max(0, self._scroll_y)
        return Point(x=0, y=y)

    def _status_total_lines(self) -> int:
        """Total printable lines currently rendered in the status area."""
        return max(0, len(_ANSI_RE.sub("", self._status._markup()).split("\n")) - 1)

    def _scroll_by(self, delta: int) -> None:
        """Adjust the scroll offset by ``delta`` lines; ``None`` means tracking bottom."""
        total = self._status_total_lines()
        current = self._scroll_y if self._scroll_y is not None else total
        new = current + delta
        if new >= total:
            self._scroll_y = None
        else:
            self._scroll_y = max(0, new)
        self._invalidate()

    def _on_status_mouse(self, mouse_event: MouseEvent) -> Any:
        """Translate wheel-up / wheel-down events into status-area scrolling."""
        if mouse_event.event_type == MouseEventType.SCROLL_UP:
            self._scroll_by(-1)
            return None
        if mouse_event.event_type == MouseEventType.SCROLL_DOWN:
            self._scroll_by(1)
            return None
        return NotImplemented

    def _input_prefix(self, line_number: int, wrap_count: int) -> AnyFormattedText:
        """Render the ``›`` input glyph; colored green when idle, dim while running."""  # noqa: RUF002
        if line_number == 0 and wrap_count == 0:
            color = "2" if self._running else "1;32"
            return ANSI(f"\x1b[{color}m›\x1b[0m ")  # noqa: RUF001
        return ""

    SELECT_SENTINEL = "\x00__select_active_question__\x00"
    APPROVAL_SENTINEL = "\x00__select_active_approval__\x00"
    PLAN_TOGGLE_SENTINEL = "\x00__toggle_plan_mode__\x00"

    def set_active_approval(self, panel: Any) -> None:
        """Pin ``panel`` as the active approval modal and invalidate the screen."""
        self._active_approval = panel
        self._status.set_active_panel(panel.compose() if panel else "")
        self._invalidate()

    def clear_active_approval(self) -> None:
        """Dismiss the active approval panel."""
        self._active_approval = None
        self._status.clear_active_panel()
        self._invalidate()

    def refresh_active_approval(self) -> None:
        """Re-render the active approval panel after a cursor move."""
        if self._active_approval is not None:
            self._status.set_active_panel(self._active_approval.compose())
            self._invalidate()

    def set_active_question(self, panel: Any) -> None:
        """Pin ``panel`` as the active question modal and invalidate the screen."""
        self._active_question = panel
        self._status.set_active_panel(panel.compose() if panel else "")
        self._invalidate()

    def clear_active_question(self) -> None:
        """Dismiss the active question panel."""
        self._active_question = None
        self._status.clear_active_panel()
        self._invalidate()

    def refresh_active_question(self) -> None:
        """Re-render the active question panel after a cursor move."""
        if self._active_question is not None:
            self._status.set_active_panel(self._active_question.compose())
            self._invalidate()

    def _build_key_bindings(self) -> KeyBindings:
        """Wire every Xerxes-specific keybinding into one :class:`KeyBindings` registry.

        Covers scroll (PageUp/PageDown/Ctrl+Home/Ctrl+End), history vs
        completion vs panel navigation overloads for arrows, completion
        cycling on Tab, Shift+Tab as the plan-mode toggle sentinel,
        Enter as submit-with-panel-fallback, and Ctrl+C / Ctrl+D as a
        two-step exit with mid-turn interrupt support."""
        kb = KeyBindings()

        def _queue_interrupt() -> None:
            """Enqueue the ``/interrupt`` sentinel so the input pump cancels the turn."""
            self._input_queue.put_nowait("/interrupt")
            self._exit_armed = False

        @kb.add(Keys.PageUp, eager=True)
        def _pgup(event: KeyPressEvent) -> None:
            """PageUp: scroll the status area up by ten lines."""
            self._scroll_by(-10)

        @kb.add(Keys.PageDown, eager=True)
        def _pgdown(event: KeyPressEvent) -> None:
            """PageDown: scroll the status area down by ten lines."""
            self._scroll_by(10)

        @kb.add(Keys.ControlEnd, eager=True)
        def _to_bottom(event: KeyPressEvent) -> None:
            """Ctrl+End: jump back to follow-the-tail mode."""
            self._scroll_y = None
            self._invalidate()

        @kb.add(Keys.ControlHome, eager=True)
        def _to_top(event: KeyPressEvent) -> None:
            """Ctrl+Home: jump to the top of the rendered history."""
            self._scroll_y = 0
            self._invalidate()

        @kb.add(Keys.Up)
        def _up(event: KeyPressEvent) -> None:
            """Up arrow: route through completion menu, modal panel, then history."""
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
            """Down arrow: route through completion menu, modal panel, then history."""
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
            """Tab: start completion or cycle to the next completion."""
            buffer = event.app.current_buffer
            if buffer.complete_state is not None:
                buffer.complete_next()
            else:
                buffer.start_completion(select_first=True)

        @kb.add(Keys.BackTab, eager=True)
        def _shift_tab(event: KeyPressEvent) -> None:
            """Shift+Tab: enqueue the plan-mode toggle sentinel without submitting input."""
            buffer = event.app.current_buffer
            if buffer.complete_state is not None:
                buffer.cancel_completion()
            self._input_queue.put_nowait(self.PLAN_TOGGLE_SENTINEL)

        @kb.add(Keys.Enter)
        def _enter(event: KeyPressEvent) -> None:
            """Enter: apply pending completion, otherwise submit or select panel option."""
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
            """Esc: cancel an in-flight turn, otherwise close any open completion menu."""
            if self._running:
                _queue_interrupt()
                return
            buffer = event.app.current_buffer
            if buffer.complete_state is not None:
                buffer.cancel_completion()

        @kb.add(Keys.ControlC, eager=True)
        def _ctrl_c(event: KeyPressEvent) -> None:
            """Ctrl+C: interrupt the running turn, clear input, then arm a confirmation exit."""
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
            """Mirror of the Ctrl+C handler for terminals that report it as ``c-c``."""
            _ctrl_c(event)

        @kb.add(Keys.ControlD)
        def _ctrl_d(event: KeyPressEvent) -> None:
            """Ctrl+D on an empty buffer exits; otherwise it's swallowed."""
            buffer = event.app.current_buffer
            if buffer.text:
                return
            event.app.exit()

        return kb

    def _accept_handler(self, buffer: Buffer) -> bool:
        """Submit handler: normalize text, push onto :attr:`input_queue`, clear buffer.

        Returns ``False`` so prompt_toolkit doesn't append the entry to
        the buffer's own history (we manage that explicitly)."""
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
        """Queue from which the consuming task pulls submitted input strings."""
        return self._input_queue

    @property
    def is_running(self) -> bool:
        """``True`` while a turn is actively streaming."""
        return self._running

    def set_running(self, running: bool) -> None:
        """Update the running flag and cascade it to the status / footer renderers."""
        self._running = running
        self._status.set_running(running)
        self._footer.set_running(running)
        if not running:
            self._status.set_spinner_label("Working")
        self._invalidate()

    def set_spinner_label(self, label: str) -> None:
        """Forward ``label`` to the status renderer (caption beside the spinner)."""
        self._status.set_spinner_label(label)
        self._invalidate()

    def reset_spinner_timer(self) -> None:
        """Restart the spinner elapsed-time counter at zero."""
        self._status.reset_spinner_timer()
        self._invalidate()

    def set_stats(self, tokens: str = "", cost: str = "") -> None:
        """Forward token / cost strings to the status renderer."""
        self._status.set_stats(tokens, cost)
        self._invalidate()

    def set_session(self, *, agent_name: str = "", model: str = "", cwd: str = "", branch: str = "") -> None:
        """Forward partial session metadata updates to the footer."""
        self._footer.set_session(agent_name, model, cwd, branch)
        self._invalidate()

    def set_skills(self, skills: list[str]) -> None:
        """Replace the slash-command skill list so completions stay in sync."""
        self._completer.set_skills(skills)

    def set_context(self, used: int, max_: int) -> None:
        """Update the footer's context-usage gauge."""
        self._footer.set_context(used, max_)
        self._invalidate()

    def set_queue_count(self, count: int) -> None:
        """Update the queued-input badge in the status area."""
        self._status.set_queue_count(count)
        self._invalidate()

    def set_plan_mode(self, plan_mode: bool) -> None:
        """Apply plan-mode tinting to both the status and footer."""
        self._status.set_plan_mode(plan_mode)
        self._footer.set_plan_mode(plan_mode)
        self._invalidate()

    def set_activity_mode(self, mode: str) -> None:
        """Apply the non-plan activity label to both status and footer."""
        self._status.set_activity_mode(mode)
        self._footer.set_activity_mode(mode)
        self._invalidate()

    def append_line(self, line: str) -> None:
        """Append ``line`` to the committed status history."""
        self._status.append_line(line)
        self._invalidate()

    def clear_content(self) -> None:
        """Wipe the committed status history."""
        self._status.clear_content()
        self._invalidate()

    def append_streaming(self, text: str) -> None:
        """Append ``text`` to the live streaming buffer."""
        self._status.append_streaming(text)
        self._invalidate()

    def commit_streaming(self) -> None:
        """Render the streaming buffer through Markdown and move it onto history."""
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
        """Discard the streaming buffer without committing it to history."""
        self._status.clear_streaming()
        self._invalidate()

    def append_thinking(self, text: str) -> None:
        """Append ``text`` to the rolling thinking preview."""
        self._status.append_thinking(text)
        self._invalidate()

    def clear_thinking(self) -> None:
        """Discard the thinking preview."""
        self._status.clear_thinking()
        self._invalidate()

    def set_active_tool(self, tool_call_id: str, render_fn: Callable[[], str]) -> None:
        """Register ``render_fn`` so the status area can re-paint the tool live."""
        self._status.set_active_tool(tool_call_id, render_fn)
        self._invalidate()

    def commit_active_tool(self, tool_call_id: str, final_text: str) -> None:
        """Remove the live tool render and append its final ``final_text`` to history."""
        self._status.pop_active_tool(tool_call_id)
        if final_text:
            self._status.append_line(final_text)
        self._invalidate()

    def clear_active_tools(self) -> None:
        """Drop every live-tool renderer."""
        self._status.clear_active_tools()
        self._invalidate()

    def set_subagent_preview(self, task_id: str, label: str, text: str) -> None:
        """Replace the live preview line for ``task_id`` (never appended)."""
        self._status.set_subagent_preview(task_id, label, text)
        self._invalidate()

    def clear_subagent_preview(self, task_id: str) -> None:
        """Drop the preview line for ``task_id``."""
        self._status.clear_subagent_preview(task_id)
        self._invalidate()

    def clear_subagent_previews(self) -> None:
        """Drop every subagent preview line."""
        self._status.clear_subagent_previews()
        self._invalidate()

    def _invalidate(self) -> None:
        """Request a redraw of the prompt application (no-op if not running)."""
        if self._app:
            self._app.invalidate()

    async def run(self) -> Application[None]:
        """Run the prompt_toolkit application until it exits; returns it for inspection.

        Set ``XERXES_MOUSE=1`` to enable mouse support (off by default
        so it doesn't fight the host terminal's text-selection)."""
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
        """Ask the running application to exit; no-op when not running."""
        if self._app and self._app.is_running:
            self._app.exit()

    async def __aenter__(self) -> PersistentPrompt:
        """Async context entry — returns ``self``; does not start the application."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context exit — calls :meth:`stop`."""
        self.stop()

    def push_modal(
        self,
        panel: Any,
    ) -> None:
        """Placeholder kept for forward-compat with a generic modal stack API."""
        pass
