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
import collections
import os
import re
import time
from collections.abc import Callable, Coroutine
from typing import Any

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.data_structures import Point
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import ANSI, AnyFormattedText, to_formatted_text
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout.containers import Float, FloatContainer, HSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl, UIContent
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType
from prompt_toolkit.styles import Style

from .skin_engine import active_fg, get_active_skin

_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

# Reset only the foreground color (leaves bold/dim/bg intact), per the skin
# contract. Use this to close an ``active_fg(role)`` run that carried no other
# SGR attributes; use ``\x1b[0m`` to fully reset a run that combined bold/dim.
_FG_RESET = "\x1b[39m"


class _ScrollableFormattedTextControl(FormattedTextControl):
    """A FormattedTextControl that only renders a visible slice of content.

    This works around prompt_toolkit's cursor-driven scrolling by:
    1. Rendering only the visible lines (not all content)
    2. Reporting line_count as the visible slice size
    3. Placing cursor at the end of the visible slice

    The Window will naturally show exactly what we render because
    there's nothing extra to scroll to.
    """

    def __init__(
        self,
        renderer: StatusRenderer,
        scroll_y_accessor: Callable[[], int | None],
        visible_rows_setter: Callable[[int], None],
    ):
        super().__init__(
            text="",
            focusable=False,
            show_cursor=False,
        )
        self._renderer = renderer
        self._scroll_y_accessor = scroll_y_accessor
        self._visible_rows_setter = visible_rows_setter

    def create_content(self, width: int, height: int | None) -> UIContent:
        """Return only the visible slice of content."""
        from prompt_toolkit.formatted_text.utils import fragment_list_width, split_lines

        scroll_y = self._scroll_y_accessor()
        visible_rows = height if height is not None and height > 0 else None
        if visible_rows is not None:
            self._visible_rows_setter(visible_rows)
        markup = self._renderer._markup(scroll_y, visible_rows=visible_rows)

        # Parse the markup into fragments
        fragments = to_formatted_text(ANSI(markup))
        fragment_lines = list(split_lines(fragments))

        # The line count is exactly what we rendered
        line_count = len(fragment_lines)

        # Place cursor at the end of the visible content
        if line_count > 0:
            last_line = fragment_lines[-1]
            cursor_x = fragment_list_width(last_line)
            cursor_pos = Point(x=cursor_x, y=line_count - 1)
        else:
            cursor_pos = Point(x=0, y=0)

        def get_line(i: int) -> Any:
            if 0 <= i < len(fragment_lines):
                return [(item[0], item[1]) for item in fragment_lines[i]]
            return []

        return UIContent(
            get_line=get_line,
            line_count=line_count,
            show_cursor=False,
            cursor_position=cursor_pos,
        )


def _mode_style(plan_mode: bool, activity_mode: str) -> str:
    """Return the ANSI escape that tints prompt chrome for the current mode.

    Routed through the active skin: plan mode -> ``system`` role, researcher
    mode -> ``accent`` role, objective mode -> ``warn`` role, everything else
    falls back to the dim ``muted`` role. Shared by both the input rule and
    the footer so the tint stays consistent across prompt chrome."""
    if plan_mode:
        return active_fg("system")
    normalized = (activity_mode or "").lower()
    if normalized == "researcher":
        return active_fg("accent")
    if normalized == "objective":
        return active_fg("warn")
    return "\x1b[2m" + active_fg("muted")


SLASH_COMMANDS: list[tuple[str, str]] = [
    ("/help", "Show available commands"),
    ("/provider", "Setup or switch provider profile"),
    ("/model", "Switch model"),
    ("/sampling", "View or set sampling parameters"),
    ("/compact", "Summarize conversation to free context"),
    ("/plan", "Toggle plan mode (or /plan OBJECTIVE)"),
    ("/objective", "Switch to objective mode (or /objective GOAL)"),
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
    ("/resume", "Resume a saved session"),
    ("/btw", "Send a side question while running"),
    ("/steer", "Inject mid-turn guidance"),
    ("/cancel", "Cancel the current turn"),
    ("/cancel-all", "Cancel all running turns"),
    ("/todo", "Show or edit the TODO list"),
    ("/skin", "List or switch the TUI skin"),
    ("/exit", "Exit Xerxes"),
]

# Max rows the input box grows to before it starts scrolling internally.
_INPUT_MAX_ROWS = 10


class SlashCompleter(Completer):
    """Completer for slash commands and registered skills."""

    _EFFORT_LEVELS = ("off", "low", "medium", "high")

    def __init__(self, commands: list[tuple[str, str]] = SLASH_COMMANDS) -> None:
        """Seed the completer with built-in ``(name, description)`` pairs."""
        self._commands = commands
        self._skills: list[str] = []
        self._models: list[str] = []
        self._active_model: str = ""

    def set_skills(self, skills: list[str]) -> None:
        """Replace the dynamic skill list (deduplicated, sorted)."""
        self._skills = sorted(set(skills))

    def set_models(self, models: list[str], active: str = "") -> None:
        """Replace the model list used for ``/model`` argument completion."""
        self._models = list(dict.fromkeys(m for m in models if m))
        self._active_model = active

    def get_completions(self, document: Document, complete_event: CompleteEvent):
        """Yield slash commands/skills, or argument values for ``/model`` & ``/thinking``.

        For a bare ``/cmd`` (no space) we complete command + skill names. After
        a space we complete arguments for commands with a known value set:
        ``/model`` → provider model ids, ``/thinking``/``/reasoning`` → effort
        levels. This is why the menu now appears as you type ``/model ``."""
        text = document.text_before_cursor

        if not text.startswith("/"):
            return

        # Argument completion (post-space) for the commands that have choices.
        arg_match = re.match(r"^/(model|thinking|reasoning)\s+(.*)$", text, re.IGNORECASE)
        if arg_match:
            arg = arg_match.group(2)
            arg_l = arg.lower()
            if arg_match.group(1).lower() == "model":
                for name in self._models:
                    if arg_l and arg_l not in name.lower():
                        continue
                    is_active = name == self._active_model
                    yield Completion(
                        name,
                        start_position=-len(arg),
                        display=("● " + name) if is_active else name,
                        display_meta="active" if is_active else "model",
                    )
            else:
                for level in self._EFFORT_LEVELS:
                    if arg_l and not level.startswith(arg_l):
                        continue
                    yield Completion(level, start_position=-len(arg), display=level, display_meta="effort")
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
    # Max agent rows shown in the spawned-agent dashboard before collapsing the
    # rest into a "+N more" line. Running agents are shown first.
    AGENT_DASHBOARD_MAX_ROWS = 14
    # Slide window over committed history so a long-running session can't
    # grow ``_content_lines`` unbounded. ~2000 entries is a few hours of
    # active use; older lines roll off (the daemon-side session history
    # remains authoritative).
    CONTENT_HISTORY_LIMIT = 2000
    # Cap the in-flight streaming buffer at ~64KB. Reasoning models can
    # emit hundreds of KB per turn; once the buffer crosses this, we keep
    # only the tail so render cost stays bounded and we don't trigger
    # quadratic blowups in downstream consumers.
    STREAMING_BUFFER_CHAR_LIMIT = 64 * 1024

    def __init__(self) -> None:
        """Build an empty renderer with the spinner parked and no panels active."""
        self._content_lines: collections.deque[str] = collections.deque(maxlen=self.CONTENT_HISTORY_LIMIT)
        self._running = False
        self._queue_count = 0
        self._plan_mode = False
        self._activity_mode = "code"
        self._token_info = ""
        self._cost_info = ""

        self._active_panel: str = ""

        # ``_streaming_parts`` is a list of chunks; we materialise to a
        # single string only when we need to render. Avoids the O(n²)
        # cost of ``self._streaming_text += text`` for every token.
        self._streaming_parts: list[str] = []
        self._streaming_chars: int = 0

        self._thinking_text: str = ""

        self._active_tools: dict[str, Callable[[], str]] = {}

        # Per-agent lane state for the spawned-agent dashboard. Keyed by task_id;
        # each value is a structured dict {label, agent_type, status, count,
        # action, result, started_at}. O(agents), never O(calls).
        self._agent_lanes: dict[str, dict[str, Any]] = {}
        self._todo_items: list[str] = []

        self._spinner_frame: int = 0
        self._spinner_started_at: float = 0.0
        self._spinner_label: str = "Working"
        self._last_render_line_count: int = 1
        self._last_render_last_line_width: int = 0
        # Version counter bumped on every state mutation. ``_markup``
        # caches its output keyed by ``(_state_version, columns, rows)``
        # so unchanged frames cost a single dict lookup instead of a
        # full ANSI rebuild.
        self._state_version: int = 0
        self._cached_markup: str | None = None
        self._cached_markup_key: tuple[int, int, int, int, int | None, int | None] | None = None
        # ``_streaming_text`` is exposed as a property to existing callers
        # that read it directly (legacy API). The setter funnels through
        # ``set_streaming_text`` to keep the parts buffer consistent.

    @property
    def _streaming_text(self) -> str:
        """Materialised in-flight streaming buffer (joins lazily)."""
        if not self._streaming_parts:
            return ""
        joined = "".join(self._streaming_parts)
        # Collapse multi-part buffer back into one so repeated reads are O(1).
        self._streaming_parts = [joined]
        return joined

    @_streaming_text.setter
    def _streaming_text(self, value: str) -> None:
        """Replace the streaming buffer; back-compat for legacy callers that assigned to it."""
        self._streaming_parts = [value] if value else []
        self._streaming_chars = len(value)
        self._mark_dirty()

    def _mark_dirty(self) -> None:
        """Invalidate the markup cache. Cheap — just bumps a counter."""
        self._state_version += 1

    def set_running(self, running: bool) -> None:
        """Flip the running indicator; rearms the spinner clock on each off→on edge."""
        was_running = self._running
        if running == was_running:
            return
        self._running = running
        if running:
            self._spinner_started_at = time.monotonic()
            self._spinner_frame = 0
        self._mark_dirty()

    def reset_spinner_timer(self) -> None:
        """Restart the spinner elapsed clock at 0.

        Call when a new tool starts or the spinner label flips so the
        timer reflects the current step, not the turn."""
        self._spinner_started_at = time.monotonic()
        self._mark_dirty()

    def set_spinner_label(self, label: str) -> None:
        """Update the spinner caption; restarts the elapsed clock if it changed."""
        new_label = label or "Working"
        if new_label != self._spinner_label:
            self.reset_spinner_timer()
        if self._spinner_label != new_label:
            self._spinner_label = new_label
            self._mark_dirty()

    def set_queue_count(self, count: int) -> None:
        """Update the queued-inputs badge (clamped at 0)."""
        new = max(0, count)
        if new != self._queue_count:
            self._queue_count = new
            self._mark_dirty()

    def set_plan_mode(self, plan_mode: bool) -> None:
        """Record whether plan mode is currently active."""
        if self._plan_mode != plan_mode:
            self._plan_mode = plan_mode
            self._mark_dirty()

    def set_activity_mode(self, mode: str) -> None:
        """Record the activity label shown in the bottom rule when plan mode is off."""
        new = mode or "code"
        if self._activity_mode != new:
            self._activity_mode = new
            self._mark_dirty()

    def set_stats(self, tokens: str = "", cost: str = "") -> None:
        """Store optional token / cost strings (currently informational only)."""
        if tokens != self._token_info or cost != self._cost_info:
            self._token_info = tokens
            self._cost_info = cost
            self._mark_dirty()

    def append_line(self, line: str) -> None:
        """Commit ``line`` to history after stripping leading/trailing blank edges."""
        line = self._strip_blank_edges(line)
        if not line:
            return
        self._content_lines.extend(self._split_render_lines(line))
        self._mark_dirty()

    def clear_content(self) -> None:
        """Drop every committed history line."""
        self._content_lines.clear()
        self._mark_dirty()

    def replace_lines(self, start: int, count: int, text: str) -> int:
        """Replace a committed rendered-line range with ``text`` split into rows."""
        if start < 0 or count <= 0:
            return 0
        lines = list(self._content_lines)
        if start >= len(lines):
            return 0
        replacement = self._split_render_lines(self._strip_blank_edges(text))
        end = min(len(lines), start + count)
        lines[start:end] = replacement
        self._content_lines.clear()
        self._content_lines.extend(lines)
        self._mark_dirty()
        return len(replacement)

    def set_active_panel(self, text: str) -> None:
        """Pin ``text`` as the active modal panel (approval / question)."""
        new = text or ""
        if new != self._active_panel:
            self._active_panel = new
            self._mark_dirty()

    def clear_active_panel(self) -> None:
        """Remove the pinned modal panel."""
        if self._active_panel:
            self._active_panel = ""
            self._mark_dirty()

    def append_streaming(self, text: str) -> None:
        """Append ``text`` to the in-flight streaming buffer in O(1)."""
        if not text:
            return
        self._streaming_parts.append(text)
        self._streaming_chars += len(text)
        if self._streaming_chars > self.STREAMING_BUFFER_CHAR_LIMIT:
            # Compact to a single tailing chunk so memory and downstream
            # render cost stay bounded for huge responses.
            joined = "".join(self._streaming_parts)
            tail = joined[-self.STREAMING_BUFFER_CHAR_LIMIT :]
            self._streaming_parts = [tail]
            self._streaming_chars = len(tail)
        self._mark_dirty()

    def commit_streaming(self) -> None:
        """Move the streaming buffer onto the committed history and clear it."""
        text = self._strip_blank_edges(self._streaming_text)
        self._streaming_parts = []
        self._streaming_chars = 0
        if text:
            self._content_lines.extend(self._split_render_lines(text))
        self._mark_dirty()

    def clear_streaming(self) -> None:
        """Discard the streaming buffer without committing it."""
        if self._streaming_parts:
            self._streaming_parts = []
            self._streaming_chars = 0
            self._mark_dirty()

    def append_thinking(self, text: str) -> None:
        """Append ``text`` to the rolling thinking preview (last ``N`` lines only)."""
        if not text:
            return
        self._thinking_text += text
        # Keep only the rendered tail in memory. Tailing here (not just at
        # render) ensures the underlying buffer stays small even when the
        # model dumps tens of KB of reasoning.
        if len(self._thinking_text) > 4096:
            self._thinking_text = self._thinking_text[-4096:]
        self._thinking_text = self._tail_lines(self._thinking_text, self.THINKING_PREVIEW_LINES)
        self._mark_dirty()

    def clear_thinking(self) -> None:
        """Discard the thinking preview."""
        if self._thinking_text:
            self._thinking_text = ""
            self._mark_dirty()

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

    @staticmethod
    def _split_render_lines(text: str) -> list[str]:
        """Split rendered ANSI text into display rows while preserving internal blanks."""
        if not text:
            return []
        lines = text.split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        return lines

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

    def set_agent_lane(
        self,
        task_id: str,
        *,
        label: str = "",
        agent_type: str = "",
        status: str = "running",
        count: int = 0,
        action: str = "",
        result: str = "",
    ) -> None:
        """Create or merge one spawned-agent's lane in the dashboard.

        Fields merge so out-of-order updates don't clobber state: identity
        (``label``/``agent_type``) sticks once set, ``count`` only grows, and
        ``action``/``result`` take the latest non-empty value. ``started_at`` is
        stamped once for the elapsed clock."""
        if not task_id:
            return
        lane = self._agent_lanes.get(task_id)
        if lane is None:
            self._agent_lanes[task_id] = {
                "label": label,
                "agent_type": agent_type,
                "status": status or "running",
                "count": int(count),
                "action": action,
                "result": result,
                "started_at": time.monotonic(),
            }
        else:
            if label:
                lane["label"] = label
            if agent_type:
                lane["agent_type"] = agent_type
            lane["status"] = status or lane["status"]
            lane["count"] = max(int(count), int(lane.get("count", 0)))
            if action:
                lane["action"] = action
            if result:
                lane["result"] = result
        self._mark_dirty()

    def set_subagent_preview(self, task_id: str, label: str, text: str) -> None:
        """Back-compat shim — route a rolling preview into the agent dashboard."""
        self.set_agent_lane(task_id, label=label, action=text)

    def clear_subagent_preview(self, task_id: str) -> None:
        """Remove one finished agent's lane from the dashboard."""
        if self._agent_lanes.pop(task_id, None) is not None:
            self._mark_dirty()

    def clear_subagent_previews(self) -> None:
        """Remove every agent lane (called when a turn ends)."""
        if self._agent_lanes:
            self._agent_lanes.clear()
            self._mark_dirty()

    def _render_agent_dashboard(self, spinner_tick: int) -> list[str]:
        """Render the spawned-agent dashboard: a header + one line per agent.

        Running agents sort first; beyond :data:`AGENT_DASHBOARD_MAX_ROWS` the
        remainder collapses into a ``+N more`` line so a large fan-out can't own
        the viewport. Each row shows status, agent name, aggregate call count,
        the current action (or final result), and elapsed time."""
        lanes = list(self._agent_lanes.values())
        total = len(lanes)
        done = sum(1 for ln in lanes if ln.get("status") != "running")
        total_calls = sum(int(ln.get("count", 0)) for ln in lanes)
        sys_fg = active_fg("system")
        accent_fg = active_fg("accent")
        warn_fg = active_fg("warn")
        error_fg = active_fg("error")
        frame = self.SPINNER_FRAMES[spinner_tick % len(self.SPINNER_FRAMES)]

        out = [f"{sys_fg}⬡ {total} agents{_FG_RESET} \x1b[2m{done}/{total} · {total_calls} calls\x1b[0m"]
        running = [ln for ln in lanes if ln.get("status") == "running"]
        finished = [ln for ln in lanes if ln.get("status") != "running"]
        ordered = running + finished
        shown = ordered[: self.AGENT_DASHBOARD_MAX_ROWS]
        now = time.monotonic()
        for ln in shown:
            status = ln.get("status", "running")
            if status == "running":
                icon, icon_fg, detail = frame, warn_fg, str(ln.get("action", ""))
            elif status in ("failed", "cancelled"):
                icon, icon_fg, detail = "✗", error_fg, str(ln.get("result") or status)
            else:
                icon, icon_fg, detail = "✓", accent_fg, str(ln.get("result") or "done")
            name = (ln.get("agent_type") or str(ln.get("label", "")).split("#")[0] or "agent")[:14]
            count = int(ln.get("count", 0))
            count_s = f"{count / 1000:.1f}k" if count >= 1000 else str(count)
            elapsed = self._fmt_elapsed(now - float(ln.get("started_at", now)))
            detail = " ".join(detail.split())[:46]
            out.append(
                f"  {icon_fg}{icon}{_FG_RESET} {sys_fg}{name:<14}{_FG_RESET} "
                f"\x1b[2m{count_s:>5}\x1b[0m  \x1b[2m{detail}\x1b[0m  \x1b[2m{elapsed}\x1b[0m"
            )
        hidden = total - len(shown)
        if hidden > 0:
            out.append(f"  \x1b[2m… +{hidden} more\x1b[0m")
        return out

    @staticmethod
    def _fmt_elapsed(secs: float) -> str:
        """Format an elapsed-seconds value as ``Ns`` or ``MmSSs``."""
        s = max(0, int(secs))
        return f"{s}s" if s < 60 else f"{s // 60}m{s % 60:02d}s"

    def set_todo_items(self, items: list[str]) -> None:
        """Replace the TODO list displayed above the input area."""
        new_items = [i for i in items if i]
        if new_items != self._todo_items:
            self._todo_items = new_items
            self._mark_dirty()

    def clear_todo_items(self) -> None:
        """Remove the TODO list from the status area."""
        if self._todo_items:
            self._todo_items = []
            self._mark_dirty()

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

    def _body_lines(self, spinner_tick: int) -> list[str]:
        """Return the scrollable status body as rendered lines without the input rule."""
        lines = list(self._content_lines)

        for render_fn in self._active_tools.values():
            try:
                tool_text = render_fn()
            except Exception:
                continue
            lines.extend(self._split_render_lines(tool_text))

        if self._thinking_text:
            lines.extend(self._split_render_lines(self._thinking_text))

        if self._agent_lanes:
            lines.extend(self._render_agent_dashboard(spinner_tick))

        streaming = self._streaming_text
        if streaming:
            lines.extend(self._split_render_lines(streaming))

        if self._active_panel:
            lines.extend(self._split_render_lines(self._active_panel))

        if self._todo_items:
            warn_fg = active_fg("warn")
            accent_fg = active_fg("accent")
            muted_fg = active_fg("muted")
            # Items carry their state as a leading glyph: ``✓`` completed,
            # ``◐`` in-progress, otherwise pending. Completed rows dim away,
            # the in-progress row stays bright so the active task stands out.
            done = sum(1 for it in self._todo_items if it.startswith("✓"))
            total = len(self._todo_items)
            lines.append(f"\x1b[1m{warn_fg}TODO\x1b[0m \x1b[2m{done}/{total}\x1b[0m")
            for item in self._todo_items[:10]:
                if item.startswith("✓"):
                    lines.append(f"  {accent_fg}✓{_FG_RESET} \x1b[2m{item[1:].strip()}\x1b[0m")
                elif item.startswith("◐"):
                    lines.append(f"  {warn_fg}◐{_FG_RESET} {item[1:].strip()}")
                else:
                    lines.append(f"  {muted_fg}☐ {item}{_FG_RESET}")
            if total > 10:
                lines.append(f"  \x1b[2m… and {total - 10} more\x1b[0m")

        if self._running:
            frame = self.SPINNER_FRAMES[spinner_tick % len(self.SPINNER_FRAMES)]
            elapsed = int(time.monotonic() - self._spinner_started_at) if self._spinner_started_at else 0
            queued = f"  ·  {self._queue_count} queued" if self._queue_count else ""
            lines.append(
                f"{active_fg('accent')}{frame}{_FG_RESET} \x1b[1m{self._spinner_label}…\x1b[0m  "
                f"\x1b[2m{elapsed}s{queued}  ·  esc to interrupt\x1b[0m"
            )

        return lines

    def body_line_count(self) -> int:
        """Return the number of rendered body lines available for scrolling."""
        return len(self._body_lines(self._current_spinner_tick()))

    def _input_rule(self, columns: int) -> str:
        """Return the fixed input separator line shown below the scrollable body."""
        title_parts = ["input"]
        if self._plan_mode:
            title_parts.append("plan")
        elif self._activity_mode == "researcher":
            title_parts.append("research")
        elif self._activity_mode == "objective":
            title_parts.append("objective")
        if self._queue_count:
            title_parts.append(f"{self._queue_count} queued")
        title = f" {' · '.join(title_parts)} "
        dash = "╌" if self._plan_mode else "─"
        border = f"{dash}{dash}{title}{dash * max(0, columns - len(title) - 2)}"
        border_style = _mode_style(self._plan_mode, self._activity_mode)
        return f"{border_style}{border}\x1b[0m"

    def _markup(self, scroll_y: int | None = None, *, visible_rows: int | None = None) -> str:
        """Compose the ANSI markup string, optionally sliced by ``scroll_y``.

        When ``scroll_y`` is ``None`` (default) the view follows the tail
        (live mode).  When ``scroll_y >= 0`` the view is pinned to that rendered
        line offset, allowing the user to scroll back through history.

        Caching: the result is keyed by ``(_state_version, columns, rows,
        spinner_tick, scroll_y, visible_rows)`` so unchanged frames re-use the cached string.
        """
        columns = self._terminal_columns()
        rows = self._terminal_rows()
        spinner_tick = self._current_spinner_tick()
        key = (self._state_version, columns, rows, spinner_tick, scroll_y, visible_rows)
        if self._cached_markup is not None and self._cached_markup_key == key:
            return self._cached_markup

        # The app keeps provider reasoning hidden by default by not forwarding
        # think chunks here; callers that explicitly use ``append_thinking`` get
        # the bounded preview promised by this renderer.
        body_lines = self._body_lines(spinner_tick)
        body_line_count = len(body_lines)
        if visible_rows is None:
            body_budget = max(20, rows * 2)
        else:
            # Leave one row for the fixed input separator.
            body_budget = max(0, visible_rows - 1)

        if body_budget <= 0:
            selected_body: list[str] = []
        elif scroll_y is None:
            start = max(0, body_line_count - body_budget)
            selected_body = body_lines[start:]
        else:
            max_start = max(0, body_line_count - body_budget)
            start = min(max(0, scroll_y), max_start)
            selected_body = body_lines[start : start + body_budget]

        # Pin the input separator to the bottom of the viewport (just above the
        # input buffer) by padding between the body and the rule; otherwise a
        # short body leaves the rule floating right under the banner.
        if visible_rows is not None:
            pad = max(0, visible_rows - len(selected_body) - 1)
            parts = [*selected_body, *([""] * pad), self._input_rule(columns)]
        else:
            parts = [*selected_body, self._input_rule(columns)]
        markup = "\n".join(parts)
        self._last_render_line_count, self._last_render_last_line_width = self._count_lines(markup)
        self._cached_markup = markup
        self._cached_markup_key = key
        return markup

    @staticmethod
    def _count_lines(markup: str) -> tuple[int, int]:
        """Count newlines and last-line width in O(n) without a regex sweep.

        The previous implementation ran ``_ANSI_RE.sub("", markup)`` then
        ``split("\\n")`` on the result — two full passes per frame. We can
        do both jobs in a single linear scan: track newlines as we go,
        and reset the last-line counter on each ``\\n``. Skipping inside
        ``\\x1b[...m`` runs gives the same plain-text width as the regex
        without allocating a stripped copy.
        """
        if not markup:
            return 1, 0
        newlines = 0
        last_width = 0
        i = 0
        n = len(markup)
        while i < n:
            ch = markup[i]
            if ch == "\x1b" and i + 1 < n and markup[i + 1] == "[":
                # Skip CSI sequence: ESC [ params terminator
                j = i + 2
                while j < n and not (0x40 <= ord(markup[j]) <= 0x7E):
                    j += 1
                i = j + 1
                continue
            if ch == "\n":
                newlines += 1
                last_width = 0
            else:
                last_width += 1
            i += 1
        return max(1, newlines + 1), last_width

    def _current_spinner_tick(self) -> int:
        """Return the spinner tick at ~4 FPS (only advances while running).

        Decoupling the spinner clock from paint count means idle screens
        cache forever, and animation rate is wall-clock-based instead of
        "however many paints happened this second".
        """
        if not self._running or not self._spinner_started_at:
            return self._spinner_frame
        # 4 FPS animation → 250ms per frame.
        tick = int((time.monotonic() - self._spinner_started_at) * 4)
        self._spinner_frame = tick
        return tick

    def line_count(self) -> int:
        """Count the printable lines in the current markup (ANSI stripped)."""
        markup = self._markup()
        if not markup:
            return 0
        plain = _ANSI_RE.sub("", markup)
        return plain.count("\n") + (0 if plain.endswith("\n") else 1)

    def __call__(self, scroll_y: int | None = None) -> AnyFormattedText:
        """Return ``ANSI(self._markup(scroll_y))`` so this object plugs into prompt_toolkit."""
        return ANSI(self._markup(scroll_y))


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
        self._reasoning_effort = "off"
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

    def set_reasoning_effort(self, effort: str) -> None:
        """Record the current thinking effort (``off``/``low``/``medium``/``high``) for the footer."""
        self._reasoning_effort = effort or "off"

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
        when both halves don't fit on one row. Per-segment color is
        routed through skin roles (agent/model/mode -> system, the
        activity dot/context badge -> accent, cwd/branch/think/tip ->
        muted) while the row-wide ``_mode_style`` tint backs the
        separators so width math (measured on the ANSI-stripped text)
        stays identical to the plain layout."""
        style = _mode_style(self._plan_mode, self._activity_mode)

        def _seg(role: str, text: str) -> str:
            """Tint ``text`` with ``role`` then restore the row's base tint."""
            return f"{active_fg(role)}{text}{_FG_RESET}{style}"

        spinner = "●" if self._running else "○"
        model = self._model or "—"
        agent = f"agent ({model} {_seg('accent', spinner)})"
        mode = f"mode: {'plan' if self._plan_mode else self._activity_mode}"
        think = f"think: {self._reasoning_effort}"
        columns = self._terminal_columns()

        cwd_max = max(20, columns // 3)
        cwd_short = self._truncate_path(self._cwd, cwd_max) if self._cwd else ""
        left_segments = [_seg("system", agent)]
        if cwd_short:
            left_segments.append(_seg("muted", cwd_short))
        if self._branch:
            left_segments.append(_seg("muted", self._branch))
        left_segments.append(_seg("system", mode))
        left_segments.append(_seg("muted", think))
        left_segments.append(_seg("muted", self._tip))
        left = "  ".join(left_segments)
        plain_left_segments = [
            f"agent ({model} {spinner})",
            *([cwd_short] if cwd_short else []),
            *([self._branch] if self._branch else []),
            mode,
            think,
            self._tip,
        ]
        plain_left = "  ".join(plain_left_segments)

        if self._context_max > 0:
            pct = (self._context_used / self._context_max) * 100
            ctx_text = (
                f"context: {pct:.1f}% "
                f"({self._format_tokens(self._context_used)}/"
                f"{self._format_tokens(self._context_max)})"
            )
        else:
            ctx_text = "context: 0.0% (0/0)"

        plain_right = ctx_text
        ctx_render = _seg("accent", ctx_text)

        rule = style + ("─" * columns) + "\x1b[0m"

        if len(plain_left) + len(plain_right) + 2 <= columns:
            gap = max(1, columns - len(plain_left) - len(plain_right))
            row = f"{style}{left}{' ' * gap}{ctx_render}\x1b[0m"
            return rule + "\n" + row + "\n"

        right_pad = max(0, columns - len(plain_right))
        line1 = f"{style}{left}\x1b[0m"
        line2 = f"{' ' * right_pad}{style}{ctx_render}\x1b[0m"
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
        # Invalidate throttling: ``_last_invalidate_ts`` records the wall
        # clock of the last forced paint; ``_pending_invalidate`` is the
        # asyncio handle for the trailing-edge flush when a paint was
        # suppressed during the throttle window.
        self._last_invalidate_ts: float = 0.0
        self._pending_invalidate: Any = None

        self._active_question: Any = None
        self._active_approval: Any = None
        self._active_resume: Any = None

        # Whose turn last touched history — ``""`` / ``"user"`` / ``"assistant"``
        # / ``"tool"``. Drives the role captions so a new voice gets a header
        # only when it actually changes hands (see ``_emit_voice_caption``).
        self._last_voice: str = ""

        self._scroll_y: int | None = None
        self._status_visible_rows = 20
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

        self._status_control = _ScrollableFormattedTextControl(
            renderer=self._status,
            scroll_y_accessor=lambda: self._scroll_y,
            visible_rows_setter=self._set_status_visible_rows,
        )

        self._status_window = Window(
            content=self._status_control,
            height=Dimension(weight=1),
            wrap_lines=True,
            always_hide_cursor=True,
        )

        self._status_control.mouse_handler = self._on_status_mouse  # type: ignore[method-assign]

        self._buffer_control = BufferControl(
            buffer=self._input_buffer,
            focusable=True,
        )

        # Input grows with its content (newlines via ctrl-j and wrapped long
        # lines), from 1 row up to _INPUT_MAX_ROWS; beyond that it scrolls.
        self._buffer_window = Window(
            content=self._buffer_control,
            height=Dimension(min=1, max=_INPUT_MAX_ROWS),
            dont_extend_height=True,
            wrap_lines=True,
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

    def _status_total_lines(self) -> int:
        """Total printable lines in the full rendered status body."""
        return max(0, self._status.body_line_count())

    def _set_status_visible_rows(self, rows: int) -> None:
        """Remember the status-window height for PageUp/PageDown clamping."""
        self._status_visible_rows = max(1, rows)

    def _scroll_by(self, delta: int) -> None:
        """Adjust the scroll offset by ``delta`` lines; ``None`` means tracking bottom.

        The scroll offset is measured in rendered body lines, not message
        blocks, so long assistant responses can be paged through line by line."""
        total = self._status_total_lines()
        visible_body_lines = max(1, self._status_visible_rows - 1)
        max_start = max(0, total - visible_body_lines)
        if max_start == 0:
            self._scroll_y = None
            self._invalidate()
            return
        current = max_start if self._scroll_y is None else min(self._scroll_y, max_start)
        new = current + delta
        if new >= max_start:
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
        """Render the branded prompt glyph; accent when idle, dim while running."""
        if line_number == 0 and wrap_count == 0:
            glyph = get_active_skin().label("prompt_symbol")
            if self._running:
                return ANSI(f"\x1b[2m{glyph}\x1b[0m ")
            return ANSI(f"\x1b[1m{active_fg('accent')}{glyph}\x1b[0m ")
        return ""

    SELECT_SENTINEL = "\x00__select_active_question__\x00"
    APPROVAL_SENTINEL = "\x00__select_active_approval__\x00"
    RESUME_SENTINEL = "\x00__select_active_resume__\x00"
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

    def set_active_resume(self, panel: Any) -> None:
        """Pin ``panel`` as the active resume picker and invalidate the screen."""
        self._active_resume = panel
        self._status.set_active_panel(panel.compose() if panel else "")
        self._invalidate()

    def clear_active_resume(self) -> None:
        """Dismiss the active resume picker."""
        self._active_resume = None
        self._status.clear_active_panel()
        self._invalidate()

    def refresh_active_resume(self) -> None:
        """Re-render the active resume picker after a cursor move."""
        if self._active_resume is not None:
            self._status.set_active_panel(self._active_resume.compose())
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
            if self._active_resume is not None:
                self._active_resume.move_cursor_up()
                self.refresh_active_resume()
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
            if self._active_resume is not None:
                self._active_resume.move_cursor_down()
                self.refresh_active_resume()
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
            if self._active_resume is not None and not buffer.text.strip():
                self._input_queue.put_nowait(self.RESUME_SENTINEL)
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
            if self._active_resume is not None:
                self._input_queue.put_nowait("/cancel")
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

    def set_models(self, models: list[str], active: str = "") -> None:
        """Feed the provider's model ids to the completer for ``/model`` completion."""
        self._completer.set_models(models, active)

    def set_context(self, used: int, max_: int) -> None:
        """Update the footer's context-usage gauge."""
        self._footer.set_context(used, max_)
        self._invalidate()

    def set_reasoning_effort(self, effort: str) -> None:
        """Update the footer's thinking-effort indicator."""
        self._footer.set_reasoning_effort(effort)
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

    def append_user_message(self, text: str) -> None:
        """Commit a user turn as a lapis-barred block so it reads as *your* voice.

        Each line carries a ``▌`` gutter in the primary color and a dim ``you``
        caption, setting the user turn apart from the assistant's flush-left
        prose and the indented tool band."""
        bar = f"{active_fg('primary')}▌{_FG_RESET}"
        self._status.append_line(" ")  # one blank row to open the turn
        self._status.append_line(f"{bar} {active_fg('primary')}\x1b[2myou\x1b[0m")
        for raw in (text or "").splitlines() or [text or ""]:
            self._status.append_line(f"{bar} {raw}")
        self._last_voice = "user"
        self._invalidate()

    def _emit_voice_caption(self, voice: str) -> None:
        """Emit a one-line role caption when the speaking voice changes hands.

        The assistant gets a faint ``xerxes`` caption (in the accent color) the
        first time it speaks after a user turn or a tool group, so a turn reads
        as alternating voices instead of one flat stream."""
        if self._last_voice != voice and voice == "assistant":
            self._status.append_line(" ")
            self._status.append_line(f"{active_fg('accent')}xerxes{_FG_RESET}")
        self._last_voice = voice

    def clear_content(self) -> None:
        """Wipe the committed status history and reset scroll to tail-follow."""
        self._status.clear_content()
        self._scroll_y = None
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
            self._emit_voice_caption("assistant")
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
            self._last_voice = "tool"
        self._invalidate()

    def clear_active_tools(self) -> None:
        """Drop every live-tool renderer."""
        self._status.clear_active_tools()
        self._invalidate()

    def set_agent_lane(
        self,
        task_id: str,
        *,
        label: str = "",
        agent_type: str = "",
        status: str = "running",
        count: int = 0,
        action: str = "",
        result: str = "",
    ) -> None:
        """Create or update one spawned-agent's lane in the live dashboard."""
        self._status.set_agent_lane(
            task_id,
            label=label,
            agent_type=agent_type,
            status=status,
            count=count,
            action=action,
            result=result,
        )
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

    def set_todo_items(self, items: list[str]) -> None:
        """Replace the TODO list displayed above the input area."""
        self._status.set_todo_items(items)
        self._invalidate()

    def clear_todo_items(self) -> None:
        """Remove the TODO list from the status area."""
        self._status.clear_todo_items()
        self._invalidate()

    # Minimum gap between forced re-paints. Streamed LLM tokens arrive
    # dozens of times per second; without throttling each chunk would
    # trigger a full status rebuild. 33ms = ~30 FPS upper bound, which
    # is smoother than a human eye notices and dramatically cheaper
    # than the previous "one paint per token" behaviour.
    _INVALIDATE_MIN_INTERVAL = 0.033

    def _invalidate(self) -> None:
        """Request a redraw — coalesced to at most :data:`_INVALIDATE_MIN_INTERVAL` apart.

        prompt_toolkit ``invalidate()`` schedules a full status rebuild on
        the next event-loop tick. When the daemon streams a hot response
        (Kimi-for-coding emits 50-100 tokens/sec) the resulting paint
        storm dominates the TUI's CPU profile. We coalesce: if a paint
        is already pending or we paint too recently, the call is a no-op
        — the trailing edge of the burst still lands as a paint thanks
        to the scheduled task below, so no content is lost.
        """
        if self._app is None:
            return
        now = time.monotonic()
        gap = now - self._last_invalidate_ts
        if gap >= self._INVALIDATE_MIN_INTERVAL:
            self._last_invalidate_ts = now
            self._app.invalidate()
            return
        # We're inside the throttle window — schedule a single trailing
        # invalidate so a final state change always reaches the screen.
        if self._pending_invalidate is not None:
            return
        loop = asyncio.get_event_loop()
        delay = self._INVALIDATE_MIN_INTERVAL - gap
        self._pending_invalidate = loop.call_later(delay, self._flush_invalidate)

    def _flush_invalidate(self) -> None:
        """Trailing-edge paint; clears the pending-invalidate handle."""
        self._pending_invalidate = None
        if self._app is None:
            return
        self._last_invalidate_ts = time.monotonic()
        self._app.invalidate()

    @staticmethod
    def _completion_style() -> Style:
        """Build the completion-menu ``Style`` from the active skin's palette.

        prompt_toolkit's named style classes accept ``#rrggbb`` values, so we
        source them straight from the skin's role hexes. Only the menu chrome is
        branded (unselected/selected rows, the meta column, the scrollbar) so
        the menu reads in-palette instead of falling back to library defaults."""
        skin = get_active_skin()
        primary = skin.color("primary")
        accent = skin.color("accent")
        muted = skin.color("muted")
        tool_name = skin.color("tool_name")
        return Style.from_dict(
            {
                "completion-menu.completion": f"bg:default {tool_name}",
                "completion-menu.completion.current": f"bg:{primary} #0a0e14 bold",
                "completion-menu.meta.completion": f"bg:default {muted}",
                "completion-menu.meta.completion.current": f"bg:{accent} #0a0e14",
                "scrollbar.background": f"bg:{muted}",
                "scrollbar.button": f"bg:{primary}",
            }
        )

    async def run(self) -> Application[None]:
        """Run the prompt_toolkit application until it exits; returns it for inspection.

        Set ``XERXES_MOUSE=0`` to disable mouse support if wheel events
        should stay with the host terminal instead of scrolling the TUI.

        ``refresh_interval`` is bumped to 500ms — prompt_toolkit only
        needs the periodic refresh to advance time-based UI (the spinner
        and elapsed counter); all content updates already trigger
        :meth:`_invalidate`. The previous 100ms tick was costing 10
        CPU-bound paints per second on every idle screen.
        """
        self._app = Application(
            self._layout,
            key_bindings=self._kb,
            style=self._completion_style(),
            erase_when_done=True,
            mouse_support=os.environ.get("XERXES_MOUSE", "1").lower() not in {"0", "false", "no", "off"},
            full_screen=True,
            refresh_interval=0.5,
        )
        await self._app.run_async(handle_sigint=False)
        return self._app

    def stop(self) -> None:
        """Ask the running application to exit; no-op when not running."""
        if self._app and self._app.is_running:
            self._app.exit()
        if self._pending_invalidate is not None:
            self._pending_invalidate.cancel()
            self._pending_invalidate = None

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
