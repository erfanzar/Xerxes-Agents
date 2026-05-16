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
"""Per-turn content blocks rendered into the Xerxes TUI status area.

Each block subclass accumulates one logical fragment of a turn —
streaming text, reasoning trace, tool call, nested subagent tool calls,
notifications — and exposes a ``compose`` method that returns ANSI
markup ready for :class:`~xerxes.tui.prompt.StatusRenderer` to display.

The :class:`ApprovalRequestPanel` and :class:`QuestionRequestPanel`
modal blocks are also defined here; they live inside the same render
slot as the streaming blocks while interactive input is pending."""

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
        """No-op stand-in used when ``markdown_it`` isn't installed."""

        def parse(self, text: str) -> list[Any]:
            """Return an empty token list regardless of input."""
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
    """Return ``True`` when ``text`` ends at a Markdown block boundary.

    Recognized as a boundary: trailing ``\\n\\n`` or a final line that
    matches a heading / fence / quote / list / horizontal-rule marker.
    Used to decide when streaming content can be split into paragraphs."""
    if not text:
        return False

    if text.endswith("\n\n"):
        return True

    return bool(_BLOCK_PATTERN.match(text.rstrip("\n").split("\n")[-1]))


def _extract_key_argument(arguments: str | dict[str, Any]) -> str:
    """Pick a short, identifying argument value to show beside a tool name.

    Prefers semantically-meaningful keys (``path``, ``file``, ``query``,
    ``url``, ``name``, ``code``, ``command``, ``target``) before falling
    back to the first non-empty value. Output is always truncated to
    keep the live tool line single-row."""
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


def _markup_safe(text: Any) -> str:
    """Return text that cannot be interpreted as prompt markup tags."""
    return str(text).replace("[", "(").replace("]", ")")


class _ToolCallLexer:
    """Incremental JSON lexer for streaming tool-call argument deltas.

    Each ``append`` retries a full ``json.loads`` of the accumulated
    buffer; once the buffer parses, :attr:`is_complete` flips and
    subsequent appends are ignored. This lets the TUI show a key
    argument the moment the JSON closes without paying the cost of a
    streaming JSON parser."""

    def __init__(self) -> None:
        """Start with an empty buffer in the incomplete state."""
        self._buffer = ""
        self._complete = False
        self._arguments: dict[str, Any] | None = None

    def append(self, chunk: str) -> None:
        """Append ``chunk`` and attempt a full re-parse; no-op once complete."""
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
        """Parsed argument dict once :attr:`is_complete` is true; otherwise ``None``."""
        return self._arguments

    @property
    def is_complete(self) -> bool:
        """``True`` once the buffered JSON has parsed successfully."""
        return self._complete


class _ContentBlock:
    """Accumulates assistant text and emits paragraphs at block boundaries.

    Attributes:
        MAX_LIVE_LINES: Soft cap retained for future trimming logic.
    """

    MAX_LIVE_LINES = 40

    def __init__(self, block_id: str) -> None:
        """Allocate an empty block tagged with ``block_id`` and start its timer."""
        self.block_id = block_id
        self._raw = ""
        self._committed = False
        self._started_at = time.monotonic()
        self._token_count = 0
        self._subagent_text_len = 0

    def append(self, text: str) -> list[str]:
        """Append ``text``; return any fully-formed paragraphs that just closed.

        Paragraphs split on ``\\n\\n``; the trailing fragment stays in
        the buffer for the next call."""
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
        """Uncommitted text currently sitting in the buffer."""
        return self._raw

    @property
    def committed_text(self) -> str:
        """Alias for :attr:`raw_text` retained for legacy callers."""
        return self._raw

    def finalize(self) -> None:
        """Mark the block as no longer accepting more text."""
        self._committed = True

    @property
    def is_finalized(self) -> bool:
        """``True`` once :meth:`finalize` has been called."""
        return self._committed

    def compose(self, *, running: bool = False) -> AnyFormattedText:
        """Render the block: spinner glyph, body text, elapsed time, token count."""
        from prompt_toolkit.formatted_text import ANSI

        if not self._raw.strip():
            return ANSI("")

        elapsed = int(time.monotonic() - self._started_at)
        spinner = "⠋" if running else " "
        tail = self._raw.strip()
        return ANSI(f"[dim]{spinner} {tail}  [dim]{elapsed}s · {self._token_count} tokens[/dim]")


@dataclass
class _ThinkingBlock:
    """Accumulates reasoning-trace fragments with an animated spinner.

    Attributes:
        block_id: Identifier matching the parent content block.
    """

    block_id: str
    _raw: str = ""
    _started_at: float = field(default_factory=lambda: time.monotonic(), repr=False)
    _frame: int = 0
    _frames: tuple[str, ...] = (".  ", ".. ", "...", " ..", "  .", "   ")
    _committed: bool = field(default=False, repr=False)

    def append(self, text: str) -> list[str]:
        """Append ``text``; return any paragraphs that closed since the last call."""
        self._raw += text

        if "\n\n" not in text and not _is_block_boundary(self._raw):
            return []
        parts = self._raw.split("\n\n")
        committed = "\n\n".join(parts[:-1])
        self._raw = parts[-1] if parts[-1].strip() else ""
        return [committed]

    def finalize(self) -> None:
        """Mark the thinking block as no longer accepting more text."""
        self._committed = True

    @property
    def raw_text(self) -> str:
        """Uncommitted text currently sitting in the buffer."""
        return self._raw

    def compose(self, *, running: bool = False) -> AnyFormattedText:
        """Render the thinking block; advances the spinner frame on every call."""
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
    """One nested tool call inside a parent (e.g. subagent) tool invocation.

    Attributes:
        tool_call_id: Unique id from the daemon.
        name: Tool name (rendered with cyan).
        key_arg: Short representative argument (path/url/etc).
        status: ``"running"`` / ``"done"`` / ``"error"``.
        result: Tool return value (only shown to debugger surfaces).
        duration_ms: Execution duration once finished.
    """

    tool_call_id: str
    name: str
    key_arg: str
    status: str = "running"
    result: str = ""
    duration_ms: float = 0.0

    def finish(self, result: str, duration_ms: float) -> None:
        """Mark this sub-call done and record its result + duration."""
        self.result = result
        self.duration_ms = duration_ms
        self.status = "done"

    def compose(self) -> str:
        """Render one bullet line: status icon, tool name, key argument, timing."""
        from .console import _prompt_text_to_ansi

        icon = {"running": "◐", "done": "✓", "error": "✗"}.get(self.status, "·")
        name = _markup_safe(self.name)
        key_arg = _markup_safe(self.key_arg)
        if self.status == "done":
            markup = f"  {icon} [cyan]{name}[/cyan] ({key_arg}) — {self.duration_ms:.0f}ms"
        else:
            markup = f"  {icon} [cyan]{name}[/cyan] ({key_arg})"
        return _prompt_text_to_ansi(markup)


class _ToolCallBlock:
    """Live + final renderer for one tool invocation and its nested sub-calls.

    Attributes:
        MAX_RESULT_LINES: How many trailing result lines to keep in the
            committed view.
        MAX_SUBAGENT_TOOL_LINES: Tail length for the sub-call list.
    """

    MAX_RESULT_LINES = 5
    MAX_SUBAGENT_TOOL_LINES = 5

    def __init__(self, block_id: str, tool_call_id: str, name: str, arguments: str | None) -> None:
        """Start a running block; ``arguments`` is fed into the incremental lexer."""
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
        """Return a short representative argument, parsing JSON when possible."""
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
        """Feed a JSON fragment into both the raw buffer and incremental lexer."""
        self._arguments = (self._arguments or "") + chunk
        self._args_lexer.append(chunk)

    def set_result(self, result: str, duration_ms: float) -> None:
        """Record the tool's return value and flip ``_status`` to done."""
        self._result = result
        self._duration_ms = duration_ms
        self._status = "done"

    def append_sub_tool_call(self, tool_call_id: str, name: str, key_arg: str) -> _SubToolCall:
        """Register a nested sub-call and return the created :class:`_SubToolCall`."""
        sub = _SubToolCall(tool_call_id=tool_call_id, name=name, key_arg=key_arg)
        self._sub_tool_calls.append(sub)
        return sub

    def finish_sub_tool_call(self, tool_call_id: str, result: str, duration_ms: float) -> None:
        """Mark the sub-call with id ``tool_call_id`` finished; no-op if unknown."""
        for sub in self._sub_tool_calls:
            if sub.tool_call_id == tool_call_id:
                sub.finish(result, duration_ms)
                break

    def set_subagent_metadata(self, agent_id: str, subagent_type: str) -> None:
        """Tag this block as a subagent invocation for downstream renderers."""
        self._subagent_id = agent_id
        self._subagent_type = subagent_type

    def compose(self) -> str:
        """Render the full block (status header, sub-call lines, truncated result)."""
        from .console import _prompt_text_to_ansi

        elapsed_s = int(time.monotonic() - self._started_at)
        duration = f"{elapsed_s}s" if self._status == "running" else f"{self._duration_ms:.0f}ms"
        icon = {"running": "◐", "done": "✓", "error": "✗"}.get(self._status, "·")
        status_word = "Using" if self._status == "running" else "Used"
        name = _markup_safe(self.name)
        key_arg = _markup_safe(self.key_arg)

        lines = [f"{icon} [bold cyan]{status_word} {name}[/bold cyan] ([dim]{key_arg}[/dim]) — {duration}"]

        sub_lines: list[str] = []
        for sub in self._sub_tool_calls[-self.MAX_SUBAGENT_TOOL_LINES :]:
            sub_lines.append(sub.compose())

        if self._status == "done" and self._result:
            result_lines = _markup_safe(self._result).strip().split("\n")[: self.MAX_RESULT_LINES]
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
    """Single notification (icon + title + body) committed to the prompt history.

    Attributes:
        MAX_BODY_LINES: Body lines kept before truncation marker.
        PROSE_CATEGORIES: Categories rendered through full Markdown
            instead of the compact icon+text layout.
    """

    MAX_BODY_LINES = 2

    def __init__(self, notification_id: str, category: str, severity: str, title: str, body: str) -> None:
        """Capture the notification fields verbatim; rendering happens in :meth:`compose`."""
        self.notification_id = notification_id
        self.category = category
        self.severity = severity
        self.title = title
        self.body = body

    PROSE_CATEGORIES = ("slash", "history")

    def compose(self) -> str:
        """Format the notification as one ANSI block.

        Slash/history categories pass through Markdown for nicer output;
        everything else uses the compact icon + title + dim-body layout."""
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
    """Return the ANSI 256-color index for ``sev`` (defaults to grey ``"245"``)."""
    return {"info": "51", "warning": "214", "error": "196"}.get(sev, "245")


class ApprovalRequestPanel:
    """Interactive approval modal: navigate options with arrows, submit with Enter.

    Attributes:
        SELECTIONS: Concrete response values sent back to the daemon.
        LABELS: Human-readable display strings keyed by selection.
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
        """Capture the daemon-sent request fields; cursor starts on ``approve``."""
        self.request_id = request_id
        self.tool_call_id = tool_call_id
        self.action = action
        self.description = description
        self.diff = diff
        self._selected = 0
        self._expanded = False

    @property
    def selected_response(self) -> str:
        """Currently highlighted response value (one of :attr:`SELECTIONS`)."""
        return self.SELECTIONS[self._selected]

    def move_cursor_up(self) -> None:
        """Highlight the previous response (wraps to the bottom)."""
        self._selected = (self._selected - 1) % len(self.SELECTIONS)

    def move_cursor_down(self) -> None:
        """Highlight the next response (wraps to the top)."""
        self._selected = (self._selected + 1) % len(self.SELECTIONS)

    def toggle_expand(self) -> None:
        """Toggle inline diff display when a diff was attached to the request."""
        if self.diff:
            self._expanded = not self._expanded

    def compose(self) -> str:
        """Render the panel: header, action description, response list, optional diff."""
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
    """Sequential question modal: walks through a list, collecting one answer each.

    Each question dict carries ``id``, ``question``, ``options``
    (multiple-choice list, possibly empty), and ``allow_free_form``
    (whether typing arbitrary text is permitted alongside the options)."""

    def __init__(
        self,
        request_id: str,
        questions: list[dict[str, Any]],
    ) -> None:
        """Initialize at question 0 with the first option highlighted."""
        self.request_id = request_id
        self.questions = questions
        self._question_index = 0
        self._option_index = 0
        self._free_text_mode = False
        self._free_text = ""
        self._answers: dict[str, str] = {}

    @property
    def current_question(self) -> dict[str, Any]:
        """Dict for the question currently being answered."""
        return self.questions[self._question_index]

    @property
    def current_options(self) -> list[str]:
        """Option labels for the current question (may be empty for free-form-only)."""
        return self.current_question.get("options", [])

    @property
    def selected_label(self) -> str:
        """Display string for the highlighted selection (option or free text)."""
        opts = self.current_options
        if self._free_text_mode:
            return self._free_text or "(type your answer)"
        if opts and self._option_index < len(opts):
            return opts[self._option_index]
        return ""

    def move_up(self) -> None:
        """Highlight the previous option (no-op while in free-text mode)."""
        if self._free_text_mode:
            return
        self._option_index = (self._option_index - 1) % max(1, len(self.current_options))

    def move_down(self) -> None:
        """Highlight the next option (no-op while in free-text mode)."""
        if self._free_text_mode:
            return
        self._option_index = (self._option_index + 1) % max(1, len(self.current_options))

    def select_other(self) -> None:
        """Switch the cursor into free-text capture mode."""
        self._free_text_mode = True

    def append_free_text(self, char: str) -> None:
        """Append ``char`` to the free-text answer buffer (only while in that mode)."""
        if self._free_text_mode:
            self._free_text += char

    def backspace_free_text(self) -> None:
        """Delete the last character of the free-text answer."""
        if self._free_text_mode and self._free_text:
            self._free_text = self._free_text[:-1]

    def confirm_current(self) -> str | None:
        """Record the highlighted/typed answer and advance to the next question.

        Returns the just-answered question id once every question is
        answered; ``None`` while more questions remain."""
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
        """``True`` once every question in :attr:`questions` has an entry in ``_answers``."""
        return len(self._answers) == len(self.questions)

    def compose(self) -> str:
        """Render the question panel: header, prompt, option list, free-form hint."""
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
