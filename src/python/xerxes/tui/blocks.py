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

from .console import severity_icon
from .skin_engine import active_fg, get_active_skin

# Map notification severities onto skin roles so notification coloring follows
# the active skin instead of the legacy fixed palette in ``console.py``.
_SEVERITY_ROLE: dict[str, str] = {
    "info": "system",
    "success": "accent",
    "warning": "warn",
    "error": "error",
    "debug": "muted",
}


def _severity_role(sev: str) -> str:
    """Return the skin role for ``sev`` (defaults to ``muted``)."""
    return _SEVERITY_ROLE.get(sev, "muted")


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
        raw_arguments = arguments
        try:
            arguments = json.loads(raw_arguments)
        except Exception:
            return raw_arguments[:60] if raw_arguments else ""

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

    Reasoning models (Kimi-for-coding, R1-style chains) can emit tens of
    KB per turn. Storing the whole stream as a single string and growing
    it with ``+=`` was O(n²) — for a 200 KB trace that's ~20 G char ops.
    We accumulate into a chunk list and only collapse when we actually
    need the text. The raw buffer is also capped at
    :data:`RAW_BUFFER_CHAR_LIMIT` since :meth:`compose` only renders the
    tail anyway, so holding more is wasted memory.

    Attributes:
        block_id: Identifier matching the parent content block.
    """

    RAW_BUFFER_CHAR_LIMIT = 8 * 1024
    THINKING_TAIL_LINES = 3
    THINKING_TAIL_CHARS = 200

    block_id: str
    _raw: str = ""
    _parts: list[str] = field(default_factory=list, repr=False)
    _raw_chars: int = field(default=0, repr=False)
    _token_count: int = field(default=0, repr=False)
    _started_at: float = field(default_factory=lambda: time.monotonic(), repr=False)
    _frame: int = 0
    _frames: tuple[str, ...] = (".  ", ".. ", "...", " ..", "  .", "   ")
    _committed: bool = field(default=False, repr=False)

    def _materialise(self) -> str:
        """Join the chunk buffer back into a single string; cached after first call."""
        if not self._parts:
            return self._raw
        if self._raw:
            self._parts.insert(0, self._raw)
        joined = "".join(self._parts)
        if len(joined) > self.RAW_BUFFER_CHAR_LIMIT:
            joined = joined[-self.RAW_BUFFER_CHAR_LIMIT :]
        self._raw = joined
        self._parts = []
        self._raw_chars = len(joined)
        return joined

    def append(self, text: str) -> list[str]:
        """Append ``text``; return any paragraphs that closed since the last call.

        Storage is O(1) — we only materialise the full string when checking
        for paragraph boundaries, and even then only the tail is kept.
        Token-count tracking moves to an incremental counter so we don't
        ``.split()`` the entire buffer every render.
        """
        if not text:
            return []
        self._parts.append(text)
        self._raw_chars += len(text)
        self._token_count += len(text.split())
        # Cheap pre-check: only materialise when the new chunk could close a
        # paragraph. ``_is_block_boundary`` previously ran on every call.
        if "\n\n" not in text and not _is_block_boundary(text):
            if self._raw_chars > self.RAW_BUFFER_CHAR_LIMIT * 2:
                # Compact periodically so the chunk list doesn't bloat.
                self._materialise()
            return []
        full = self._materialise()
        if "\n\n" not in full and not _is_block_boundary(full):
            return []
        parts = full.split("\n\n")
        committed = "\n\n".join(parts[:-1])
        tail = parts[-1] if parts[-1].strip() else ""
        self._raw = tail
        self._raw_chars = len(tail)
        self._parts = []
        return [committed]

    def finalize(self) -> None:
        """Mark the thinking block as no longer accepting more text."""
        self._committed = True

    @property
    def raw_text(self) -> str:
        """Uncommitted text currently sitting in the buffer."""
        return self._materialise()

    def compose(self, *, running: bool = False) -> AnyFormattedText:
        """Render the thinking block; advances the spinner frame on every call.

        Reasoning traces from models like Kimi-for-coding can run for tens of
        thousands of characters; rendering the whole buffer pushes everything
        else off-screen and traps the user in scrollback. We tail to the last
        few lines / chars instead — enough to show what the model is currently
        chewing on, without owning the viewport. Token count comes from the
        incremental counter, so we don't ``.split()`` the whole buffer on
        every frame.
        """
        from prompt_toolkit.formatted_text import ANSI

        self._frame = (self._frame + 1) % len(self._frames)
        frame = self._frames[self._frame]
        elapsed = int(time.monotonic() - self._started_at)
        raw = self._materialise().strip()
        if raw:
            content = self._tail(raw, self.THINKING_TAIL_LINES, self.THINKING_TAIL_CHARS)
        else:
            content = "[thinking...]"
        return ANSI(
            f"[dim i]Thinking {frame}  {elapsed}s · {self._token_count} tokens[/dim i]\n[dim i]{content}[/dim i]"
        )

    @staticmethod
    def _tail(text: str, max_lines: int, max_chars: int) -> str:
        """Return the last ``max_lines`` lines (and ``max_chars`` chars) of ``text``.

        Truncated text is prefixed with ``…`` so the user can see at a glance
        that there is more reasoning above the displayed window.
        """
        truncated = False
        if len(text) > max_chars:
            text = text[-max_chars:]
            truncated = True
        lines = text.splitlines()
        if len(lines) > max_lines:
            lines = lines[-max_lines:]
            truncated = True
        rendered = "\n".join(lines)
        return ("…" + rendered) if truncated else rendered


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
        """Render one compact sub-call line: status, name, key arg, timing."""
        from .console import _prompt_text_to_ansi

        icon = {"running": "◐", "done": "✓", "error": "✗"}.get(self.status, "○")
        name = _markup_safe(self.name)
        key_arg = _markup_safe(self.key_arg)
        if self.status == "done":
            markup = f"  {icon} [tool_name]{name}[/tool_name] ({key_arg}) — {self.duration_ms:.0f}ms"
        else:
            markup = f"  {icon} [tool_name]{name}[/tool_name] ({key_arg})"
        return _prompt_text_to_ansi(markup)


class _ToolCallBlock:
    """Live + final renderer for one tool invocation and its nested sub-calls.

    Attributes:
        MAX_RESULT_LINES: How many trailing result lines to keep in the
            committed view.
        MAX_SUBAGENT_TOOL_LINES: Tail length for the sub-call list.
    """

    MAX_RESULT_LINES = 1
    MAX_SUBAGENT_TOOL_LINES = 5
    MAX_RESULT_CHARS = 40  # single-line cap for tool output; collapsed + truncated with "..."
    # Tool calls are an "actions" band — indented one step under the assistant's
    # prose so the eye groups them apart from the model's voice.
    INDENT = "  "
    # Capped-preview diff: show this many changed lines inline, then a
    # "… N more lines" marker. Keeps an edit-heavy turn scannable without
    # flooding the viewport; the full diff stays in the result payload.
    MAX_DIFF_LINES = 12

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
        # Rich display blocks attached to the tool result (currently diffs).
        # Arrive over the wire as plain dicts; may also be dataclasses locally.
        self._display_blocks: list[Any] = []

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

    def set_result(self, result: str, duration_ms: float, display_blocks: list[Any] | None = None) -> None:
        """Record the tool's return value and flip ``_status`` to done.

        ``display_blocks`` carries rich payloads (diffs) the daemon attaches to
        the result; they render below the one-line summary."""
        self._result = result
        self._duration_ms = duration_ms
        self._status = "done"
        if display_blocks:
            self._display_blocks = list(display_blocks)

    def append_sub_tool_call(self, tool_call_id: str, name: str, key_arg: str) -> _SubToolCall:
        """Register a nested sub-call and return the created :class:`_SubToolCall`.

        The retained list is bounded: ``compose`` only ever renders the last few
        sub-calls, so a subagent making thousands of calls must not grow this
        list without limit (it previously did — an O(calls) leak per spawn)."""
        sub = _SubToolCall(tool_call_id=tool_call_id, name=name, key_arg=key_arg)
        self._sub_tool_calls.append(sub)
        cap = self.MAX_SUBAGENT_TOOL_LINES * 2
        if len(self._sub_tool_calls) > cap:
            del self._sub_tool_calls[:-cap]
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

    @staticmethod
    def _block_field(block: Any, key: str) -> Any:
        """Read ``key`` from a display block that may be a wire dict or a dataclass."""
        if isinstance(block, dict):
            return block.get(key)
        return getattr(block, key, None)

    def _diff_text(self) -> str | None:
        """Return the unified-diff text from the first attached diff block, if any."""
        for block in self._display_blocks:
            if self._block_field(block, "type") == "diff":
                diff = self._block_field(block, "diff")
                if diff:
                    return str(diff)
        return None

    @staticmethod
    def _diff_counts(diff_text: str) -> tuple[int, int]:
        """Count added / removed body lines, skipping the ``+++`` / ``---`` headers."""
        adds = dels = 0
        for ln in diff_text.splitlines():
            if ln.startswith("+") and not ln.startswith("+++"):
                adds += 1
            elif ln.startswith("-") and not ln.startswith("---"):
                dels += 1
        return adds, dels

    def _compose_diff_markup(self, diff_text: str) -> list[str]:
        """Render a capped, colorized preview of ``diff_text`` as indented markup lines."""
        gutter = f"{self.INDENT}  [muted]│[/muted] "
        body: list[tuple[str, str]] = []
        for raw in diff_text.splitlines():
            if raw.startswith(("--- ", "+++ ")):
                continue  # file headers — the path is already shown on the tool line
            if raw.startswith("@@"):
                body.append(("hunk", raw))
            elif raw.startswith("+"):
                body.append(("add", raw))
            elif raw.startswith("-"):
                body.append(("del", raw))
            else:
                body.append(("ctx", raw))
        shown = body[: self.MAX_DIFF_LINES]
        hidden = len(body) - len(shown)
        role = {"add": "diff_add", "del": "diff_del", "hunk": "muted", "ctx": "dim"}
        out = [f"{gutter}[{role[kind]}]{_markup_safe(text)}[/{role[kind]}]" for kind, text in shown]
        if hidden > 0:
            out.append(f"{gutter}[muted]… {hidden} more line(s)[/muted]")
        return out

    def compose(self) -> str:
        """Render the tool call: an indented status line plus an optional diff body.

        The line lives in the "actions" band — indented one step under the
        assistant's prose so it reads as a tool invocation, not the model's
        voice::

            ✓ {name} ({key_arg}) — {summary} — {duration}

        When the result carries a diff, the summary becomes ``+N -M`` and a
        capped, colorized diff preview is rendered beneath it.
        """
        from .console import _prompt_text_to_ansi

        elapsed_s = int(time.monotonic() - self._started_at)
        duration = f"{elapsed_s}s" if self._status == "running" else f"{self._duration_ms:.0f}ms"
        name = _markup_safe(self.name)
        key_arg = _markup_safe(self.key_arg)

        # Status icon (unified set): running=◐, done=✓, error=✗
        if self._status == "running":
            icon = "◐"
            icon_color = "warn"
        elif self._status == "done":
            icon = "✓"
            icon_color = "accent"
        else:
            icon = "✗"
            icon_color = "error"

        line = f"[{icon_color}]{icon}[/{icon_color}] [tool_name]{name}[/tool_name] ([dim]{key_arg}[/dim])"

        diff_text = self._diff_text() if self._status == "done" else None
        if diff_text:
            adds, dels = self._diff_counts(diff_text)
            line += f" — [diff_add]+{adds}[/diff_add] [diff_del]-{dels}[/diff_del]"
        elif self._status == "done" and self._result:
            collapsed = _markup_safe(self._result).replace("\n", " ").strip()
            while "  " in collapsed:
                collapsed = collapsed.replace("  ", " ")
            if len(collapsed) > self.MAX_RESULT_CHARS:
                collapsed = collapsed[: self.MAX_RESULT_CHARS - 1] + "…"
            line += f" — [dim]{collapsed}[/dim]"
        elif self._status == "running":
            line += " — [dim]…[/dim]"

        line += f" — [dim]{duration}[/dim]"

        out_lines = [self.INDENT + _prompt_text_to_ansi(line)]
        if diff_text:
            out_lines.extend(_prompt_text_to_ansi(markup) for markup in self._compose_diff_markup(diff_text))
        for sub in self._sub_tool_calls[-self.MAX_SUBAGENT_TOOL_LINES :]:
            out_lines.append(self.INDENT + sub.compose())
        return "\n".join(out_lines)


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

        role = _severity_role(self.severity)
        icon = severity_icon(self.severity)
        all_lines = self.body.strip().split("\n")

        if self.category in self.PROSE_CATEGORIES and not self.body.lstrip().startswith("✨"):
            try:
                body_text = markdown_to_ansi(self.body)
                return f"{active_fg(role)}{icon}\x1b[39m {body_text}"
            except Exception:
                pass

        if self.category == "slash" or not self.title:
            body_text = "\n".join(all_lines)
            markup = f"[{role}]{icon}[/{role}] {body_text}"
        else:
            body_text = "\n".join(all_lines[: self.MAX_BODY_LINES])
            if len(all_lines) > self.MAX_BODY_LINES:
                body_text += " [dim](truncated)[/dim]"
            markup = f"[{role}]{icon} [bold]{self.title}[/bold][/{role}]\n[dim]{body_text}[/dim]"
        return _prompt_text_to_ansi(markup)


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
            "[warn bold]? Permission required[/warn bold]",
            "",
            f"  [bold]{self.action}[/bold]",
            f"  [dim]{self.description}[/dim]",
            "",
        ]

        for i, response in enumerate(self.SELECTIONS):
            marker = "▸" if i == self._selected else " "
            label = self.LABELS[response]
            role = "accent" if response == "approve" else ("tool_name" if response == "approve_for_session" else "error")
            lines.append(f"  {marker} [{role}]{label}[/{role}]")

        if self.diff and self._expanded:
            lines.append("")
            lines.append("[dim]--- diff (Ctrl+E to collapse) ---[/dim]")
            for diff_line in self.diff.split("\n"):
                if diff_line.startswith("+") and not diff_line.startswith("+++"):
                    lines.append(f"[accent]{diff_line}[/accent]")
                elif diff_line.startswith("-") and not diff_line.startswith("---"):
                    lines.append(f"[error]{diff_line}[/error]")
                else:
                    lines.append(diff_line)

        return _prompt_text_to_ansi("\n".join(lines))


class ResumeSessionPanel:
    """Interactive saved-session picker for ``/resume``."""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        """Store display records and start with the newest session highlighted."""
        self.records = records
        self._selected = 0

    @property
    def selected_session_id(self) -> str:
        """Session id for the highlighted row."""
        if not self.records:
            return ""
        record = self.records[self._selected]
        return str(record.get("session_id") or record.get("id") or "")

    def move_cursor_up(self) -> None:
        """Highlight the previous session."""
        if self.records:
            self._selected = (self._selected - 1) % len(self.records)

    def move_cursor_down(self) -> None:
        """Highlight the next session."""
        if self.records:
            self._selected = (self._selected + 1) % len(self.records)

    @staticmethod
    def _clip(value: Any, limit: int) -> str:
        """Return ``value`` collapsed to one short display line."""
        text = " ".join(str(value or "").split())
        return text[: limit - 1] + "…" if len(text) > limit else text

    @staticmethod
    def _updated_label(value: Any) -> str:
        """Compact an ISO timestamp for session-list display."""
        text = str(value or "")
        if "T" not in text:
            return text
        date, rest = text.split("T", 1)
        return f"{date} {rest[:5]}"

    def compose(self) -> str:
        """Render the picker panel with one selectable row per session."""
        from .console import _prompt_text_to_ansi

        lines = [
            "[bold accent]? Resume Session[/bold accent]",
            "",
            "[dim]Use Up/Down then Enter. Type a number, id, or title to resume. Type /cancel to close.[/dim]",
            "",
        ]
        if not self.records:
            lines.append("  [dim]No saved sessions found.[/dim]")
            return _prompt_text_to_ansi("\n".join(lines))

        for i, record in enumerate(self.records):
            selected = i == self._selected
            marker = "▸" if selected else " "
            role = "accent" if selected else "system"
            sid = str(record.get("session_id") or record.get("id") or "?")
            title = self._clip(record.get("title") or sid, 68)
            updated = self._updated_label(record.get("updated_at"))
            turns = int(record.get("turn_count", 0) or 0)
            messages = int(record.get("messages", 0) or 0)
            lines.append(f"  {marker} [{role}]{i + 1}. {title}[/{role}]")
            lines.append(f"      [dim]{sid} · {turns} turn(s), {messages} message(s), updated {updated}[/dim]")

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
            f"[bold accent]? Question ({self._question_index + 1}/{len(self.questions)})[/bold accent]",
            "",
            f"  [bold]{self.current_question.get('question', '')}[/bold]",
            "",
        ]

        for i, option in enumerate(self.current_options):
            marker = "▸" if i == self._option_index else " "
            lines.append(f"  {i + 1}. {marker} {option}")

        if self.current_question.get("allow_free_form"):
            other_marker = "▸" if self._free_text_mode else " "
            lines.append(f"     {other_marker} [dim]Other (type your answer)[/dim]")
            if self._free_text_mode:
                prompt_glyph = get_active_skin().label("prompt_symbol")
                lines.append(f"       [accent]{prompt_glyph} {self._free_text}_[/accent]")

        lines.append("")
        lines.append("[dim]Type a number to pick, free text for custom, or /cancel to abort.[/dim]")

        return _prompt_text_to_ansi("\n".join(lines))
