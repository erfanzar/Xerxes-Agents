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
"""Console rendering utilities for the Xerxes TUI.

This module provides helpers for converting markup, Markdown, and syntax
highlighted code into ANSI-escaped strings suitable for terminal display.
It also defines severity colors, icons, and a simple spinner state dataclass.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from typing import IO

from rich.console import Console as RichConsole
from rich.markdown import Markdown as RichMarkdown
from rich.syntax import Syntax as RichSyntax
from rich.theme import Theme as RichTheme


def _ansi_color_map() -> dict[str, str]:
    """Return the name-to-ANSI-256 mapping used by the bracket-tag parser."""
    return {
        "black": "16",
        "red": "196",
        "green": "46",
        "yellow": "226",
        "blue": "21",
        "magenta": "201",
        "cyan": "51",
        "white": "231",
        "brightblack": "242",
        "brightred": "9",
        "brightgreen": "10",
        "brightyellow": "11",
        "brightblue": "12",
        "brightmagenta": "13",
        "brightcyan": "14",
        "brightwhite": "15",
    }


_ANSI_FG_RESET = "\x1b[39m"
_ANSI_BG_RESET = "\x1b[49m"
_ANSI_STYLE_MAP = {
    "bold": "\x1b[1m",
    "/bold": "\x1b[22m",
    "italic": "\x1b[3m",
    "/italic": "\x1b[23m",
    "underline": "\x1b[4m",
    "/underline": "\x1b[24m",
    "strike": "\x1b[9m",
    "/strike": "\x1b[29m",
    "dim": "\x1b[2m",
    "/dim": "\x1b[22m",
    "reverse": "\x1b[7m",
    "/reverse": "\x1b[27m",
}


def _css_to_ansi(color: str) -> str:
    """Convert a CSS-style color to an ANSI 256-color index string.

    Supports ``#RRGGBB``, ``rgb(r, g, b)``, and the named colors in
    :func:`_ansi_color_map`. Returns ``""`` when ``color`` cannot be
    parsed."""
    color = color.strip().lower()
    if not color:
        return ""
    ansi_map = _ansi_color_map()

    hex_m = re.match(r"#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$", color)
    if hex_m:
        r, g, b = (int(h, 16) for h in hex_m.groups())
        idx = 16 + 36 * (r // 43) + 6 * (g // 43) + (b // 43)
        return str(idx)

    rgb_m = re.match(r"rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)$", color)
    if rgb_m:
        r, g, b = (int(x) for x in rgb_m.groups())
        idx = 16 + 36 * (r // 43) + 6 * (g // 43) + (b // 43)
        return str(idx)

    if color in ansi_map:
        return ansi_map[color]

    return ""


def _resolve_tag(tag: str) -> str | None:
    """Resolve a single rich-style markup tag into ANSI escape sequences.

    Accepts space-separated tags (e.g. ``"bold red"``) and ``"/tag"``
    closers. Returns ``None`` when ``tag`` contains no recognizable
    keyword so the caller can pass the literal ``"[tag]"`` through."""
    parts = [p for p in tag.split() if p]
    if not parts:
        return None
    out = []
    color_map = _ansi_color_map()
    is_closing = parts[0].startswith("/")
    if is_closing:
        for p in parts:
            stem = p[1:] if p.startswith("/") else p
            if stem in _ANSI_STYLE_MAP:
                close_key = "/" + stem
                out.append(_ANSI_STYLE_MAP.get(close_key, "\x1b[22m"))
            elif stem in color_map:
                out.append("\x1b[39m")
        return "".join(out) if out else None

    for p in parts:
        if p in _ANSI_STYLE_MAP:
            out.append(_ANSI_STYLE_MAP[p])
        elif p in color_map:
            idx = _css_to_ansi(p)
            if idx:
                out.append(f"\x1b[38;5;{idx}m")
    return "".join(out) if out else None


def _prompt_text_to_ansi(markup: str) -> str:
    """Convert bracket-style markup (``[bold]...[/bold]``) into ANSI escapes.

    Unknown tags are passed through verbatim — callers can therefore
    mix literal ``[..]`` text with markup without escaping."""
    output = ""
    i = 0
    while i < len(markup):
        open_idx = markup.find("[", i)
        if open_idx == -1:
            output += markup[i:]
            break

        output += markup[i:open_idx]
        i = open_idx

        close_idx = markup.find("]", i)
        if close_idx == -1:
            output += markup[i:]
            break

        tag = markup[i + 1 : close_idx]
        i = close_idx + 1

        ansi = _resolve_tag(tag)
        if ansi is not None:
            output += ansi
        else:
            output += f"[{tag}]"

    return output


def render_to_ansi(text: str, columns: int) -> str:
    """Public wrapper around :func:`_prompt_text_to_ansi`.

    ``columns`` is reserved for a future wrapping pass and currently ignored."""
    return _prompt_text_to_ansi(text)


def text_to_ansi_escaped(text: str) -> str:
    """Public alias for :func:`_prompt_text_to_ansi`."""
    return _prompt_text_to_ansi(text)


_rich_console: RichConsole | None = None


def _get_rich_console() -> RichConsole:
    """Return a lazily-initialized process-wide :class:`RichConsole`.

    Configured with a Markdown-tuned theme and the live terminal width.
    Cached so theme construction happens once per process."""
    global _rich_console
    if _rich_console is None:
        _rich_console = RichConsole(
            theme=RichTheme(
                {
                    "markdown.code": "dim",
                    "markdown.code_block": "dim",
                    "markdown.fence": "dim",
                    "markdown.link": "cyan underline",
                    "markdown.h1": "bold",
                    "markdown.h2": "bold",
                    "markdown.h3": "bold",
                    "markdown.h4": "bold",
                    "markdown.h5": "bold",
                    "markdown.h6": "bold",
                    "markdown.quote": "italic dim",
                    "markdown.list.bullet": "cyan",
                    "markdown.list.number": "cyan",
                    "markdown.em": "italic",
                    "markdown.strong": "bold",
                }
            ),
            force_terminal=True,
            width=shutil.get_terminal_size().columns,
        )
    return _rich_console


def markdown_to_ansi(text: str, *, columns: int | None = None) -> str:
    """Render Markdown ``text`` to ANSI-escaped plain text via a one-shot Rich console.

    The default width is the live terminal column count — rendering at
    a fixed wide canvas (e.g. 1000) causes Rich to pad code-block
    backgrounds and rule lines far past the visible area, which then
    wrap and stack as vertical glitches.

    Args:
        text: Markdown source.
        columns: Override the wrap width; falls back to the live
            terminal columns (min 40) when ``None`` or non-positive.
    """
    import io

    if columns and columns > 0:
        width = columns
    else:
        width = max(40, shutil.get_terminal_size((100, 24)).columns)
    buf = io.StringIO()
    console = RichConsole(
        file=buf,
        force_terminal=True,
        color_system="256",
        width=width,
        record=False,
        soft_wrap=True,
    )
    console.print(RichMarkdown(text, code_theme="monokai"))

    lines = [line.rstrip(" ") for line in buf.getvalue().split("\n")]
    collapsed: list[str] = []
    blank_run = 0
    for line in lines:
        if line == "":
            blank_run += 1
            if blank_run <= 1:
                collapsed.append(line)
        else:
            blank_run = 0
            collapsed.append(line)
    return "\n".join(collapsed).rstrip("\n")


def print_markdown(text: str, *, file: IO[str] | None = None) -> None:
    """Print Markdown ``text`` via the shared Rich console (or ``file`` if given)."""
    console = _get_rich_console()
    md = RichMarkdown(
        text,
        code_theme="monokai",
        verify_code_blocks=False,
    )
    if file:
        console.print(md, file=file)
    else:
        console.print(md)


def print_syntax(code: str, language: str = "", label: str = "") -> None:
    """Print syntax-highlighted ``code`` inside a box border.

    Args:
        code: Source code to display.
        language: Lexer name passed to Rich Syntax; falls back to plain text.
        label: Optional header label drawn into the top border.
    """
    console = _get_rich_console()
    width = shutil.get_terminal_size().columns

    if label or language:
        header = f" {label or language} "
    else:
        header = ""

    border_char = "─"
    border = border_char * max(0, width - 2)

    syntax = RichSyntax(
        code,
        lexer=language or "text",
        theme="monokai",
        line_numbers=False,
        word_wrap=True,
    )

    if header:
        console.print(f"┌{border}┐")
        console.print(f"│{header:<{width - 2}}│")
        console.print(f"├{border}┤")
        console.print(syntax)
        console.print(f"└{border}┘")
    else:
        console.print(f"┌{border}┐")
        console.print(syntax)
        console.print(f"└{border}┘")


@dataclass
class SpinnerState:
    """Tracks spinner animation frame and associated text.

    Attributes:
        frame (int): Current frame index.
        frames (tuple[str, ...]): Sequence of spinner characters.
        text (str): Optional text displayed alongside the spinner.
    """

    frame: int = 0
    frames: tuple[str, ...] = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
    text: str = ""

    def tick(self) -> str:
        """Advance one frame and return the character that should be drawn now."""
        result = self.frames[self.frame % len(self.frames)]
        self.frame += 1
        return result

    def current(self) -> str:
        """Return the current spinner character without advancing."""
        return self.frames[self.frame % len(self.frames)]


SEVERITY_COLORS: dict[str, str] = {
    "info": "cyan",
    "success": "green",
    "warning": "yellow",
    "error": "red",
    "debug": "dim",
}


def severity_color(severity: str) -> str:
    """Return the color name for ``severity`` (defaults to ``"white"``)."""
    return SEVERITY_COLORS.get(severity.lower(), "white")


def severity_icon(severity: str) -> str:
    """Return the icon glyph for ``severity`` (defaults to a middle dot)."""
    icons = {
        "info": "ℹ",  # noqa: RUF001
        "success": "✓",
        "warning": "⚠",
        "error": "✗",
        "debug": "⚙",
    }
    return icons.get(severity.lower(), "·")
