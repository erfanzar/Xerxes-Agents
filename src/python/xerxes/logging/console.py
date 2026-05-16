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
"""Colorized console logger for the Xerxes runtime.

Defines the ANSI ``COLORS`` palette, a level-coloring :class:`ColorFormatter`,
the process-wide :class:`XerxesLogger` singleton, and a family of
``log_step`` / ``log_success`` / ``log_error`` etc. shortcuts that render
emoji + ANSI when stdout is a TTY and degrade gracefully otherwise.

Also exposes :func:`stream_callback`, the legacy streaming pretty-printer
used by the non-TUI demo loop to render :mod:`xerxes.types` events
(``StreamChunk``, ``FunctionCallsExtracted``, ``Completion``, ...)."""

import datetime
import json
import logging
import os
import re
import shutil
import sys
import textwrap
import threading
import time
from pprint import pformat

from ..types import (
    AgentSwitch,
    Completion,
    FunctionCallsExtracted,
    FunctionDetection,
    FunctionExecutionComplete,
    FunctionExecutionStart,
    ReinvokeSignal,
    StreamChunk,
)

COLORS = {
    "BLACK": "\033[30m",
    "RED": "\033[31m",
    "GREEN": "\033[32m",
    "YELLOW": "\033[33m",
    "BLUE": "\033[34m",
    "MAGENTA": "\033[35m",
    "CYAN": "\033[36m",
    "WHITE": "\033[37m",
    "LIGHT_BLACK": "\033[90m",
    "LIGHT_RED": "\033[91m",
    "LIGHT_GREEN": "\033[92m",
    "LIGHT_YELLOW": "\033[93m",
    "LIGHT_BLUE": "\033[94m",
    "LIGHT_MAGENTA": "\033[95m",
    "LIGHT_CYAN": "\033[96m",
    "LIGHT_WHITE": "\033[97m",
    "RESET": "\033[0m",
    "BOLD": "\033[1m",
    "DIM": "\033[2m",
    "ITALIC": "\033[3m",
    "UNDERLINE": "\033[4m",
    "BLUE_PURPLE": "\033[38;5;99m",
}

LEVEL_COLORS = {
    "DEBUG": COLORS["LIGHT_BLUE"],
    "INFO": COLORS["BLUE_PURPLE"],
    "WARNING": COLORS["YELLOW"],
    "ERROR": COLORS["LIGHT_RED"],
    "CRITICAL": COLORS["RED"] + COLORS["BOLD"],
}


class ColorFormatter(logging.Formatter):
    """``logging.Formatter`` that colorizes the level name and prepends a timestamp tag.

    Each line of a multi-line message is decorated with ``(HH:MM:SS name)``
    so streamed output stays attributable in mixed-source logs."""

    def format(self, record: logging.LogRecord) -> str:
        """Return ``record`` formatted with ANSI color escapes per level."""

        orig_levelname = record.levelname
        color = LEVEL_COLORS.get(record.levelname, COLORS["RESET"])
        record.levelname = f"{color}{record.levelname:<8}{COLORS['RESET']}"
        current_time = datetime.datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        formatted_name = f"{color}({current_time} {record.name}){COLORS['RESET']}"
        message = record.getMessage()
        lines = message.split("\n")
        formatted_lines = [f"{formatted_name} {line}" if line else formatted_name for line in lines]
        result = "\n".join(formatted_lines)

        record.levelname = orig_levelname
        return result


class XerxesLogger:
    """Thread-safe singleton wrapper around the ``Xerxes`` :mod:`logging` logger.

    Holds a single console handler with a :class:`ColorFormatter`. The
    effective level is sourced from ``$XERXES_LOG_LEVEL`` (default ``INFO``)
    and can be adjusted at runtime via :meth:`set_level`."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Return the singleton instance, constructing it on first access."""

        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Set up the underlying logger on first construction only."""

        if not hasattr(self, "_initialized"):
            self._initialized = True
            self._setup_logger()

    def _setup_logger(self):
        """Attach the ANSI-colored stdout handler at the env-configured level."""

        self.logger = logging.getLogger("Xerxes")
        self.logger.setLevel(logging.DEBUG)

        self.logger.handlers = []

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self._get_log_level())

        console_handler.setFormatter(ColorFormatter())
        self.logger.addHandler(console_handler)

    def _get_log_level(self) -> int:
        """Resolve the effective log level from ``$XERXES_LOG_LEVEL``."""

        level_str = os.environ.get("XERXES_LOG_LEVEL", "INFO").upper()
        return getattr(logging, level_str, logging.INFO)

    def debug(self, message: str, *args, **kwargs):
        """Forward ``message`` and ``args``/``kwargs`` to the underlying logger at DEBUG."""

        self.logger.debug(message, *args, **kwargs)

    def info(self, message: str, *args, **kwargs):
        """Forward ``message`` and ``args``/``kwargs`` to the underlying logger at INFO."""

        self.logger.info(message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs):
        """Forward ``message`` and ``args``/``kwargs`` to the underlying logger at WARNING."""

        self.logger.warning(message, *args, **kwargs)

    def error(self, message: str, *args, **kwargs):
        """Forward ``message`` and ``args``/``kwargs`` to the underlying logger at ERROR."""

        self.logger.error(message, *args, **kwargs)

    def critical(self, message: str, *args, **kwargs):
        """Forward ``message`` and ``args``/``kwargs`` to the underlying logger at CRITICAL."""

        self.logger.critical(message, *args, **kwargs)

    def set_level(self, level: str):
        """Update the logger and every attached handler to ``level``.

        Args:
            level: Case-insensitive level name (``"DEBUG"``, ``"INFO"``, ...).
                Unknown values fall back to ``INFO``.
        """

        numeric_level = getattr(logging, level.upper(), logging.INFO)
        self.logger.setLevel(numeric_level)
        for handler in self.logger.handlers:
            handler.setLevel(numeric_level)


def get_logger() -> XerxesLogger:
    """Return the process-wide :class:`XerxesLogger` singleton."""

    return XerxesLogger()


def set_verbosity(level: str):
    """Shortcut to update the singleton logger's level (see :meth:`XerxesLogger.set_level`)."""

    logger = get_logger()
    logger.set_level(level)


def log_step(step_name: str, description: str = "", color: str = "CYAN"):
    """Log a labelled pipeline step at INFO.

    Args:
        step_name: Short tag bracketed in the output (e.g. ``"plan"``).
        description: Optional follow-up sentence appended after the tag.
        color: ANSI color key from :data:`COLORS`; falls back to cyan
            when the name is unknown.
    """

    logger = get_logger()
    color_code = COLORS.get(color.upper(), COLORS["CYAN"])
    reset = COLORS["RESET"]

    if sys.stdout.isatty():
        message = f"{color_code}[{step_name}]{reset}"
        if description:
            message += f" {description}"
    else:
        message = f"[{step_name}]"
        if description:
            message += f" {description}"

    logger.info(message)


def log_thinking(agent_name: str):
    """Log that ``agent_name`` is generating, with brain emoji on TTYs."""

    logger = get_logger()
    if sys.stdout.isatty():
        logger.info(
            f"{COLORS['BLUE']}  (🧠 {agent_name}){COLORS['RESET']}{COLORS['BLUE_PURPLE']} is thinking...{COLORS['RESET']}"
        )
    else:
        logger.info(f"  (🧠 {agent_name}) is thinking...")


def log_success(message: str):
    """Log a success ``message`` at INFO with a rocket prefix."""

    logger = get_logger()
    if sys.stdout.isatty():
        logger.info(f"{COLORS['BLUE']}🚀 {message}{COLORS['RESET']}")
    else:
        logger.info(f"🚀 {message}")


def log_error(message: str):
    """Log an error ``message`` at ERROR with a cross-mark prefix."""

    logger = get_logger()
    if sys.stdout.isatty():
        logger.error(f"{COLORS['LIGHT_RED']}❌ {message}{COLORS['RESET']}")
    else:
        logger.error(f"❌ {message}")


def log_warning(message: str):
    """Log a warning ``message`` at WARNING with a triangle prefix."""

    logger = get_logger()
    if sys.stdout.isatty():
        logger.warning(f"{COLORS['YELLOW']}⚠️ {message}{COLORS['RESET']}")
    else:
        logger.warning(f"⚠️ {message}")


def log_retry(attempt: int, max_attempts: int, error: str):
    """Log a retry attempt at WARNING with the offending error message.

    Args:
        attempt: 1-based attempt counter.
        max_attempts: Hard cap so the user can see ``attempt/max_attempts``.
        error: Stringified exception associated with the failure.
    """

    logger = get_logger()
    if sys.stdout.isatty():
        message = f"{COLORS['YELLOW']}⏳ Retry {attempt}/{max_attempts}: {COLORS['LIGHT_RED']}{error}{COLORS['RESET']}"
    else:
        message = f"⏳ Retry {attempt}/{max_attempts}: {error}"
    logger.warning(message)


def log_delegation(from_agent: str, to_agent: str):
    """Log an agent-to-agent handoff at INFO with an arrow glyph."""

    logger = get_logger()
    if sys.stdout.isatty():
        message = (
            f"{COLORS['MAGENTA']}📌 Delegation: "
            f"{COLORS['CYAN']}{from_agent}{COLORS['RESET']} → "
            f"{COLORS['CYAN']}{to_agent}{COLORS['RESET']}"
        )
    else:
        message = f"📌 Delegation: {from_agent} → {to_agent}"
    logger.info(message)


def log_agent_start(agent: str | None = None):
    """Log that ``agent`` has been started (omit the name for an anonymous start)."""

    logger = get_logger()
    if sys.stdout.isatty():
        message = f" {COLORS['BLUE_PURPLE']}{agent} Agent is started.{COLORS['RESET']}"
    else:
        message = f" {agent} Agent is started."
    logger.info(message)


def log_task_start(task_name: str, agent: str | None = None):
    """Log a task-start marker, optionally tagging it with the owning agent."""

    logger = get_logger()
    if sys.stdout.isatty():
        message = f"{COLORS['BLUE']} Task Started: {COLORS['BOLD']}{task_name}{COLORS['RESET']}"
        if agent:
            message += f" {COLORS['DIM']}(Agent: {agent}){COLORS['RESET']}"
    else:
        message = f" Task Started: {task_name}"
        if agent:
            message += f" (Agent: {agent})"
    logger.info(message)


def log_task_complete(task_name: str, duration: float | None = None):
    """Log a task-complete marker; ``duration`` (seconds) renders as ``(x.xxs)``."""

    logger = get_logger()
    if sys.stdout.isatty():
        message = f"{COLORS['GREEN']}🚀 Task Completed: {task_name}{COLORS['RESET']}"
        if duration:
            message += f" {COLORS['DIM']}({duration:.2f}s){COLORS['RESET']}"
    else:
        message = f"🚀 Task Completed: {task_name}"
        if duration:
            message += f" ({duration:.2f}s)"
    logger.info(message)


logger = get_logger()


def stream_callback(chunk):
    """Pretty-print one streaming event from the cortex/agent loop.

    Dispatches on the ``chunk`` subclass (``StreamChunk``, ``FunctionDetection``,
    ``FunctionCallsExtracted``, ``FunctionExecutionStart``/``Complete``,
    ``AgentSwitch``, ``ReinvokeSignal``, ``Completion``) to render the
    appropriate banner / spinner / result block to stdout. Keeps per-call
    state on the function object so repeated invocations stitch the
    output together coherently."""

    COL = COLORS
    ACCENT = COL["BLUE_PURPLE"]
    BOLD = COL["BOLD"]
    DIM = COL["DIM"]
    ITALIC = COL["ITALIC"]
    RESET = COL["RESET"]
    LWHITE = COL["LIGHT_WHITE"]
    LGREEN = COL["LIGHT_GREEN"]
    LRED = COL["LIGHT_RED"]

    if not hasattr(stream_callback, "_state"):
        stream_callback._state = {
            "open_line": False,
            "tool_headers_printed": set(),
            "tool_indents": {},
            "exec_start_times": {},
        }

    state = stream_callback._state

    ANSI_RE = re.compile(r"\x1b```math[0-9;]*m")

    def strip_ansi(s: str) -> str:
        """Drop ANSI escape sequences so ``s`` can be measured for layout."""

        return ANSI_RE.sub("", s)

    def term_width() -> int:
        """Return current terminal column count clamped to a sensible range."""

        try:
            return max(60, shutil.get_terminal_size(fallback=(100, 24)).columns)
        except Exception:
            return 100

    def paint(text: object, *styles: str) -> str:
        """Wrap ``text`` in concatenated ANSI ``styles`` plus a final reset."""

        return "".join(styles) + str(text) + RESET

    def tag(agent_id: str) -> str:
        """Render ``[agent_id]`` as a bold accent-colored bracket prefix."""

        return f"{BOLD}{ACCENT}[{agent_id}]{RESET}"

    def bullet() -> str:
        """Return the accent-styled ``•`` glyph used in list items."""

        return paint("•", ACCENT, BOLD)

    def ensure_newline() -> None:
        """Emit a newline if the previous ``write`` left the line open."""

        if state["open_line"]:
            print("", flush=True)
            state["open_line"] = False

    def write(s: str) -> None:
        """Print ``s`` without a newline and remember the line is still open."""

        print(s, end="", flush=True)
        state["open_line"] = True

    def writeln(s: str) -> None:
        """Print ``s`` followed by a newline, closing any in-progress line first."""

        ensure_newline()
        print(s, flush=True)
        state["open_line"] = False

    def indent_newlines(s: str, indent: str) -> str:
        """Re-indent every newline in ``s`` with ``indent`` for hanging blocks."""

        if not s or "\n" not in s:
            return s
        return s.replace("\n", "\n" + indent)

    def preview(text: object, max_len: int = 100) -> str:
        """Return ``text`` capped at ``max_len`` chars with an ellipsis if cut."""

        s = str(text)
        return (s[:max_len] + "…") if len(s) > max_len else s

    def pretty_result(value: object) -> str:
        """Return a 2-space JSON dump of ``value`` with pprint as fallback."""

        try:
            if isinstance(value, dict | list):
                return json.dumps(value, indent=2, ensure_ascii=False)
            return json.dumps(value, indent=2, ensure_ascii=False, default=str)
        except Exception:
            return pformat(value, width=max(60, term_width() - 8), compact=False)

    def hr(title: str | None = None) -> None:
        """Print a horizontal rule, optionally centered around ``title``."""

        width = term_width()
        if not title:
            print(paint("─" * width, ACCENT), flush=True)
            return
        title_str = f" {title} "
        left = (width - len(title_str)) // 2
        right = max(0, width - left - len(title_str))
        print(paint("─" * left, ACCENT) + paint(title_str, LWHITE, BOLD) + paint("─" * right, ACCENT), flush=True)

    if isinstance(chunk, StreamChunk):
        if getattr(chunk, "reasoning_content", None):
            write(paint(chunk.reasoning_content, COL["MAGENTA"]))
            if chunk.reasoning_content.endswith("\n"):
                state["open_line"] = False

        if getattr(chunk, "content", None):
            if hasattr(chunk, "is_thinking") and chunk.is_thinking:
                write(paint(chunk.content, COL["MAGENTA"]))
            else:
                write(chunk.content)
            if chunk.content.endswith("\n"):
                state["open_line"] = False

        if getattr(chunk, "streaming_tool_calls", None):
            for tool_call in chunk.streaming_tool_calls:
                tool_id = (
                    getattr(tool_call, "id", None) or f"{getattr(chunk, 'agent_id', '')}:{tool_call.function_name or ''}"
                )

                if tool_call.function_name is not None and tool_id not in state["tool_headers_printed"]:
                    ensure_newline()
                    line = (
                        f"{paint('🛠️', ACCENT)}  {tag(chunk.agent_id)} "
                        f"{paint('Calling', ACCENT, BOLD)} "
                        f"{paint(tool_call.function_name, LWHITE, BOLD)} : "
                    )
                    write(line)

                    visible_len = len(strip_ansi(line))
                    state["tool_indents"][tool_id] = " " * visible_len
                    state["tool_headers_printed"].add(tool_id)

                if tool_call.arguments is not None:
                    if tool_id not in state.get("tool_args_buf", {}):
                        state.setdefault("tool_args_buf", {})[tool_id] = ""
                    state["tool_args_buf"][tool_id] += tool_call.arguments

    elif isinstance(chunk, FunctionDetection):
        ensure_newline()
        writeln(f"{paint('🔍', ACCENT)} {tag(chunk.agent_id)} {paint(chunk.message, LWHITE)}")

    elif isinstance(chunk, FunctionCallsExtracted):
        ensure_newline()
        tool_args_buf = state.get("tool_args_buf", {})
        if tool_args_buf:
            for _tid, raw_args in tool_args_buf.items():
                indent = state["tool_indents"].get(_tid, "")
                try:
                    formatted = json.dumps(json.loads(raw_args), indent=2)
                except Exception:
                    formatted = raw_args
                for i, arg_line in enumerate(formatted.splitlines()):
                    if i == 0:
                        write(paint(arg_line, LWHITE))
                    else:
                        write("\n" + indent + paint(arg_line, LWHITE))
                write("\n")
            state["tool_args_buf"] = {}
            state["open_line"] = False
        writeln(
            f"{paint('📋', ACCENT)} {tag(chunk.agent_id)} "
            f"{paint(f'Found {len(chunk.function_calls)} function(s) to execute:', ACCENT, BOLD)}"
        )
        for fc in chunk.function_calls:
            writeln(f"   {bullet()} {paint(fc.name, LWHITE, BOLD)} {paint(f'(id: {fc.id})', DIM)}")

    elif isinstance(chunk, FunctionExecutionStart):
        ensure_newline()
        state["tool_headers_printed"].clear()
        state["tool_indents"].clear()

        key = (getattr(chunk, "agent_id", ""), getattr(chunk, "function_name", ""))
        state["exec_start_times"][key] = time.perf_counter()

        progress = f" {paint(chunk.progress, ACCENT)}" if getattr(chunk, "progress", None) else ""
        writeln(
            f"{paint('⚡', ACCENT)} {tag(chunk.agent_id)} "
            f"{paint('Executing', ACCENT, BOLD)} {paint(chunk.function_name, LWHITE, BOLD)}{progress}..."
        )

    elif isinstance(chunk, FunctionExecutionComplete):
        ensure_newline()

        key = (getattr(chunk, "agent_id", ""), getattr(chunk, "function_name", ""))
        started = state["exec_start_times"].pop(key, None)
        dur = f" in {time.perf_counter() - started:.2f}s" if started else ""

        status_icon = paint("✅", LGREEN) if chunk.status == "success" else paint("❌", LRED)
        writeln(
            f"{status_icon} {tag(chunk.agent_id)} "
            f"{paint(chunk.function_name, LWHITE, BOLD)} {paint('completed', ACCENT)}{paint(dur, DIM)}"
        )

        if getattr(chunk, "error", None):
            writeln(f"   {paint('⚠️ Error:', LRED, BOLD)} {paint(chunk.error, LWHITE)}")
        elif getattr(chunk, "result", None):
            formatted = pretty_result(chunk.result)
            if "\n" in formatted or len(formatted) > 200:
                writeln(paint("   ⋮ Result", ACCENT, BOLD))
                print(paint(textwrap.indent(formatted, prefix="   "), LWHITE), flush=True)
            else:
                writeln(f"   {paint('→ Result:', ACCENT, BOLD)} {paint(preview(formatted, 100), LWHITE)}")

    elif isinstance(chunk, AgentSwitch):
        ensure_newline()
        writeln(
            f"{paint('🔄', ACCENT)} "
            f"{paint('Switching', ACCENT, BOLD)} {paint('from', DIM)} "
            f"{paint(f'[{chunk.from_agent}]', ACCENT, BOLD)} {paint('to', DIM)} "
            f"{paint(f'[{chunk.to_agent}]', ACCENT, BOLD)}"
        )
        if getattr(chunk, "reason", None):
            writeln(f"   {paint('Reason:', ACCENT, BOLD)} {paint(chunk.reason, ITALIC, LWHITE)}")

    elif isinstance(chunk, ReinvokeSignal):
        ensure_newline()
        writeln(f"{paint('🔁', ACCENT)} {tag(chunk.agent_id)} {paint(chunk.message, LWHITE)}")

    elif isinstance(chunk, Completion):
        ensure_newline()
        if getattr(chunk, "agent_id", "") == "cortex":
            hr("Pipeline completed")
            writeln(
                f"   {bullet()} {paint('Functions executed:', ACCENT, BOLD)} "
                f"{paint(chunk.function_calls_executed, LWHITE)}"
            )
            if hasattr(chunk, "execution_history") and chunk.execution_history:
                writeln(
                    f"   {bullet()} {paint('Execution steps:', ACCENT, BOLD)} "
                    f"{paint(len(chunk.execution_history), LWHITE)}"
                )
            hr()
        else:
            writeln(f"{paint('✓', LGREEN)} {tag(chunk.agent_id)} {paint('Task completed', ACCENT, BOLD)}")
            writeln(
                f"   {bullet()} {paint('Functions called:', ACCENT, BOLD)} "
                f"{paint(chunk.function_calls_executed, LWHITE)}"
            )
            if getattr(chunk, "final_content", None):
                preview_text = (
                    chunk.final_content[:100] + "..." if len(chunk.final_content) > 100 else chunk.final_content
                )
                writeln(f"   {bullet()} {paint('Output preview:', ACCENT, BOLD)} {paint(preview_text, LWHITE)}")
