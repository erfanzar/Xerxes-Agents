# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0


# distributed under the License is distributed on an "AS IS" BASIS,

# See the License for the specific language governing permissions and
# limitations under the License.


"""Lightweight logging system for Xerxes using ANSI colors.

This module provides a centralized, color-coded logging system for the
Xerxes framework. It includes:
- ANSI color support for terminal output
- Thread-safe singleton logger implementation
- Colored log level formatting
- Utility functions for common logging patterns
- Streaming callback for real-time event display

The logging system automatically detects TTY terminals and applies
ANSI colors accordingly, falling back to plain text in non-TTY contexts.

Example:
    >>> from xerxes.logging.console import get_logger, log_step
    >>> logger = get_logger()
    >>> logger.info("Starting process")
    >>> log_step("INIT", "Initializing components", color="GREEN")
"""

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
    """Custom log formatter that adds ANSI color codes to log output.

    This formatter colorizes log messages based on their severity level
    and adds timestamps and logger names with appropriate styling.

    The formatter handles multi-line messages by prepending the formatted
    name to each line, ensuring consistent visual alignment.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record with ANSI color codes.

        Args:
            record: The log record to format.

        Returns:
            Formatted string with ANSI color codes applied based on log level.
        """
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
    """Centralized logger for Xerxes with colored output support.

    Thread-safe singleton logger that provides colored console output
    with configurable log levels. Uses the singleton pattern to ensure
    a single logger instance across the application.

    Attributes:
        logger: The underlying Python logging.Logger instance.

    Note:
        Log level can be configured via the XERXES_LOG_LEVEL environment
        variable. Defaults to INFO if not set.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Create or return the singleton logger instance.

        Returns:
            The singleton XerxesLogger instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the logger if not already initialized."""
        if not hasattr(self, "_initialized"):
            self._initialized = True
            self._setup_logger()

    def _setup_logger(self):
        """Set up the main logger with colored console handler.

        Configures the internal logger with a ColorFormatter and sets
        the log level from environment variables.
        """
        self.logger = logging.getLogger("Xerxes")
        self.logger.setLevel(logging.DEBUG)

        self.logger.handlers = []

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self._get_log_level())

        console_handler.setFormatter(ColorFormatter())
        self.logger.addHandler(console_handler)

    def _get_log_level(self) -> int:
        """Get log level from XERXES_LOG_LEVEL environment variable.

        Returns:
            Integer log level corresponding to the environment variable
            value, defaulting to INFO if not set or invalid.
        """
        level_str = os.environ.get("XERXES_LOG_LEVEL", "INFO").upper()
        return getattr(logging, level_str, logging.INFO)

    def debug(self, message: str, *args, **kwargs):
        """Log a message at DEBUG level.

        Args:
            message: The message to log.
            *args: Positional arguments for string formatting.
            **kwargs: Keyword arguments passed to the logger.
        """
        self.logger.debug(message, *args, **kwargs)

    def info(self, message: str, *args, **kwargs):
        """Log a message at INFO level.

        Args:
            message: The message to log.
            *args: Positional arguments for string formatting.
            **kwargs: Keyword arguments passed to the logger.
        """
        self.logger.info(message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs):
        """Log a message at WARNING level.

        Args:
            message: The message to log.
            *args: Positional arguments for string formatting.
            **kwargs: Keyword arguments passed to the logger.
        """
        self.logger.warning(message, *args, **kwargs)

    def error(self, message: str, *args, **kwargs):
        """Log a message at ERROR level.

        Args:
            message: The message to log.
            *args: Positional arguments for string formatting.
            **kwargs: Keyword arguments passed to the logger.
        """
        self.logger.error(message, *args, **kwargs)

    def critical(self, message: str, *args, **kwargs):
        """Log a message at CRITICAL level.

        Args:
            message: The message to log.
            *args: Positional arguments for string formatting.
            **kwargs: Keyword arguments passed to the logger.
        """
        self.logger.critical(message, *args, **kwargs)

    def set_level(self, level: str):
        """Set the logging level for the logger and all handlers.

        Args:
            level: Log level name ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
        """
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        self.logger.setLevel(numeric_level)
        for handler in self.logger.handlers:
            handler.setLevel(numeric_level)


def get_logger() -> XerxesLogger:
    """Get the singleton XerxesLogger instance.

    Returns:
        The global XerxesLogger singleton instance.

    Example:
        >>> logger = get_logger()
        >>> logger.info("Application started")
    """
    return XerxesLogger()


def set_verbosity(level: str):
    """Set the global verbosity level for the Xerxes logger.

    Args:
        level: Log level name ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').

    Example:
        >>> set_verbosity('DEBUG')
        >>> set_verbosity('ERROR')
    """
    logger = get_logger()
    logger.set_level(level)


def log_step(step_name: str, description: str = "", color: str = "CYAN"):
    """Log a step in the process with formatted, colored output.

    Displays a step name in brackets with optional description.
    Colors are applied only when output is to a TTY terminal.

    Args:
        step_name: Name of the step (displayed in brackets).
        description: Optional description following the step name.
        color: Color name from COLORS dict (default: "CYAN").

    Example:
        >>> log_step("INIT", "Loading configuration")
        >>> log_step("BUILD", "Compiling assets", color="GREEN")
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
    """Log an agent's thinking state with colored brain emoji.

    Displays a visual indicator that an agent is processing or thinking,
    with blue/purple coloring in TTY terminals.

    Args:
        agent_name: Name of the agent that is currently thinking.
    """
    logger = get_logger()
    if sys.stdout.isatty():
        logger.info(
            f"{COLORS['BLUE']}  (🧠 {agent_name}){COLORS['RESET']}{COLORS['BLUE_PURPLE']} is thinking...{COLORS['RESET']}"
        )
    else:
        logger.info(f"  (🧠 {agent_name}) is thinking...")


def log_success(message: str):
    """Log a success message with rocket emoji and blue color.

    Args:
        message: The success message to display.
    """
    logger = get_logger()
    if sys.stdout.isatty():
        logger.info(f"{COLORS['BLUE']}🚀 {message}{COLORS['RESET']}")
    else:
        logger.info(f"🚀 {message}")


def log_error(message: str):
    """Log an error message with cross emoji and red color.

    Args:
        message: The error message to display.
    """
    logger = get_logger()
    if sys.stdout.isatty():
        logger.error(f"{COLORS['LIGHT_RED']}❌ {message}{COLORS['RESET']}")
    else:
        logger.error(f"❌ {message}")


def log_warning(message: str):
    """Log a warning message with warning emoji and yellow color.

    Args:
        message: The warning message to display.
    """
    logger = get_logger()
    if sys.stdout.isatty():
        logger.warning(f"{COLORS['YELLOW']}⚠️ {message}{COLORS['RESET']}")
    else:
        logger.warning(f"⚠️ {message}")


def log_retry(attempt: int, max_attempts: int, error: str):
    """Log a retry attempt with attempt counter and error details.

    Args:
        attempt: Current attempt number (1-based).
        max_attempts: Maximum number of attempts allowed.
        error: Description of the error that triggered the retry.
    """
    logger = get_logger()
    if sys.stdout.isatty():
        message = f"{COLORS['YELLOW']}⏳ Retry {attempt}/{max_attempts}: {COLORS['LIGHT_RED']}{error}{COLORS['RESET']}"
    else:
        message = f"⏳ Retry {attempt}/{max_attempts}: {error}"
    logger.warning(message)


def log_delegation(from_agent: str, to_agent: str):
    """Log an agent delegation event with arrow visualization.

    Displays a formatted message showing control transfer from
    one agent to another.

    Args:
        from_agent: Name of the delegating agent.
        to_agent: Name of the agent receiving delegation.
    """
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
    """Log the initialization of an agent.

    Args:
        agent: Name of the agent being started.
    """
    logger = get_logger()
    if sys.stdout.isatty():
        message = f" {COLORS['BLUE_PURPLE']}{agent} Agent is started.{COLORS['RESET']}"
    else:
        message = f" {agent} Agent is started."
    logger.info(message)


def log_task_start(task_name: str, agent: str | None = None):
    """Log the start of a task with optional agent context.

    Args:
        task_name: Name or description of the task being started.
        agent: Optional name of the agent executing the task.
    """
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
    """Log task completion with optional duration.

    Args:
        task_name: Name of the completed task.
        duration: Optional execution duration in seconds.
    """
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
    """Purple-accented streaming callback for real-time event display.

    Handles various event types from the Xerxes streaming system and
    displays them with appropriate formatting, colors, and timing
    information. Maintains internal state for tracking tool calls
    and execution times.

    Supports the following event types:
    - StreamChunk: Content streaming and tool call arguments
    - FunctionDetection: Detection of function calls in output
    - FunctionCallsExtracted: List of extracted function calls
    - FunctionExecutionStart: Beginning of function execution
    - FunctionExecutionComplete: Completion of function execution
    - AgentSwitch: Agent transition events
    - ReinvokeSignal: Re-invocation signals
    - Completion: Task/pipeline completion events

    Args:
        chunk: Event object from the streaming system. Can be any of
            the supported event types defined in xerxes.types.

    Note:
        Uses internal state to track tool call headers, indentation,
        and execution timing across multiple calls.
    """

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
        """Strip ANSI escape codes from a string.

        Args:
            s: The string potentially containing ANSI escape sequences.

        Returns:
            The input string with all ANSI escape sequences removed.
        """
        return ANSI_RE.sub("", s)

    def term_width() -> int:
        """Return the current terminal width in columns.

        Queries the terminal size via ``shutil.get_terminal_size`` and clamps
        the result to a minimum of 60 columns.

        Returns:
            The terminal width in columns, with a minimum of 60 and a
            fallback default of 100 if the terminal size cannot be determined.
        """
        try:
            return max(60, shutil.get_terminal_size(fallback=(100, 24)).columns)
        except Exception:
            return 100

    def paint(text: object, *styles: str) -> str:
        """Wrap text with ANSI style codes and a reset suffix.

        Args:
            text: The object to render as styled text. Converted to ``str``
                before wrapping.
            *styles: One or more ANSI escape code strings to prepend.

        Returns:
            The styled string with all style codes prepended and an
            ANSI reset code appended.
        """
        return "".join(styles) + str(text) + RESET

    def tag(agent_id: str) -> str:
        """Format an agent identifier as a bold accented bracketed label.

        Args:
            agent_id: The agent identifier string to format.

        Returns:
            A string of the form ``[agent_id]`` wrapped in bold and accent
            ANSI codes.
        """
        return f"{BOLD}{ACCENT}[{agent_id}]{RESET}"

    def bullet() -> str:
        """Return a styled bullet point character.

        Returns:
            A bullet character (``'\\u2022'``) wrapped in accent and bold
            ANSI codes.
        """
        return paint("•", ACCENT, BOLD)

    def ensure_newline() -> None:
        """Print a newline if the current output line is still open.

        Checks the internal ``open_line`` state flag and, if a line is
        currently incomplete, flushes a newline to ``stdout`` and resets
        the flag to ``False``.
        """
        if state["open_line"]:
            print("", flush=True)
            state["open_line"] = False

    def write(s: str) -> None:
        """Print text without a trailing newline and mark the line as open.

        Args:
            s: The string to write to ``stdout``.
        """
        print(s, end="", flush=True)
        state["open_line"] = True

    def writeln(s: str) -> None:
        """Ensure any open line is closed, then print a complete line.

        Calls :func:`ensure_newline` first so that any partial output is
        properly terminated before writing the new line.

        Args:
            s: The string to print as a full line to ``stdout``.
        """
        ensure_newline()
        print(s, flush=True)
        state["open_line"] = False

    def indent_newlines(s: str, indent: str) -> str:
        """Replace embedded newlines with newline-plus-indent sequences.

        Args:
            s: The source string that may contain embedded newlines.
            indent: The indentation string to insert after each newline.

        Returns:
            The string with all newline characters replaced by the
            newline followed by *indent*. Returns *s* unchanged if it
            is empty or contains no newlines.
        """
        if not s or "\n" not in s:
            return s
        return s.replace("\n", "\n" + indent)

    def preview(text: object, max_len: int = 100) -> str:
        """Truncate text to *max_len* characters, appending an ellipsis if needed.

        Args:
            text: The object to preview. Converted to ``str`` before
                truncation.
            max_len: Maximum allowed length before truncation. Defaults
                to 100.

        Returns:
            The string representation of *text*, truncated to *max_len*
            characters with a trailing ellipsis (``'\\u2026'``) if the
            original exceeds the limit.
        """
        s = str(text)
        return (s[:max_len] + "…") if len(s) > max_len else s

    def pretty_result(value: object) -> str:
        """Serialize a value to a human-readable string.

        Attempts JSON serialization first for ``dict`` and ``list`` types,
        then falls back to ``pprint.pformat`` if JSON encoding fails.

        Args:
            value: The object to serialize. Dicts and lists are serialized
                as indented JSON; other types use a ``str`` default encoder.

        Returns:
            A formatted, human-readable string representation of *value*.
        """
        try:
            if isinstance(value, dict | list):
                return json.dumps(value, indent=2, ensure_ascii=False)
            return json.dumps(value, indent=2, ensure_ascii=False, default=str)
        except Exception:
            return pformat(value, width=max(60, term_width() - 8), compact=False)

    def hr(title: str | None = None) -> None:
        """Print a full-width horizontal rule, optionally with a centred title.

        Renders a line of ``'\\u2500'`` characters spanning the terminal
        width. When *title* is provided, it is centered within the rule.

        Args:
            title: Optional title text to center within the horizontal
                rule. If ``None``, a plain line is printed.
        """
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
