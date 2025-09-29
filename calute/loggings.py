# Copyright 2025 The EasyDeL/Calute Author @erfanzar (Erfan Zare Chavoshi).
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


"""Lightweight logging system for Calute using ANSI colors"""

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

from .types import (
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
    def format(self, record: logging.LogRecord) -> str:
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


class CaluteLogger:
    """Centralized logger for Calute with colored output support"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the logger"""
        if not hasattr(self, "_initialized"):
            self._initialized = True
            self._setup_logger()

    def _setup_logger(self):
        """Set up the main logger"""

        self.logger = logging.getLogger("Calute")
        self.logger.setLevel(logging.DEBUG)

        self.logger.handlers = []

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self._get_log_level())

        console_handler.setFormatter(ColorFormatter())
        self.logger.addHandler(console_handler)

    def _get_log_level(self) -> int:
        """Get log level from environment variable"""
        level_str = os.environ.get("CALUTE_LOG_LEVEL", "INFO").upper()
        return getattr(logging, level_str, logging.INFO)

    def debug(self, message: str, *args, **kwargs):
        """Log debug message"""
        self.logger.debug(message, *args, **kwargs)

    def info(self, message: str, *args, **kwargs):
        """Log info message"""
        self.logger.info(message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs):
        """Log warning message"""
        self.logger.warning(message, *args, **kwargs)

    def error(self, message: str, *args, **kwargs):
        """Log error message"""
        self.logger.error(message, *args, **kwargs)

    def critical(self, message: str, *args, **kwargs):
        """Log critical message"""
        self.logger.critical(message, *args, **kwargs)

    def set_level(self, level: str):
        """Set logging level"""
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        self.logger.setLevel(numeric_level)
        for handler in self.logger.handlers:
            handler.setLevel(numeric_level)


def get_logger() -> CaluteLogger:
    """Get the singleton CaluteLogger instance"""
    return CaluteLogger()


def set_verbosity(level: str):
    """Set the global verbosity level

    Args:
        level: One of 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    """
    logger = get_logger()
    logger.set_level(level)


def log_step(step_name: str, description: str = "", color: str = "CYAN"):
    """Log a step in the process with formatting

    Args:
        step_name: Name of the step
        description: Optional description
        color: Color name from COLORS dict
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
    """Log a thinking message with green color"""
    logger = get_logger()
    if sys.stdout.isatty():
        logger.info(
            f"{COLORS['BLUE']}  (🧠 {agent_name}){COLORS['RESET']}{COLORS['BLUE_PURPLE']} is thinking...{COLORS['RESET']}"
        )
    else:
        logger.info(f"  (🧠 {agent_name}) is thinking...")


def log_success(message: str):
    """Log a success message with green color"""
    logger = get_logger()
    if sys.stdout.isatty():
        logger.info(f"{COLORS['BLUE']}🚀 {message}{COLORS['RESET']}")
    else:
        logger.info(f"🚀 {message}")


def log_error(message: str):
    """Log an error message with red color"""
    logger = get_logger()
    if sys.stdout.isatty():
        logger.error(f"{COLORS['LIGHT_RED']}❌ {message}{COLORS['RESET']}")
    else:
        logger.error(f"❌ {message}")


def log_warning(message: str):
    """Log a warning message with yellow color"""
    logger = get_logger()
    if sys.stdout.isatty():
        logger.warning(f"{COLORS['YELLOW']}⚠️ {message}{COLORS['RESET']}")
    else:
        logger.warning(f"⚠️ {message}")


def log_retry(attempt: int, max_attempts: int, error: str):
    """Log a retry attempt"""
    logger = get_logger()
    if sys.stdout.isatty():
        message = f"{COLORS['YELLOW']}⏳ Retry {attempt}/{max_attempts}: {COLORS['LIGHT_RED']}{error}{COLORS['RESET']}"
    else:
        message = f"⏳ Retry {attempt}/{max_attempts}: {error}"
    logger.warning(message)


def log_delegation(from_agent: str, to_agent: str):
    """Log agent delegation"""
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
    """Log the start of a agent"""
    logger = get_logger()
    if sys.stdout.isatty():
        message = f" {COLORS['BLUE_PURPLE']}{agent} Agent is started.{COLORS['RESET']}"
    else:
        message = f" {agent} Agent is started."
    logger.info(message)


def log_task_start(task_name: str, agent: str | None = None):
    """Log the start of a task"""
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
    """Log task completion"""
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
    """Purple-accented streaming callback. Keeps your exact tool-call style; adds timing, pretty results, and clean spacing."""

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

    def strip_ansi(s):
        return ANSI_RE.sub("", s)

    def term_width():
        try:
            return max(60, shutil.get_terminal_size(fallback=(100, 24)).columns)
        except Exception:
            return 100

    def paint(text, *styles):
        return "".join(styles) + str(text) + RESET

    def tag(agent_id):
        return f"{BOLD}{ACCENT}[{agent_id}]{RESET}"

    def bullet():
        return paint("•", ACCENT, BOLD)

    def ensure_newline():
        if state["open_line"]:
            print("", flush=True)
            state["open_line"] = False

    def write(s):
        print(s, end="", flush=True)
        state["open_line"] = True

    def writeln(s):
        ensure_newline()
        print(s, flush=True)
        state["open_line"] = False

    def indent_newlines(s, indent):
        if not s or "\n" not in s:
            return s
        return s.replace("\n", "\n" + indent)

    def preview(text, max_len=100):
        s = str(text)
        return (s[:max_len] + "…") if len(s) > max_len else s

    def pretty_result(value):
        try:
            if isinstance(value, dict | list):
                return json.dumps(value, indent=2, ensure_ascii=False)
            return json.dumps(value, indent=2, ensure_ascii=False, default=str)
        except Exception:
            return pformat(value, width=max(60, term_width() - 8), compact=False)

    def hr(title=None):
        width = term_width()
        if not title:
            print(paint("─" * width, ACCENT), flush=True)
            return
        title_str = f" {title} "
        left = (width - len(title_str)) // 2
        right = max(0, width - left - len(title_str))
        print(paint("─" * left, ACCENT) + paint(title_str, LWHITE, BOLD) + paint("─" * right, ACCENT), flush=True)

    if isinstance(chunk, StreamChunk):
        if getattr(chunk, "content", None):
            # Check if we're in thinking mode
            if hasattr(chunk, "is_thinking") and chunk.is_thinking:
                # Write thinking content in purple/magenta
                write(paint(chunk.content, COL["MAGENTA"]))
            else:
                # Write normal content
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
                    indent = state["tool_indents"].get(tool_id, "")
                    arg_text = indent_newlines(tool_call.arguments, indent)
                    write(paint(arg_text, LWHITE))

    elif isinstance(chunk, FunctionDetection):
        ensure_newline()
        writeln(f"{paint('🔍', ACCENT)} {tag(chunk.agent_id)} {paint(chunk.message, LWHITE)}")

    elif isinstance(chunk, FunctionCallsExtracted):
        ensure_newline()
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
