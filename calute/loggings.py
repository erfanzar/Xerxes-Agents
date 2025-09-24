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
import logging
import os
import sys
import threading

# ANSI color codes
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

# Level colors mapping
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
        # Create logger
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


# Create a global logger instance
logger = get_logger()
