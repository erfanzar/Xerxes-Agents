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


"""Logging subsystem for Calute.

This package provides two complementary logging modules for the Calute framework:

- ``console``: Lightweight ANSI-colored terminal logging with emoji helpers,
  formatted step/task output, and a streaming callback for real-time event display.
- ``structured``: Advanced structured logging with optional integration for
  OpenTelemetry distributed tracing, Prometheus metrics collection, and JSON
  log formatting.

The public API re-exports the most commonly used symbols from the ``console``
module for convenient access.

Example:
    >>> from calute.logging import get_logger, log_step, set_verbosity
    >>> set_verbosity("DEBUG")
    >>> logger = get_logger()
    >>> logger.info("Application started")
    >>> log_step("INIT", "Loading configuration", color="GREEN")
"""

from .console import (
    CaluteLogger,
    ColorFormatter,
    get_logger,
    log_agent_start,
    log_delegation,
    log_error,
    log_retry,
    log_step,
    log_success,
    log_task_complete,
    log_task_start,
    log_thinking,
    log_warning,
    set_verbosity,
    stream_callback,
)

__all__ = [
    "CaluteLogger",
    "ColorFormatter",
    "get_logger",
    "log_agent_start",
    "log_delegation",
    "log_error",
    "log_retry",
    "log_step",
    "log_success",
    "log_task_complete",
    "log_task_start",
    "log_thinking",
    "log_warning",
    "set_verbosity",
    "stream_callback",
]
