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
"""Console-oriented logging helpers for Xerxes.

Re-exports the colorized :class:`XerxesLogger` singleton, ``stream_callback``
for live execution streams, and the family of ``log_*`` shortcuts (step,
thinking, success, error, retry, delegation, agent_start, task_start,
task_complete, warning) used throughout the agent runtime."""

from .console import (
    ColorFormatter,
    XerxesLogger,
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
    "ColorFormatter",
    "XerxesLogger",
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
