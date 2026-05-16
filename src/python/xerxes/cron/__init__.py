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
"""Built-in cron scheduler for Xerxes.

Persisted jobs (cron expression or one-shot), a scheduler thread, and
platform-aware delivery routing. The scheduler is transport-agnostic:
supply a ``run_job`` callable that knows how to start a Xerxes turn
for a given prompt + workspace.

All times are stored as UTC ISO timestamps; conversion to local time
is the caller's responsibility (typically a TUI / channel formatter).
"""

from .delivery import DeliveryTarget, route_output
from .jobs import CronJob, JobStore, next_fire_at
from .scheduler import CronScheduler

__all__ = [
    "CronJob",
    "CronScheduler",
    "DeliveryTarget",
    "JobStore",
    "next_fire_at",
    "route_output",
]
