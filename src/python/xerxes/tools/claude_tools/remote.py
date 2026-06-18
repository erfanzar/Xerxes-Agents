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
"""Remote trigger and scheduled cron tools."""

from __future__ import annotations

from ...types import AgentBaseFn


class RemoteTriggerTool(AgentBaseFn):
    """Trigger a named remote endpoint.

    Example:
        >>> RemoteTriggerTool.static_call(trigger_name="notify-slack", payload="Build complete")
    """

    @staticmethod
    def static_call(
        trigger_name: str,
        payload: str = "",
        **context_variables,
    ) -> str:
        """Trigger a remote endpoint.

        Args:
            trigger_name: Name of the trigger to invoke.
            payload: Data to send with the trigger.
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Status message.
        """
        return f"[RemoteTrigger] name={trigger_name} payload={payload[:100]}\nRemote triggers require configured remote endpoints."


class ScheduleCronTool(AgentBaseFn):
    """Schedule a task for cron-based execution.

    Example:
        >>> ScheduleCronTool.static_call(schedule="0 9 * * *", prompt="Daily report")
    """

    @staticmethod
    def static_call(
        schedule: str,
        prompt: str,
        name: str = "",
        **context_variables,
    ) -> str:
        """Schedule a cron task.

        Args:
            schedule: Cron expression (e.g., "0 9 * * *").
            prompt: Task prompt for the scheduled execution.
            name: Optional name for the scheduled task.
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Status message.
        """
        return (
            f"[ScheduleCron] schedule={schedule} name={name or '(unnamed)'}\n"
            f"Prompt: {prompt[:100]}\n"
            "Cron scheduling requires a persistent scheduler service."
        )
