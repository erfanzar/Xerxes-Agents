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
"""Route cron job output to messaging platforms or workspace files.

Every fire archives the agent's response under
``<archive_dir>/<job_id>/<utc-stamp>.md``. When the job's delivery
target names a real channel adapter, a caller-supplied ``sender`` is
invoked to forward the same content to that platform.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass
class DeliveryTarget:
    """Where the output of a cron job should land.

    Attributes:
        platform: ``"none"`` (archive only), ``"workspace"``, or a
            channel adapter name (``"telegram"``, ``"discord"``, ...).
        recipient: platform-specific id (chat id, email, ...).
    """

    platform: str
    recipient: str = ""


def _archive_dir(base_dir: Path, job_id: str) -> Path:
    """Ensure and return the per-job archive directory."""
    target = base_dir / job_id
    target.mkdir(parents=True, exist_ok=True)
    return target


def archive_output(base_dir: Path, job_id: str, content: str, *, now: datetime | None = None) -> Path:
    """Write ``content`` to ``<base>/<job_id>/<UTC-stamp>.md``.

    Returns the path written. ``now`` defaults to the current UTC time
    and is exposed for deterministic testing.
    """
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%S")
    out = _archive_dir(base_dir, job_id) / f"{stamp}.md"
    out.write_text(content, encoding="utf-8")
    return out


def route_output(
    target: DeliveryTarget,
    content: str,
    *,
    archive_dir: Path,
    job_id: str,
    sender: Callable[[str, str, str], None] | None = None,
) -> Path:
    """Archive ``content`` and optionally forward it to a channel.

    ``sender(platform, recipient, content)`` is the channel-adapter
    callback supplied by the daemon. Archiving always happens; sender
    is only invoked when ``target.platform`` isn't ``"none"`` or
    ``"workspace"``.
    """
    path = archive_output(archive_dir, job_id, content)
    if target.platform not in ("none", "workspace") and sender is not None:
        sender(target.platform, target.recipient, content)
    return path


__all__ = ["DeliveryTarget", "archive_output", "route_output"]
