# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).
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
"""Identity unification across channels.

Each platform has its own user IDs (Telegram numeric, Slack U…).
:class:`IdentityResolver` maps ``(channel, channel_user_id)`` pairs to
a stable Xerxes ``user_id`` so memory + profile remain consistent
when the same person reaches the agent through multiple channels.

Persistence is optional — a :class:`MemoryStorage` instance is consulted
when provided. Without one, mappings live in memory only.
"""

from __future__ import annotations

import threading
import typing as tp
import uuid
from dataclasses import dataclass
from datetime import datetime

if tp.TYPE_CHECKING:
    from ..memory.storage import MemoryStorage
IDENTITY_KEY_PREFIX = "_identity_"


@dataclass
class IdentityRecord:
    """A resolved identity mapping.

    Attributes:
        user_id: Stable Xerxes identifier.
        channel: Origin channel name.
        channel_user_id: Platform-native user id.
        display_name: Best-effort display name.
        first_seen: ISO timestamp of first observation.
    """

    user_id: str
    channel: str
    channel_user_id: str
    display_name: str = ""
    first_seen: str = ""

    def to_dict(self) -> dict[str, tp.Any]:
        """Serialise to a plain JSON-friendly dict with all five fields."""
        return {
            "user_id": self.user_id,
            "channel": self.channel,
            "channel_user_id": self.channel_user_id,
            "display_name": self.display_name,
            "first_seen": self.first_seen,
        }

    @classmethod
    def from_dict(cls, data: dict[str, tp.Any]) -> IdentityRecord:
        """Rebuild an :class:`IdentityRecord` from its :meth:`to_dict` shape."""
        return cls(
            user_id=data["user_id"],
            channel=data["channel"],
            channel_user_id=data["channel_user_id"],
            display_name=data.get("display_name", ""),
            first_seen=data.get("first_seen", ""),
        )


def _key(channel: str, channel_user_id: str) -> str:
    """Compose the storage key used to index a ``(channel, user)`` pair."""
    return f"{IDENTITY_KEY_PREFIX}{channel}::{channel_user_id}"


class IdentityResolver:
    """Map per-channel users to stable Xerxes user IDs.

    Designed to be safe under concurrent inbound traffic: every public
    method takes the internal lock before mutating state.

    Example:
        >>> r = IdentityResolver()
        >>> r.resolve("telegram", "12345", display_name="Alice")
        IdentityRecord(user_id='...', channel='telegram', ...)
    """

    def __init__(self, storage: MemoryStorage | None = None) -> None:
        """Initialise the resolver, optionally rehydrating from ``storage``.

        Args:
            storage: Optional :class:`MemoryStorage` used to persist and
                reload identity records across process restarts.
        """
        self.storage = storage
        self._index: dict[str, IdentityRecord] = {}
        self._lock = threading.RLock()
        self._hydrate()

    def _hydrate(self) -> None:
        """Load previously persisted identity records from ``self.storage``."""
        if self.storage is None:
            return
        try:
            keys = self.storage.list_keys(IDENTITY_KEY_PREFIX)
        except Exception:
            return
        for k in keys:
            if not k.startswith(IDENTITY_KEY_PREFIX):
                continue
            try:
                data = self.storage.load(k)
                if data:
                    rec = IdentityRecord.from_dict(data)
                    self._index[k] = rec
            except Exception:
                continue

    def resolve(
        self,
        channel: str,
        channel_user_id: str,
        *,
        display_name: str = "",
    ) -> IdentityRecord:
        """Look up an existing mapping or create a fresh one."""
        if not channel or not channel_user_id:
            raise ValueError("channel and channel_user_id are required")
        key = _key(channel, channel_user_id)
        with self._lock:
            rec = self._index.get(key)
            if rec is not None:
                if display_name and not rec.display_name:
                    rec.display_name = display_name
                    self._persist(key, rec)
                return rec
            rec = IdentityRecord(
                user_id=str(uuid.uuid4()),
                channel=channel,
                channel_user_id=channel_user_id,
                display_name=display_name,
                first_seen=datetime.now().isoformat(),
            )
            self._index[key] = rec
            self._persist(key, rec)
            return rec

    def link(self, user_id: str, channel: str, channel_user_id: str) -> IdentityRecord:
        """Bind an existing Xerxes ``user_id`` to a new channel identity."""
        key = _key(channel, channel_user_id)
        with self._lock:
            existing = self._index.get(key)
            if existing is not None:
                if existing.user_id == user_id:
                    return existing
                existing.user_id = user_id
                self._persist(key, existing)
                return existing
            rec = IdentityRecord(
                user_id=user_id,
                channel=channel,
                channel_user_id=channel_user_id,
                first_seen=datetime.now().isoformat(),
            )
            self._index[key] = rec
            self._persist(key, rec)
            return rec

    def get(self, channel: str, channel_user_id: str) -> IdentityRecord | None:
        """Return the record for ``(channel, channel_user_id)`` or ``None``."""
        with self._lock:
            return self._index.get(_key(channel, channel_user_id))

    def channels_for(self, user_id: str) -> list[IdentityRecord]:
        """Return every known record linked to a Xerxes ``user_id``."""
        with self._lock:
            return [r for r in self._index.values() if r.user_id == user_id]

    def all(self) -> list[IdentityRecord]:
        """Return a snapshot list of every resolved identity record."""
        with self._lock:
            return list(self._index.values())

    def _persist(self, key: str, rec: IdentityRecord) -> None:
        """Best-effort write of ``rec`` to ``self.storage`` under ``key``."""
        if self.storage is None:
            return
        try:
            self.storage.save(key, rec.to_dict())
        except Exception:
            pass


__all__ = ["IDENTITY_KEY_PREFIX", "IdentityRecord", "IdentityResolver"]
