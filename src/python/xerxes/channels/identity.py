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
"""Identity resolution for cross-channel users.

Maps ``(channel, channel_user_id)`` pairs to stable ``user_id`` values,
with optional persistence via ``MemoryStorage``.
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
    """Persistent mapping between a channel-specific user and a global user ID."""

    user_id: str
    channel: str
    channel_user_id: str
    display_name: str = ""
    first_seen: str = ""

    def to_dict(self) -> dict[str, tp.Any]:
        """Serialize the record to a plain dictionary.

        Returns:
            dict[str, Any]: OUT: field names mapped to their values.
        """
        return {
            "user_id": self.user_id,
            "channel": self.channel,
            "channel_user_id": self.channel_user_id,
            "display_name": self.display_name,
            "first_seen": self.first_seen,
        }

    @classmethod
    def from_dict(cls, data: dict[str, tp.Any]) -> IdentityRecord:
        """Deserialize a record from a plain dictionary.

        Args:
            data (dict[str, Any]): IN: dictionary produced by ``to_dict``.

        Returns:
            IdentityRecord: OUT: reconstructed identity record.
        """
        return cls(
            user_id=data["user_id"],
            channel=data["channel"],
            channel_user_id=data["channel_user_id"],
            display_name=data.get("display_name", ""),
            first_seen=data.get("first_seen", ""),
        )


def _key(channel: str, channel_user_id: str) -> str:
    """Build a storage key for an identity record.

    Args:
        channel (str): IN: channel name.
        channel_user_id (str): IN: platform-specific user identifier.

    Returns:
        str: OUT: prefixed composite key.
    """
    return f"{IDENTITY_KEY_PREFIX}{channel}::{channel_user_id}"


class IdentityResolver:
    """Resolves and persists identity mappings across channels."""

    def __init__(self, storage: MemoryStorage | None = None) -> None:
        """Initialize the resolver and hydrate the in-memory index.

        Args:
            storage (MemoryStorage | None): IN: optional storage backend for
                persistence. OUT: used to load and save ``IdentityRecord``
                instances.
        """
        self.storage = storage
        self._index: dict[str, IdentityRecord] = {}
        self._lock = threading.RLock()
        self._hydrate()

    def _hydrate(self) -> None:
        """Load existing identity records from storage into memory.

        Skips silently on storage errors or when no storage is configured.
        """
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
        """Resolve an identity, creating a new record if necessary.

        Args:
            channel (str): IN: channel name.
            channel_user_id (str): IN: platform-specific user identifier.
            display_name (str): IN: human-readable name to associate with the
                identity. OUT: persisted if the record is new or has no name.

        Returns:
            IdentityRecord: OUT: existing or newly created identity record.

        Raises:
            ValueError: If ``channel`` or ``channel_user_id`` is empty.
        """
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
        """Explicitly link a global user ID to a channel-specific ID.

        Args:
            user_id (str): IN: global user identifier.
            channel (str): IN: channel name.
            channel_user_id (str): IN: platform-specific user identifier.

        Returns:
            IdentityRecord: OUT: the linked identity record (updated or new).
        """
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
        """Lookup an existing identity record.

        Args:
            channel (str): IN: channel name.
            channel_user_id (str): IN: platform-specific user identifier.

        Returns:
            IdentityRecord | None: OUT: the record if found, otherwise ``None``.
        """
        with self._lock:
            return self._index.get(_key(channel, channel_user_id))

    def channels_for(self, user_id: str) -> list[IdentityRecord]:
        """Return all identity records linked to a global user ID.

        Args:
            user_id (str): IN: global user identifier.

        Returns:
            list[IdentityRecord]: OUT: matching identity records.
        """
        with self._lock:
            return [r for r in self._index.values() if r.user_id == user_id]

    def all(self) -> list[IdentityRecord]:
        """Return all known identity records.

        Returns:
            list[IdentityRecord]: OUT: snapshot of the in-memory index.
        """
        with self._lock:
            return list(self._index.values())

    def _persist(self, key: str, rec: IdentityRecord) -> None:
        """Persist an identity record to storage.

        Args:
            key (str): IN: storage key.
            rec (IdentityRecord): IN: record to save.
        """
        if self.storage is None:
            return
        try:
            self.storage.save(key, rec.to_dict())
        except Exception:
            pass


__all__ = ["IDENTITY_KEY_PREFIX", "IdentityRecord", "IdentityResolver"]
