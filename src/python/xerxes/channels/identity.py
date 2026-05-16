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

Maps every ``(channel, channel_user_id)`` pair the agent has ever seen to a
stable global ``user_id`` (UUID). The same human reaching out via Slack and
Telegram can be linked to a single Xerxes user with ``link``, after which
``channels_for`` returns every alias. Records are kept in memory for lookup
speed and optionally persisted through ``MemoryStorage``.
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
    """One link between a platform identity and a global Xerxes user.

    Attributes:
        user_id: Stable global Xerxes user id (UUID string).
        channel: Channel name, e.g. ``"slack"`` or ``"telegram"``.
        channel_user_id: Platform-specific user id as the adapter sees it.
        display_name: Human-readable name when known; otherwise empty.
        first_seen: ISO timestamp captured on creation.
    """

    user_id: str
    channel: str
    channel_user_id: str
    display_name: str = ""
    first_seen: str = ""

    def to_dict(self) -> dict[str, tp.Any]:
        """Serialise the record as a plain JSON-friendly dict."""
        return {
            "user_id": self.user_id,
            "channel": self.channel,
            "channel_user_id": self.channel_user_id,
            "display_name": self.display_name,
            "first_seen": self.first_seen,
        }

    @classmethod
    def from_dict(cls, data: dict[str, tp.Any]) -> IdentityRecord:
        """Rebuild a record from the output of ``to_dict``.

        Args:
            data: Mapping previously produced by ``to_dict`` (any extra keys
                are ignored; missing optional keys fall back to defaults).

        Returns:
            The reconstructed identity record.
        """
        return cls(
            user_id=data["user_id"],
            channel=data["channel"],
            channel_user_id=data["channel_user_id"],
            display_name=data.get("display_name", ""),
            first_seen=data.get("first_seen", ""),
        )


def _key(channel: str, channel_user_id: str) -> str:
    """Build the prefixed storage key for one ``(channel, user)`` pair."""
    return f"{IDENTITY_KEY_PREFIX}{channel}::{channel_user_id}"


class IdentityResolver:
    """Thread-safe in-memory index of identity records, optionally backed by storage.

    On construction the resolver hydrates from ``storage`` if provided so
    that restarts preserve user identities. ``resolve`` is the hot-path
    entry: it returns an existing record or mints a fresh UUID-backed one.
    All mutating operations hold an ``RLock`` so the resolver is safe to
    share across daemon threads handling different channels.
    """

    def __init__(self, storage: MemoryStorage | None = None) -> None:
        """Build the resolver and load any persisted records.

        Args:
            storage: Optional persistence backend. When provided the resolver
                hydrates its in-memory index from existing keys and writes
                every mutation back; when ``None`` the resolver is purely
                in-memory.
        """
        self.storage = storage
        self._index: dict[str, IdentityRecord] = {}
        self._lock = threading.RLock()
        self._hydrate()

    def _hydrate(self) -> None:
        """Pull every persisted ``IdentityRecord`` into the in-memory index.

        Tolerant of storage errors and partially corrupt records — anything
        that fails to deserialise is skipped silently rather than raising.
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
        """Return the record for a channel user, creating it on first sight.

        When the record exists and lacks a display name, the supplied
        ``display_name`` is back-filled and persisted. This is the call
        adapters make once per inbound message.

        Args:
            channel: Channel name.
            channel_user_id: Platform-specific user identifier.
            display_name: Optional human-readable name; back-filled into an
                existing record only when it currently has no name.

        Returns:
            The existing or freshly minted identity record.

        Raises:
            ValueError: ``channel`` or ``channel_user_id`` is empty.
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
        """Force a channel identity to point at the given global ``user_id``.

        Used to merge previously distinct identities — once two channel ids
        share a ``user_id`` they will both surface via ``channels_for``. If
        the channel id already mapped to a different global id, that mapping
        is overwritten.

        Args:
            user_id: Global Xerxes user id to associate.
            channel: Channel name.
            channel_user_id: Platform-specific user id.

        Returns:
            The updated or newly created identity record.
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
        """Look up an existing record without creating one.

        Args:
            channel: Channel name.
            channel_user_id: Platform-specific user id.

        Returns:
            The record if known, otherwise ``None``.
        """
        with self._lock:
            return self._index.get(_key(channel, channel_user_id))

    def channels_for(self, user_id: str) -> list[IdentityRecord]:
        """Return every channel-side alias linked to ``user_id``.

        Args:
            user_id: Global Xerxes user id.

        Returns:
            All ``IdentityRecord`` instances whose ``user_id`` matches. Empty
            when no aliases exist.
        """
        with self._lock:
            return [r for r in self._index.values() if r.user_id == user_id]

    def all(self) -> list[IdentityRecord]:
        """Return a snapshot of every record currently in the index."""
        with self._lock:
            return list(self._index.values())

    def _persist(self, key: str, rec: IdentityRecord) -> None:
        """Best-effort write of one record to ``storage`` (swallows errors)."""
        if self.storage is None:
            return
        try:
            self.storage.save(key, rec.to_dict())
        except Exception:
            pass


__all__ = ["IDENTITY_KEY_PREFIX", "IdentityRecord", "IdentityResolver"]
