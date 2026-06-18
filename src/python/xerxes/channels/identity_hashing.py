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
"""Stable, salt-aware identity hashing across messaging platforms.

Used by audit logs, history, and metric labels to refer to humans without
storing raw platform ids. Hashes are deterministic per ``XERXES_IDENTITY_SALT``
so the same user produces the same hash across process restarts, while a
hash leaked from logs cannot be reversed without the salt.

.. warning::
    The irreversibility guarantee only holds when ``XERXES_IDENTITY_SALT`` is
    set to a secret, per-install value. When it is unset, the public default
    salt below is used and leaked hashes can be brute-forced back to their raw
    platform ids (platform strings are known and many id spaces — e.g. Telegram
    integer ids — are small/enumerable). Generate a per-install secret once and
    export it, for example::

        export XERXES_IDENTITY_SALT="$(python -c 'import secrets; print(secrets.token_hex(32))')"

Layout:
    ``hash_user(platform, raw_user_id)``  → ``"user_<sha16>"``
    ``hash_chat(platform, raw_chat_id)``  → ``"<platform>:<sha16>"``
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os

logger = logging.getLogger(__name__)

# Public, well-known fallback salt. Kept unchanged so existing hashes computed
# without a configured salt remain stable; it provides NO privacy guarantee.
_DEFAULT_SALT = "xerxes-default-identity-salt"

_warned_default_salt = False


def _salt() -> bytes:
    """Return the active identity-hashing salt as bytes.

    Reads ``XERXES_IDENTITY_SALT`` each call so operators can rotate it
    without restarting (though doing so invalidates every previously
    computed hash).

    When the variable is unset, falls back to a public default salt and emits a
    loud one-time warning, because the documented irreversibility guarantee does
    not hold with a non-secret salt.
    """
    configured = os.environ.get("XERXES_IDENTITY_SALT")
    if configured is None:
        global _warned_default_salt
        if not _warned_default_salt:
            _warned_default_salt = True
            logger.warning(
                "XERXES_IDENTITY_SALT is not set; using the public default identity salt. "
                "Identity hashes are NOT irreversible in this configuration: leaked user/chat "
                "hashes can be brute-forced back to raw platform ids. Set a secret per-install "
                "salt, e.g. export XERXES_IDENTITY_SALT=\"$(python -c 'import secrets; "
                "print(secrets.token_hex(32))')\""
            )
        configured = _DEFAULT_SALT
    return configured.encode("utf-8")


def hash_user(platform: str, raw_user_id: str | int) -> str:
    """Hash a user id into the stable ``user_<sha16>`` form.

    Args:
        platform: Platform name (``"telegram"``, ``"slack"`` …); included in
            the HMAC input so identical raw ids on different platforms hash
            differently.
        raw_user_id: The platform's raw user id.

    Returns:
        ``"user_"`` plus the first 16 hex characters of a salted SHA-256.
    """
    digest = hmac.new(_salt(), f"{platform}|{raw_user_id}".encode(), hashlib.sha256).hexdigest()
    return f"user_{digest[:16]}"


def hash_chat(platform: str, raw_chat_id: str | int) -> str:
    """Hash a chat/room id into the platform-prefixed form ``<platform>:<sha16>``.

    The prefix is preserved so downstream consumers can route by platform
    without needing the raw id.
    """
    digest = hmac.new(_salt(), f"{platform}|chat|{raw_chat_id}".encode(), hashlib.sha256).hexdigest()
    return f"{platform}:{digest[:16]}"


def matches_user(platform: str, raw_user_id: str | int, candidate: str) -> bool:
    """Return whether ``candidate`` equals ``hash_user(platform, raw_user_id)``.

    Helpful for checking that a stored hash still corresponds to a known raw
    id without re-hashing in caller code.
    """
    return hash_user(platform, raw_user_id) == candidate


__all__ = ["hash_chat", "hash_user", "matches_user"]
