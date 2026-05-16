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
"""Persist OAuth tokens to disk under ``$XERXES_HOME/credentials``.

Each provider gets one ``<provider>.json`` file holding the serialised
:class:`OAuthToken` (access/refresh tokens, scopes, expiry). Files are
chmod'd to ``0600`` after write where the OS supports it. Module-level
helpers wrap a default ``CredentialStorage`` for convenience.
"""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path

from .._compat_shims import xerxes_subdir_safe
from ..mcp.oauth import OAuthToken


@dataclass
class CredentialStorage:
    """Filesystem-backed token store rooted at ``base_dir``.

    Attributes:
        base_dir: Directory containing one ``<provider>.json`` per record.
    """

    base_dir: Path

    @classmethod
    def default(cls) -> CredentialStorage:
        """Return a storage rooted at ``$XERXES_HOME/credentials``."""
        return cls(base_dir=xerxes_subdir_safe("credentials"))

    def _path(self, provider: str) -> Path:
        """Return the on-disk path used for ``provider``."""
        return self.base_dir / f"{provider}.json"

    def save(self, provider: str, token: OAuthToken) -> Path:
        """Write ``token`` for ``provider`` and chmod the file to ``0600``."""
        path = self._path(provider)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(token.to_dict(), indent=2), encoding="utf-8")
        try:
            os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass
        return path

    def load(self, provider: str) -> OAuthToken | None:
        """Load the token for ``provider``, or ``None`` if missing/corrupt."""
        path = self._path(provider)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return OAuthToken(
            access_token=data["access_token"],
            refresh_token=data.get("refresh_token"),
            token_type=data.get("token_type", "Bearer"),
            expires_at=data.get("expires_at"),
            scopes=tuple(data.get("scopes", [])),
        )

    def remove(self, provider: str) -> bool:
        """Delete the token for ``provider``; return ``True`` if removed."""
        path = self._path(provider)
        if not path.exists():
            return False
        path.unlink()
        return True

    def list_providers(self) -> list[str]:
        """Return the alphabetically sorted set of stored provider names."""
        if not self.base_dir.exists():
            return []
        return sorted(p.stem for p in self.base_dir.glob("*.json"))


_default = CredentialStorage.default()


def save(provider: str, token: OAuthToken) -> Path:
    """Save ``token`` for ``provider`` via the default storage."""
    return _default.save(provider, token)


def load(provider: str) -> OAuthToken | None:
    """Load a token for ``provider`` from the default storage."""
    return _default.load(provider)


def remove(provider: str) -> bool:
    """Remove ``provider`` from the default storage."""
    return _default.remove(provider)


def list_providers() -> list[str]:
    """List providers present in the default storage."""
    return _default.list_providers()


__all__ = ["CredentialStorage", "list_providers", "load", "remove", "save"]
