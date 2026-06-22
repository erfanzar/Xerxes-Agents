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

import base64
import hashlib
import json
import os
import stat
import tempfile
import threading
import warnings
from dataclasses import dataclass
from pathlib import Path

from .._compat_shims import xerxes_subdir_safe
from ..mcp.oauth import OAuthToken


def _get_fernet():
    """Return a Fernet instance or None if encryption is unavailable."""
    try:
        from cryptography.fernet import Fernet
    except ImportError:
        warnings.warn(
            "cryptography not installed; OAuth tokens stored in plaintext",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    key = os.environ.get("XERXES_CREDENTIAL_KEY")
    if key:
        raw = hashlib.sha256(key.encode()).digest()
        fernet_key = base64.urlsafe_b64encode(raw)
        return Fernet(fernet_key)

    key_path = Path.home() / ".xerxes" / ".credential_key"
    key_path.parent.mkdir(parents=True, exist_ok=True)
    if key_path.exists():
        fernet_key = key_path.read_bytes().strip()
    else:
        fernet_key = Fernet.generate_key()
        key_path.write_bytes(fernet_key)
        try:
            os.chmod(key_path, stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass
    return Fernet(fernet_key)


def _lock_file(fd, exclusive: bool = False):
    """Acquire an advisory file lock if the platform supports it."""
    try:
        import fcntl

        op = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        fcntl.flock(fd, op)
    except ImportError:
        try:
            import portalocker

            op = portalocker.LOCK_EX if exclusive else portalocker.LOCK_SH
            portalocker.lock(fd, op)
        except ImportError:
            pass


def _unlock_file(fd):
    """Release an advisory file lock if the platform supports it."""
    try:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)
    except ImportError:
        try:
            import portalocker

            portalocker.unlock(fd)
        except ImportError:
            pass


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
        data = json.dumps(token.to_dict(), indent=2)
        fernet = _get_fernet()
        if fernet is not None:
            data = fernet.encrypt(data.encode("utf-8")).decode("utf-8")
        tmp_fd, tmp_path_str = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        tmp_path = Path(tmp_path_str)
        try:
            encoded = data.encode("utf-8")
            total = 0
            while total < len(encoded):
                n = os.write(tmp_fd, encoded[total:])
                if n == 0:
                    raise OSError("os.write returned 0")
                total += n
            try:
                os.chmod(tmp_path, stat.S_IRUSR | stat.S_IWUSR)
            except OSError:
                pass
            # Acquire exclusive lock on temp file and hold across replace
            _lock_file(tmp_fd, exclusive=True)
            try:
                os.replace(tmp_path, path)
            finally:
                _unlock_file(tmp_fd)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        finally:
            try:
                os.close(tmp_fd)
            except OSError:
                pass
        return path

    def load(self, provider: str) -> OAuthToken | None:
        """Load the token for ``provider``, or ``None`` if missing/corrupt."""
        path = self._path(provider)
        if not path.exists():
            return None
        try:
            with open(path, "r") as f:
                fd = f.fileno()
                _lock_file(fd, exclusive=False)
                try:
                    raw = f.read()
                finally:
                    _unlock_file(fd)
        except (OSError, json.JSONDecodeError):
            return None
        fernet = _get_fernet()
        if fernet is not None:
            try:
                decrypted = fernet.decrypt(raw.encode("utf-8")).decode("utf-8")
                data = json.loads(decrypted)
            except Exception:
                # Fallback: maybe the file was written before encryption was enabled
                try:
                    data = json.loads(raw)
                except Exception:
                    return None
        else:
            try:
                data = json.loads(raw)
            except Exception:
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


_instance = None
_lock = threading.Lock()


def default_instance():
    global _instance
    with _lock:
        if _instance is None:
            _instance = CredentialStorage.default()
        return _instance


def save(provider: str, token: OAuthToken) -> Path:
    """Save ``token`` for ``provider`` via the default storage."""
    return default_instance().save(provider, token)


def load(provider: str) -> OAuthToken | None:
    """Load a token for ``provider`` from the default storage."""
    return default_instance().load(provider)


def remove(provider: str) -> bool:
    """Remove ``provider`` from the default storage."""
    return default_instance().remove(provider)


def list_providers() -> list[str]:
    """List providers present in the default storage."""
    return default_instance().list_providers()


__all__ = ["CredentialStorage", "list_providers", "load", "remove", "save"]
