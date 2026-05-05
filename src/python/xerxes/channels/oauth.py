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
"""OAuth 2.0 client for channel integrations.

Manages authorization URLs, state validation, token exchange, refresh, and
persistent token storage.
"""

from __future__ import annotations

import json
import logging
import secrets
import threading
import time
import typing as tp
import urllib.parse
from dataclasses import dataclass

if tp.TYPE_CHECKING:
    from ..memory.storage import MemoryStorage

logger = logging.getLogger(__name__)


@dataclass
class OAuthProvider:
    """Configuration for an OAuth 2.0 identity provider."""

    name: str
    client_id: str
    client_secret: str
    authorize_url: str
    token_url: str
    scopes: list[str]
    redirect_uri: str
    extra_authorize_params: dict[str, str] | None = None


@dataclass
class OAuthToken:
    """OAuth 2.0 access token with metadata."""

    provider: str
    access_token: str
    refresh_token: str = ""
    expires_at: float = 0.0
    scopes: list[str] | None = None
    raw: dict[str, tp.Any] | None = None

    def __post_init__(self) -> None:
        """Ensure ``scopes`` and ``raw`` are initialized to empty collections."""
        if self.scopes is None:
            self.scopes = []
        if self.raw is None:
            self.raw = {}

    def is_expired(self, *, now: float | None = None) -> bool:
        """Check whether the token is expired (with a 60-second buffer).

        Args:
            now (float | None): IN: optional Unix timestamp. Defaults to the
                current time.

        Returns:
            bool: OUT: ``True`` if expired or about to expire.
        """
        now = time.time() if now is None else now
        if self.expires_at == 0.0:
            return False
        return now + 60.0 >= self.expires_at

    def to_dict(self) -> dict[str, tp.Any]:
        """Serialize the token to a plain dictionary.

        Returns:
            dict[str, Any]: OUT: field names mapped to their values.
        """
        return {
            "provider": self.provider,
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "expires_at": self.expires_at,
            "scopes": list(self.scopes or []),
            "raw": dict(self.raw or {}),
        }

    @classmethod
    def from_dict(cls, data: dict[str, tp.Any]) -> OAuthToken:
        """Deserialize a token from a plain dictionary.

        Args:
            data (dict[str, Any]): IN: dictionary produced by ``to_dict``.

        Returns:
            OAuthToken: OUT: reconstructed token instance.
        """
        return cls(
            provider=data["provider"],
            access_token=data["access_token"],
            refresh_token=data.get("refresh_token", ""),
            expires_at=float(data.get("expires_at", 0.0)),
            scopes=list(data.get("scopes", [])),
            raw=dict(data.get("raw", {})),
        )


class OAuthClient:
    """OAuth 2.0 authorization-code flow client with token persistence."""

    STATE_TTL_SECONDS = 600.0

    def __init__(
        self,
        provider: OAuthProvider,
        *,
        storage: MemoryStorage | None = None,
        http_client: tp.Any | None = None,
    ) -> None:
        """Initialize the OAuth client.

        Args:
            provider (OAuthProvider): IN: provider configuration.
                OUT: stored for building authorization and token requests.
            storage (MemoryStorage | None): IN: optional storage backend for
                persisting tokens.
            http_client (Any | None): IN: optional HTTP client callable used
                for token endpoint requests. OUT: forwarded to ``_post_form``.
        """
        self.provider = provider
        self.storage = storage
        self._http = http_client
        self._lock = threading.Lock()
        self._states: dict[str, float] = {}

    def authorize_url(self) -> tuple[str, str]:
        """Build the provider authorization URL and a CSRF state token.

        Returns:
            tuple[str, str]: OUT: ``(full_authorize_url, state)``. The state
            is tracked internally for later validation.
        """
        state = secrets.token_urlsafe(24)
        with self._lock:
            self._states[state] = time.time()
            self._gc_states()
        params = {
            "client_id": self.provider.client_id,
            "redirect_uri": self.provider.redirect_uri,
            "scope": " ".join(self.provider.scopes),
            "state": state,
            "response_type": "code",
        }
        if self.provider.extra_authorize_params:
            params.update(self.provider.extra_authorize_params)
        return f"{self.provider.authorize_url}?{urllib.parse.urlencode(params)}", state

    def consume_state(self, state: str) -> bool:
        """Validate and consume a CSRF state token.

        Args:
            state (str): IN: state value received from the OAuth callback.

        Returns:
            bool: OUT: ``True`` if the state exists and has not expired.
        """
        with self._lock:
            ts = self._states.pop(state, None)
            self._gc_states()
        if ts is None:
            return False
        return (time.time() - ts) <= self.STATE_TTL_SECONDS

    def _gc_states(self) -> None:
        """Remove expired state entries."""
        cutoff = time.time() - self.STATE_TTL_SECONDS
        for s, ts in list(self._states.items()):
            if ts < cutoff:
                self._states.pop(s, None)

    def exchange_code(
        self,
        code: str,
        *,
        install_id: str = "default",
        state_received: str | None = None,
        expected_state: str | None = None,
    ) -> OAuthToken:
        """Exchange an authorization code for an access token.

        Args:
            code (str): IN: authorization code from the OAuth callback.
            install_id (str): IN: installation identifier for token storage.
                Defaults to "default".
            state_received (str | None): IN: state received with the callback.
            expected_state (str | None): IN: state originally issued. OUT:
                compared against ``state_received`` when both are provided.

        Returns:
            OAuthToken: OUT: newly obtained token, persisted to storage.

        Raises:
            ValueError: If the received state does not match the expected state.
        """
        if expected_state is not None and state_received != expected_state:
            raise ValueError("OAuth state mismatch — refusing token exchange")
        body = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.provider.redirect_uri,
            "client_id": self.provider.client_id,
            "client_secret": self.provider.client_secret,
        }
        payload = self._post_form(self.provider.token_url, body)
        token = self._token_from_payload(payload)
        self._save_token(install_id, token)
        return token

    def refresh(self, install_id: str = "default") -> OAuthToken:
        """Refresh the access token for an installation.

        Args:
            install_id (str): IN: installation identifier. Defaults to
                "default".

        Returns:
            OAuthToken: OUT: newly refreshed token, persisted to storage.

        Raises:
            RuntimeError: If no stored token or refresh token is available.
        """
        token = self.get_token(install_id)
        if token is None or not token.refresh_token:
            raise RuntimeError("No refresh_token available — re-authorise")
        body = {
            "grant_type": "refresh_token",
            "refresh_token": token.refresh_token,
            "client_id": self.provider.client_id,
            "client_secret": self.provider.client_secret,
        }
        payload = self._post_form(self.provider.token_url, body)
        new_token = self._token_from_payload(payload)
        if not new_token.refresh_token:
            new_token.refresh_token = token.refresh_token
        self._save_token(install_id, new_token)
        return new_token

    def get_valid_token(self, install_id: str = "default") -> OAuthToken | None:
        """Retrieve a non-expired token, refreshing if necessary.

        Args:
            install_id (str): IN: installation identifier. Defaults to
                "default".

        Returns:
            OAuthToken | None: OUT: valid token, or ``None`` if unavailable
            or refresh failed.
        """
        token = self.get_token(install_id)
        if token is None:
            return None
        if not token.is_expired():
            return token
        try:
            return self.refresh(install_id)
        except Exception:
            logger.warning("OAuth refresh failed", exc_info=True)
            return None

    def _store_key(self, install_id: str) -> str:
        """Build the storage key for an installation's token.

        Args:
            install_id (str): IN: installation identifier.

        Returns:
            str: OUT: prefixed storage key.
        """
        return f"_oauth_{self.provider.name}_{install_id}"

    def get_token(self, install_id: str = "default") -> OAuthToken | None:
        """Load a token from storage.

        Args:
            install_id (str): IN: installation identifier. Defaults to
                "default".

        Returns:
            OAuthToken | None: OUT: deserialized token, or ``None``.
        """
        if self.storage is None:
            return None
        try:
            data = self.storage.load(self._store_key(install_id))
        except Exception:
            return None
        if not data:
            return None
        try:
            return OAuthToken.from_dict(data)
        except Exception:
            return None

    def _save_token(self, install_id: str, token: OAuthToken) -> None:
        """Persist a token to storage.

        Args:
            install_id (str): IN: installation identifier.
            token (OAuthToken): IN: token to save.
        """
        if self.storage is None:
            return
        try:
            self.storage.save(self._store_key(install_id), token.to_dict())
        except Exception:
            logger.warning("Failed to persist OAuth token", exc_info=True)

    def _post_form(self, url: str, data: dict[str, tp.Any]) -> dict[str, tp.Any]:
        """POST form data to a URL and return the JSON response.

        Args:
            url (str): IN: target URL.
            data (dict[str, Any]): IN: form fields.

        Returns:
            dict[str, Any]: OUT: parsed JSON response.

        Raises:
            RuntimeError: If ``httpx`` is required but not installed.
        """
        if self._http is not None:
            response = self._http(url, data=data)
            return response if isinstance(response, dict) else json.loads(response)
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("httpx is required for OAuthClient HTTP calls; install with `pip install httpx`") from exc
        resp = httpx.post(url, data=data, timeout=15.0)
        resp.raise_for_status()
        try:
            return resp.json()
        except Exception:
            return dict(urllib.parse.parse_qsl(resp.text))

    def _token_from_payload(self, payload: dict[str, tp.Any]) -> OAuthToken:
        """Build an ``OAuthToken`` from a token-endpoint response.

        Args:
            payload (dict[str, Any]): IN: raw JSON response from the token URL.

        Returns:
            OAuthToken: OUT: populated token instance.

        Raises:
            RuntimeError: If the payload lacks an ``access_token``.
        """
        access = payload.get("access_token") or payload.get("authed_user", {}).get("access_token", "")
        if not access:
            raise RuntimeError(f"Token endpoint response missing access_token: {payload}")
        expires_in = payload.get("expires_in")
        expires_at = (time.time() + float(expires_in)) if expires_in else 0.0
        scopes_field = payload.get("scope") or payload.get("scopes") or ""
        scopes = scopes_field.split() if isinstance(scopes_field, str) else list(scopes_field)
        return OAuthToken(
            provider=self.provider.name,
            access_token=access,
            refresh_token=payload.get("refresh_token", ""),
            expires_at=expires_at,
            scopes=scopes,
            raw=payload,
        )


__all__ = ["OAuthClient", "OAuthProvider", "OAuthToken"]
