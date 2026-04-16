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
"""OAuth helper for channel adapters.

A small framework around the OAuth 2.0 *authorization code* flow that
Slack / Discord / Microsoft Teams etc. all use. The helper intentionally
covers only the spec-shaped bits — building the authorization URL,
exchanging codes for tokens, refreshing them, and persisting the tokens
to a :class:`MemoryStorage`.

Each provider passes its endpoints + scopes to :class:`OAuthClient`;
the result is a simple ``authorize_url`` / ``exchange_code`` /
``get_valid_token`` API the adapter can call.
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
    """OAuth2 endpoints + scopes for a single provider.

    Attributes:
        name: Stable identifier (e.g. ``"slack"``, ``"discord"``).
        client_id: OAuth client identifier.
        client_secret: OAuth client secret.
        authorize_url: Authorization endpoint URL.
        token_url: Token exchange endpoint URL.
        scopes: List of scopes to request.
        redirect_uri: Configured callback URL.
        extra_authorize_params: Optional extra query params for the
            authorize step (e.g. ``user_scope`` for Slack).
    """

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
    """Decoded OAuth access token bundle.

    Attributes:
        provider: Provider name from :class:`OAuthProvider`.
        access_token: Bearer token.
        refresh_token: Refresh token (when issued).
        expires_at: Epoch seconds at which ``access_token`` expires.
        scopes: Granted scopes.
        raw: Original JSON body returned by the token endpoint.
    """

    provider: str
    access_token: str
    refresh_token: str = ""
    expires_at: float = 0.0
    scopes: list[str] = None  # type: ignore[assignment]
    raw: dict[str, tp.Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """Normalise ``scopes`` and ``raw`` to empty containers when ``None``."""
        if self.scopes is None:
            self.scopes = []
        if self.raw is None:
            self.raw = {}

    def is_expired(self, *, now: float | None = None) -> bool:
        """Whether the token has expired (or expires within 60 s)."""
        now = time.time() if now is None else now
        if self.expires_at == 0.0:
            return False
        return now + 60.0 >= self.expires_at

    def to_dict(self) -> dict[str, tp.Any]:
        """Serialise to a dict suitable for :class:`MemoryStorage` persistence."""
        return {
            "provider": self.provider,
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "expires_at": self.expires_at,
            "scopes": list(self.scopes),
            "raw": dict(self.raw),
        }

    @classmethod
    def from_dict(cls, data: dict[str, tp.Any]) -> OAuthToken:
        """Rebuild an :class:`OAuthToken` from its :meth:`to_dict` form."""
        return cls(
            provider=data["provider"],
            access_token=data["access_token"],
            refresh_token=data.get("refresh_token", ""),
            expires_at=float(data.get("expires_at", 0.0)),
            scopes=list(data.get("scopes", [])),
            raw=dict(data.get("raw", {})),
        )


class OAuthClient:
    """OAuth2 helper bound to one provider.

    Storage layout (when a :class:`MemoryStorage` is supplied):
    ``_oauth_<provider>_<install_id>`` → JSON of :class:`OAuthToken`.

    HTTP work is performed via ``httpx`` lazily; the constructor does
    no network I/O.

    Example:
        >>> p = OAuthProvider(name="slack", ...)
        >>> client = OAuthClient(p, storage=mem)
        >>> url, state = client.authorize_url()
        >>> # ... user grants in browser ...
        >>> tok = client.exchange_code(code="abc", state_received=state, expected_state=state)
    """

    STATE_TTL_SECONDS = 600.0

    def __init__(
        self,
        provider: OAuthProvider,
        *,
        storage: MemoryStorage | None = None,
        http_client: tp.Any | None = None,
    ) -> None:
        """Bind the client to one provider.

        Args:
            provider: OAuth2 endpoints + credentials for a single service.
            storage: Optional persistence for issued tokens.
            http_client: Optional callable ``(url, data=...)`` used instead
                of ``httpx`` (primarily for tests).
        """
        self.provider = provider
        self.storage = storage
        self._http = http_client
        self._lock = threading.Lock()
        self._states: dict[str, float] = {}

    def authorize_url(self) -> tuple[str, str]:
        """Build the authorization URL plus a fresh ``state`` token."""
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
        """Validate + remove a previously issued state token."""
        with self._lock:
            ts = self._states.pop(state, None)
            self._gc_states()
        if ts is None:
            return False
        return (time.time() - ts) <= self.STATE_TTL_SECONDS

    def _gc_states(self) -> None:
        """Drop issued state tokens older than :attr:`STATE_TTL_SECONDS`."""
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
        """Exchange the authorization code for an access token bundle."""
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
        """Refresh the stored token in-place."""
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
        """Return a non-expired token, refreshing it when needed."""
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
        """Storage key used for tokens belonging to ``install_id``."""
        return f"_oauth_{self.provider.name}_{install_id}"

    def get_token(self, install_id: str = "default") -> OAuthToken | None:
        """Load the persisted token for ``install_id``, or ``None`` if absent.

        Args:
            install_id: Tenant / installation identifier; defaults to the
                single-tenant ``"default"`` slot.

        Returns:
            The stored :class:`OAuthToken`, or ``None`` when no storage is
            configured, nothing is saved, or the record is unreadable.
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
        """Best-effort persist of ``token`` under the slot for ``install_id``."""
        if self.storage is None:
            return
        try:
            self.storage.save(self._store_key(install_id), token.to_dict())
        except Exception:
            logger.warning("Failed to persist OAuth token", exc_info=True)

    def _post_form(self, url: str, data: dict[str, tp.Any]) -> dict[str, tp.Any]:
        """Submit ``data`` as ``application/x-www-form-urlencoded`` to ``url``.

        Args:
            url: Target token endpoint.
            data: Form fields (client credentials, code, grant_type, ...).

        Returns:
            The parsed JSON response (or ``parse_qsl`` dict when the server
            returns form-encoded bodies, as legacy providers sometimes do).
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
        """Translate a token-endpoint JSON payload into :class:`OAuthToken`.

        Handles provider-specific quirks — Slack nests its user token under
        ``authed_user``, ``expires_in`` is optional, and ``scope`` may be a
        space-delimited string or a list.
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
