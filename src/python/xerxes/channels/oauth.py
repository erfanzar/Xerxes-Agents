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
"""OAuth 2.0 authorization-code client used by channel integrations.

Handles authorisation URL construction, CSRF state issuance and
validation, ``code → token`` exchange, refresh, and per-installation
token persistence through ``MemoryStorage``. Refresh is serialised per
``install_id`` because most providers rotate the refresh token on every
call — without the lock two concurrent refreshes would race and one
caller would end up holding an invalidated refresh token.
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
    """Static configuration describing one OAuth 2.0 identity provider.

    Attributes:
        name: Short identifier; used as the storage-key prefix.
        client_id: OAuth client id issued by the provider.
        client_secret: OAuth client secret. Treat as sensitive.
        authorize_url: Endpoint hosting the user-facing consent UI.
        token_url: Endpoint that exchanges codes/refresh tokens for access
            tokens.
        scopes: Scopes requested when building the authorize URL.
        redirect_uri: Callback URL registered with the provider.
        extra_authorize_params: Provider-specific extras merged into the
            authorize URL (e.g. ``access_type=offline`` for Google).
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
    """A persisted OAuth 2.0 access token plus metadata.

    Attributes:
        provider: Name of the issuing ``OAuthProvider``.
        access_token: Bearer token used on API calls.
        refresh_token: Long-lived token used by ``OAuthClient.refresh``.
            Empty when the provider does not return one.
        expires_at: Absolute Unix timestamp at which ``access_token`` stops
            working. ``0.0`` means "unknown / non-expiring".
        scopes: Granted scope list.
        raw: Full token-endpoint payload, retained for provider-specific
            fields the dataclass does not model.
    """

    provider: str
    access_token: str
    refresh_token: str = ""
    expires_at: float = 0.0
    scopes: list[str] | None = None
    raw: dict[str, tp.Any] | None = None

    def __post_init__(self) -> None:
        """Coerce ``scopes`` and ``raw`` from ``None`` to empty containers."""
        if self.scopes is None:
            self.scopes = []
        if self.raw is None:
            self.raw = {}

    def is_expired(self, *, now: float | None = None) -> bool:
        """Return whether the token has expired (with a 60-second skew buffer).

        The 60-second buffer means the token is treated as expired slightly
        early so callers can refresh before in-flight requests fail.

        Args:
            now: Optional Unix timestamp override; defaults to ``time.time()``.

        Returns:
            ``True`` if the token is expired or expiring within 60 seconds.
            Always ``False`` when ``expires_at`` is ``0.0`` (unknown).
        """
        now = time.time() if now is None else now
        if self.expires_at == 0.0:
            return False
        return now + 60.0 >= self.expires_at

    def to_dict(self) -> dict[str, tp.Any]:
        """Serialise the token to a JSON-friendly dict for ``MemoryStorage``."""
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
        """Rebuild a token from the output of ``to_dict``.

        Args:
            data: Mapping previously written by ``to_dict``.

        Returns:
            Reconstructed token.
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
    """OAuth 2.0 authorization-code flow with token persistence and refresh locking.

    Issues CSRF-protected authorize URLs, consumes the matching state on
    callback, exchanges codes for tokens, refreshes tokens before they
    expire, and persists tokens through ``MemoryStorage`` keyed by
    ``(provider, install_id)`` so different workspaces / accounts can keep
    independent credentials.
    """

    STATE_TTL_SECONDS = 600.0

    def __init__(
        self,
        provider: OAuthProvider,
        *,
        storage: MemoryStorage | None = None,
        http_client: tp.Any | None = None,
    ) -> None:
        """Build the client.

        Args:
            provider: Provider configuration used for every authorize and
                token request.
            storage: Optional backend that persists tokens. Without it
                ``get_token`` always returns ``None`` and tokens live only
                inside the current process.
            http_client: Optional callable replacing ``httpx`` for the token
                endpoint. Receives ``(url, data=...)`` and must return a dict
                or JSON-decodable string. Used heavily in tests.
        """
        self.provider = provider
        self.storage = storage
        self._http = http_client
        self._lock = threading.Lock()
        self._states: dict[str, float] = {}
        self._refresh_locks: dict[str, threading.Lock] = {}

    def authorize_url(self) -> tuple[str, str]:
        """Build the provider's consent URL and a fresh CSRF state token.

        The state token is recorded internally with a TTL and must be passed
        back through ``consume_state`` once the OAuth callback returns.

        Returns:
            ``(full_authorize_url, state)``. The state must be round-tripped
            through the callback to defeat CSRF.
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
        """Validate and atomically consume a previously issued CSRF state.

        States can only be used once; reuse returns ``False``. Stale entries
        beyond ``STATE_TTL_SECONDS`` are GC'd opportunistically here.

        Args:
            state: Value received from the OAuth callback.

        Returns:
            ``True`` only when the state was outstanding and within its TTL.
        """
        with self._lock:
            ts = self._states.pop(state, None)
            self._gc_states()
        if ts is None:
            return False
        return (time.time() - ts) <= self.STATE_TTL_SECONDS

    def _gc_states(self) -> None:
        """Drop state tokens older than ``STATE_TTL_SECONDS``."""
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
        """Trade an authorization code for an access token and persist it.

        When both ``state_received`` and ``expected_state`` are supplied the
        client checks they match before talking to the provider — callers
        that already validated via ``consume_state`` may omit them.

        Args:
            code: Authorization code returned to the OAuth callback.
            install_id: Per-installation key used for token storage. Distinct
                values give per-workspace / per-account isolation.
            state_received: Value echoed back by the provider on the callback.
            expected_state: Value originally issued by ``authorize_url``.

        Returns:
            The newly obtained token, also written to storage.

        Raises:
            ValueError: ``state_received`` does not match ``expected_state``.
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
        """Refresh the access token for ``install_id`` using its refresh token.

        Carries the existing refresh token forward when the provider does
        not return a new one in the response.

        Args:
            install_id: Installation identifier used at exchange time.

        Returns:
            The refreshed token, also persisted to storage.

        Raises:
            RuntimeError: No stored token, or the stored token has no
                refresh token (the user must re-authorise).
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
        """Return a fresh token, refreshing transparently when needed.

        Concurrent refreshes for the same ``install_id`` are serialised by a
        per-installation lock; without that two callers can both POST to the
        token endpoint, the provider rotates the refresh token on each call,
        and one of the responses wins the storage write — the other caller
        is then holding an invalidated refresh token and the next refresh
        fails permanently.

        Args:
            install_id: Installation identifier.

        Returns:
            A non-expired token, or ``None`` when no token is stored or the
            refresh attempt failed.
        """
        token = self.get_token(install_id)
        if token is None:
            return None
        if not token.is_expired():
            return token
        # Serialize refreshes per install_id. Without this, two concurrent
        # callers can both POST to the token endpoint; the provider may
        # rotate refresh_token on each call and one of the responses wins
        # the storage write — the other caller now holds an invalidated
        # refresh_token and the next refresh fails permanently.
        with self._lock:
            lock = self._refresh_locks.setdefault(install_id, threading.Lock())
        with lock:
            # Re-read inside the lock: a concurrent caller may have just
            # refreshed for us.
            current = self.get_token(install_id)
            if current is not None and not current.is_expired():
                return current
            try:
                return self.refresh(install_id)
            except Exception:
                logger.warning("OAuth refresh failed", exc_info=True)
                return None

    def _store_key(self, install_id: str) -> str:
        """Return the ``MemoryStorage`` key for one installation's token."""
        return f"_oauth_{self.provider.name}_{install_id}"

    def get_token(self, install_id: str = "default") -> OAuthToken | None:
        """Read the stored token for ``install_id`` without touching the network.

        Tolerant of storage and deserialisation errors — any failure yields
        ``None`` so callers can fall back to a fresh authorize flow.

        Args:
            install_id: Installation identifier.

        Returns:
            The deserialised token, or ``None`` when missing or invalid.
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
        """Best-effort persist of one token; storage errors are logged not raised."""
        if self.storage is None:
            return
        try:
            self.storage.save(self._store_key(install_id), token.to_dict())
        except Exception:
            logger.warning("Failed to persist OAuth token", exc_info=True)

    def _post_form(self, url: str, data: dict[str, tp.Any]) -> dict[str, tp.Any]:
        """POST ``application/x-www-form-urlencoded`` to ``url`` and decode the body.

        Falls back to parsing the response as a query string when JSON
        decoding fails — some providers (notably old Slack tier) reply with
        form-encoded payloads.

        Args:
            url: Target URL.
            data: Form fields.

        Returns:
            Parsed response, either from JSON or from query-string decoding.

        Raises:
            RuntimeError: ``httpx`` is needed but not installed.
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
        """Build an ``OAuthToken`` from a raw token-endpoint response.

        Handles two access-token shapes: top-level (most providers) and the
        nested ``authed_user.access_token`` used by Slack user-tokens.

        Args:
            payload: Raw JSON response from ``token_url``.

        Returns:
            Populated token. ``expires_at`` is ``0.0`` when the provider
            omitted ``expires_in``.

        Raises:
            RuntimeError: The payload contains no usable ``access_token``.
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
