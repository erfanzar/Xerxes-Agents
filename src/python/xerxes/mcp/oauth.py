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
"""OAuth 2.1 (PKCE) helpers used to authorise MCP servers.

Implements the slice of RFC 6749 + RFC 7636 needed for the
authorization-code-with-PKCE flow modern MCP servers use. The flow is split
into pure helpers (``generate_pkce_pair``, ``build_authorize_url``) so tests
can drive it deterministically; network IO is confined to
``exchange_code`` and ``refresh_token`` which both accept an injected
``httpx.Client`` for testing.
"""

from __future__ import annotations

import base64
import hashlib
import os
import time
import urllib.parse
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx


@dataclass
class OAuthConfig:
    """OAuth 2.1 client metadata for one MCP server.

    Attributes:
        client_id: Registered OAuth client identifier.
        authorize_url: Provider's authorization endpoint.
        token_url: Provider's token-exchange endpoint.
        redirect_uri: Must exactly match the registered redirect URI.
        scopes: Requested scopes (space-joined automatically).
    """

    client_id: str
    authorize_url: str
    token_url: str
    redirect_uri: str = "http://127.0.0.1:5454/callback"
    scopes: tuple[str, ...] = ()


@dataclass
class OAuthToken:
    """One issued access/refresh token pair.

    Attributes:
        access_token: Opaque bearer token sent with each MCP request.
        refresh_token: Long-lived token used to mint new access tokens.
        token_type: Token type, usually ``"Bearer"``.
        expires_at: Epoch seconds when ``access_token`` expires.
        scopes: Scopes the provider actually granted (may differ from those requested).
    """

    access_token: str
    refresh_token: str | None = None
    token_type: str = "Bearer"
    expires_at: float | None = None
    scopes: tuple[str, ...] = field(default_factory=tuple)

    def is_expired(self, *, skew_seconds: float = 30.0) -> bool:
        """True when the token expires within ``skew_seconds``; tokens without an expiry never expire."""
        if self.expires_at is None:
            return False
        return time.time() >= self.expires_at - skew_seconds

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict with ``scopes`` as a plain list."""
        d = asdict(self)
        d["scopes"] = list(self.scopes)
        return d

    @classmethod
    def from_response(cls, payload: dict[str, Any]) -> OAuthToken:
        """Construct a token from the provider's ``/token`` response body."""
        expires_in = payload.get("expires_in")
        expires_at = time.time() + float(expires_in) if expires_in else None
        scope = payload.get("scope", "")
        scopes = tuple(scope.split()) if scope else ()
        return cls(
            access_token=payload["access_token"],
            refresh_token=payload.get("refresh_token"),
            token_type=payload.get("token_type", "Bearer"),
            expires_at=expires_at,
            scopes=scopes,
        )


def generate_pkce_pair(*, _verifier: str | None = None) -> tuple[str, str]:
    """Return a fresh ``(verifier, challenge)`` for PKCE S256.

    The verifier is base64url-encoded randomness (96 bytes of entropy); the
    challenge is the URL-safe base64 SHA-256 of the verifier with padding
    stripped. ``_verifier`` is only meant for deterministic tests.
    """
    if _verifier is None:
        verifier = base64.urlsafe_b64encode(os.urandom(96)).decode("ascii").rstrip("=")
    else:
        verifier = _verifier
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return verifier, challenge


def build_authorize_url(config: OAuthConfig, *, state: str, code_challenge: str) -> str:
    """Compose the ``authorize_url`` (with PKCE + ``state``) the user opens in a browser."""
    params: dict[str, str] = {
        "response_type": "code",
        "client_id": config.client_id,
        "redirect_uri": config.redirect_uri,
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }
    if config.scopes:
        params["scope"] = " ".join(config.scopes)
    return f"{config.authorize_url}?{urllib.parse.urlencode(params)}"


def exchange_code(
    config: OAuthConfig,
    *,
    code: str,
    code_verifier: str,
    client: httpx.Client | None = None,
) -> OAuthToken:
    """Trade an authorization code + PKCE verifier for a fresh :class:`OAuthToken`."""
    payload = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": config.redirect_uri,
        "client_id": config.client_id,
        "code_verifier": code_verifier,
    }
    own_client = client is None
    c = client or httpx.Client(timeout=30.0)
    try:
        resp = c.post(config.token_url, data=payload)
        resp.raise_for_status()
        return OAuthToken.from_response(resp.json())
    finally:
        if own_client:
            c.close()


def refresh_token(
    config: OAuthConfig,
    *,
    token: OAuthToken,
    client: httpx.Client | None = None,
) -> OAuthToken:
    """Exchange a refresh token for a new access token, preserving the original refresh token when the provider omits it."""
    if token.refresh_token is None:
        raise ValueError("no refresh_token available")
    payload = {
        "grant_type": "refresh_token",
        "refresh_token": token.refresh_token,
        "client_id": config.client_id,
    }
    own_client = client is None
    c = client or httpx.Client(timeout=30.0)
    try:
        resp = c.post(config.token_url, data=payload)
        resp.raise_for_status()
        new_token = OAuthToken.from_response(resp.json())
        # Some providers omit refresh_token on refresh — keep the old one.
        if new_token.refresh_token is None:
            new_token.refresh_token = token.refresh_token
        return new_token
    finally:
        if own_client:
            c.close()


__all__ = [
    "OAuthConfig",
    "OAuthToken",
    "build_authorize_url",
    "exchange_code",
    "generate_pkce_pair",
    "refresh_token",
]
