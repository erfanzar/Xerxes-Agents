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
"""Provider-agnostic OAuth helpers.

Wraps the MCP OAuth helpers with a thin ``OAuthClient`` that targets
common providers (OpenAI, Anthropic, GitHub Copilot, GitHub PAT).
Each provider sets a different ``OAuthConfig``; the client object
exposes ``begin_authorize``, ``finish_authorize``, and ``refresh``.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass

import httpx

from ..mcp.oauth import OAuthConfig as McpOAuthConfig
from ..mcp.oauth import OAuthToken, generate_pkce_pair
from ..mcp.oauth import build_authorize_url as _build_url
from ..mcp.oauth import exchange_code as _exchange
from ..mcp.oauth import refresh_token as _refresh


@dataclass
class OAuthConfig(McpOAuthConfig):
    """Provider configuration alias of :class:`mcp.oauth.OAuthConfig`."""

    pass


@dataclass
class _AuthorizeContext:
    """Material returned by :meth:`OAuthClient.begin_authorize`.

    Attributes:
        url: Authorize URL to open in the user's browser.
        state: Anti-CSRF token; must be echoed in the callback.
        code_verifier: PKCE verifier required to exchange the code.
    """

    url: str
    state: str
    code_verifier: str


class OAuthClient:
    """Authorization-code-with-PKCE client bound to one provider config."""

    def __init__(self, config: OAuthConfig, *, http_client: httpx.Client | None = None) -> None:
        """Bind to ``config``; reuse ``http_client`` if supplied."""
        self._config = config
        self._http = http_client

    def begin_authorize(self) -> _AuthorizeContext:
        """Start the flow: generate PKCE pair and the authorize URL."""
        verifier, challenge = generate_pkce_pair()
        state = secrets.token_urlsafe(24)
        url = _build_url(self._config, state=state, code_challenge=challenge)
        return _AuthorizeContext(url=url, state=state, code_verifier=verifier)

    def finish_authorize(self, *, code: str, code_verifier: str) -> OAuthToken:
        """Exchange the redirect-supplied ``code`` for an access token."""
        return _exchange(self._config, code=code, code_verifier=code_verifier, client=self._http)

    def refresh(self, token: OAuthToken) -> OAuthToken:
        """Refresh ``token`` using its refresh grant, returning a new token."""
        return _refresh(self._config, token=token, client=self._http)


# ---------------------------- provider presets -----------------------------


def github_pat_preset(client_id: str) -> OAuthConfig:
    """OAuth config for the standard GitHub user-token flow."""
    return OAuthConfig(
        client_id=client_id,
        authorize_url="https://github.com/login/oauth/authorize",
        token_url="https://github.com/login/oauth/access_token",
        scopes=("read:user",),
    )


def openai_preset(client_id: str) -> OAuthConfig:
    """OAuth config for OpenAI's auth-code flow."""
    return OAuthConfig(
        client_id=client_id,
        authorize_url="https://auth.openai.com/oauth/authorize",
        token_url="https://auth.openai.com/oauth/token",
        scopes=("openid", "profile"),
    )


def anthropic_preset(client_id: str) -> OAuthConfig:
    """OAuth config for the Anthropic / Claude Code creds bridge."""
    return OAuthConfig(
        client_id=client_id,
        authorize_url="https://console.anthropic.com/oauth/authorize",
        token_url="https://console.anthropic.com/oauth/token",
        scopes=("openid",),
    )


def copilot_preset(client_id: str) -> OAuthConfig:
    """OAuth config for GitHub Copilot bearer-token issuance."""
    return OAuthConfig(
        client_id=client_id,
        authorize_url="https://github.com/login/oauth/authorize",
        token_url="https://github.com/login/oauth/access_token",
        scopes=("copilot",),
    )


__all__ = [
    "OAuthClient",
    "OAuthConfig",
    "OAuthToken",
    "anthropic_preset",
    "copilot_preset",
    "github_pat_preset",
    "openai_preset",
]
