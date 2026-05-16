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
"""Tests for xerxes.mcp.oauth."""

from __future__ import annotations

import base64
import hashlib
import urllib.parse

import httpx
import pytest
from xerxes.mcp.oauth import (
    OAuthConfig,
    OAuthToken,
    build_authorize_url,
    exchange_code,
    generate_pkce_pair,
    refresh_token,
)


@pytest.fixture
def config():
    return OAuthConfig(
        client_id="cid",
        authorize_url="https://example.test/authorize",
        token_url="https://example.test/token",
        scopes=("read", "write"),
    )


class TestPkce:
    def test_pair_deterministic_with_explicit_verifier(self):
        _v, c = generate_pkce_pair(_verifier="fixed-verifier")
        expected = base64.urlsafe_b64encode(hashlib.sha256(b"fixed-verifier").digest()).decode().rstrip("=")
        assert c == expected

    def test_random_verifier_long_enough(self):
        v, _c = generate_pkce_pair()
        # base64url(96 bytes) = 128 chars before padding strip
        assert len(v) >= 100

    def test_no_padding(self):
        v, c = generate_pkce_pair()
        assert "=" not in v
        assert "=" not in c


class TestBuildAuthorizeUrl:
    def test_includes_required_params(self, config):
        url = build_authorize_url(config, state="abc", code_challenge="cc")
        parsed = urllib.parse.urlparse(url)
        qs = dict(urllib.parse.parse_qsl(parsed.query))
        assert qs["response_type"] == "code"
        assert qs["client_id"] == "cid"
        assert qs["redirect_uri"] == "http://127.0.0.1:5454/callback"
        assert qs["state"] == "abc"
        assert qs["code_challenge"] == "cc"
        assert qs["code_challenge_method"] == "S256"
        assert qs["scope"] == "read write"


class TestOAuthToken:
    def test_from_response_parses(self):
        tok = OAuthToken.from_response(
            {"access_token": "a", "refresh_token": "r", "expires_in": 3600, "scope": "read write"}
        )
        assert tok.access_token == "a"
        assert tok.refresh_token == "r"
        assert tok.scopes == ("read", "write")
        assert tok.expires_at is not None

    def test_is_expired_handles_missing_expires_at(self):
        tok = OAuthToken(access_token="a")
        assert tok.is_expired() is False

    def test_is_expired_true_when_past(self):
        import time as _time

        tok = OAuthToken(access_token="a", expires_at=_time.time() - 1)
        assert tok.is_expired() is True


class TestExchangeAndRefresh:
    def test_exchange_code_posts_pkce(self, config):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["body"] = dict(urllib.parse.parse_qsl(request.content.decode()))
            return httpx.Response(200, json={"access_token": "abc", "expires_in": 60})

        client = httpx.Client(transport=httpx.MockTransport(handler))
        tok = exchange_code(config, code="auth-code", code_verifier="ver", client=client)
        assert tok.access_token == "abc"
        assert captured["body"]["grant_type"] == "authorization_code"
        assert captured["body"]["code_verifier"] == "ver"
        client.close()

    def test_refresh_token_requires_refresh(self, config):
        with pytest.raises(ValueError):
            refresh_token(config, token=OAuthToken(access_token="x"))

    def test_refresh_preserves_refresh_when_missing(self, config):
        def handler(req):
            return httpx.Response(200, json={"access_token": "new", "expires_in": 60})

        client = httpx.Client(transport=httpx.MockTransport(handler))
        tok = OAuthToken(access_token="old", refresh_token="rt")
        new = refresh_token(config, token=tok, client=client)
        assert new.access_token == "new"
        assert new.refresh_token == "rt"
        client.close()
