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
"""Tests for auth/oauth.py + auth/storage.py."""

from __future__ import annotations

import urllib.parse

import httpx
from xerxes.auth.oauth import (
    OAuthClient,
    anthropic_preset,
    copilot_preset,
    github_pat_preset,
    openai_preset,
)
from xerxes.auth.storage import CredentialStorage
from xerxes.mcp.oauth import OAuthToken


class TestPresets:
    def test_github_preset(self):
        cfg = github_pat_preset("client-x")
        assert cfg.client_id == "client-x"
        assert "github.com" in cfg.authorize_url

    def test_openai_preset(self):
        cfg = openai_preset("c")
        assert "openai.com" in cfg.token_url

    def test_anthropic_preset(self):
        cfg = anthropic_preset("c")
        assert "anthropic.com" in cfg.token_url

    def test_copilot_preset(self):
        cfg = copilot_preset("c")
        assert "copilot" in cfg.scopes


class TestOAuthClient:
    def test_begin_authorize_produces_url_and_pkce(self):
        client = OAuthClient(github_pat_preset("cid"))
        ctx = client.begin_authorize()
        assert ctx.url.startswith("https://github.com/login/oauth/authorize?")
        # Parse the URL — state matches what we got back; challenge present.
        qs = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(ctx.url).query))
        assert qs["state"] == ctx.state
        assert "code_challenge" in qs

    def test_finish_authorize_exchanges_code(self):
        called = {}

        def handler(request):
            called["url"] = str(request.url)
            return httpx.Response(200, json={"access_token": "abc", "expires_in": 60})

        c = httpx.Client(transport=httpx.MockTransport(handler))
        client = OAuthClient(github_pat_preset("cid"), http_client=c)
        tok = client.finish_authorize(code="ac", code_verifier="ver")
        assert tok.access_token == "abc"
        assert "/login/oauth/access_token" in called["url"]
        c.close()


class TestCredentialStorage:
    def test_save_and_load(self, tmp_path):
        s = CredentialStorage(base_dir=tmp_path)
        s.save("github", OAuthToken(access_token="abc", refresh_token="r"))
        out = s.load("github")
        assert out is not None and out.access_token == "abc"
        assert out.refresh_token == "r"

    def test_load_missing_returns_none(self, tmp_path):
        s = CredentialStorage(base_dir=tmp_path)
        assert s.load("ghost") is None

    def test_list_providers(self, tmp_path):
        s = CredentialStorage(base_dir=tmp_path)
        s.save("a", OAuthToken(access_token="x"))
        s.save("b", OAuthToken(access_token="y"))
        assert s.list_providers() == ["a", "b"]

    def test_remove(self, tmp_path):
        s = CredentialStorage(base_dir=tmp_path)
        s.save("a", OAuthToken(access_token="x"))
        assert s.remove("a") is True
        assert s.remove("a") is False

    def test_file_permissions_restricted(self, tmp_path):
        s = CredentialStorage(base_dir=tmp_path)
        path = s.save("a", OAuthToken(access_token="x"))
        mode = path.stat().st_mode & 0o777
        # On POSIX, expect 0o600 (-rw-------).
        if hasattr(__import__("os"), "geteuid"):
            assert mode == 0o600
