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
"""Tests for the channel framework — registry, identity, OAuth, webhooks."""

from __future__ import annotations

import asyncio

import pytest
from xerxes.channels import (
    Channel,
    ChannelMessage,
    ChannelRegistry,
    IdentityResolver,
    OAuthClient,
    OAuthProvider,
    OAuthToken,
    WebhookDispatcher,
    WebhookResponse,
)
from xerxes.memory import SimpleStorage


class DummyChannel(Channel):
    name = "dummy"

    def __init__(self):
        self.started = False
        self.sent: list[ChannelMessage] = []
        self.handler = None

    async def start(self, on_inbound):
        self.started = True
        self.handler = on_inbound

    async def stop(self):
        self.started = False

    async def send(self, message):
        self.sent.append(message)


class TestRegistry:
    def test_register_get_unregister(self):
        r = ChannelRegistry()
        c = DummyChannel()
        r.register("dummy", c)
        assert r.get("dummy") is c
        assert r.names() == ["dummy"]
        r.unregister("dummy")
        assert r.get("dummy") is None

    def test_start_all_requires_handler(self):
        r = ChannelRegistry()
        r.register("d", DummyChannel())
        with pytest.raises(RuntimeError):
            asyncio.run(r.start_all())

    def test_start_stop_lifecycle(self):
        r = ChannelRegistry()
        c = DummyChannel()
        r.register("d", c)
        r.set_handler(lambda msg: asyncio.sleep(0))
        asyncio.run(r.start_all())
        assert c.started is True
        asyncio.run(r.stop_all())
        assert c.started is False

    def test_send_routes_by_message_channel(self):
        r = ChannelRegistry()
        c = DummyChannel()
        r.register("dummy", c)
        msg = ChannelMessage(text="hi", channel="dummy")
        asyncio.run(r.send(msg))
        assert c.sent == [msg]

    def test_send_unknown_raises(self):
        r = ChannelRegistry()
        with pytest.raises(KeyError):
            asyncio.run(r.send(ChannelMessage(text="x", channel="nope")))


class TestIdentityResolver:
    def test_creates_stable_user_id(self):
        r = IdentityResolver()
        a = r.resolve("telegram", "12345", display_name="Alice")
        b = r.resolve("telegram", "12345")
        assert a.user_id == b.user_id

    def test_different_channel_users_get_different_ids(self):
        r = IdentityResolver()
        a = r.resolve("telegram", "1")
        b = r.resolve("slack", "U1")
        assert a.user_id != b.user_id

    def test_link_unifies_existing_to_new(self):
        r = IdentityResolver()
        u = r.resolve("telegram", "1").user_id
        r.link(u, "slack", "U1")
        assert r.get("slack", "U1").user_id == u

    def test_persistence_round_trip(self):
        store = SimpleStorage()
        r1 = IdentityResolver(storage=store)
        rec = r1.resolve("telegram", "1", display_name="Alice")
        r2 = IdentityResolver(storage=store)
        assert r2.get("telegram", "1").user_id == rec.user_id

    def test_channels_for_lists_all(self):
        r = IdentityResolver()
        u = r.resolve("telegram", "1").user_id
        r.link(u, "slack", "U1")
        recs = r.channels_for(u)
        assert {x.channel for x in recs} == {"telegram", "slack"}


class TestOAuthClient:
    def _provider(self):
        return OAuthProvider(
            name="slack",
            client_id="cid",
            client_secret="csec",
            authorize_url="https://slack.example/oauth/v2/authorize",
            token_url="https://slack.example/oauth/v2/access",
            scopes=["chat:write", "users:read"],
            redirect_uri="https://example.com/cb",
        )

    def test_authorize_url_includes_state(self):
        c = OAuthClient(self._provider())
        url, state = c.authorize_url()
        assert "state=" in url
        assert "client_id=cid" in url
        assert "scope=chat%3Awrite+users%3Aread" in url
        assert state and len(state) > 8

    def test_consume_state_validates_and_invalidates(self):
        c = OAuthClient(self._provider())
        _, state = c.authorize_url()
        assert c.consume_state(state) is True
        assert c.consume_state(state) is False  # one-shot

    def test_exchange_code_with_injected_http(self):
        store = SimpleStorage()

        def fake_http(url, data):
            assert data["code"] == "abc"
            return {
                "access_token": "tok",
                "refresh_token": "rt",
                "expires_in": 3600,
                "scope": "chat:write users:read",
            }

        c = OAuthClient(self._provider(), storage=store, http_client=fake_http)
        token = c.exchange_code(code="abc", install_id="t1")
        assert token.access_token == "tok"
        assert token.refresh_token == "rt"
        assert "chat:write" in token.scopes
        # Persisted:
        assert c.get_token("t1").access_token == "tok"

    def test_state_mismatch_rejects(self):
        c = OAuthClient(self._provider())
        with pytest.raises(ValueError):
            c.exchange_code(code="x", state_received="a", expected_state="b")

    def test_get_valid_token_refreshes(self):
        store = SimpleStorage()
        seen_grants: list[str] = []

        def fake_http(url, data):
            seen_grants.append(data["grant_type"])
            return {
                "access_token": "fresh",
                "refresh_token": "rt2",
                "expires_in": 3600,
            }

        c = OAuthClient(self._provider(), storage=store, http_client=fake_http)
        # Plant an expired token
        old = OAuthToken(provider="slack", access_token="old", refresh_token="rt", expires_at=1.0)
        store.save(c._store_key("default"), old.to_dict())
        new = c.get_valid_token()
        assert new.access_token == "fresh"
        assert "refresh_token" in seen_grants


class TestWebhookDispatcher:
    def test_unknown_returns_404(self):
        d = WebhookDispatcher()
        resp = asyncio.run(d.dispatch("nope", {}, b""))
        assert resp.status == 404

    def test_handler_dispatch(self):
        d = WebhookDispatcher()

        async def handler(headers, body):
            assert headers["x-test"] == "1"
            return WebhookResponse(status=200, body=body.decode())

        d.register("svc", handler)
        resp = asyncio.run(d.dispatch("svc", {"x-test": "1"}, b"ok"))
        assert resp.status == 200
        assert resp.body == "ok"

    def test_handler_exception_becomes_500(self):
        d = WebhookDispatcher()

        async def boom(headers, body):
            raise RuntimeError("kaboom")

        d.register("svc", boom)
        resp = asyncio.run(d.dispatch("svc", {}, b""))
        assert resp.status == 500
