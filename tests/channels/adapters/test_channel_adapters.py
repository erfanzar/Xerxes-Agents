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
"""Smoke + integration tests for every Hermes channel adapter."""

from __future__ import annotations

import asyncio
import json

import pytest
from xerxes.channels import ChannelMessage, MessageDirection
from xerxes.channels.adapters import (
    BlueBubblesChannel,
    DingTalkChannel,
    DiscordChannel,
    EmailChannel,
    FeishuChannel,
    HomeAssistantChannel,
    MatrixChannel,
    MattermostChannel,
    SignalChannel,
    SlackChannel,
    TelegramChannel,
    TwilioSMSChannel,
    WeComChannel,
    WhatsAppChannel,
)


class CapturingHTTP:
    def __init__(self, response=None):
        self.calls: list[dict] = []
        self.response = response if response is not None else {"ok": True}

    def __call__(self, url, json=None, headers=None, data=None):
        self.calls.append({"url": url, "json": json, "headers": headers, "data": data})
        return self.response


def _start(channel):
    received = []

    async def handler(msg):
        received.append(msg)

    asyncio.run(channel.start(handler))
    return received


def _post(channel, headers, body):
    return asyncio.run(channel.handle_webhook(headers, body))


class TestTelegram:
    def test_inbound(self):
        c = TelegramChannel("TOK")
        rx = _start(c)
        body = json.dumps(
            {
                "message": {
                    "message_id": 7,
                    "chat": {"id": 42, "type": "private"},
                    "from": {"id": 99, "username": "u"},
                    "text": "hi",
                }
            }
        ).encode()
        resp = _post(c, {}, body)
        assert resp.status == 200
        assert rx[0].text == "hi"
        assert rx[0].room_id == "42"
        assert rx[0].channel_user_id == "99"
        assert rx[0].direction == MessageDirection.INBOUND

    def test_outbound(self):
        http = CapturingHTTP()
        c = TelegramChannel("TOK", http_client=http)
        asyncio.run(c.send(ChannelMessage(text="ok", channel="telegram", room_id="42")))
        assert http.calls[0]["json"]["chat_id"] == "42"
        assert "sendMessage" in http.calls[0]["url"]


class TestDiscord:
    def test_inbound(self):
        c = DiscordChannel("BOT")
        rx = _start(c)
        body = json.dumps(
            {"id": "1", "channel_id": "C1", "content": "hi", "author": {"id": "U1"}, "guild_id": "G"}
        ).encode()
        _post(c, {}, body)
        assert rx[0].text == "hi" and rx[0].room_id == "C1" and rx[0].channel_user_id == "U1"

    def test_outbound_uses_bot_auth(self):
        http = CapturingHTTP()
        c = DiscordChannel("BOT", http_client=http)
        asyncio.run(c.send(ChannelMessage(text="hi", channel="discord", room_id="C1")))
        assert http.calls[0]["headers"]["Authorization"] == "Bot BOT"


class TestSlack:
    def test_url_verification_handshake_returns_no_messages(self):
        c = SlackChannel("xoxb-tok")
        rx = _start(c)
        body = json.dumps({"type": "url_verification", "challenge": "x"}).encode()
        _post(c, {}, body)
        assert rx == []

    def test_inbound_message_event(self):
        c = SlackChannel("xoxb-tok")
        rx = _start(c)
        body = json.dumps(
            {"team_id": "T", "event": {"type": "message", "user": "U", "channel": "C", "ts": "1.0", "text": "hi"}}
        ).encode()
        _post(c, {}, body)
        assert rx[0].text == "hi"

    def test_inbound_skips_bot_messages(self):
        c = SlackChannel("xoxb-tok")
        rx = _start(c)
        body = json.dumps(
            {"event": {"type": "message", "user": "U", "channel": "C", "ts": "1.0", "text": "hi", "bot_id": "B"}}
        ).encode()
        _post(c, {}, body)
        assert rx == []

    def test_outbound_uses_bearer(self):
        http = CapturingHTTP()
        c = SlackChannel("xoxb-tok", http_client=http)
        asyncio.run(c.send(ChannelMessage(text="hi", channel="slack", room_id="C")))
        assert http.calls[0]["headers"]["Authorization"] == "Bearer xoxb-tok"


class TestEmail:
    def test_inbound(self):
        c = EmailChannel(from_address="bot@ex.com")
        rx = _start(c)
        body = json.dumps({"from": "u@x", "to": "bot@ex.com", "subject": "Hi", "text": "body"}).encode()
        _post(c, {}, body)
        assert rx[0].text == "body"
        assert rx[0].metadata["subject"] == "Hi"

    def test_outbound_uses_injected_sender(self):
        sent = []

        def fake(f, t, s, b):
            sent.append((f, t, s, b))

        c = EmailChannel(from_address="bot@ex.com", smtp_sender=fake)
        asyncio.run(c.send(ChannelMessage(text="reply", channel="email", room_id="user@x")))
        assert sent == [("bot@ex.com", "user@x", "Re:", "reply")]


class TestMatrix:
    def test_inbound(self):
        c = MatrixChannel("https://m.example", "tok")
        rx = _start(c)
        body = json.dumps(
            {
                "events": [
                    {
                        "type": "m.room.message",
                        "sender": "@a:m",
                        "room_id": "!r:m",
                        "event_id": "$e",
                        "content": {"body": "hi"},
                    }
                ]
            }
        ).encode()
        _post(c, {}, body)
        assert rx[0].text == "hi" and rx[0].room_id == "!r:m"

    def test_outbound(self):
        http = CapturingHTTP()
        c = MatrixChannel("https://m.example", "tok", http_client=http)
        asyncio.run(c.send(ChannelMessage(text="hi", channel="matrix", room_id="!r:m")))
        assert "/rooms/!r:m/send/m.room.message/" in http.calls[0]["url"]
        assert http.calls[0]["json"]["msgtype"] == "m.text"


class TestMattermost:
    def test_inbound(self):
        c = MattermostChannel("https://mm.example", "tok")
        rx = _start(c)
        body = json.dumps({"text": "hi", "user_id": "U", "channel_id": "C", "post_id": "P", "team_id": "T"}).encode()
        _post(c, {}, body)
        assert rx[0].text == "hi"

    def test_outbound(self):
        http = CapturingHTTP()
        c = MattermostChannel("https://mm.example", "tok", http_client=http)
        asyncio.run(c.send(ChannelMessage(text="hi", channel="mattermost", room_id="C")))
        assert http.calls[0]["json"]["channel_id"] == "C"


class TestSMS:
    def test_inbound_form_encoded(self):
        c = TwilioSMSChannel("AC", "tok", "+15550000000")
        rx = _start(c)
        body = b"From=%2B15551234567&To=%2B15550000000&Body=hi&MessageSid=SM"
        _post(c, {}, body)
        assert rx[0].text == "hi"
        assert rx[0].channel_user_id == "+15551234567"

    def test_outbound(self):
        http = CapturingHTTP()
        c = TwilioSMSChannel("AC", "tok", "+15550000000", http_client=http)
        asyncio.run(c.send(ChannelMessage(text="hi", channel="sms", room_id="+1")))
        assert "twilio.com" in http.calls[0]["url"]


class TestWhatsApp:
    def test_inbound(self):
        c = WhatsAppChannel("TOK", "PHONE_ID")
        rx = _start(c)
        body = json.dumps(
            {"entry": [{"changes": [{"value": {"messages": [{"id": "wamid", "from": "+1", "text": {"body": "hi"}}]}}]}]}
        ).encode()
        _post(c, {}, body)
        assert rx[0].text == "hi" and rx[0].channel_user_id == "+1"

    def test_outbound(self):
        http = CapturingHTTP()
        c = WhatsAppChannel("TOK", "PHONE_ID", http_client=http)
        asyncio.run(c.send(ChannelMessage(text="hi", channel="whatsapp", room_id="+1")))
        body = http.calls[0]["json"]
        assert body["messaging_product"] == "whatsapp"
        assert body["to"] == "+1"


class TestSignal:
    def test_inbound(self):
        c = SignalChannel("http://signal-cli", "+15550000000")
        rx = _start(c)
        body = json.dumps(
            {"envelope": {"sourceNumber": "+1", "timestamp": 1, "dataMessage": {"message": "hi"}}}
        ).encode()
        _post(c, {}, body)
        assert rx[0].text == "hi"

    def test_outbound(self):
        http = CapturingHTTP()
        c = SignalChannel("http://signal-cli", "+1", http_client=http)
        asyncio.run(c.send(ChannelMessage(text="hi", channel="signal", room_id="+2")))
        body = http.calls[0]["json"]
        assert body["recipients"] == ["+2"]


class TestDingTalk:
    def test_inbound(self):
        c = DingTalkChannel("https://oapi.dingtalk.example/robot/send")
        rx = _start(c)
        body = json.dumps({"text": {"content": "hi"}, "senderId": "U", "conversationId": "C", "msgId": "M"}).encode()
        _post(c, {}, body)
        assert rx[0].text == "hi"

    def test_outbound(self):
        http = CapturingHTTP()
        c = DingTalkChannel("https://wh", http_client=http)
        asyncio.run(c.send(ChannelMessage(text="hi", channel="dingtalk")))
        assert http.calls[0]["json"]["msgtype"] == "text"


class TestFeishu:
    def test_inbound(self):
        c = FeishuChannel("tok")
        rx = _start(c)
        body = json.dumps(
            {
                "event": {
                    "sender": {"sender_id": {"open_id": "ou_1"}},
                    "message": {"chat_id": "oc_1", "message_id": "om_1", "content": json.dumps({"text": "hi"})},
                }
            }
        ).encode()
        _post(c, {}, body)
        assert rx[0].text == "hi"
        assert rx[0].room_id == "oc_1"

    def test_outbound_uses_token_provider(self):
        http = CapturingHTTP()
        c = FeishuChannel(token_provider=lambda: "fresh", http_client=http)
        asyncio.run(c.send(ChannelMessage(text="hi", channel="feishu", room_id="oc_1")))
        assert http.calls[0]["headers"]["Authorization"] == "Bearer fresh"


class TestWeCom:
    def test_inbound(self):
        c = WeComChannel("tok", agent_id=1000)
        rx = _start(c)
        body = json.dumps({"Content": "hi", "FromUserName": "U", "MsgId": "M"}).encode()
        _post(c, {}, body)
        assert rx[0].text == "hi"

    def test_outbound(self):
        http = CapturingHTTP()
        c = WeComChannel("tok", agent_id=1000, http_client=http)
        asyncio.run(c.send(ChannelMessage(text="hi", channel="wecom", channel_user_id="U")))
        assert http.calls[0]["json"]["touser"] == "U"


class TestBlueBubbles:
    def test_inbound(self):
        c = BlueBubblesChannel("http://bb", "secret")
        rx = _start(c)
        body = json.dumps(
            {"data": {"text": "hi", "guid": "g", "handle": {"address": "+1"}, "chats": [{"guid": "gg"}]}}
        ).encode()
        _post(c, {}, body)
        assert rx[0].text == "hi" and rx[0].room_id == "gg"

    def test_outbound(self):
        http = CapturingHTTP()
        c = BlueBubblesChannel("http://bb", "secret", http_client=http)
        asyncio.run(c.send(ChannelMessage(text="hi", channel="bluebubbles", room_id="gg")))
        assert "password=secret" in http.calls[0]["url"]


class TestHomeAssistant:
    def test_inbound(self):
        c = HomeAssistantChannel("http://ha", "tok")
        rx = _start(c)
        body = json.dumps({"text": "turn off the lights"}).encode()
        _post(c, {}, body)
        assert "lights" in rx[0].text

    def test_outbound_persistent_notification(self):
        http = CapturingHTTP()
        c = HomeAssistantChannel("http://ha", "tok", http_client=http)
        asyncio.run(c.send(ChannelMessage(text="hi", channel="home_assistant")))
        assert "/persistent_notification/create" in http.calls[0]["url"]


@pytest.mark.parametrize(
    "channel",
    [
        TelegramChannel("T"),
        DiscordChannel("T"),
        SlackChannel("T"),
        EmailChannel(),
        MatrixChannel("http://m", "T"),
        MattermostChannel("http://m", "T"),
        TwilioSMSChannel("AC", "T", "+1"),
        WhatsAppChannel("T", "P"),
        SignalChannel("http://s", "+1"),
        DingTalkChannel("http://d"),
        FeishuChannel("T"),
        WeComChannel("T"),
        BlueBubblesChannel("http://b", "p"),
        HomeAssistantChannel("http://h", "T"),
    ],
)
class TestAdapterContract:
    def test_returns_503_when_not_started(self, channel):
        resp = asyncio.run(channel.handle_webhook({}, b"{}"))
        assert resp.status == 503

    def test_garbage_inbound_returns_200_with_no_messages(self, channel):
        rx = _start(channel)
        resp = asyncio.run(channel.handle_webhook({}, b"this is not json"))
        assert resp.status in (200, 400)

        if resp.status == 200:
            assert rx == []
