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
"""Tests for media tools (image_generate / vision_analyze / text_to_speech)."""

from __future__ import annotations

import base64

import pytest
from xerxes.tools.media_tools import (
    configure_media,
    image_generate,
    set_media_client,
    text_to_speech,
    vision_analyze,
)


class _FakeHTTP:
    def __init__(self, mapping):
        self.mapping = mapping
        self.calls = []

    def post(self, url, json=None, headers=None):
        self.calls.append({"url": url, "json": json, "headers": headers})
        spec = self.mapping[url]
        if isinstance(spec, bytes | bytearray):
            return _BytesResp(bytes(spec))
        return _JSONResp(spec)


class _JSONResp:
    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload

    @property
    def text(self):
        import json as _j

        return _j.dumps(self.payload)


class _BytesResp:
    def __init__(self, data: bytes):
        self.content = data
        self.text = ""


@pytest.fixture
def fake_image():
    img_b64 = base64.b64encode(b"FAKEPNGDATA").decode()
    http = _FakeHTTP(
        {"https://api.test/v1/images/generations": {"data": [{"b64_json": img_b64, "revised_prompt": "fox"}]}}
    )
    configure_media(base_url="https://api.test/v1", api_key="key", image_model="fake-img")
    set_media_client(http)
    yield http
    set_media_client(None)


@pytest.fixture
def fake_chat():
    http = _FakeHTTP(
        {"https://api.test/v1/chat/completions": {"choices": [{"message": {"content": "It's a cat on a mat."}}]}}
    )
    configure_media(base_url="https://api.test/v1", api_key="key", vision_model="fake-vision")
    set_media_client(http)
    yield http
    set_media_client(None)


@pytest.fixture
def fake_audio():
    http = _FakeHTTP({"https://api.test/v1/audio/speech": b"FAKEMP3"})
    configure_media(base_url="https://api.test/v1", api_key="key", tts_model="fake-tts", tts_voice="alloy")
    set_media_client(http)
    yield http
    set_media_client(None)


class TestImageGenerate:
    def test_returns_b64(self, fake_image):
        out = image_generate.static_call(prompt="a fox in autumn")
        assert out["count"] == 1
        assert out["images"][0]["b64"] == base64.b64encode(b"FAKEPNGDATA").decode()
        assert out["model"] == "fake-img"

    def test_passes_size(self, fake_image):
        image_generate.static_call(prompt="x", size="512x512")
        body = fake_image.calls[0]["json"]
        assert body["size"] == "512x512"

    def test_failure_returns_error(self):
        class Broken:
            def post(self, *a, **kw):
                raise RuntimeError("403 forbidden")

        configure_media(base_url="https://api.test/v1", api_key="k")
        set_media_client(Broken())
        out = image_generate.static_call(prompt="x")
        assert "error" in out
        set_media_client(None)


class TestVisionAnalyze:
    def test_uses_url_form(self, fake_chat):
        out = vision_analyze.static_call(image_url="https://x.test/cat.jpg", question="What is this?")
        assert out["answer"] == "It's a cat on a mat."
        body = fake_chat.calls[0]["json"]
        assert body["messages"][0]["content"][1]["image_url"]["url"] == "https://x.test/cat.jpg"

    def test_uses_b64_form(self, fake_chat):
        out = vision_analyze.static_call(image_b64=base64.b64encode(b"PNG").decode())
        body = fake_chat.calls[0]["json"]
        url = body["messages"][0]["content"][1]["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")
        assert out["answer"]

    def test_requires_image(self, fake_chat):
        out = vision_analyze.static_call()
        assert "error" in out


class TestTextToSpeech:
    def test_returns_audio_bytes(self, fake_audio):
        out = text_to_speech.static_call(text="Hello world")
        assert out["bytes"] == len(b"FAKEMP3")
        assert base64.b64decode(out["audio_b64"]) == b"FAKEMP3"
        assert out["voice"] == "alloy"

    def test_voice_override(self, fake_audio):
        text_to_speech.static_call(text="hi", voice="verse")
        body = fake_audio.calls[0]["json"]
        assert body["voice"] == "verse"

    def test_failure_returns_error(self):
        class Broken:
            def post(self, *a, **kw):
                raise RuntimeError("audio engine down")

        configure_media(base_url="https://api.test/v1", api_key="k")
        set_media_client(Broken())
        out = text_to_speech.static_call(text="x")
        assert "error" in out
        set_media_client(None)
