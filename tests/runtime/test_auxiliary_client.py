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
"""Tests for xerxes.runtime.auxiliary_client."""

from __future__ import annotations

import pytest
from xerxes.runtime.auxiliary_client import AuxiliaryClient, AuxiliaryRequest


class TestAuxiliaryClient:
    def _client(self, response: str = "ok"):
        captured = {}

        def backend(messages, max_tokens, model):
            captured["messages"] = messages
            captured["max_tokens"] = max_tokens
            captured["model"] = model
            return response

        return AuxiliaryClient(backend, model="test-aux-model"), captured

    def test_default_model(self) -> None:
        client, _ = self._client()
        assert client.model == "test-aux-model"

    def test_call_returns_response(self) -> None:
        client, captured = self._client("summary text")
        req = AuxiliaryRequest(purpose="x", messages=[{"role": "user", "content": "hi"}])
        resp = client.call(req)
        assert resp.text == "summary text"
        assert resp.purpose == "x"
        assert resp.model == "test-aux-model"
        assert resp.duration_ms >= 0.0
        assert captured["model"] == "test-aux-model"

    def test_summarize_uses_summary_instruction(self) -> None:
        client, captured = self._client("S")
        out = client.summarize([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "there"}])
        assert out == "S"
        system_msg = captured["messages"][0]
        assert system_msg["role"] == "system"
        assert "Summarize" in system_msg["content"]
        # User content has both messages joined.
        user_msg = captured["messages"][1]
        assert "[user] hi" in user_msg["content"]
        assert "[assistant] there" in user_msg["content"]

    def test_summarize_budget_override(self) -> None:
        client, captured = self._client("S")
        client.summarize([{"role": "user", "content": "hi"}], budget_tokens=42)
        assert captured["max_tokens"] == 42

    def test_title_strips_quotes(self) -> None:
        client, _ = self._client('"My Cool Title"')
        out = client.title([{"role": "user", "content": "hi"}])
        assert out == "My Cool Title"

    def test_extract_uses_instruction_as_system(self) -> None:
        client, captured = self._client("extracted")
        out = client.extract("body text", instruction="Extract emails")
        assert out == "extracted"
        assert captured["messages"][0]["content"] == "Extract emails"
        assert captured["messages"][1]["content"] == "body text"

    def test_backend_exception_propagates(self) -> None:
        def backend(*a, **kw):
            raise RuntimeError("nope")

        client = AuxiliaryClient(backend, model="m")
        with pytest.raises(RuntimeError, match="nope"):
            client.summarize([{"role": "user", "content": "x"}])
