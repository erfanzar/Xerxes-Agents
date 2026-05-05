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
"""Tests for daemon WebSocket gateway bearer-token auth."""

from __future__ import annotations

from xerxes.daemon.gateway import WebSocketGateway


class TestGatewayAuth:
    def test_no_token_means_auth_disabled(self):
        g = WebSocketGateway("127.0.0.1", 0)
        assert g._is_authorized("GET / HTTP/1.1\r\n", {}) is True

    def test_empty_string_token_disables_auth(self):
        g = WebSocketGateway("127.0.0.1", 0, auth_token="")
        assert g._is_authorized("GET / HTTP/1.1\r\n", {}) is True

    def test_authorization_header_accepts_correct_bearer(self):
        g = WebSocketGateway("127.0.0.1", 0, auth_token="s3cret")
        headers = {"authorization": "Bearer s3cret"}
        assert g._is_authorized("GET / HTTP/1.1\r\n", headers) is True

    def test_authorization_header_rejects_wrong_bearer(self):
        g = WebSocketGateway("127.0.0.1", 0, auth_token="s3cret")
        headers = {"authorization": "Bearer wrong"}
        assert g._is_authorized("GET / HTTP/1.1\r\n", headers) is False

    def test_authorization_header_rejects_non_bearer_scheme(self):
        g = WebSocketGateway("127.0.0.1", 0, auth_token="s3cret")
        headers = {"authorization": "Basic czNjcmV0"}
        assert g._is_authorized("GET / HTTP/1.1\r\n", headers) is False

    def test_query_string_token_accepted(self):
        g = WebSocketGateway("127.0.0.1", 0, auth_token="s3cret")
        request = "GET /?token=s3cret HTTP/1.1\r\n"
        assert g._is_authorized(request, {}) is True

    def test_query_string_token_rejected(self):
        g = WebSocketGateway("127.0.0.1", 0, auth_token="s3cret")
        request = "GET /?token=wrong HTTP/1.1\r\n"
        assert g._is_authorized(request, {}) is False

    def test_missing_token_with_auth_required_rejects(self):
        g = WebSocketGateway("127.0.0.1", 0, auth_token="s3cret")
        assert g._is_authorized("GET / HTTP/1.1\r\n", {}) is False

    def test_query_with_other_params(self):
        g = WebSocketGateway("127.0.0.1", 0, auth_token="s3cret")
        request = "GET /?foo=bar&token=s3cret&baz=qux HTTP/1.1\r\n"
        assert g._is_authorized(request, {}) is True
