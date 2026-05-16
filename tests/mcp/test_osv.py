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
"""Tests for xerxes.mcp.osv."""

from __future__ import annotations

import httpx
from xerxes.mcp.osv import Vulnerability, check_package, is_blocked


def _client(payload):
    def handler(req):
        return httpx.Response(200, json=payload)

    return httpx.Client(transport=httpx.MockTransport(handler))


class TestCheckPackage:
    def test_returns_vulns_from_response(self):
        client = _client(
            {
                "vulns": [
                    {
                        "id": "GHSA-1234",
                        "summary": "RCE",
                        "database_specific": {"severity": "CRITICAL"},
                    }
                ]
            }
        )
        out = check_package("npm", "foo", "1.0.0", client=client)
        assert len(out) == 1
        assert out[0].id == "GHSA-1234"
        assert out[0].severity == "CRITICAL"
        client.close()

    def test_no_vulns(self):
        client = _client({"vulns": []})
        assert check_package("PyPI", "requests", "2.32.0", client=client) == []
        client.close()

    def test_network_error_returns_empty(self):
        def handler(req):
            raise httpx.RequestError("boom")

        client = httpx.Client(transport=httpx.MockTransport(handler))
        # Doesn't raise — returns empty so we don't block legitimate installs.
        assert check_package("npm", "x", client=client) == []
        client.close()

    def test_cache_returns_same_result(self, tmp_path):
        client = _client({"vulns": [{"id": "GHSA-1"}]})
        first = check_package("npm", "x", "1.0", cache_dir=tmp_path, client=client)
        client.close()
        # Second call: don't pass a client; cache must satisfy it.
        second = check_package("npm", "x", "1.0", cache_dir=tmp_path)
        assert len(second) == 1
        assert second[0].id == first[0].id


class TestIsBlocked:
    def test_malware_advisory_blocks(self):
        assert is_blocked([Vulnerability(id="MAL-2026-001")]) is True

    def test_critical_severity_blocks(self):
        assert is_blocked([Vulnerability(id="GHSA-x", severity="CRITICAL")]) is True

    def test_high_severity_blocks(self):
        assert is_blocked([Vulnerability(id="GHSA-x", severity="HIGH 9.0")]) is True

    def test_moderate_does_not_block(self):
        assert is_blocked([Vulnerability(id="GHSA-x", severity="MODERATE")]) is False

    def test_empty_does_not_block(self):
        assert is_blocked([]) is False
