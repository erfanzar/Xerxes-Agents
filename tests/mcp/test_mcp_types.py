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
"""Tests for xerxes.mcp.types module."""

from xerxes.mcp.types import (
    MCPPrompt,
    MCPResource,
    MCPServerConfig,
    MCPTool,
    MCPTransportType,
)


class TestMCPTransportType:
    def test_values(self):
        assert MCPTransportType.STDIO.value == "stdio"
        assert MCPTransportType.SSE.value == "sse"
        assert MCPTransportType.STREAMABLE_HTTP.value == "streamable_http"

    def test_backwards_compat(self):
        assert MCPTransportType.HTTP.value == "sse"
        assert MCPTransportType.WEBSOCKET.value == "streamable_http"


class TestMCPServerConfig:
    def test_defaults(self):
        config = MCPServerConfig(name="test")
        assert config.name == "test"
        assert config.command is None
        assert config.transport == MCPTransportType.STDIO
        assert config.enabled is True
        assert config.timeout == 30.0

    def test_stdio(self):
        config = MCPServerConfig(name="fs", command="npx", args=["-y", "server"])
        assert config.command == "npx"
        assert len(config.args) == 2

    def test_sse(self):
        config = MCPServerConfig(
            name="remote",
            url="https://example.com/sse",
            transport=MCPTransportType.SSE,
            headers={"Authorization": "Bearer token"},
        )
        assert config.url == "https://example.com/sse"
        assert "Authorization" in config.headers


class TestMCPTool:
    def test_basic(self):
        tool = MCPTool(
            name="read_file",
            description="Read a file",
            input_schema={"type": "object"},
            server_name="fs",
        )
        assert tool.name == "read_file"
        assert tool.server_name == "fs"


class TestMCPResource:
    def test_basic(self):
        resource = MCPResource(
            uri="file:///doc.txt",
            name="Document",
            description="A document",
            mime_type="text/plain",
            server_name="fs",
        )
        assert resource.uri == "file:///doc.txt"
        assert resource.mime_type == "text/plain"

    def test_defaults(self):
        resource = MCPResource(uri="x", name="y", description="z")
        assert resource.mime_type is None
        assert resource.server_name == ""


class TestMCPPrompt:
    def test_basic(self):
        prompt = MCPPrompt(
            name="summarize",
            description="Summarize text",
            arguments=[{"name": "text", "type": "string"}],
            server_name="tools",
        )
        assert prompt.name == "summarize"
        assert len(prompt.arguments) == 1

    def test_defaults(self):
        prompt = MCPPrompt(name="test", description="desc")
        assert prompt.arguments == []
        assert prompt.server_name == ""
