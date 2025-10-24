"""Tests for MCP integration."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from calute.cortex import CortexAgent
from calute.llms import OpenAILLM
from calute.mcp import MCPClient, MCPManager, MCPServerConfig
from calute.mcp.integration import add_mcp_tools_to_agent, mcp_tool_to_calute_function
from calute.mcp.types import MCPPrompt, MCPResource, MCPTool, MCPTransportType


@pytest.fixture
def mock_mcp_config():
    """Create a mock MCP server configuration."""
    return MCPServerConfig(
        name="test_server",
        command="test_command",
        args=["arg1", "arg2"],
        transport=MCPTransportType.STDIO,
        enabled=True,
    )


@pytest.fixture
def mock_mcp_tool():
    """Create a mock MCP tool."""
    return MCPTool(
        name="test_tool",
        description="A test tool",
        input_schema={
            "type": "object",
            "properties": {"param1": {"type": "string"}, "param2": {"type": "integer"}},
        },
        server_name="test_server",
    )


@pytest.fixture
def mock_mcp_resource():
    """Create a mock MCP resource."""
    return MCPResource(
        uri="file:///test/resource",
        name="Test Resource",
        description="A test resource",
        mime_type="text/plain",
        server_name="test_server",
    )


@pytest.fixture
def mock_mcp_prompt():
    """Create a mock MCP prompt."""
    return MCPPrompt(
        name="test_prompt",
        description="A test prompt",
        arguments=[{"name": "arg1", "type": "string"}],
        server_name="test_server",
    )


class TestMCPClient:
    """Tests for MCPClient."""

    @pytest.mark.asyncio
    async def test_client_initialization(self, mock_mcp_config):
        """Test MCP client initialization."""
        client = MCPClient(mock_mcp_config)

        assert client.config == mock_mcp_config
        assert client.process is None
        assert client.session_id is None
        assert not client.connected
        assert client.tools == []
        assert client.resources == []
        assert client.prompts == []

    @pytest.mark.asyncio
    async def test_connect_stdio_failure_no_command(self):
        """Test connection failure when no command is provided."""
        config = MCPServerConfig(
            name="test",
            command=None,
            transport=MCPTransportType.STDIO,
        )
        client = MCPClient(config)

        result = await client.connect()
        assert result is False

    @pytest.mark.asyncio
    async def test_connect_http_not_implemented(self, mock_mcp_config):
        """Test HTTP transport not implemented."""
        mock_mcp_config.transport = MCPTransportType.HTTP
        mock_mcp_config.url = "http://localhost:8000"
        client = MCPClient(mock_mcp_config)

        result = await client.connect()
        assert result is False

    @pytest.mark.asyncio
    async def test_connect_websocket_not_implemented(self, mock_mcp_config):
        """Test WebSocket transport not implemented."""
        mock_mcp_config.transport = MCPTransportType.WEBSOCKET
        mock_mcp_config.url = "ws://localhost:8000"
        client = MCPClient(mock_mcp_config)

        result = await client.connect()
        assert result is False


class TestMCPManager:
    """Tests for MCPManager."""

    @pytest.mark.asyncio
    async def test_manager_initialization(self):
        """Test MCP manager initialization."""
        manager = MCPManager()

        assert manager.servers == {}
        assert manager.logger is not None

    @pytest.mark.asyncio
    async def test_add_server_disabled(self, mock_mcp_config):
        """Test adding a disabled server."""
        manager = MCPManager()
        mock_mcp_config.enabled = False

        result = await manager.add_server(mock_mcp_config)

        assert result is False
        assert mock_mcp_config.name not in manager.servers

    @pytest.mark.asyncio
    async def test_add_server_duplicate(self, mock_mcp_config):
        """Test adding a server with duplicate name."""
        manager = MCPManager()

        # Mock a successful connection
        with patch.object(MCPClient, "connect", return_value=True):
            await manager.add_server(mock_mcp_config)

            # Try to add again
            result = await manager.add_server(mock_mcp_config)

            assert result is False

    @pytest.mark.asyncio
    async def test_remove_server(self, mock_mcp_config):
        """Test removing a server."""
        manager = MCPManager()

        # Mock a client
        mock_client = MagicMock()
        mock_client.disconnect = AsyncMock()
        manager.servers[mock_mcp_config.name] = mock_client

        await manager.remove_server(mock_mcp_config.name)

        assert mock_mcp_config.name not in manager.servers
        mock_client.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_all_tools(self, mock_mcp_tool):
        """Test getting all tools from all servers."""
        manager = MCPManager()

        # Mock clients with tools
        client1 = MagicMock()
        client1.tools = [mock_mcp_tool]
        client2 = MagicMock()
        client2.tools = [
            MCPTool(
                name="tool2",
                description="Tool 2",
                input_schema={},
                server_name="server2",
            )
        ]

        manager.servers["server1"] = client1
        manager.servers["server2"] = client2

        all_tools = manager.get_all_tools()

        assert len(all_tools) == 2
        assert mock_mcp_tool in all_tools

    @pytest.mark.asyncio
    async def test_get_all_resources(self, mock_mcp_resource):
        """Test getting all resources from all servers."""
        manager = MCPManager()

        # Mock client with resources
        mock_client = MagicMock()
        mock_client.resources = [mock_mcp_resource]
        manager.servers["test"] = mock_client

        resources = manager.get_all_resources()

        assert len(resources) == 1
        assert mock_mcp_resource in resources

    @pytest.mark.asyncio
    async def test_get_all_prompts(self, mock_mcp_prompt):
        """Test getting all prompts from all servers."""
        manager = MCPManager()

        # Mock client with prompts
        mock_client = MagicMock()
        mock_client.prompts = [mock_mcp_prompt]
        manager.servers["test"] = mock_client

        prompts = manager.get_all_prompts()

        assert len(prompts) == 1
        assert mock_mcp_prompt in prompts

    @pytest.mark.asyncio
    async def test_call_tool(self, mock_mcp_tool):
        """Test calling a tool through the manager."""
        manager = MCPManager()

        # Mock client
        mock_client = MagicMock()
        mock_client.tools = [mock_mcp_tool]
        mock_client.call_tool = AsyncMock(return_value={"result": "success"})
        manager.servers["test_server"] = mock_client

        result = await manager.call_tool("test_tool", {"param": "value"})

        assert result == {"result": "success"}
        mock_client.call_tool.assert_called_once_with("test_tool", {"param": "value"})

    @pytest.mark.asyncio
    async def test_call_tool_not_found(self):
        """Test calling a non-existent tool."""
        manager = MCPManager()

        with pytest.raises(ValueError, match="Tool .* not found"):
            await manager.call_tool("nonexistent", {})

    @pytest.mark.asyncio
    async def test_list_servers(self):
        """Test listing server names."""
        manager = MCPManager()

        manager.servers["server1"] = MagicMock()
        manager.servers["server2"] = MagicMock()

        servers = manager.list_servers()

        assert len(servers) == 2
        assert "server1" in servers
        assert "server2" in servers

    @pytest.mark.asyncio
    async def test_get_capabilities_summary(self, mock_mcp_tool, mock_mcp_resource):
        """Test getting capabilities summary."""
        manager = MCPManager()

        # Mock client
        mock_client = MagicMock()
        mock_client.tools = [mock_mcp_tool]
        mock_client.resources = [mock_mcp_resource]
        mock_client.prompts = []
        manager.servers["test"] = mock_client

        summary = manager.get_capabilities_summary()

        assert "test" in summary
        assert summary["test"]["tools"] == 1
        assert summary["test"]["resources"] == 1
        assert summary["test"]["prompts"] == 0


class TestMCPIntegration:
    """Tests for MCP integration helpers."""

    def test_mcp_tool_to_calute_function(self, mock_mcp_tool):
        """Test converting MCP tool to Calute function."""
        manager = MagicMock()

        func = mcp_tool_to_calute_function(mock_mcp_tool, manager)

        assert callable(func)
        assert func.__name__ == "test_tool"
        assert "A test tool" in func.__doc__
        assert hasattr(func, "__annotations__")

    @pytest.mark.asyncio
    async def test_add_mcp_tools_to_agent(self, mock_mcp_tool):
        """Test adding MCP tools to an agent."""
        manager = MCPManager()

        # Mock client with tools
        mock_client = MagicMock()
        mock_client.tools = [mock_mcp_tool]
        manager.servers["test_server"] = mock_client

        # Create agent
        agent = CortexAgent(
            role="Test Agent",
            goal="Test",
            backstory="Test agent",
            model="gpt-4",
            llm=MagicMock(spec=OpenAILLM),
        )

        # Add tools
        await add_mcp_tools_to_agent(agent, manager)

        # Verify tools were added
        assert agent.functions is not None
        assert len(agent.functions) > 0

    @pytest.mark.asyncio
    async def test_add_mcp_tools_filtered_by_server(self, mock_mcp_tool):
        """Test adding MCP tools filtered by server name."""
        manager = MCPManager()

        # Mock clients
        client1 = MagicMock()
        client1.tools = [mock_mcp_tool]
        manager.servers["test_server"] = client1

        client2 = MagicMock()
        client2.tools = [
            MCPTool(
                name="other_tool",
                description="Other tool",
                input_schema={},
                server_name="other_server",
            )
        ]
        manager.servers["other_server"] = client2

        # Create agent
        agent = CortexAgent(
            role="Test Agent",
            goal="Test",
            backstory="Test agent",
            model="gpt-4",
            llm=MagicMock(spec=OpenAILLM),
        )

        # Add tools only from test_server
        await add_mcp_tools_to_agent(agent, manager, server_names=["test_server"])

        # Verify only tools from test_server were added
        assert agent.functions is not None
        assert len(agent.functions) == 1


class TestMCPTypes:
    """Tests for MCP type definitions."""

    def test_mcp_server_config_defaults(self):
        """Test MCPServerConfig default values."""
        config = MCPServerConfig(name="test")

        assert config.name == "test"
        assert config.command is None
        assert config.args == []
        assert config.env == {}
        assert config.transport == MCPTransportType.STDIO
        assert config.url is None
        assert config.headers == {}
        assert config.enabled is True

    def test_mcp_tool_creation(self, mock_mcp_tool):
        """Test MCPTool creation."""
        assert mock_mcp_tool.name == "test_tool"
        assert mock_mcp_tool.description == "A test tool"
        assert "properties" in mock_mcp_tool.input_schema
        assert mock_mcp_tool.server_name == "test_server"

    def test_mcp_resource_creation(self, mock_mcp_resource):
        """Test MCPResource creation."""
        assert mock_mcp_resource.uri == "file:///test/resource"
        assert mock_mcp_resource.name == "Test Resource"
        assert mock_mcp_resource.mime_type == "text/plain"
        assert mock_mcp_resource.server_name == "test_server"

    def test_mcp_prompt_creation(self, mock_mcp_prompt):
        """Test MCPPrompt creation."""
        assert mock_mcp_prompt.name == "test_prompt"
        assert mock_mcp_prompt.description == "A test prompt"
        assert len(mock_mcp_prompt.arguments) == 1
        assert mock_mcp_prompt.server_name == "test_server"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
