# Copyright 2025 The EasyDeL/Calute Author @erfanzar (Erfan Zare Chavoshi).
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


"""Chainlit application entry point for Calute UI.

This module provides the main application setup and message handling
for the Chainlit-based chat interface.
"""

from __future__ import annotations

import builtins
import os
import sys
import typing
from pathlib import Path

# Handle both package import and direct run by chainlit
try:
    from .themes import APP_TITLE, setup_chainlit_theme
except ImportError:
    from calute.ui.themes import APP_TITLE, setup_chainlit_theme

if typing.TYPE_CHECKING:
    from calute.calute import Calute
    from calute.cortex import Cortex, CortexAgent, CortexTask
    from calute.cortex.dynamic import DynamicCortex
    from calute.cortex.task_creator import TaskCreator
    from calute.types.agent_types import Agent

# Use builtins to store executor config - this survives module reimport
# when Chainlit's load_module() reimports this file
_EXECUTOR_CONFIG_KEY = "_calute_executor_config"


def _get_executor_config() -> dict:
    """Get executor config from builtins (survives module reimport)."""
    return getattr(builtins, _EXECUTOR_CONFIG_KEY, {"executor": None, "agent": None})


def _set_executor_config(executor, agent) -> None:
    """Set executor config in builtins."""
    setattr(builtins, _EXECUTOR_CONFIG_KEY, {"executor": executor, "agent": agent})


class ChainlitLauncher:
    """Wrapper class to provide Gradio-like .launch() API for Chainlit.

    This class enables backward compatibility with existing code that expects
    a .launch() method to be called on the UI object.
    """

    def __init__(
        self,
        executor: Calute | CortexAgent | CortexTask | Cortex | TaskCreator | DynamicCortex,
        agent: Agent | Cortex | DynamicCortex | None = None,
    ):
        """Initialize the launcher with executor and agent.

        Args:
            executor: Calute instance or Cortex component for managing conversations.
            agent: Optional agent configuration for specialized behavior.
        """
        self.executor = executor
        self.agent = agent

    def launch(
        self,
        server_name: str = "localhost",
        server_port: int = 8000,
        **kwargs,
    ):
        """Launch the Chainlit application.

        Args:
            server_name: Host to bind the server to.
            server_port: Port to run the server on.
            **kwargs: Additional arguments (watch, headless).

        Returns:
            None - runs the server until stopped.
        """
        # Set executor config in builtins BEFORE chainlit loads this module
        # This survives module reimport by chainlit's load_module
        _set_executor_config(self.executor, self.agent)

        # Setup theme configuration files BEFORE chainlit is imported
        setup_chainlit_theme()

        # Set environment variables for chainlit
        os.environ["CHAINLIT_HOST"] = server_name
        os.environ["CHAINLIT_PORT"] = str(server_port)

        # Import chainlit CLI and run
        from chainlit.cli import run_chainlit

        # Get the path to this module for Chainlit to run
        module_path = str(Path(__file__).resolve())

        # Run chainlit in same process
        run_chainlit(module_path)


def launch_application(
    executor: Calute | CortexAgent | CortexTask | Cortex | TaskCreator | DynamicCortex,
    agent: Agent | Cortex | DynamicCortex | None = None,
    server_name: str = "localhost",
    server_port: int = 8000,
    **kwargs,
):
    """Launch the Chainlit application with the given executor.

    This function can be called directly to launch the application, or it returns
    a ChainlitLauncher object that provides a .launch() method for backward
    compatibility with Gradio-style API.

    Args:
        executor: Calute instance or Cortex component for managing conversations.
        agent: Optional agent configuration for specialized behavior.
        server_name: Host to bind the server to (only used if called directly).
        server_port: Port to run the server on (only used if called directly).
        **kwargs: Additional arguments passed to Chainlit.

    Returns:
        ChainlitLauncher object with .launch() method.
    """
    return ChainlitLauncher(executor=executor, agent=agent)


# Chainlit handlers - only registered when chainlit loads this module
if "chainlit" in sys.modules or os.environ.get("CHAINLIT_ROOT_PATH"):
    import chainlit as cl
    from chainlit.input_widget import Slider, Switch

    from calute.types.messages import MessagesHistory

    try:
        from .helpers import process_message_chainlit
    except ImportError:
        from calute.ui.helpers import process_message_chainlit

    # Default settings values
    DEFAULT_SETTINGS = {
        "temperature": 0.7,
        "max_tokens": 8192,
        "top_p": 0.95,
        "streaming": True,
    }

    def _get_agents_from_executor(executor) -> dict:
        """Extract registered agents from executor."""
        # Check for Calute's orchestrator.agents pattern
        if hasattr(executor, "orchestrator") and hasattr(executor.orchestrator, "agents"):
            return executor.orchestrator.agents
        # Check for direct _agents attribute
        if hasattr(executor, "_agents"):
            return executor._agents
        # Check for agents list (Cortex pattern)
        if hasattr(executor, "agents"):
            agents = executor.agents
            if isinstance(agents, dict):
                return agents
            return {a.id if hasattr(a, "id") else str(i): a for i, a in enumerate(agents)}
        return {}

    def _apply_settings_to_agent(executor, agent_id, settings: dict) -> None:
        """Apply settings to the active agent."""
        agents = _get_agents_from_executor(executor)
        agent = agents.get(agent_id) if agent_id else None

        # Try to get default agent if no specific agent
        if agent is None and hasattr(executor, "default_agent"):
            agent = executor.default_agent

        if agent is None:
            return

        # Apply settings to agent
        if hasattr(agent, "temperature"):
            agent.temperature = settings.get("temperature", DEFAULT_SETTINGS["temperature"])
        if hasattr(agent, "max_tokens"):
            agent.max_tokens = int(settings.get("max_tokens", DEFAULT_SETTINGS["max_tokens"]))
        if hasattr(agent, "top_p"):
            agent.top_p = settings.get("top_p", DEFAULT_SETTINGS["top_p"])

    @cl.set_chat_profiles
    async def set_chat_profiles():
        """Define available chat profiles based on registered agents."""
        config = _get_executor_config()
        executor = config.get("executor")

        profiles = [
            cl.ChatProfile(
                name="Default",
                markdown_description="Default conversation mode with automatic agent selection",
            )
        ]

        if executor:
            agents = _get_agents_from_executor(executor)
            for agent_id, agent in agents.items():
                name = getattr(agent, "name", None) or agent_id
                instructions = getattr(agent, "instructions", "") or ""
                if callable(instructions):
                    instructions = "Custom agent with dynamic instructions"
                else:
                    instructions = str(instructions)[:150] + "..." if len(str(instructions)) > 150 else str(instructions)

                profiles.append(
                    cl.ChatProfile(
                        name=name,
                        markdown_description=instructions or f"Agent: {agent_id}",
                    )
                )

        return profiles

    @cl.on_chat_start
    async def on_chat_start():
        """Initialize session state when a new chat starts."""
        # Get executor config from builtins (set before chainlit loaded this module)
        config = _get_executor_config()
        executor = config["executor"]

        # Initialize message history
        cl.user_session.set("calute_msgs", MessagesHistory(messages=[]))

        # Store executor in session
        cl.user_session.set("executor", executor)

        # Handle chat profile selection
        profile = cl.user_session.get("chat_profile")
        selected_agent = config["agent"]  # Default from config

        if profile and profile != "Default" and executor:
            agents = _get_agents_from_executor(executor)
            for agent_id, agent in agents.items():
                agent_name = getattr(agent, "name", None) or agent_id
                if agent_name == profile:
                    selected_agent = agent_id
                    break

        cl.user_session.set("agent", selected_agent)

        # Setup ChatSettings
        settings = await cl.ChatSettings(
            [
                Slider(
                    id="temperature",
                    label="Temperature",
                    initial=DEFAULT_SETTINGS["temperature"],
                    min=0,
                    max=2,
                    step=0.1,
                    description="Controls randomness in responses",
                ),
                Slider(
                    id="max_tokens",
                    label="Max Tokens",
                    initial=DEFAULT_SETTINGS["max_tokens"],
                    min=-1,
                    max=131072,
                    step=1024,
                    description="Maximum length of response (-1 for unlimited)",
                ),
                Slider(
                    id="top_p",
                    label="Top P",
                    initial=DEFAULT_SETTINGS["top_p"],
                    min=0,
                    max=1,
                    step=0.05,
                    description="Nucleus sampling threshold",
                ),
                Switch(
                    id="streaming",
                    label="Enable Streaming",
                    initial=DEFAULT_SETTINGS["streaming"],
                    description="Stream responses token by token",
                ),
            ]
        ).send()

        cl.user_session.set("settings", settings)

        # Apply initial settings
        _apply_settings_to_agent(executor, selected_agent, settings)

        # Send welcome message with profile info
        profile_msg = f" (Profile: **{profile}**)" if profile and profile != "Default" else ""
        await cl.Message(
            content=f"Welcome to **{APP_TITLE}**{profile_msg}! How can I help you today?",
            author="system",
        ).send()

    @cl.on_settings_update
    async def on_settings_update(settings: dict):
        """Handle settings updates from the UI."""
        cl.user_session.set("settings", settings)

        executor = cl.user_session.get("executor")
        agent_id = cl.user_session.get("agent")

        if executor:
            _apply_settings_to_agent(executor, agent_id, settings)

    @cl.action_callback("regenerate")
    async def on_regenerate(action: cl.Action):
        """Regenerate the last assistant response."""
        calute_msgs = cl.user_session.get("calute_msgs")

        if not calute_msgs or len(calute_msgs.messages) < 2:
            await cl.Message(content="Nothing to regenerate.").send()
            return

        # Remove last assistant message
        calute_msgs.messages.pop()
        last_user_msg = calute_msgs.messages[-1].content

        executor = cl.user_session.get("executor")
        agent = cl.user_session.get("agent")

        # Re-process the last user message
        updated_msgs = await process_message_chainlit(
            message=last_user_msg,
            calute_msgs=calute_msgs,
            executor=executor,
            agent=agent,
        )

        cl.user_session.set("calute_msgs", updated_msgs)

    @cl.action_callback("clear_history")
    async def on_clear_history(action: cl.Action):
        """Clear the conversation history."""
        cl.user_session.set("calute_msgs", MessagesHistory(messages=[]))
        await cl.Message(content="Conversation history cleared.", author="system").send()

    @cl.on_message
    async def on_message(message: cl.Message):
        """Handle incoming user messages.

        Args:
            message: The incoming Chainlit message from the user.
        """
        executor = cl.user_session.get("executor")
        agent = cl.user_session.get("agent")
        calute_msgs = cl.user_session.get("calute_msgs")
        settings = cl.user_session.get("settings") or DEFAULT_SETTINGS

        if not message.content.strip():
            return

        if executor is None:
            await cl.Message(content="Error: No executor configured. Please restart the application.").send()
            return

        # Apply current settings before processing
        _apply_settings_to_agent(executor, agent, settings)

        # Process the message with streaming
        updated_msgs = await process_message_chainlit(
            message=message.content,
            calute_msgs=calute_msgs,
            agent=agent,
            executor=executor,
        )

        # Update session state
        cl.user_session.set("calute_msgs", updated_msgs)

    @cl.on_stop
    async def on_stop():
        """Handle when the user stops a response generation."""
        # Clean up any pending operations if needed
        pass

    @cl.on_chat_end
    async def on_chat_end():
        """Handle when a chat session ends."""
        cl.user_session.set("calute_msgs", None)
        cl.user_session.set("executor", None)
        cl.user_session.set("agent", None)
        cl.user_session.set("settings", None)
        cl.user_session.set("mcp_tools", None)

    # MCP (Model Context Protocol) handlers
    @cl.on_mcp_connect
    async def on_mcp_connect(connection, session):
        """Handle MCP server connection.

        This is called when a user connects to an MCP server through
        the Chainlit UI. It discovers available tools from the connected
        server and stores them in the session for use during chat.

        Args:
            connection: The MCP connection configuration
            session: The MCP ClientSession for interacting with the server
        """
        try:
            # Discover tools from the connected MCP server
            result = await session.list_tools()
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description or "",
                        "parameters": t.inputSchema if hasattr(t, "inputSchema") else {},
                    },
                }
                for t in result.tools
            ]

            # Store tools in session, keyed by connection name
            mcp_tools = cl.user_session.get("mcp_tools") or {}
            mcp_tools[connection.name] = {"tools": tools, "session": session}
            cl.user_session.set("mcp_tools", mcp_tools)

            # Notify user of connected tools
            tool_names = [t["function"]["name"] for t in tools]
            await cl.Message(
                content=f"Connected to MCP server **{connection.name}** with {len(tools)} tools: {', '.join(tool_names[:5])}{'...' if len(tool_names) > 5 else ''}",
                author="system",
            ).send()

        except Exception as e:
            await cl.Message(
                content=f"Error connecting to MCP server {connection.name}: {e}",
                author="system",
            ).send()

    @cl.on_mcp_disconnect
    async def on_mcp_disconnect(name: str, session):
        """Handle MCP server disconnection.

        Args:
            name: The name of the disconnected MCP server
            session: The MCP ClientSession being disconnected
        """
        # Remove tools from session
        mcp_tools = cl.user_session.get("mcp_tools") or {}
        if name in mcp_tools:
            del mcp_tools[name]
            cl.user_session.set("mcp_tools", mcp_tools)

        await cl.Message(
            content=f"Disconnected from MCP server **{name}**",
            author="system",
        ).send()
