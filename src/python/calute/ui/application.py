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
for the Chainlit-based chat interface. It includes:
- ChainlitLauncher class with Gradio-compatible .launch() API
- Chainlit lifecycle handlers (on_chat_start, on_message, etc.)
- Chat profile management for multi-agent selection
- Settings management for runtime configuration
- MCP (Model Context Protocol) connection handlers

The module supports dual operation modes: as a library import for
application setup, and as a Chainlit module when loaded by the
Chainlit runtime.

Example:
    >>> from calute.ui import launch_application
    >>> from calute import Calute
    >>> calute = Calute(llm=my_llm, agents=[my_agent])
    >>> app = launch_application(executor=calute)
    >>> app.launch(server_port=8000)
"""

from __future__ import annotations

import builtins
import os
import sys
import typing
from pathlib import Path

try:
    from .themes import APP_TITLE, setup_chainlit_theme
except ImportError:
    from calute.ui.themes import APP_TITLE, setup_chainlit_theme

if typing.TYPE_CHECKING:
    from calute.calute import Calute
    from calute.cortex import Cortex, CortexAgent, CortexTask
    from calute.cortex.orchestration.dynamic import DynamicCortex
    from calute.cortex.orchestration.task_creator import TaskCreator
    from calute.types.agent_types import Agent

_EXECUTOR_CONFIG_KEY = "_calute_executor_config"


def _get_executor_config() -> dict:
    """Get executor config from builtins (survives module reimport).

    Retrieves the executor configuration stored in builtins. This
    mechanism is used because Chainlit reimports this module,
    which would lose normal module-level variables.

    Returns:
        Dictionary with 'executor' and 'agent' keys.
    """
    return getattr(builtins, _EXECUTOR_CONFIG_KEY, {"executor": None, "agent": None})


def _set_executor_config(executor, agent) -> None:
    """Set executor config in builtins.

    Stores the executor configuration in builtins so it persists
    across module reimports by Chainlit's load_module().

    Args:
        executor: The Calute executor instance to store.
        agent: The agent configuration to store.
    """
    setattr(builtins, _EXECUTOR_CONFIG_KEY, {"executor": executor, "agent": agent})


class ChainlitLauncher:
    """Wrapper class to provide Gradio-like .launch() API for Chainlit.

    This class enables backward compatibility with existing code that expects
    a .launch() method to be called on the UI object. It handles the setup
    of theme configuration and Chainlit server initialization.

    Attributes:
        executor: The Calute executor instance for processing conversations.
        agent: Optional agent configuration for specialized behavior.
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

        Starts the Chainlit server with the configured executor and theme.
        This method blocks until the server is stopped (e.g., by Ctrl+C).

        Args:
            server_name: Host to bind the server to (default: "localhost").
            server_port: Port to run the server on (default: 8000).
            **kwargs: Additional arguments (watch, headless) - currently unused.

        Returns:
            None - runs the server until stopped.

        Note:
            The method sets up theme files and environment variables before
            starting Chainlit. The executor config is stored in builtins to
            survive module reimport by Chainlit.
        """
        _set_executor_config(self.executor, self.agent)

        setup_chainlit_theme()

        os.environ["CHAINLIT_HOST"] = server_name
        os.environ["CHAINLIT_PORT"] = str(server_port)

        from chainlit.cli import run_chainlit

        module_path = str(Path(__file__).resolve())

        run_chainlit(module_path)


def launch_application(
    executor: Calute | CortexAgent | CortexTask | Cortex | TaskCreator | DynamicCortex,
    agent: Agent | Cortex | DynamicCortex | None = None,
    server_name: str = "localhost",
    server_port: int = 8000,
    **kwargs,
):
    """Launch the Chainlit application with the given executor.

    Factory function that creates a ChainlitLauncher configured with
    the given executor. Returns a launcher object for deferred launch,
    allowing the caller to customize server settings before starting.

    Args:
        executor: Calute instance or Cortex component for managing conversations.
            Supported types include Calute, CortexAgent, CortexTask, Cortex,
            TaskCreator, and DynamicCortex.
        agent: Optional agent configuration for specialized behavior. Can be
            an Agent instance, Cortex, DynamicCortex, or agent ID string.
        server_name: Host to bind the server to (passed to launcher).
        server_port: Port to run the server on (passed to launcher).
        **kwargs: Additional arguments passed to the launcher.

    Returns:
        ChainlitLauncher object with .launch() method for starting the server.

    Example:
        >>> app = launch_application(executor=calute, agent=my_agent)
        >>> app.launch(server_port=3000)  # Start on custom port
    """
    return ChainlitLauncher(executor=executor, agent=agent)


if "chainlit" in sys.modules or os.environ.get("CHAINLIT_ROOT_PATH"):
    import chainlit as cl
    from chainlit.input_widget import Slider, Switch

    from calute.types.messages import MessagesHistory

    try:
        from .helpers import process_message_chainlit
    except ImportError:
        from calute.ui.helpers import process_message_chainlit

    DEFAULT_SETTINGS = {
        "temperature": 0.7,
        "max_tokens": 8192,
        "top_p": 0.95,
        "top_k": 0,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
        "streaming": True,
    }

    def _get_agents_from_executor(executor) -> dict:
        """Extract registered agents from executor.

        Supports multiple executor patterns for agent discovery:
        - Calute's orchestrator.agents
        - Direct _agents attribute
        - Cortex agents list/dict

        Args:
            executor: The executor instance to extract agents from.

        Returns:
            Dictionary mapping agent IDs to agent objects.
        """
        if hasattr(executor, "orchestrator") and hasattr(executor.orchestrator, "agents"):
            return executor.orchestrator.agents
        if hasattr(executor, "_agents"):
            return executor._agents
        if hasattr(executor, "agents"):
            agents = executor.agents
            if isinstance(agents, dict):
                return agents
            return {a.id if hasattr(a, "id") else str(i): a for i, a in enumerate(agents)}
        return {}

    def _apply_settings_to_agent(executor, agent_id, settings: dict) -> None:
        """Apply settings to the active agent.

        Updates agent parameters based on UI settings. Supports
        sampling and penalty parameters.

        Args:
            executor: The executor instance containing agents.
            agent_id: ID of the agent to configure, or None for default.
            settings: Dictionary of settings from ChatSettings widget.
        """
        agents = _get_agents_from_executor(executor)
        agent = agents.get(agent_id) if agent_id else None

        if agent is None and hasattr(executor, "default_agent"):
            agent = executor.default_agent

        if agent is None:
            return

        if hasattr(agent, "temperature"):
            agent.temperature = settings.get("temperature", DEFAULT_SETTINGS["temperature"])
        if hasattr(agent, "max_tokens"):
            agent.max_tokens = int(settings.get("max_tokens", DEFAULT_SETTINGS["max_tokens"]))
        if hasattr(agent, "top_p"):
            agent.top_p = settings.get("top_p", DEFAULT_SETTINGS["top_p"])
        if hasattr(agent, "top_k"):
            agent.top_k = int(settings.get("top_k", DEFAULT_SETTINGS["top_k"]))
        if hasattr(agent, "min_p"):
            agent.min_p = settings.get("min_p", DEFAULT_SETTINGS["min_p"])
        if hasattr(agent, "presence_penalty"):
            agent.presence_penalty = settings.get("presence_penalty", DEFAULT_SETTINGS["presence_penalty"])
        if hasattr(agent, "frequency_penalty"):
            agent.frequency_penalty = settings.get("frequency_penalty", DEFAULT_SETTINGS["frequency_penalty"])
        if hasattr(agent, "repetition_penalty"):
            agent.repetition_penalty = settings.get("repetition_penalty", DEFAULT_SETTINGS["repetition_penalty"])

    def _build_initial_settings(executor, agent_id) -> dict:
        """Build UI settings from defaults overridden by the active agent.

        Constructs the initial settings dictionary by starting with DEFAULT_SETTINGS
        and then overriding values with attributes from the active agent (if found).
        This ensures the UI sliders reflect the agent's current configuration when
        a chat session starts.

        Args:
            executor: The executor instance containing registered agents.
            agent_id: ID of the currently selected agent, or None to use the
                default agent.

        Returns:
            Dictionary of settings values suitable for initializing ChatSettings
            widgets. Keys include 'temperature', 'max_tokens', 'top_p', 'top_k',
            'min_p', 'presence_penalty', 'frequency_penalty', 'repetition_penalty',
            and 'streaming'.
        """
        settings = dict(DEFAULT_SETTINGS)
        agents = _get_agents_from_executor(executor)
        agent = agents.get(agent_id) if agent_id else None

        if agent is None and hasattr(executor, "default_agent"):
            agent = executor.default_agent

        if agent is None:
            return settings

        for key in (
            "temperature",
            "max_tokens",
            "top_p",
            "top_k",
            "min_p",
            "presence_penalty",
            "frequency_penalty",
            "repetition_penalty",
        ):
            value = getattr(agent, key, None)
            if value is not None:
                settings[key] = value

        return settings

    @cl.set_chat_profiles
    async def set_chat_profiles():
        """Define available chat profiles based on registered agents.

        Creates a "Default" profile plus one profile for each registered
        agent in the executor. Profiles allow users to select which agent
        to interact with from the Chainlit UI.

        Returns:
            List of ChatProfile objects for the profile selector.
        """
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
        """Initialize session state when a new chat starts.

        Sets up the user session with:
        - Empty message history
        - Executor reference from config
        - Selected agent based on chat profile
        - ChatSettings widgets for runtime configuration
        - Welcome message with profile info
        """
        config = _get_executor_config()
        executor = config["executor"]

        cl.user_session.set("calute_msgs", MessagesHistory(messages=[]))

        cl.user_session.set("executor", executor)

        profile = cl.user_session.get("chat_profile")
        selected_agent = config["agent"]

        if profile and profile != "Default" and executor:
            agents = _get_agents_from_executor(executor)
            for agent_id, agent in agents.items():
                agent_name = getattr(agent, "name", None) or agent_id
                if agent_name == profile:
                    selected_agent = agent_id
                    break

        cl.user_session.set("agent", selected_agent)

        initial_settings = _build_initial_settings(executor, selected_agent)

        settings = await cl.ChatSettings(
            [
                Slider(
                    id="temperature",
                    label="Temperature",
                    initial=initial_settings["temperature"],
                    min=0,
                    max=2,
                    step=0.1,
                    description="Controls randomness in responses",
                ),
                Slider(
                    id="max_tokens",
                    label="Max Tokens",
                    initial=initial_settings["max_tokens"],
                    min=-1,
                    max=131072,
                    step=1024,
                    description="Maximum length of response (-1 for unlimited)",
                ),
                Slider(
                    id="top_p",
                    label="Top P",
                    initial=initial_settings["top_p"],
                    min=0,
                    max=1,
                    step=0.05,
                    description="Nucleus sampling threshold",
                ),
                Slider(
                    id="top_k",
                    label="Top K",
                    initial=initial_settings["top_k"],
                    min=0,
                    max=200,
                    step=1,
                    description="Limit sampling to the top K candidate tokens",
                ),
                Slider(
                    id="min_p",
                    label="Min P",
                    initial=initial_settings["min_p"],
                    min=0,
                    max=1,
                    step=0.01,
                    description="Minimum probability threshold for sampling",
                ),
                Slider(
                    id="presence_penalty",
                    label="Presence Penalty",
                    initial=initial_settings["presence_penalty"],
                    min=-2,
                    max=2,
                    step=0.1,
                    description="Penalize tokens that have already appeared",
                ),
                Slider(
                    id="frequency_penalty",
                    label="Frequency Penalty",
                    initial=initial_settings["frequency_penalty"],
                    min=-2,
                    max=2,
                    step=0.1,
                    description="Penalize repeated token frequency",
                ),
                Slider(
                    id="repetition_penalty",
                    label="Repetition Penalty",
                    initial=initial_settings["repetition_penalty"],
                    min=0.5,
                    max=2,
                    step=0.05,
                    description="Multiplicative repetition penalty",
                ),
                Switch(
                    id="streaming",
                    label="Enable Streaming",
                    initial=initial_settings["streaming"],
                    description="Stream responses token by token",
                ),
            ]
        ).send()

        cl.user_session.set("settings", settings)

        _apply_settings_to_agent(executor, selected_agent, settings)

        profile_msg = f" (Profile: **{profile}**)" if profile and profile != "Default" else ""
        await cl.Message(
            content=f"Welcome to **{APP_TITLE}**{profile_msg}! How can I help you today?",
            author="system",
        ).send()

    @cl.on_settings_update
    async def on_settings_update(settings: dict):
        """Handle settings updates from the UI.

        Called when users modify ChatSettings widgets. Updates
        the session settings and applies them to the active agent.

        Args:
            settings: Dictionary of updated settings values.
        """
        cl.user_session.set("settings", settings)

        executor = cl.user_session.get("executor")
        agent_id = cl.user_session.get("agent")

        if executor:
            _apply_settings_to_agent(executor, agent_id, settings)

    @cl.action_callback("regenerate")
    async def on_regenerate(action: cl.Action):
        """Regenerate the last assistant response.

        Removes the last assistant message and re-processes the
        last user message to generate a fresh response.

        Args:
            action: The Chainlit action that triggered this callback.
        """
        calute_msgs = cl.user_session.get("calute_msgs")

        if not calute_msgs or len(calute_msgs.messages) < 2:
            await cl.Message(content="Nothing to regenerate.").send()
            return

        calute_msgs.messages.pop()
        last_user_msg = calute_msgs.messages[-1].content

        executor = cl.user_session.get("executor")
        agent = cl.user_session.get("agent")

        updated_msgs = await process_message_chainlit(
            message=last_user_msg,
            calute_msgs=calute_msgs,
            executor=executor,
            agent=agent,
        )

        cl.user_session.set("calute_msgs", updated_msgs)

    @cl.action_callback("clear_history")
    async def on_clear_history(action: cl.Action):
        """Clear the conversation history.

        Resets the message history to an empty state and notifies
        the user.

        Args:
            action: The Chainlit action that triggered this callback.
        """
        cl.user_session.set("calute_msgs", MessagesHistory(messages=[]))
        await cl.Message(content="Conversation history cleared.", author="system").send()

    @cl.on_message
    async def on_message(message: cl.Message):
        """Handle incoming user messages.

        Processes user messages through the configured executor with
        streaming support. Applies current settings, processes the
        message, and updates the session history.

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

        _apply_settings_to_agent(executor, agent, settings)

        updated_msgs = await process_message_chainlit(
            message=message.content,
            calute_msgs=calute_msgs,
            agent=agent,
            executor=executor,
        )

        cl.user_session.set("calute_msgs", updated_msgs)

    @cl.on_stop
    async def on_stop():
        """Handle when the user stops a response generation.

        Called when the user clicks the stop button during streaming.
        Cleans up any active streaming buffers and signals cancellation.
        """
        executor = cl.user_session.get("executor")

        if executor and hasattr(executor, "streamer_buffer"):
            buf = executor.streamer_buffer
            if buf and hasattr(buf, "kill"):
                buf.kill()

        streamer = cl.user_session.get("active_streamer")
        if streamer and hasattr(streamer, "kill"):
            streamer.kill()
            cl.user_session.set("active_streamer", None)

    @cl.on_chat_end
    async def on_chat_end():
        """Handle when a chat session ends.

        Cleans up all session state including message history,
        executor reference, agent configuration, and MCP tools.
        """
        cl.user_session.set("calute_msgs", None)
        cl.user_session.set("executor", None)
        cl.user_session.set("agent", None)
        cl.user_session.set("settings", None)
        cl.user_session.set("mcp_tools", None)

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

            mcp_tools = cl.user_session.get("mcp_tools") or {}
            mcp_tools[connection.name] = {"tools": tools, "session": session}
            cl.user_session.set("mcp_tools", mcp_tools)

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
        mcp_tools = cl.user_session.get("mcp_tools") or {}
        if name in mcp_tools:
            del mcp_tools[name]
            cl.user_session.set("mcp_tools", mcp_tools)

        await cl.Message(
            content=f"Disconnected from MCP server **{name}**",
            author="system",
        ).send()
