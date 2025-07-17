# Copyright 2023 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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
"""Defines Pydantic models for the vSurge API, mimicking OpenAI's structure."""

import typing as tp

from pydantic import BaseModel

from .function_execution_types import AgentCapability, AgentSwitchTrigger, FunctionCallStrategy

AgentFunction = tp.Callable[[], tp.Union[str, "Agent", dict]]  # type:ignore


class Agent(BaseModel):
    """Agent with function calling and switching capabilities"""

    model: str
    id: str | None = None
    name: str | None = None
    instructions: str | tp.Callable[[], str] = "You are a helpful agent."
    rules: list[str] | tp.Callable[[], list[str]] | None = None
    examples: list[str] | None = None
    functions: list[tp.Callable] = []
    capabilities: list[AgentCapability] = []

    function_call_strategy: FunctionCallStrategy = FunctionCallStrategy.SEQUENTIAL
    tool_choice: str | list[str] = None
    parallel_tool_calls: bool = True
    function_timeout: float | None = 30.0
    max_function_retries: int = 3

    top_p: float = 0.95
    max_tokens: int = 2048
    temperature: float = 0.7
    top_k: int = 0
    min_p: float = 0.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0

    stop: str | list[str] | None = None

    switch_triggers: list[AgentSwitchTrigger] = []
    fallback_agent_id: str | None = None

    def add_capability(self, capability: AgentCapability):
        """Add a capability to the agent"""
        self.capabilities.append(capability)

    def has_capability(self, capability_name: str) -> bool:
        """Check if agent has a specific capability"""
        return any(cap.name == capability_name for cap in self.capabilities)

    def get_available_functions(self) -> list[str]:
        """Get list of available function names"""
        return [func.__name__ for func in self.functions]


class Response(BaseModel):
    """
    Represents a response from an agent or a snapshot of the interaction state,
    containing messages and contextual information.

        Attributes:
            messages (tp.List):
                A list of messages representing the conversation history up to this response.
                Each message is typically a dictionary or a dedicated message object
                (e.g., `{'role': 'user', 'content': '...'}`,
                `{'role': 'assistant', 'content': '...', 'tool_calls': [...]}`).
                Defaults to an empty list.
            agent (tp.Optional[Agent]):
                An optional reference to the `Agent` instance that generated this response
                or is associated with this state. This can be useful for understanding
                the configuration that led to the response. Defaults to None.
            context_variables (dict):
                A dictionary for storing arbitrary contextual variables related to this
                response or the ongoing conversation state. Useful for carrying state
                between turns, logging, or passing information to downstream processes.
                Defaults to an empty dictionary.
    """

    messages: list = []
    agent: Agent | None = None
    context_variables: dict = {}


class Result(BaseModel):
    """
    Encapsulates the possible return values for an agent function.

    Attributes:
        value (str): The result value as a string.
        agent (Agent): The agent instance, if applicable.
        context_variables (dict): A dictionary of context variables.
    """

    value: str = ""
    agent: Agent | None = None
    context_variables: dict = {}


__all__ = "Agent", "AgentFunction", "Result"
