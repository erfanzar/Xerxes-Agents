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

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest  # type:ignore

from calute import Agent, Calute, MemoryStore
from calute.client import OpenAIClient
from calute.types import AgentCapability


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client for testing."""
    client = Mock(spec=OpenAIClient)
    client.generate_completion = AsyncMock()
    client.extract_content = Mock(return_value="Test response")
    client.process_streaming_response = AsyncMock(return_value="Streamed response")
    return client


@pytest.fixture
def sample_agent():
    """Create a sample agent for testing."""

    def sample_function(text: str) -> str:
        """A sample function for testing."""
        return f"Processed: {text}"

    return Agent(
        id="test_agent",
        name="Test Agent",
        model="gpt-4",
        instructions="You are a test agent.",
        functions=[sample_function],
        capabilities=[AgentCapability.FUNCTION_CALLING],
    )


@pytest.fixture
def calute_instance(mock_openai_client):
    """Create a Calute instance with mock client."""
    return Calute(client=mock_openai_client, enable_memory=True)


@pytest.fixture
def memory_store():
    """Create a memory store for testing."""
    return MemoryStore(max_short_term=10, max_working=5)


@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset any singleton instances between tests."""
    # Add any singleton reset logic here
    yield
