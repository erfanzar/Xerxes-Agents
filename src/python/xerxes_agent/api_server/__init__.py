# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).
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


"""Modular OpenAI-compatible API server for Xerxes agents with Cortex support.

This module provides a FastAPI-based server that exposes Xerxes agents through
HTTP endpoints following the OpenAI API specification. This allows seamless
integration with any OpenAI-compatible client library. The server supports
both single-agent operations and multi-agent orchestration through Cortex.

Key Features:
    - OpenAI-compatible API endpoints (/v1/chat/completions, /v1/models)
    - Support for streaming and non-streaming responses
    - Multi-agent orchestration via Cortex integration
    - Modular router architecture for extensibility
    - Health check and model listing endpoints

Example:
    >>> from xerxes_agent import Xerxes, OpenAILLM
    >>> from xerxes_agent.api_server import XerxesAPIServer
    >>> from xerxes_agent.types import Agent
    >>>
    >>> llm = OpenAILLM(api_key="your-api-key")
    >>> xerxes = Xerxes(client=llm.client)
    >>> agent = Agent(id="assistant", model="gpt-4", instructions="Help users")
    >>> server = XerxesAPIServer(xerxes)
    >>> server.register_agent(agent)
    >>> server.run(port=8000)
"""

from .cortex_completion_service import CortexCompletionService
from .server import XerxesAPIServer

__all__ = ["CortexCompletionService", "XerxesAPIServer"]
