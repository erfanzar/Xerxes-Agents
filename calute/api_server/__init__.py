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


"""Modular OpenAI-compatible API server for Calute agents with Cortex support.

This module provides a FastAPI-based server that exposes Calute agents through
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
    >>> from calute import Calute, OpenAILLM
    >>> from calute.api_server import CaluteAPIServer
    >>> from calute.types import Agent
    >>>
    >>> llm = OpenAILLM(api_key="your-api-key")
    >>> calute = Calute(client=llm.client)
    >>> agent = Agent(id="assistant", model="gpt-4", instructions="Help users")
    >>> server = CaluteAPIServer(calute)
    >>> server.register_agent(agent)
    >>> server.run(port=8000)
"""

from .cortex_completion_service import CortexCompletionService
from .server import CaluteAPIServer

__all__ = ["CaluteAPIServer", "CortexCompletionService"]
