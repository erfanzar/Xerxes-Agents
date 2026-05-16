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
"""OpenAI-compatible HTTP API in front of Xerxes agents and Cortex.

The server is the third surface Xerxes exposes (alongside the daemon
and bridge). It hosts FastAPI routers that translate OpenAI chat
completion requests into either single-agent runs (:class:`Xerxes`)
or multi-agent Cortex executions, including SSE streaming. Endpoints:

* ``POST /v1/chat/completions`` — sync or streamed completions.
* ``GET  /v1/models`` — list registered agents and Cortex variants.
* ``GET  /health`` — server liveness plus registered agent count.
"""

from .cortex_completion_service import CortexCompletionService
from .server import XerxesAPIServer

__all__ = ["CortexCompletionService", "XerxesAPIServer"]
