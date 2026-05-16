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
"""Foundational utilities shared by every Xerxes subsystem.

This package gathers the cross-cutting primitives the rest of the codebase
imports: filesystem path helpers (``paths``), the Pydantic-based config
model (``config``), the exception hierarchy (``errors``), the
``XerxesBase`` strict model + tool-schema reflection (``utils``), the
thread-safe streaming response buffer (``streamer_buffer``), multimodal
image helpers (``multimodal``), and the registry decorators used to wire
modules into the agent kernel (``basics``). Names are exposed via a lazy
``__getattr__`` so importing :mod:`xerxes.core` itself stays cheap.
"""

import importlib as _importlib

from .prompt_template import PromptSection, PromptTemplate

__all__ = [
    "AGENTS_REGISTRY",
    "CLIENT_REGISTRY",
    "KILL_TAG",
    "REGISTRY",
    "XERXES_REGISTRY",
    "AgentError",
    "ClientError",
    "ConfigurationError",
    "FunctionExecutionError",
    "PromptSection",
    "PromptTemplate",
    "RateLimitError",
    "SerializableImage",
    "StreamerBuffer",
    "ValidationError",
    "XerxesBase",
    "XerxesConfig",
    "XerxesError",
    "XerxesMemoryError",
    "XerxesTimeoutError",
    "_pretty_print",
    "basic_registry",
    "debug_print",
    "function_to_json",
    "get_config",
    "load_config",
    "run_sync",
    "set_config",
]

_SUBMODULE_MAP = {
    "AGENTS_REGISTRY": ".basics",
    "XERXES_REGISTRY": ".basics",
    "CLIENT_REGISTRY": ".basics",
    "REGISTRY": ".basics",
    "_pretty_print": ".basics",
    "basic_registry": ".basics",
    "XerxesConfig": ".config",
    "get_config": ".config",
    "load_config": ".config",
    "set_config": ".config",
    "AgentError": ".errors",
    "XerxesError": ".errors",
    "XerxesMemoryError": ".errors",
    "XerxesTimeoutError": ".errors",
    "ClientError": ".errors",
    "ConfigurationError": ".errors",
    "FunctionExecutionError": ".errors",
    "RateLimitError": ".errors",
    "ValidationError": ".errors",
    "SerializableImage": ".multimodal",
    "KILL_TAG": ".streamer_buffer",
    "StreamerBuffer": ".streamer_buffer",
    "XerxesBase": ".utils",
    "debug_print": ".utils",
    "function_to_json": ".utils",
    "run_sync": ".utils",
}


def __getattr__(name: str) -> object:
    """Resolve an exported name to its owning submodule on first access.

    Raises ``AttributeError`` for names outside :data:`_SUBMODULE_MAP`.
    """
    if name in _SUBMODULE_MAP:
        module = _importlib.import_module(_SUBMODULE_MAP[name], __package__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
