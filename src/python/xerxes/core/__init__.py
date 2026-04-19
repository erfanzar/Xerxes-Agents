# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# distributed under the License is distributed on an "AS IS" BASIS,
# See the License for the specific language governing permissions and
# limitations under the License.


"""Core building blocks for Xerxes.

Provides prompt template structures, formatting utilities, configuration,
error types, multimodal handling, streaming buffers, and base utilities.

Submodules (use direct imports to avoid circular dependencies):
- xerxes.core.basics
- xerxes.core.config
- xerxes.core.errors
- xerxes.core.multimodal
- xerxes.core.prompt_template
- xerxes.core.streamer_buffer
- xerxes.core.utils
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
    """Lazily import attributes from submodules to avoid circular imports.

    Args:
        name: The attribute name to resolve.

    Returns:
        The resolved attribute from the appropriate submodule.

    Raises:
        AttributeError: If the attribute is not found in any submodule.
    """
    if name in _SUBMODULE_MAP:
        module = _importlib.import_module(_SUBMODULE_MAP[name], __package__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
