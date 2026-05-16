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
"""Module-level registries and the :func:`basic_registry` decorator.

Three small dicts (``CLIENT_REGISTRY``, ``AGENTS_REGISTRY``,
``XERXES_REGISTRY``) hold legacy components keyed by name. The
:func:`basic_registry` decorator both inserts a class into one of those
registries and mixes in a uniform ``to_dict`` / ``__str__`` / ``__repr__``
that other layers (logging, debug dumps) rely on.
"""

import pprint
from collections.abc import Callable
from typing import Any, Literal, TypeVar, cast

CLIENT_REGISTRY: dict[str, Any] = dict()

AGENTS_REGISTRY: dict[str, Any] = dict()

XERXES_REGISTRY: dict[str, Any] = dict()

REGISTRY: dict[str, dict[str, Any]] = {
    "client": CLIENT_REGISTRY,
    "agents": AGENTS_REGISTRY,
    "xerxes": XERXES_REGISTRY,
}

T = TypeVar("T")


def _pretty_print(dict_in: dict[str, Any], indent: int = 0) -> str:
    """Render a nested dict as an indented multi-line string for debug output."""
    result = []
    for key, value in dict_in.items():
        result.append(" " * indent + str(key) + ":")
        if isinstance(value, dict):
            result.append(_pretty_print(value, indent + 2))
        else:
            result.append(" " * (indent + 2) + str(value))
    return "\n".join(result)


def basic_registry(
    register_type: Literal["xerxes", "agents", "client"],
    register_name: str,
) -> Callable[[T], T]:
    """Decorate a class to register it under ``register_type[register_name]``.

    Also attaches uniform ``to_dict`` / ``__str__`` / ``__repr__`` helpers.
    """
    assert register_type in ["xerxes", "agents", "client"], "Unknown Registry!"

    def to_dict(self) -> dict[str, Any]:
        """Return public (non-underscore) instance attributes as a dict."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def str_func(self) -> str:
        """Return an indented ``ClassName(...)`` representation built from :func:`to_dict`."""
        return f"{self.__class__.__name__}(\n\t" + pprint.pformat(self.to_dict(), indent=2).replace("\n", "\n\t") + "\n)"

    def wraper(obj: T) -> T:
        """Mutate ``obj`` with the helper methods and register it; returns ``obj``."""
        any_obj = cast(Any, obj)
        any_obj.to_dict = to_dict
        any_obj.__str__ = str_func
        any_obj.__repr__ = str_func
        REGISTRY[register_type][register_name] = obj
        return obj

    return wraper
