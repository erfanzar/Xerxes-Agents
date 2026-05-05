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
"""Basic registries and decorators for Xerxes component discovery.

Provides ``REGISTRY`` and its sub-registries, plus ``basic_registry`` — a
decorator that registers classes and injects ``to_dict`` / ``__str__`` helpers.
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
    """Recursively render a dictionary as an indented string.

    Args:
        dict_in (dict[str, Any]): IN: dictionary to render.
        indent (int): IN: current indentation level in spaces.
            Defaults to 0.

    Returns:
        str: OUT: formatted multi-line string.
    """
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
    """Class decorator that registers a class and adds dict/string helpers.

    Args:
        register_type (Literal["xerxes", "agents", "client"]): IN: which
            sub-registry to store the class in.
        register_name (str): IN: key under which the class is stored.

    Returns:
        Callable[[T], T]: OUT: decorator that mutates and returns the class.
    """
    assert register_type in ["xerxes", "agents", "client"], "Unknown Registry!"

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary of public attributes.

        Returns:
            dict[str, Any]: OUT: attribute names mapped to values, excluding
            private attributes.
        """
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def str_func(self) -> str:
        """Return a pretty-printed string representation.

        Returns:
            str: OUT: formatted representation using ``to_dict``.
        """
        return f"{self.__class__.__name__}(\n\t" + pprint.pformat(self.to_dict(), indent=2).replace("\n", "\n\t") + "\n)"

    def wraper(obj: T) -> T:
        """Wrap the class with helper methods and register it.

        Args:
            obj (T): IN: class to decorate.

        Returns:
            T: OUT: the mutated class.
        """
        any_obj = cast(Any, obj)
        any_obj.to_dict = to_dict
        any_obj.__str__ = str_func
        any_obj.__repr__ = str_func
        REGISTRY[register_type][register_name] = obj
        return obj

    return wraper
