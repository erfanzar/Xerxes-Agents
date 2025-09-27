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


"""Basic utilities and registration system for Calute components.

This module provides a registration system for tracking and managing
different Calute components including clients, agents, and calute instances.
It also includes utility functions for pretty-printing and object serialization.
"""

import pprint
from typing import Any, Literal, TypeVar

CLIENT_REGISTERY: dict[str, Any] = dict()
"""Registry for client instances."""

AGENTS_REGISTERY: dict[str, Any] = dict()
"""Registry for agent instances."""

calute_REGISTERY: dict[str, Any] = dict()
"""Registry for calute instances."""

REGISTERY: dict[str, dict[str, Any]] = {
    "client": CLIENT_REGISTERY,
    "agents": AGENTS_REGISTERY,
    "calute": calute_REGISTERY,
}
"""Master registry containing all component registries."""

T = TypeVar("T")


def _pretty_print(dict_in: dict[str, Any], indent: int = 0) -> str:
    """Helper function for pretty-printing a dictionary.

    Recursively formats a dictionary with proper indentation for
    improved readability.

    Args:
        dict_in: The dictionary to pretty-print.
        indent: The current indentation level (spaces).

    Returns:
        The pretty-printed string representation of the dictionary.

    Example:
        >>> d = {"key1": "value1", "nested": {"key2": "value2"}}
        >>> print(_pretty_print(d))
        key1:
          value1
        nested:
          key2:
            value2
    """
    result = []
    for key, value in dict_in.items():
        result.append(" " * indent + str(key) + ":")
        if isinstance(value, dict):
            result.append(_pretty_print(value, indent + 2))
        else:
            result.append(" " * (indent + 2) + str(value))
    return "\n".join(result)


def basic_registery(
    register_type: Literal["calute", "agents", "client"],
    register_name: str,
) -> callable:
    """Decorator for registering Calute components and adding utility methods.

    This decorator registers a class in the appropriate registry and adds
    utility methods for serialization and string representation.

    Args:
        register_type: The type of component to register ('calute', 'agents', or 'client').
        register_name: The name to register the component under.

    Returns:
        A decorator function that registers the class and adds utility methods.

    Raises:
        AssertionError: If register_type is not one of the valid options.

    Example:
        >>> @basic_registery("agents", "my_agent")
        ... class MyAgent:
        ...     def __init__(self, name):
        ...         self.name = name
        >>>
        >>> agent = MyAgent("test")
        >>> print(agent.to_dict())
        {'name': 'test'}
    """
    assert register_type in ["calute", "agents", "client"], "Unknown Registery!"

    def to_dict(self) -> dict[str, Any]:
        """Convert the class instance into a dictionary.

        Returns:
            A dictionary representation of the instance, excluding private attributes.
        """
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def str_func(self) -> str:
        """Return a formatted string representation of the instance.

        Returns:
            A formatted string showing the instance configuration.
        """
        return f"{self.__class__.__name__}(\n\t" + pprint.pformat(self.to_dict(), indent=2).replace("\n", "\n\t") + "\n)"

    def wraper(obj: T) -> T:
        """Inner wrapper that applies the registration and method additions.

        Args:
            obj: The class to be registered and enhanced.

        Returns:
            The enhanced class with added utility methods.
        """
        obj.to_dict = to_dict
        obj.__str__ = str_func
        obj.__repr__ = str_func
        REGISTERY[register_type][register_name] = obj
        return obj

    return wraper
