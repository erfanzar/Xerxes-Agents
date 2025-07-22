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

import pprint
from typing import Any, Literal, TypeVar

CLIENT_REGISTERY = dict()
AGENTS_REGISTERY = dict()
calute_REGISTERY = dict()

REGISTERY = {
    "client": CLIENT_REGISTERY,
    "agents": AGENTS_REGISTERY,
    "calute": calute_REGISTERY,
}

T = TypeVar("T")


def _pretty_print(dict_in: dict[str, Any], indent: int = 0) -> str:
    """
    Helper function for pretty-printing a dictionary.

    Args:
                d (Dict[str, Any]): The dictionary to pretty-print.
                indent (int): The indentation level.

    Returns:
                str: The pretty-printed string representation of the dictionary.
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
    register_name,
):
    assert register_type in ["calute", "agents", "client"], "Unknown Registery!"

    def to_dict(self) -> dict[str, Any]:
        """
        Converts the Class object into a dictionary.

        Returns:
            Dict[str, Any]: A dictionary representation of the TrainingArguments.
        """
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def str_func(self):
        """
        Returns a formatted string representation of the configuration.

        Returns:
            str: A formatted string showing the configuration settings.
        """
        return f"{self.__class__.__name__}(\n\t" + pprint.pformat(self.to_dict(), indent=2).replace("\n", "\n\t") + "\n)"

    def wraper(obj: T) -> T:
        obj.to_dict = to_dict
        obj.__str__ = str_func
        obj.__repr__ = str_func
        REGISTERY[register_type][register_name] = obj
        return obj

    return wraper
