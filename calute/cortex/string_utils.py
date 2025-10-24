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


"""String utility functions for template interpolation"""

import re
from typing import Any


def interpolate_inputs(
    input_string: str | None,
    inputs: dict[str, str | int | float | dict[str, Any] | list[Any]],
) -> str:
    """
    Interpolate placeholders (e.g., {key}) in a string with provided values.

    Only interpolates placeholders that follow the pattern {variable_name} where
    variable_name starts with a letter/underscore and contains only letters, numbers, and underscores.

    Args:
        input_string: The string containing template variables to interpolate.
                     Can be None or empty, in which case an empty string is returned.
        inputs: Dictionary mapping template variables to their values.
               Supported value types are strings, integers, floats, and dicts/lists
               containing only these types and other nested dicts/lists.

    Returns:
        The interpolated string with all template variables replaced with their values.
        Empty string if input_string is None or empty.

    Raises:
        KeyError: If a template variable is missing from inputs
        ValueError: If a value contains unsupported types

    Examples:
        >>> interpolate_inputs("Hello {name}!", {"name": "World"})
        "Hello World!"

        >>> interpolate_inputs("Year: {year}, Topic: {topic}", {"year": 2025, "topic": "AI"})
        "Year: 2025, Topic: AI"
    """
    if not input_string:
        return ""

    pattern = r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}"

    def replacer(match):
        key = match.group(1)
        if key not in inputs:
            raise KeyError(f"Missing required template variable '{key}'")

        value = inputs[key]

        if isinstance(value, str | int | float | bool):
            return str(value)
        elif isinstance(value, dict | list):
            import json

            try:
                return json.dumps(value, ensure_ascii=False)
            except (TypeError, ValueError):
                return str(value)
        elif value is None:
            return ""
        else:
            raise ValueError(f"Unsupported type {type(value).__name__} for template variable '{key}'")

    return re.sub(pattern, replacer, input_string)


def extract_template_variables(input_string: str) -> set[str]:
    """
    Extract all template variable names from a string.

    Args:
        input_string: String potentially containing {variable} placeholders

    Returns:
        Set of variable names found in the string

    Example:
        >>> extract_template_variables("Hello {name}, year {year}")
        {'name', 'year'}
    """
    if not input_string:
        return set()

    pattern = r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}"
    return set(re.findall(pattern, input_string))


def validate_inputs_for_template(
    template_string: str, inputs: dict[str, Any], allow_extra: bool = True
) -> tuple[bool, list[str]]:
    """
    Validate that all required template variables are present in inputs.

    Args:
        template_string: String containing template variables
        inputs: Dictionary of provided inputs
        allow_extra: Whether to allow extra keys in inputs not used in template

    Returns:
        Tuple of (is_valid, list_of_missing_or_errors)

    Example:
        >>> validate_inputs_for_template("Hello {name}", {"name": "World"})
        (True, [])

        >>> validate_inputs_for_template("Hello {name}", {})
        (False, ["Missing required variable: name"])
    """
    required_vars = extract_template_variables(template_string)
    provided_keys = set(inputs.keys())

    errors = []

    missing = required_vars - provided_keys
    if missing:
        for var in missing:
            errors.append(f"Missing required variable: {var}")

    if not allow_extra:
        extra = provided_keys - required_vars
        if extra:
            for var in extra:
                errors.append(f"Unexpected variable: {var}")

    return (len(errors) == 0, errors)
