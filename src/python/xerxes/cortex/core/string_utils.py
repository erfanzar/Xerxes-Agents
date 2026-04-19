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


"""String utility functions for template interpolation.

This module provides utilities for working with template strings that use
simple placeholder syntax (e.g., {variable_name}). It includes:
- Template variable interpolation with type-safe value substitution
- Template variable extraction for validation
- Input validation against template requirements

Unlike the Jinja2-based PromptTemplate class, these utilities use a simpler
placeholder syntax that is compatible with Python's str.format() but with
additional type safety and validation features.

Example:
    >>> from xerxes.cortex.string_utils import interpolate_inputs
    >>> result = interpolate_inputs(
    ...     "Hello {name}, you have {count} messages.",
    ...     {"name": "Alice", "count": 5}
    ... )
    >>> print(result)
    Hello Alice, you have 5 messages.

    >>> from xerxes.cortex.string_utils import validate_inputs_for_template
    >>> is_valid, errors = validate_inputs_for_template(
    ...     "Hello {name}",
    ...     {"name": "World"}
    ... )
    >>> print(is_valid)
    True
"""

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

    def replacer(match) -> str:
        """Replace a single regex match with its corresponding input value.

        Args:
            match: A regex match object whose group(1) is the placeholder name.

        Returns:
            String representation of the input value for the matched key.

        Raises:
            KeyError: If the matched key is not present in inputs.
            ValueError: If the value type is not supported for serialization.
        """
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
    """Extract all template variable names from a string.

    Scans the input string for placeholders following the ``{variable_name}``
    pattern and returns a set of all unique variable names found. Variable
    names must start with a letter or underscore and contain only alphanumeric
    characters and underscores.

    Args:
        input_string: String potentially containing ``{variable}`` placeholders.
            If None or empty, returns an empty set.

    Returns:
        Set of unique variable name strings found in the template. Returns
        an empty set if no placeholders are found or input is empty.

    Example:
        >>> extract_template_variables("Hello {name}, year {year}")
        {'name', 'year'}
        >>> extract_template_variables("No placeholders here")
        set()
        >>> extract_template_variables("")
        set()
    """
    if not input_string:
        return set()

    pattern = r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}"
    return set(re.findall(pattern, input_string))


def validate_inputs_for_template(
    template_string: str, inputs: dict[str, Any], allow_extra: bool = True
) -> tuple[bool, list[str]]:
    """Validate that all required template variables are present in inputs.

    Checks whether the provided inputs dictionary contains values for every
    template variable found in the template string. Optionally validates
    that no extra (unused) keys exist in the inputs.

    Args:
        template_string: String containing ``{variable}`` template placeholders
            to validate against.
        inputs: Dictionary of provided input values. Keys should correspond
            to template variable names.
        allow_extra: Whether to allow extra keys in inputs that are not
            referenced in the template. When False, extra keys generate
            error messages. Defaults to True.

    Returns:
        Tuple of (is_valid, errors) where:
        - is_valid: True if all required variables are present and no
          disallowed extra keys exist.
        - errors: List of error message strings. Empty list if valid.
          Messages follow the format "Missing required variable: {name}"
          or "Unexpected variable: {name}".

    Example:
        >>> validate_inputs_for_template("Hello {name}", {"name": "World"})
        (True, [])
        >>> validate_inputs_for_template("Hello {name}", {})
        (False, ["Missing required variable: name"])
        >>> validate_inputs_for_template("{x}", {"x": 1, "y": 2}, allow_extra=False)
        (False, ["Unexpected variable: y"])
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
