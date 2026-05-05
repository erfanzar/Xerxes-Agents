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
"""String interpolation and template validation utilities."""

import re
from typing import Any


def interpolate_inputs(
    input_string: str | None,
    inputs: dict[str, str | int | float | dict[str, Any] | list[Any]],
) -> str:
    """Interpolate template variables into a string.

    Replaces ``{variable_name}`` placeholders in *input_string* with values
    from *inputs*. Supports scalar values, dicts, and lists (serialized as JSON).

    Args:
        input_string (str | None): The template string containing ``{var}``
            placeholders. IN: A template with named placeholders or ``None``.
            OUT: Parsed to extract placeholder keys for substitution.
        inputs (dict): Mapping of variable names to replacement values.
            IN: Keys must match placeholders in *input_string*; values may be
            ``str``, ``int``, ``float``, ``bool``, ``dict``, or ``list``.
            OUT: Values are converted to strings and substituted into the template.

    Returns:
        str: The interpolated string with all placeholders replaced.
            OUT: A fully resolved string ready for use.

    Raises:
        KeyError: If a placeholder key is missing from *inputs*.
        ValueError: If a value has an unsupported type.
    """

    if not input_string:
        return ""

    pattern = r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}"

    def replacer(match) -> str:
        """Replace a single regex match with the corresponding input value.

        Args:
            match: A regex ``Match`` object for a ``{key}`` placeholder.
                IN: Contains the placeholder key as group 1.
                OUT: Used to look up the key in *inputs* and return its string form.

        Returns:
            str: The string representation of the matched input value.
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

    Args:
        input_string (str): The template string to scan.
            IN: A string potentially containing ``{var}`` placeholders.
            OUT: Parsed to identify all unique placeholder keys.

    Returns:
        set[str]: A set of unique variable names found in the template.
            OUT: Empty set if no placeholders exist.
    """

    if not input_string:
        return set()

    pattern = r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}"
    return set(re.findall(pattern, input_string))


def validate_inputs_for_template(
    template_string: str, inputs: dict[str, Any], allow_extra: bool = True
) -> tuple[bool, list[str]]:
    """Validate that inputs satisfy the requirements of a template.

    Args:
        template_string (str): The template string to validate against.
            IN: Contains ``{var}`` placeholders defining required variables.
            OUT: Scanned to determine required variable names.
        inputs (dict): The candidate inputs to check.
            IN: Mapping of variable names to values provided by the caller.
            OUT: Compared against required variables to detect missing or extra keys.
        allow_extra (bool): Whether extra keys in *inputs* are permitted.
            IN: ``True`` to ignore extra keys; ``False`` to treat them as errors.
            OUT: Controls whether unexpected variables generate validation errors.

    Returns:
        tuple[bool, list[str]]: A boolean indicating overall validity and a list
            of error messages. OUT: ``(True, [])`` when valid; otherwise
            ``(False, [error1, ...])``.
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
