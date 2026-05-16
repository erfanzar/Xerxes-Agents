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
    """Substitute ``{var}`` placeholders in ``input_string`` with values.

    Scalars (``str``, ``int``, ``float``, ``bool``) are stringified directly;
    ``dict`` and ``list`` values are JSON-encoded (falling back to ``str``);
    ``None`` becomes the empty string.

    Args:
        input_string: Template string; ``None`` is treated as ``""``.
        inputs: Mapping from placeholder key to replacement value.

    Returns:
        The fully interpolated string.

    Raises:
        KeyError: A placeholder key is missing from ``inputs``.
        ValueError: A value has an unsupported type.
    """

    if not input_string:
        return ""

    pattern = r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}"

    def replacer(match) -> str:
        """Return the stringified value for one matched ``{key}`` placeholder."""
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
    """Return the set of unique ``{var}`` placeholder names in a template."""

    if not input_string:
        return set()

    pattern = r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}"
    return set(re.findall(pattern, input_string))


def validate_inputs_for_template(
    template_string: str, inputs: dict[str, Any], allow_extra: bool = True
) -> tuple[bool, list[str]]:
    """Check whether ``inputs`` satisfies the placeholders in a template.

    Args:
        template_string: Template whose required variables are derived from
            its ``{var}`` placeholders.
        inputs: Candidate input mapping to validate.
        allow_extra: When ``False``, keys in ``inputs`` that are not used by
            the template are flagged as errors.

    Returns:
        ``(is_valid, errors)`` — ``errors`` is empty on success and contains
        one human-readable string per missing or extra key on failure.
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
