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

"""Python version compatibility shims.

Backports for features added in newer Python versions so the codebase
remains compatible with the minimum supported runtime (3.10).
"""

try:
    from enum import StrEnum
except ImportError:
    from enum import Enum

    class StrEnum(str, Enum):
        """Backport of enum.StrEnum for Python < 3.11."""

        def __str__(self) -> str:
            return self.value

__all__ = ["StrEnum"]
