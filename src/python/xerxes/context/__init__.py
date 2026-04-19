# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0


# distributed under the License is distributed on an "AS IS" BASIS,

# See the License for the specific language governing permissions and
# limitations under the License.

"""Context management — compaction strategies and token counting."""

from .advanced_compressor import HermesCompressionStrategy
from .compaction_strategies import (
    BaseCompactionStrategy,
    PriorityBasedStrategy,
    SlidingWindowStrategy,
    SummarizationStrategy,
    TruncateStrategy,
    get_compaction_strategy,
)
from .token_counter import SmartTokenCounter

__all__ = [
    "BaseCompactionStrategy",
    "HermesCompressionStrategy",
    "PriorityBasedStrategy",
    "SlidingWindowStrategy",
    "SmartTokenCounter",
    "SummarizationStrategy",
    "TruncateStrategy",
    "get_compaction_strategy",
]
