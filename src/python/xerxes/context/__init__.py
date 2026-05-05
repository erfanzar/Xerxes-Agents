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
"""Public exports for the Xerxes context compaction package.

Provides token counting and various strategies for compacting conversation
history when it exceeds a target token budget.
"""

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
