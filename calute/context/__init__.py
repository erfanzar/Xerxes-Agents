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


"""Context management system for Calute framework.

This module provides comprehensive context management capabilities
for managing conversation history and token budgets in LLM applications.
It includes:

- Multiple compaction strategies for managing conversation context
- Smart token counting with multi-provider support
- Automatic strategy selection based on compression requirements
- Support for preserving important messages during compaction

The context management system helps maintain conversation coherence
while staying within token limits of various LLM providers.

Available Strategies:
    - SummarizationStrategy: Uses LLM to summarize older messages
    - SlidingWindowStrategy: Keeps only the most recent messages
    - PriorityBasedStrategy: Retains messages based on importance scores
    - SmartCompactionStrategy: Hybrid approach combining multiple strategies
    - TruncateStrategy: Simple truncation for emergency compaction

Example:
    >>> from calute.context import (
    ...     SmartCompactionStrategy,
    ...     SmartTokenCounter,
    ...     get_compaction_strategy,
    ... )
    >>> counter = SmartTokenCounter(model="gpt-4")
    >>> token_count = counter.count_tokens("Hello, world!")
    >>> strategy = get_compaction_strategy(
    ...     strategy=CompactionStrategy.SMART,
    ...     target_tokens=4000,
    ...     model="gpt-4"
    ... )
"""

from .compaction_strategies import (
    BaseCompactionStrategy,
    PriorityBasedStrategy,
    SlidingWindowStrategy,
    SmartCompactionStrategy,
    SummarizationStrategy,
    TruncateStrategy,
    get_compaction_strategy,
)
from .token_counter import SmartTokenCounter

__all__ = [
    "BaseCompactionStrategy",
    "PriorityBasedStrategy",
    "SlidingWindowStrategy",
    "SmartCompactionStrategy",
    "SmartTokenCounter",
    "SummarizationStrategy",
    "TruncateStrategy",
    "get_compaction_strategy",
]
