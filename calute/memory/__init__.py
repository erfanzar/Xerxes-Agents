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


"""Memory management system for Calute AI agents.

This module provides a comprehensive memory subsystem for AI agents, enabling
them to store, retrieve, and manage information across different contexts and
time scales. The memory system supports multiple storage backends and memory
types optimized for different use cases.

Memory Types:
    - ShortTermMemory: Temporary storage for current conversation context
    - LongTermMemory: Persistent storage for important information across sessions
    - ContextualMemory: Context-aware memory that adapts to the current situation
    - EntityMemory: Specialized memory for tracking entities and their attributes
    - UserMemory: User-specific memory for personalization

Storage Backends:
    - SimpleStorage: In-memory storage for development and testing
    - SQLiteStorage: Persistent local storage using SQLite
    - RAGStorage: Vector-based storage for semantic retrieval

Example:
    >>> from calute.memory import ShortTermMemory, MemoryItem
    >>>
    >>> memory = ShortTermMemory()
    >>> item = MemoryItem(content="Important information", metadata={"source": "user"})
    >>> memory.add(item)
    >>> results = memory.search("information")
"""

from .base import Memory, MemoryItem
from .compat import MemoryStore, MemoryType
from .contextual_memory import ContextualMemory
from .entity_memory import EntityMemory
from .long_term_memory import LongTermMemory
from .short_term_memory import ShortTermMemory
from .storage import MemoryStorage, RAGStorage, SimpleStorage, SQLiteStorage
from .user_memory import UserMemory

MemoryEntry = MemoryItem

__all__ = [
    "ContextualMemory",
    "EntityMemory",
    "LongTermMemory",
    "Memory",
    "MemoryEntry",
    "MemoryItem",
    "MemoryStorage",
    "MemoryStore",
    "MemoryType",
    "RAGStorage",
    "SQLiteStorage",
    "ShortTermMemory",
    "SimpleStorage",
    "UserMemory",
]
