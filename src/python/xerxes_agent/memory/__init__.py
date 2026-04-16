# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).
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


"""Memory management system for Xerxes AI agents.

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
    >>> from xerxes_agent.memory import ShortTermMemory, MemoryItem
    >>>
    >>> memory = ShortTermMemory()
    >>> item = MemoryItem(content="Important information", metadata={"source": "user"})
    >>> memory.add(item)
    >>> results = memory.search("information")
"""

from .base import Memory, MemoryItem
from .compat import MemoryEntry, MemoryStore, MemoryType
from .contextual_memory import ContextualMemory
from .embedders import (
    Embedder,
    HashEmbedder,
    OllamaEmbedder,
    OpenAIEmbedder,
    SentenceTransformerEmbedder,
    cosine_similarity,
    get_default_embedder,
    reset_default_embedder,
)
from .entity_memory import EntityMemory
from .long_term_memory import LongTermMemory
from .retrieval import HybridRetriever, RetrievalResult, RetrievalWeights
from .short_term_memory import ShortTermMemory
from .storage import MemoryStorage, RAGStorage, SimpleStorage, SQLiteStorage
from .turn_indexer import make_memory_provider, make_turn_indexer_hook
from .user_memory import UserMemory
from .user_profile import ConfidentValue, UserProfile, UserProfileStore
from .vector_storage import SQLiteVectorStorage

__all__ = [
    "ConfidentValue",
    "ContextualMemory",
    "Embedder",
    "EntityMemory",
    "HashEmbedder",
    "HybridRetriever",
    "LongTermMemory",
    "Memory",
    "MemoryEntry",
    "MemoryItem",
    "MemoryStorage",
    "MemoryStore",
    "MemoryType",
    "OllamaEmbedder",
    "OpenAIEmbedder",
    "RAGStorage",
    "RetrievalResult",
    "RetrievalWeights",
    "SQLiteStorage",
    "SQLiteVectorStorage",
    "SentenceTransformerEmbedder",
    "ShortTermMemory",
    "SimpleStorage",
    "UserMemory",
    "UserProfile",
    "UserProfileStore",
    "cosine_similarity",
    "get_default_embedder",
    "make_memory_provider",
    "make_turn_indexer_hook",
    "reset_default_embedder",
]
