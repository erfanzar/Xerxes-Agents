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

from datetime import datetime, timedelta

import numpy as np

from calute import MemoryEntry, MemoryStore, MemoryType


class TestMemory:
    """Test suite for memory management."""

    def test_memory_entry_creation(self):
        """Test creating a memory entry."""
        entry = MemoryEntry(
            content="Test memory",
            timestamp=datetime.now(),
            memory_type=MemoryType.SHORT_TERM,
            agent_id="test_agent",
            context={"key": "value"},
            importance_score=0.8,
            tags=["test", "memory"],
        )

        assert entry.content == "Test memory"
        assert entry.memory_type == MemoryType.SHORT_TERM
        assert entry.agent_id == "test_agent"
        assert entry.context == {"key": "value"}
        assert entry.importance_score == 0.8
        assert entry.tags == ["test", "memory"]
        assert entry.access_count == 0
        assert entry.last_accessed is None

    def test_memory_entry_to_dict(self):
        """Test converting memory entry to dictionary."""
        timestamp = datetime.now()
        entry = MemoryEntry(
            content="Test content",
            timestamp=timestamp,
            memory_type=MemoryType.LONG_TERM,
            agent_id="agent_1",
            importance_score=0.7,
        )

        entry_dict = entry.to_dict()
        assert entry_dict["content"] == "Test content"
        assert entry_dict["timestamp"] == timestamp.isoformat()
        assert entry_dict["memory_type"] == "long_term"
        assert entry_dict["agent_id"] == "agent_1"
        assert entry_dict["importance_score"] == 0.7

    def test_memory_store_initialization(self):
        """Test memory store initialization."""
        store = MemoryStore(max_short_term=15, max_working=8)

        assert store.max_short_term == 15
        assert store.max_working == 8
        assert len(store.memories) == len(MemoryType)
        for memory_type in MemoryType:
            assert memory_type in store.memories
            assert store.memories[memory_type] == []

    def test_add_memory(self, memory_store):
        """Test adding memories to the store."""
        entry = memory_store.add_memory(
            content="Test memory content",
            memory_type=MemoryType.SHORT_TERM,
            agent_id="test_agent",
            context={"session": "123"},
            importance_score=0.9,
            tags=["important"],
        )

        assert entry.content == "Test memory content"
        assert len(memory_store.memories[MemoryType.SHORT_TERM]) == 1
        assert memory_store.memories[MemoryType.SHORT_TERM][0] == entry

    def test_memory_limits_short_term(self, memory_store):
        """Test that short-term memory respects limits."""
        # Add memories up to the limit
        for i in range(memory_store.max_short_term + 5):
            memory_store.add_memory(
                content=f"Memory {i}",
                memory_type=MemoryType.SHORT_TERM,
                agent_id="test_agent",
                importance_score=0.5,
            )

        # Check that we haven't exceeded the limit
        assert len(memory_store.memories[MemoryType.SHORT_TERM]) <= memory_store.max_short_term

    def test_memory_limits_working(self, memory_store):
        """Test that working memory respects limits."""
        # Add memories up to the limit
        for i in range(memory_store.max_working + 3):
            memory_store.add_memory(
                content=f"Working memory {i}",
                memory_type=MemoryType.WORKING,
                agent_id="test_agent",
            )

        # Check that we haven't exceeded the limit
        assert len(memory_store.memories[MemoryType.WORKING]) <= memory_store.max_working

    def test_retrieve_memories_by_type(self, memory_store):
        """Test retrieving memories by type."""
        # Add different types of memories
        memory_store.add_memory("Short term", MemoryType.SHORT_TERM, "agent1")
        memory_store.add_memory("Long term", MemoryType.LONG_TERM, "agent1")
        memory_store.add_memory("Episodic", MemoryType.EPISODIC, "agent1")

        short_term = memory_store.retrieve_memories(memory_type=MemoryType.SHORT_TERM)
        long_term = memory_store.retrieve_memories(memory_type=MemoryType.LONG_TERM)
        episodic = memory_store.retrieve_memories(memory_type=MemoryType.EPISODIC)

        assert len(short_term) == 1
        assert short_term[0].content == "Short term"
        assert len(long_term) == 1
        assert long_term[0].content == "Long term"
        assert len(episodic) == 1
        assert episodic[0].content == "Episodic"

    def test_retrieve_memories_by_agent(self, memory_store):
        """Test retrieving memories by agent ID."""
        memory_store.add_memory("Agent 1 memory", MemoryType.SHORT_TERM, "agent1")
        memory_store.add_memory("Agent 2 memory", MemoryType.SHORT_TERM, "agent2")
        memory_store.add_memory("Another Agent 1", MemoryType.SHORT_TERM, "agent1")

        agent1_memories = memory_store.retrieve_memories(agent_id="agent1")
        agent2_memories = memory_store.retrieve_memories(agent_id="agent2")

        assert len(agent1_memories) == 2
        assert len(agent2_memories) == 1
        assert agent2_memories[0].content == "Agent 2 memory"

    def test_retrieve_memories_by_tags(self, memory_store):
        """Test retrieving memories by tags."""
        memory_store.add_memory(
            "Tagged memory 1", MemoryType.SEMANTIC, "agent1", tags=["important", "urgent"]
        )
        memory_store.add_memory("Tagged memory 2", MemoryType.SEMANTIC, "agent1", tags=["important"])
        memory_store.add_memory("Tagged memory 3", MemoryType.SEMANTIC, "agent1", tags=["trivial"])

        important_memories = memory_store.retrieve_memories(tags=["important"])
        urgent_memories = memory_store.retrieve_memories(tags=["urgent"])

        assert len(important_memories) == 2
        assert len(urgent_memories) == 1

    def test_retrieve_recent_memories(self, memory_store):
        """Test retrieving recent memories."""
        # Add memories with different timestamps
        now = datetime.now()
        old_entry = MemoryEntry(
            content="Old memory",
            timestamp=now - timedelta(hours=2),
            memory_type=MemoryType.SHORT_TERM,
            agent_id="agent1",
        )
        memory_store.memories[MemoryType.SHORT_TERM].append(old_entry)

        memory_store.add_memory("Recent memory", MemoryType.SHORT_TERM, "agent1")

        recent = memory_store.retrieve_recent(minutes_ago=60)
        assert len(recent) == 1
        assert recent[0].content == "Recent memory"

    def test_clear_memories(self, memory_store):
        """Test clearing memories."""
        # Add various memories
        memory_store.add_memory("Memory 1", MemoryType.SHORT_TERM, "agent1")
        memory_store.add_memory("Memory 2", MemoryType.LONG_TERM, "agent1")
        memory_store.add_memory("Memory 3", MemoryType.SHORT_TERM, "agent2")

        # Clear specific type
        memory_store.clear_memories(memory_type=MemoryType.SHORT_TERM)
        assert len(memory_store.memories[MemoryType.SHORT_TERM]) == 0
        assert len(memory_store.memories[MemoryType.LONG_TERM]) == 1

        # Clear all
        memory_store.clear_memories()
        for memory_type in MemoryType:
            assert len(memory_store.memories[memory_type]) == 0

    def test_memory_with_embedding(self, memory_store):
        """Test memory with embeddings."""
        rng = np.random.default_rng()
        embedding = rng.standard_normal(768)  # Typical embedding size
        entry = memory_store.add_memory(
            content="Memory with embedding",
            memory_type=MemoryType.SEMANTIC,
            agent_id="agent1",
        )
        entry.embedding = embedding

        assert entry.embedding is not None
        assert entry.embedding.shape == (768,)

    def test_consolidate_memories(self, memory_store):
        """Test memory consolidation from short-term to long-term."""
        # Add high-importance short-term memories
        for i in range(3):
            memory_store.add_memory(
                content=f"Important memory {i}",
                memory_type=MemoryType.SHORT_TERM,
                agent_id="agent1",
                importance_score=0.9,
            )

        # Consolidate high-importance memories
        memory_store.consolidate_memories(threshold=0.8)

        # Check that high-importance memories were moved to long-term
        long_term = memory_store.retrieve_memories(memory_type=MemoryType.LONG_TERM)
        assert len(long_term) > 0
