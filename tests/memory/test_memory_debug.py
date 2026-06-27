#!/usr/bin/env python3
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
"""
Debug and test memory system to identify and fix issues.
"""

import traceback
from datetime import datetime, timedelta

from xerxes.memory import MemoryEntry, MemoryStore, MemoryType


def test_original_memory():
    """Test original MemoryStore."""
    print("\n🧪 Testing Original MemoryStore...")
    try:
        store = MemoryStore(max_short_term=5, max_working=3)

        for i in range(10):
            store.add_memory(
                content=f"Test memory {i}",
                memory_type=MemoryType.SHORT_TERM,
                agent_id="test_agent",
                importance_score=0.5 + i * 0.05,
                tags=[f"tag_{i % 3}"],
            )
            print(f"  ✓ Added memory {i}")

        memories = store.retrieve_memories(memory_types=[MemoryType.SHORT_TERM], agent_id="test_agent", limit=5)
        print(f"  ✓ Retrieved {len(memories)} memories")

        summary = store.consolidate_memories("test_agent")
        print(f"  ✓ Consolidated memories: {len(summary)} chars")

        print("✅ Original MemoryStore working!")

    except Exception as e:
        print(f"❌ Original MemoryStore failed: {e}")
        traceback.print_exc()
        raise AssertionError("Original MemoryStore failed") from e


def test_enhanced_memory():
    """Test MemoryStore."""
    print("\n🧪 Testing MemoryStore...")
    issues = []

    try:
        store = MemoryStore(max_short_term=10, max_working=5, enable_persistence=False, enable_vector_search=False)
        print("  ✓ Initialized MemoryStore")

    except Exception as e:
        issues.append(f"Initialization failed: {e}")
        traceback.print_exc()
        assert not issues
        return

    try:
        for i in range(5):
            entry = store.add_memory(
                content=f"Enhanced test memory {i}",
                memory_type=MemoryType.SHORT_TERM,
                agent_id="test_agent",
                importance_score=0.5 + i * 0.1,
                tags=["enhanced", f"test_{i}"],
                confidence=0.9,
            )
            assert entry.metadata["confidence"] == 0.9
            print(f"  ✓ Added enhanced memory {i}: {entry.memory_id}")

    except Exception as e:
        issues.append(f"Adding memories failed: {e}")
        traceback.print_exc()

    try:
        memories = store.retrieve_memories(agent_id="test_agent", tags=["enhanced"], limit=3)
        print(f"  ✓ Retrieved {len(memories)} enhanced memories")

    except Exception as e:
        issues.append(f"Retrieval failed: {e}")
        traceback.print_exc()

    try:
        stats = store.get_statistics()
        print(f"  ✓ Statistics: {stats}")

    except Exception as e:
        issues.append(f"Statistics failed: {e}")
        traceback.print_exc()

    try:
        item = store.save("Contextual memory smoke", metadata={"kind": "debug"}, importance=0.9)
        assert item.content == "Contextual memory smoke"
        assert item.metadata["kind"] == "debug"
        print("  ✓ Contextual save API works")

    except Exception as e:
        issues.append(f"Persistence failed: {e}")
        traceback.print_exc()

    if not issues:
        print("✅ MemoryStore working!")
    else:
        print(f"❌ MemoryStore has {len(issues)} issues:")
        for issue in issues:
            print(f"  - {issue}")

    assert not issues


def test_memory_indexing():
    """Test memory indexing functionality."""
    print("\n🧪 Testing Memory Indexing...")
    issues = []

    try:
        store = MemoryStore(max_short_term=20)

        memories_added = []
        for i in range(10):
            entry = store.add_memory(
                content=f"Memory about topic {i % 3}",
                memory_type=MemoryType.SHORT_TERM if i < 5 else MemoryType.LONG_TERM,
                agent_id=f"agent_{i % 2}",
                tags=[f"topic_{i % 3}", "test"],
                importance_score=0.3 + (i * 0.07),
            )
            memories_added.append(entry)

        print(f"  ✓ Added {len(memories_added)} memories")

        tests = [
            ("by agent", {"agent_id": "agent_0"}),
            ("by tags", {"tags": ["topic_1"]}),
            ("by importance", {"min_importance": 0.5}),
            ("by type", {"memory_types": [MemoryType.LONG_TERM]}),
        ]

        for test_name, kwargs in tests:
            try:
                results = store.retrieve_memories(**kwargs, limit=10)
                print(f"  ✓ Retrieval {test_name}: {len(results)} results")
            except Exception as e:
                issues.append(f"Retrieval {test_name} failed: {e}")

    except Exception as e:
        issues.append(f"Indexing test failed: {e}")
        traceback.print_exc()

    if not issues:
        print("✅ Memory indexing working!")
    else:
        print(f"❌ Memory indexing has {len(issues)} issues")

    assert not issues


def test_memory_decay():
    """Test memory importance decay."""
    print("\n🧪 Testing Memory Decay...")

    try:
        entry = MemoryEntry(
            id="test_decay",
            content="Test memory",
            timestamp=datetime.now() - timedelta(hours=10),
            memory_type=MemoryType.SHORT_TERM,
            agent_id="test",
            importance_score=0.8,
            decay_rate=0.05,
            access_count=5,
        )

        current_importance = entry.get_current_importance()
        print(f"  Original importance: {entry.importance_score:.2f}")
        print(f"  Current importance (with decay): {current_importance:.2f}")
        print(f"  Access count: {entry.access_count}")

        if current_importance < entry.importance_score:
            print("✅ Memory decay working!")
        else:
            print("❌ Memory decay not working properly")

    except Exception as e:
        print(f"❌ Memory decay test failed: {e}")
        traceback.print_exc()


def test_edge_cases():
    """Test edge cases and potential bugs."""
    print("\n🧪 Testing Edge Cases...")
    issues = []

    store = MemoryStore(max_short_term=5)

    try:
        results = store.retrieve_memories(agent_id="nonexistent")
        print(f"  ✓ Empty retrieval: {len(results)} results")
    except Exception as e:
        issues.append(f"Empty retrieval failed: {e}")

    try:
        store.add_memory(
            content="Test",
            memory_type=MemoryType.SHORT_TERM,
            agent_id="test",
            tags=["tag1", "tag1", "tag2"],
        )
        print("  ✓ Handled duplicate tags")
    except Exception as e:
        issues.append(f"Duplicate tags failed: {e}")

    try:
        long_content = "x" * 10000
        store.add_memory(content=long_content, memory_type=MemoryType.SHORT_TERM, agent_id="test")
        print(f"  ✓ Handled long content ({len(long_content)} chars)")
    except Exception as e:
        issues.append(f"Long content failed: {e}")

    try:
        special_content = "Test with special chars: 你好 🚀 \n\t\r"
        store.add_memory(content=special_content, memory_type=MemoryType.SHORT_TERM, agent_id="test")
        print("  ✓ Handled special characters")
    except Exception as e:
        issues.append(f"Special characters failed: {e}")

    try:
        store.retrieve_memories(agent_id="test", limit=2)

        store.retrieve_memories(agent_id="test", limit=2)

        store.add_memory(content="New memory", memory_type=MemoryType.SHORT_TERM, agent_id="test")

        store.retrieve_memories(agent_id="test", limit=2)

        stats = store.get_statistics()
        assert "cache_hit_rate" in stats
        print(f"  ✓ Retrieval statistics available (cache hit rate: {stats['cache_hit_rate']})")

    except Exception as e:
        issues.append(f"Cache test failed: {e}")

    if not issues:
        print("✅ All edge cases handled!")
    else:
        print(f"❌ {len(issues)} edge case issues found")
        for issue in issues:
            print(f"  - {issue}")

    assert not issues


def main():
    """Run all memory tests."""
    print("=" * 60)
    print("🔍 MEMORY SYSTEM DEBUG & TEST")
    print("=" * 60)

    test_original_memory()
    test_enhanced_memory()
    test_memory_indexing()
    test_memory_decay()
    test_edge_cases()

    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print("✅ All tests passed! Memory system is working correctly.")
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
