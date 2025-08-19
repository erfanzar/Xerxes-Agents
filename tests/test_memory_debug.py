#!/usr/bin/env python3
"""
Debug and test memory system to identify and fix issues.
"""

import traceback
from datetime import datetime, timedelta
from pathlib import Path

# Test original and enhanced memory systems
from calute.memory import MemoryEntry, MemoryStore, MemoryType


def test_original_memory():
    """Test original MemoryStore."""
    print("\n🧪 Testing Original MemoryStore...")
    try:
        store = MemoryStore(max_short_term=5, max_working=3)

        # Add memories
        for i in range(10):
            store.add_memory(
                content=f"Test memory {i}",
                memory_type=MemoryType.SHORT_TERM,
                agent_id="test_agent",
                importance_score=0.5 + i * 0.05,
                tags=[f"tag_{i % 3}"],
            )
            print(f"  ✓ Added memory {i}")

        # Retrieve memories
        memories = store.retrieve_memories(memory_types=[MemoryType.SHORT_TERM], agent_id="test_agent", limit=5)
        print(f"  ✓ Retrieved {len(memories)} memories")

        # Test consolidation
        summary = store.consolidate_memories("test_agent")
        print(f"  ✓ Consolidated memories: {len(summary)} chars")

        print("✅ Original MemoryStore working!")
        return True

    except Exception as e:
        print(f"❌ Original MemoryStore failed: {e}")
        traceback.print_exc()
        return False


def test_enhanced_memory():
    """Test MemoryStore."""
    print("\n🧪 Testing MemoryStore...")
    issues = []

    try:
        # Test initialization
        store = MemoryStore(max_short_term=10, max_working=5, enable_persistence=False, enable_vector_search=False)
        print("  ✓ Initialized MemoryStore")

    except Exception as e:
        issues.append(f"Initialization failed: {e}")
        traceback.print_exc()
        return issues

    # Test adding memories
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
            print(f"  ✓ Added enhanced memory {i}: {entry.id}")

    except Exception as e:
        issues.append(f"Adding memories failed: {e}")
        traceback.print_exc()

    # Test retrieval
    try:
        memories = store.retrieve_memories(agent_id="test_agent", tags=["enhanced"], limit=3)
        print(f"  ✓ Retrieved {len(memories)} enhanced memories")

    except Exception as e:
        issues.append(f"Retrieval failed: {e}")
        traceback.print_exc()

    # Test statistics
    try:
        stats = store.get_statistics()
        print(f"  ✓ Statistics: {stats}")

    except Exception as e:
        issues.append(f"Statistics failed: {e}")
        traceback.print_exc()

    # Test persistence
    try:
        temp_path = Path("/tmp/test_memory.pkl")
        store.persistence_path = temp_path
        store.save()
        print(f"  ✓ Saved to {temp_path}")

        # Load back
        MemoryStore(enable_persistence=True, persistence_path=temp_path)
        print(f"  ✓ Loaded from {temp_path}")

        # Clean up
        temp_path.unlink(missing_ok=True)

    except Exception as e:
        issues.append(f"Persistence failed: {e}")
        traceback.print_exc()

    if not issues:
        print("✅ MemoryStore working!")
    else:
        print(f"❌ MemoryStore has {len(issues)} issues:")
        for issue in issues:
            print(f"  - {issue}")

    return issues


def test_memory_indexing():
    """Test memory indexing functionality."""
    print("\n🧪 Testing Memory Indexing...")
    issues = []

    try:
        store = MemoryStore(max_short_term=20)

        # Add memories with different properties
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

        # Test retrieval by different criteria
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

    return issues


def test_memory_decay():
    """Test memory importance decay."""
    print("\n🧪 Testing Memory Decay...")

    try:
        # Create a memory entry
        entry = MemoryEntry(
            id="test_decay",
            content="Test memory",
            timestamp=datetime.now() - timedelta(hours=10),  # 10 hours old
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

    # Test 1: Empty retrieval
    try:
        results = store.retrieve_memories(agent_id="nonexistent")
        print(f"  ✓ Empty retrieval: {len(results)} results")
    except Exception as e:
        issues.append(f"Empty retrieval failed: {e}")

    # Test 2: Duplicate tags
    try:
        store.add_memory(
            content="Test",
            memory_type=MemoryType.SHORT_TERM,
            agent_id="test",
            tags=["tag1", "tag1", "tag2"],  # Duplicate tags
        )
        print("  ✓ Handled duplicate tags")
    except Exception as e:
        issues.append(f"Duplicate tags failed: {e}")

    # Test 3: Very long content
    try:
        long_content = "x" * 10000
        store.add_memory(content=long_content, memory_type=MemoryType.SHORT_TERM, agent_id="test")
        print(f"  ✓ Handled long content ({len(long_content)} chars)")
    except Exception as e:
        issues.append(f"Long content failed: {e}")

    # Test 4: Special characters in content
    try:
        special_content = "Test with special chars: 你好 🚀 \n\t\r"
        store.add_memory(content=special_content, memory_type=MemoryType.SHORT_TERM, agent_id="test")
        print("  ✓ Handled special characters")
    except Exception as e:
        issues.append(f"Special characters failed: {e}")

    # Test 5: Cache invalidation
    try:
        # First retrieval (miss)
        store.retrieve_memories(agent_id="test", limit=2)

        # Second retrieval (should hit cache)
        store.retrieve_memories(agent_id="test", limit=2)

        # Add new memory (should invalidate cache)
        store.add_memory(content="New memory", memory_type=MemoryType.SHORT_TERM, agent_id="test")

        # Third retrieval (should miss cache)
        store.retrieve_memories(agent_id="test", limit=2)

        print(f"  ✓ Cache working (hits: {store.cache_hits}, misses: {store.cache_misses})")

    except Exception as e:
        issues.append(f"Cache test failed: {e}")

    if not issues:
        print("✅ All edge cases handled!")
    else:
        print(f"❌ {len(issues)} edge case issues found")
        for issue in issues:
            print(f"  - {issue}")

    return issues


def main():
    """Run all memory tests."""
    print("=" * 60)
    print("🔍 MEMORY SYSTEM DEBUG & TEST")
    print("=" * 60)

    all_issues = []

    # Test both implementations
    test_original_memory()

    enhanced_issues = test_enhanced_memory()
    all_issues.extend(enhanced_issues)

    indexing_issues = test_memory_indexing()
    all_issues.extend(indexing_issues)

    test_memory_decay()

    edge_issues = test_edge_cases()
    all_issues.extend(edge_issues)

    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)

    if not all_issues:
        print("✅ All tests passed! Memory system is working correctly.")
    else:
        print(f"❌ Found {len(all_issues)} issues to fix:")
        for i, issue in enumerate(all_issues, 1):
            print(f"{i}. {issue}")

    return len(all_issues) == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
