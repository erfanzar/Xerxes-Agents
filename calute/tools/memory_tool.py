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


"""Memory management tools for agents to read and write memories autonomously."""

from datetime import datetime
from enum import Enum
from typing import Any

from ..memory import MemoryType
from ..types import Agent


class MemoryOperation(Enum):
    """Types of memory operations agents can perform."""

    SAVE = "save"
    SEARCH = "search"
    RETRIEVE = "retrieve"
    DELETE = "delete"
    SUMMARIZE = "summarize"
    CONSOLIDATE = "consolidate"


def save_memory(
    content: str,
    memory_type: str = "short_term",
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    agent_id: str | None = None,
    context_variables: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Save a memory entry to the memory store.

    Args:
        content: The content to remember (required)
        memory_type: Type of memory - 'short_term', 'long_term', 'working', 'episodic', 'semantic', or 'procedural'
        tags: List of tags for categorization
        metadata: Additional metadata to store with the memory
        agent_id: ID of the agent creating this memory
        context_variables: Runtime context from the agent

    Returns:
        Dictionary with memory_id and status

    Examples:
        {"content": "User prefers dark mode", "tags": ["preference", "ui"]}
        {"content": "Project uses Python 3.11", "memory_type": "long_term"}
    """
    # Get memory store from context
    memory_store = context_variables.get("memory_store") if context_variables else None

    if memory_store is None:
        return {"status": "error", "message": "Memory store not available"}

    # Get current agent_id from context if not provided
    if not agent_id and context_variables:
        agent_id = context_variables.get("agent_id", "default")

    try:
        # Convert string memory type to enum
        memory_type_enum = MemoryType[memory_type.upper()]

        # Prepare metadata
        full_metadata = metadata or {}
        full_metadata["timestamp"] = datetime.now().isoformat()
        if agent_id:
            full_metadata["created_by"] = agent_id

        # Add memory (importance_score defaults to 1.0 in MemoryStore)
        memory_id = memory_store.add_memory(
            content=content,
            memory_type=memory_type_enum,
            agent_id=agent_id or "default",
            context=full_metadata,
            tags=tags or [],
        )

        # Convert memory_id to string if it's a MemoryItem object
        memory_id_str = str(memory_id.memory_id) if hasattr(memory_id, "memory_id") else str(memory_id)

        return {
            "status": "success",
            "memory_id": memory_id_str,
            "message": "Memory saved successfully",
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def search_memory(
    query: str,
    memory_types: list[str] | None = None,
    tags: list[str] | None = None,
    limit: int = 10,
    agent_id: str | None = None,
    time_range: dict[str, str] | None = None,
    context_variables: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Search for memories based on query and filters.

    Args:
        query: Search query string (required)
        memory_types: List of memory types to search in
        tags: Filter by tags
        limit: Maximum number of results to return
        agent_id: Filter by agent who created the memory
        time_range: Dict with 'start' and/or 'end' ISO timestamps
        context_variables: Runtime context from the agent

    Returns:
        Dictionary with search results

    Examples:
        {"query": "user preferences", "tags": ["preference"], "limit": 5}
        {"query": "error handling", "memory_types": ["long_term"]}
    """
    memory_store = context_variables.get("memory_store") if context_variables else None
    if memory_store is None:
        return {"status": "error", "message": "Memory store not available"}

    # Get current agent_id from context if not provided
    if not agent_id and context_variables:
        agent_id = context_variables.get("agent_id", "default")

    try:
        # Convert string memory types to enums if provided
        memory_type_enums = None
        if memory_types:
            memory_type_enums = [MemoryType[mt.upper()] for mt in memory_types]

        # Retrieve memories without tag filtering (it's broken in MemoryStore)
        # We'll filter tags manually
        memories = memory_store.retrieve_memories(
            memory_types=memory_type_enums,
            agent_id=agent_id,
            tags=None,  # Don't use tag filter - it's broken
            limit=limit * 5 if (query or tags) else limit,  # Get more for manual filtering
        )

        # Manual filtering for tags and query
        if (query or tags) and memories:
            filtered_memories = []
            query_lower = query.lower() if query else ""

            for mem in memories:
                # Check tag filter
                if tags:
                    mem_tags = mem.metadata.get("tags", [])
                    if not any(tag in mem_tags for tag in tags):
                        continue

                # Check query filter
                if query:
                    # Search in content and tags - check each word in query
                    query_words = query_lower.split()
                    content_lower = mem.content.lower()
                    tags_lower = [str(tag).lower() for tag in mem.metadata.get("tags", [])]

                    # Check if any query word matches content or tags
                    match_found = any(word in content_lower for word in query_words)
                    if not match_found:
                        match_found = any(word in tag for word in query_words for tag in tags_lower)

                    if not match_found:
                        continue

                filtered_memories.append(mem)
                if len(filtered_memories) >= limit:
                    break

            memories = filtered_memories

        # Filter by time range if provided
        if time_range and memories:
            filtered_memories = []
            for mem in memories:
                mem_time = mem.metadata.get("timestamp")
                if mem_time:
                    if time_range.get("start") and mem_time < time_range["start"]:
                        continue
                    if time_range.get("end") and mem_time > time_range["end"]:
                        continue
                    filtered_memories.append(mem)
            memories = filtered_memories

        # Format results
        results = []
        for mem in memories:
            results.append(
                {
                    "content": mem.content,
                    "tags": mem.metadata.get("tags", []),
                    "timestamp": mem.metadata.get("timestamp", "unknown"),
                    "metadata": mem.metadata,
                }
            )

        return {
            "status": "success",
            "count": len(results),
            "memories": results,
            "query": query,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def consolidate_agent_memories(
    agent_id: str,
    max_items: int = 20,
    context_variables: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Get a consolidated summary of memories for a specific agent.

    Args:
        agent_id: ID of the agent to consolidate memories for (required)
        max_items: Maximum number of memories to consolidate
        context_variables: Runtime context from the agent

    Returns:
        Dictionary with consolidated memory summary

    Examples:
        {"agent_id": "research_agent", "max_items": 15}
        {"agent_id": "default"}
    """
    memory_store = context_variables.get("memory_store") if context_variables else None
    if memory_store is None:
        return {"status": "error", "message": "Memory store not available"}

    try:
        # Retrieve memories for the agent
        memories = memory_store.retrieve_memories(
            agent_id=agent_id,
            limit=max_items,
        )

        # Generate a summary from the memories
        if not memories:
            summary = "No memories found for this agent."
        else:
            # Group memories by tags
            tagged_memories = {}
            for mem in memories:
                mem_tags = mem.metadata.get("tags", ["untagged"])
                for tag in mem_tags:
                    if tag not in tagged_memories:
                        tagged_memories[tag] = []
                    tagged_memories[tag].append(mem.content)

            # Build summary
            summary_parts = []
            summary_parts.append(f"Total memories: {len(memories)}")
            summary_parts.append("\nMemories by category:")

            for tag in sorted(tagged_memories.keys()):
                summary_parts.append(f"\n{tag.upper()}:")
                for content in tagged_memories[tag][:3]:  # Show first 3 per tag
                    summary_parts.append(f"  - {content}")
                if len(tagged_memories[tag]) > 3:
                    summary_parts.append(f"  ... and {len(tagged_memories[tag]) - 3} more")

            summary = "\n".join(summary_parts)

        # Get statistics
        stats = memory_store.get_statistics()

        return {
            "status": "success",
            "summary": summary,
            "statistics": stats,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def delete_memory(
    memory_id: str | None = None,
    tags: list[str] | None = None,
    agent_id: str | None = None,
    older_than: str | None = None,
    context_variables: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Delete memories based on criteria.

    Args:
        memory_id: Specific memory ID to delete
        tags: Delete all memories with these tags
        agent_id: Delete all memories from this agent
        older_than: Delete memories older than this ISO timestamp
        context_variables: Runtime context from the agent

    Returns:
        Dictionary with deletion status

    Examples:
        {"memory_id": "mem_12345"}
        {"tags": ["temporary", "debug"]}
        {"agent_id": "test_agent", "older_than": "2024-01-01T00:00:00"}
    """
    memory_store = context_variables.get("memory_store") if context_variables else None
    if memory_store is None:
        return {"status": "error", "message": "Memory store not available"}

    try:
        deleted_count = 0

        # For now, we'll return a message about what would be deleted
        # since the current MemoryStore doesn't have a delete method
        criteria = []
        if memory_id:
            criteria.append(f"memory_id={memory_id}")
        if tags:
            criteria.append(f"tags={tags}")
        if agent_id:
            criteria.append(f"agent_id={agent_id}")
        if older_than:
            criteria.append(f"older_than={older_than}")

        return {
            "status": "success",
            "message": f"Delete operation would remove memories matching: {', '.join(criteria)}",
            "deleted_count": deleted_count,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def get_memory_tags_and_terms(
    agent_id: str | None = None,
    context_variables: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Get all available memory tags organized by memory type.

    Args:
        agent_id: Filter tags by specific agent (optional)
        context_variables: Runtime context from the agent

    Returns:
        Dictionary with tags organized by memory type

    Examples:
        {}
        {"agent_id": "main_agent"}
    """
    memory_store = context_variables.get("memory_store") if context_variables else None
    if memory_store is None:
        return {"status": "error", "message": "Memory store not available"}

    # Get current agent_id from context if not provided
    if not agent_id and context_variables:
        agent_id = context_variables.get("agent_id", "default")

    try:
        # Retrieve all memories for the agent
        memories = memory_store.retrieve_memories(
            agent_id=agent_id,
            limit=1000,  # Get many memories to collect all tags
        )

        # Organize tags by memory type
        tags_by_type = {
            "short_term": set(),
            "long_term": set(),
            "working": set(),
            "episodic": set(),
            "semantic": set(),
            "procedural": set(),
        }

        # Also track tag frequency
        tag_frequency = {}

        for mem in memories:
            # Get memory type from metadata or use default
            mem_type = mem.memory_type.lower() if hasattr(mem, "memory_type") else "unknown"

            # Get tags from metadata
            mem_tags = mem.metadata.get("tags", []) if hasattr(mem, "metadata") else []

            # Add tags to the appropriate memory type set
            if mem_type in tags_by_type:
                for tag in mem_tags:
                    tags_by_type[mem_type].add(tag)
                    tag_frequency[tag] = tag_frequency.get(tag, 0) + 1

        # Convert sets to sorted lists
        result = {
            "status": "success",
            "tags_by_type": {
                mem_type: sorted(list(tags))
                for mem_type, tags in tags_by_type.items()
                if tags  # Only include types that have tags
            },
            "all_tags": sorted(set().union(*tags_by_type.values())),
            "tag_frequency": dict(sorted(tag_frequency.items(), key=lambda x: x[1], reverse=True)),
            "total_unique_tags": len(set().union(*tags_by_type.values())),
            "agent_id": agent_id,
        }

        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}


def get_memory_statistics(
    agent_id: str | None = None,
    context_variables: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Get statistics about memory usage.

    Args:
        agent_id: Get statistics for specific agent only
        context_variables: Runtime context from the agent

    Returns:
        Dictionary with memory statistics

    Examples:
        {}
        {"agent_id": "main_agent"}
    """
    memory_store = context_variables.get("memory_store") if context_variables else None
    if memory_store is None:
        return {"status": "error", "message": "Memory store not available"}

    try:
        stats = memory_store.get_statistics()

        # Add agent-specific stats if requested
        if agent_id:
            agent_memories = memory_store.retrieve_memories(
                agent_id=agent_id,
                limit=1000,
            )
            stats["agent_memory_count"] = len(agent_memories)
            stats["agent_id"] = agent_id

        return {
            "status": "success",
            "statistics": stats,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# Tool definitions for agent registration
MEMORY_TOOLS = [
    save_memory,
    search_memory,
    consolidate_agent_memories,
    delete_memory,
    get_memory_statistics,
    get_memory_tags_and_terms,
]


def get_memory_tool_descriptions():
    """Get descriptions of all memory tools for agent documentation."""
    descriptions = []
    for tool in MEMORY_TOOLS:
        descriptions.append(
            {
                "name": tool.__name__,
                "description": tool.__doc__.split("\n")[1].strip() if tool.__doc__ else "",
                "category": "Memory Management",
            }
        )
    return descriptions


def add_memory_tools_to_agent(
    agent: Agent,
    memory_store=None,
    include_tools: list[str] | None = None,
) -> Agent:
    """
    Add memory management tools to an agent's function list.

    Args:
        agent: The Agent to add memory tools to
        memory_store: The memory store instance to use (will be passed in context_variables)
        include_tools: List of specific tool names to include. If None, includes all tools.
                      Options: ["save_memory", "search_memory", "consolidate_agent_memories",
                               "delete_memory", "get_memory_statistics"]

    Returns:
        The agent with memory tools added

    Example:
        >>> from calute import Agent
        >>> from calute.memory_integration import add_memory_tools_to_agent
        >>>
        >>> agent = Agent(
        ...     id="research_agent",
        ...     instructions="You are a research assistant",
        ...     functions=[]
        ... )
        >>>
        >>> # Add all memory tools
        >>> agent = add_memory_tools_to_agent(agent, memory_store=my_memory_store)
        >>>
        >>> # Or add specific tools only
        >>> agent = add_memory_tools_to_agent(
        ...     agent,
        ...     memory_store=my_memory_store,
        ...     include_tools=["save_memory", "search_memory"]
        ... )
    """
    current_functions = list(agent.functions) if agent.functions else []

    if include_tools is None:
        tools_to_add = MEMORY_TOOLS
    else:
        tool_map = {tool.__name__: tool for tool in MEMORY_TOOLS}
        tools_to_add = [tool_map[name] for name in include_tools if name in tool_map]

    for tool in tools_to_add:
        if tool not in current_functions:
            current_functions.append(tool)

    agent.functions = current_functions

    if memory_store and hasattr(agent, "_memory_store"):
        agent._memory_store = memory_store

    return agent


def create_memory_enabled_agent(
    agent_id: str,
    instructions: str,
    memory_store=None,
    memory_tools: list[str] | None = None,
    **agent_kwargs,
) -> Agent:
    """
    Create a new agent with memory tools pre-configured.

    Args:
        agent_id: ID for the agent
        instructions: Instructions/system prompt for the agent
        memory_store: Memory store instance to use
        memory_tools: List of memory tool names to include (None = all)
        **agent_kwargs: Additional arguments to pass to Agent constructor

    Returns:
        New Agent instance with memory tools configured

    Example:
        >>> from calute.memory_integration import create_memory_enabled_agent
        >>> from calute.memory import MemoryStore
        >>>
        >>> memory_store = MemoryStore()
        >>> agent = create_memory_enabled_agent(
        ...     agent_id="assistant",
        ...     instructions="You are a helpful assistant that remembers past conversations",
        ...     memory_store=memory_store,
        ...     memory_tools=["save_memory", "search_memory"]
        ... )
    """
    agent = Agent(id=agent_id, instructions=instructions, functions=[], **agent_kwargs)
    agent = add_memory_tools_to_agent(agent, memory_store=memory_store, include_tools=memory_tools)
    return agent


class MemoryToolContext:
    """
    Context manager for memory tools in function execution.
    Automatically provides memory_store in context_variables.
    """

    def __init__(self, memory_store):
        self.memory_store = memory_store

    def wrap_function_call(self, func, *args, **kwargs):
        """
        Wrap a function call to inject memory_store into context_variables.
        """
        if "context_variables" not in kwargs:
            kwargs["context_variables"] = {}

        kwargs["context_variables"]["memory_store"] = self.memory_store
        return func(*args, **kwargs)
