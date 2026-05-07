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
"""Memory management tools for saving, searching, and organizing agent memory.

This module provides tools for persistent memory storage, semantic search,
and memory consolidation for agents to maintain context across sessions.

Example:
    >>> from xerxes.tools.memory_tool import save_memory, search_memory
    >>> save_memory(content="User prefers dark mode")
    >>> results = search_memory(query="user preferences")
"""

from collections.abc import Callable
from datetime import datetime
from enum import Enum
from typing import Any

from ..memory import MemoryType
from ..types import Agent


class MemoryOperation(Enum):
    """Enumeration of memory operation types.

    Attributes:
        SAVE: Save a new memory entry.
        SEARCH: Search for memories matching a query.
        RETRIEVE: Retrieve specific memories.
        DELETE: Delete memory entries.
        SUMMARIZE: Generate summary of memories.
        CONSOLIDATE: Consolidate memories from different sources.
    """

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
    """Save a memory entry to the memory store.

    Stores information for later retrieval, enabling agents to remember
    important facts and context across interactions.

    Args:
        content: The memory content to store.
        memory_type: Type of memory (short_term, long_term, working,
            episodic, semantic, procedural). Defaults to "short_term".
        tags: Optional list of tags for categorization.
        metadata: Additional metadata dictionary.
        agent_id: ID of the agent creating this memory.
        context_variables: Context including memory_store reference.

    Returns:
        Dictionary with status and memory_id on success, or error message.

    Example:
        >>> save_memory(
        ...     content="User's name is Alice",
        ...     tags=["user-info", "preferences"],
        ...     agent_id="assistant"
        ... )
    """
    memory_store = context_variables.get("memory_store") if context_variables else None

    if memory_store is None:
        return {"status": "error", "message": "Memory store not available"}

    if not agent_id and context_variables:
        agent_id = context_variables.get("agent_id", "default")

    try:
        memory_type_enum = MemoryType[memory_type.upper()]

        full_metadata = metadata or {}
        full_metadata["timestamp"] = datetime.now().isoformat()
        if agent_id:
            full_metadata["created_by"] = agent_id

        memory_id = memory_store.add_memory(
            content=content,
            memory_type=memory_type_enum,
            agent_id=agent_id or "default",
            context=full_metadata,
            tags=tags or [],
        )

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
    """Search for memories matching a query.

    Performs semantic or keyword search across stored memories.

    Args:
        query: Search query string.
        memory_types: Filter by memory types.
        tags: Filter by tags (memories must have at least one matching tag).
        limit: Maximum results to return. Defaults to 10.
        agent_id: Filter by agent ID.
        time_range: Filter by time range {"start": "...", "end": "..."}.
        context_variables: Context including memory_store reference.

    Returns:
        Dictionary with status, count, and memories list.

    Example:
        >>> search_memory(query="authentication setup", limit=5)
    """
    memory_store = context_variables.get("memory_store") if context_variables else None
    if memory_store is None:
        return {"status": "error", "message": "Memory store not available"}

    if not agent_id and context_variables:
        agent_id = context_variables.get("agent_id", "default")

    try:
        memory_type_enums = None
        if memory_types:
            memory_type_enums = [MemoryType[mt.upper()] for mt in memory_types]

        memories = memory_store.retrieve_memories(
            memory_types=memory_type_enums,
            agent_id=agent_id,
            tags=None,
            limit=limit * 5 if (query or tags) else limit,
        )

        if (query or tags) and memories:
            filtered_memories = []
            query_lower = query.lower() if query else ""

            for mem in memories:
                if tags:
                    mem_tags = mem.metadata.get("tags", [])
                    if not any(tag in mem_tags for tag in tags):
                        continue

                if query:
                    query_words = query_lower.split()
                    content_lower = mem.content.lower()
                    tags_lower = [str(tag).lower() for tag in mem.metadata.get("tags", [])]

                    match_found = any(word in content_lower for word in query_words)
                    if not match_found:
                        match_found = any(word in tag for word in query_words for tag in tags_lower)

                    if not match_found:
                        continue

                filtered_memories.append(mem)
                if len(filtered_memories) >= limit:
                    break

            memories = filtered_memories

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
    """Consolidate and summarize an agent's memories.

    Retrieves recent memories and organizes them by category for easier review.

    Args:
        agent_id: ID of the agent whose memories to consolidate.
        max_items: Maximum memories to include. Defaults to 20.
        context_variables: Context including memory_store reference.

    Returns:
        Dictionary with status, summary, and statistics.

    Example:
        >>> consolidate_agent_memories(agent_id="assistant")
    """
    memory_store = context_variables.get("memory_store") if context_variables else None
    if memory_store is None:
        return {"status": "error", "message": "Memory store not available"}

    try:
        memories = memory_store.retrieve_memories(
            agent_id=agent_id,
            limit=max_items,
        )

        if not memories:
            summary = "No memories found for this agent."
        else:
            tagged_memories: dict[str, list[str]] = {}
            for mem in memories:
                mem_tags = mem.metadata.get("tags", ["untagged"])
                for tag in mem_tags:
                    if tag not in tagged_memories:
                        tagged_memories[tag] = []
                    tagged_memories[tag].append(mem.content)

            summary_parts = []
            summary_parts.append(f"Total memories: {len(memories)}")
            summary_parts.append("\nMemories by category:")

            for tag in sorted(tagged_memories.keys()):
                summary_parts.append(f"\n{tag.upper()}:")
                for content in tagged_memories[tag][:3]:
                    summary_parts.append(f"  - {content}")
                if len(tagged_memories[tag]) > 3:
                    summary_parts.append(f"  ... and {len(tagged_memories[tag]) - 3} more")

            summary = "\n".join(summary_parts)

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
    """Delete memory entries by ID or filter criteria.

    Args:
        memory_id: Specific memory ID to delete.
        tags: Delete memories with these tags.
        agent_id: Delete memories by this agent.
        older_than: Delete memories older than this ISO timestamp.
        context_variables: Context including memory_store reference.

    Returns:
        Dictionary with status and deleted count.

    Example:
        >>> delete_memory(tags=["temporary"], older_than="2024-01-01T00:00:00")
    """
    memory_store = context_variables.get("memory_store") if context_variables else None
    if memory_store is None:
        return {"status": "error", "message": "Memory store not available"}

    try:
        deleted_count = 0

        filters: dict[str, Any] = {}
        if tags:
            filters["tags"] = tags
        if agent_id:
            filters["agent_id"] = agent_id
        if older_than:
            from datetime import datetime

            try:
                cutoff = datetime.fromisoformat(older_than.replace("Z", "+00:00"))
                filters["created_before"] = cutoff
            except ValueError:
                return {"status": "error", "message": f"Invalid timestamp format: {older_than}"}

        if memory_id:
            deleted_count = memory_store.delete(memory_id=memory_id)
        elif filters:
            deleted_count = memory_store.delete(filters=filters)
        else:
            return {"status": "error", "message": "No deletion criteria provided"}

        return {
            "status": "success",
            "message": f"Successfully deleted {deleted_count} memories",
            "deleted_count": deleted_count,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def get_memory_tags_and_terms(
    agent_id: str | None = None,
    context_variables: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Get all tags and their usage frequency across memories.

    Args:
        agent_id: Filter by agent ID.
        context_variables: Context including memory_store reference.

    Returns:
        Dictionary with tags organized by type and frequency.

    Example:
        >>> get_memory_tags_and_terms(agent_id="assistant")
    """
    memory_store = context_variables.get("memory_store") if context_variables else None
    if memory_store is None:
        return {"status": "error", "message": "Memory store not available"}

    if not agent_id and context_variables:
        agent_id = context_variables.get("agent_id", "default")

    try:
        memories = memory_store.retrieve_memories(
            agent_id=agent_id,
            limit=1000,
        )

        tags_by_type: dict[str, set[str]] = {
            "short_term": set(),
            "long_term": set(),
            "working": set(),
            "episodic": set(),
            "semantic": set(),
            "procedural": set(),
        }

        tag_frequency: dict[str, int] = {}

        for mem in memories:
            mem_type = mem.memory_type.lower() if hasattr(mem, "memory_type") else "unknown"

            mem_tags = mem.metadata.get("tags", []) if hasattr(mem, "metadata") else []

            if mem_type in tags_by_type:
                for tag in mem_tags:
                    tags_by_type[mem_type].add(tag)
                    tag_frequency[tag] = tag_frequency.get(tag, 0) + 1

        result = {
            "status": "success",
            "tags_by_type": {mem_type: sorted(list(tags)) for mem_type, tags in tags_by_type.items() if tags},
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
    """Get statistics about the memory store.

    Args:
        agent_id: Include agent-specific statistics.
        context_variables: Context including memory_store reference.

    Returns:
        Dictionary with memory statistics.

    Example:
        >>> get_memory_statistics(agent_id="assistant")
    """
    memory_store = context_variables.get("memory_store") if context_variables else None
    if memory_store is None:
        return {"status": "error", "message": "Memory store not available"}

    try:
        stats = memory_store.get_statistics()

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


MEMORY_TOOLS: list[Callable[..., Any]] = [
    save_memory,
    search_memory,
    consolidate_agent_memories,
    delete_memory,
    get_memory_statistics,
    get_memory_tags_and_terms,
]


def get_memory_tool_descriptions() -> list[dict[str, str]]:
    """Get descriptions of all available memory tools.

    Returns:
        List of dictionaries with tool name, description, and category.

    Example:
        >>> for tool in get_memory_tool_descriptions():
        ...     print(f"{tool['name']}: {tool['description']}")
    """
    descriptions = []
    for tool in MEMORY_TOOLS:
        doc = tool.__doc__ or ""
        first_line = doc.strip().split("\n")[0] if doc else ""
        descriptions.append(
            {
                "name": tool.__name__,
                "description": first_line.strip(),
                "category": "Memory Management",
            }
        )
    return descriptions


def add_memory_tools_to_agent(
    agent: Agent,
    memory_store=None,
    include_tools: list[str] | None = None,
) -> Agent:
    """Add memory tools to an agent's function list.

    Args:
        agent: Agent instance to add tools to.
        memory_store: Memory store for tools to use.
        include_tools: List of tool names to include, or None for all.

    Returns:
        The modified agent with memory tools attached.

    Example:
        >>> agent = Agent(id="assistant", instructions="You are helpful")
        >>> add_memory_tools_to_agent(agent, memory_store=store)
    """
    current_functions = list(agent.functions) if agent.functions else []

    if include_tools is None:
        tools_to_add = MEMORY_TOOLS
    else:
        tool_map: dict[str, Callable[..., Any]] = {tool.__name__: tool for tool in MEMORY_TOOLS}
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
    """Create a new agent with memory tools pre-configured.

    Args:
        agent_id: Unique identifier for the agent.
        instructions: System instructions for the agent.
        memory_store: Memory store for the agent to use.
        memory_tools: Specific memory tools to include.
        **agent_kwargs: Additional agent configuration.

    Returns:
        A new Agent instance with memory tools attached.

    Example:
        >>> agent = create_memory_enabled_agent(
        ...     agent_id="assistant",
        ...     instructions="You are helpful",
        ...     memory_store=store
        ... )
    """
    agent = Agent(id=agent_id, instructions=instructions, functions=[], **agent_kwargs)
    agent = add_memory_tools_to_agent(agent, memory_store=memory_store, include_tools=memory_tools)
    return agent


class MemoryToolContext:
    """Context manager for memory tool execution.

    Provides automatic memory store injection for memory tool calls.

    Example:
        >>> with MemoryToolContext(memory_store) as ctx:
        ...     ctx.wrap_function_call(save_memory, content="Hello")
    """

    def __init__(self, memory_store: Any) -> None:
        """Initialize with a memory store.

        Args:
            memory_store: Memory store instance to use.
        """
        self.memory_store = memory_store

    def wrap_function_call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Wrap a function call to inject memory store.

        Args:
            func: Function to wrap.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            Result of the wrapped function call.
        """
        if "context_variables" not in kwargs:
            kwargs["context_variables"] = {}

        kwargs["context_variables"]["memory_store"] = self.memory_store
        return func(*args, **kwargs)
