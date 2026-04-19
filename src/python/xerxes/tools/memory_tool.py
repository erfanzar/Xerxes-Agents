# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# distributed under the License is distributed on an "AS IS" BASIS,
# See the License for the specific language governing permissions and
# limitations under the License.


"""Memory management tools for agents to read and write memories autonomously.

This module provides a complete memory management toolkit for Xerxes agents,
enabling them to store, retrieve, search, and manage memories. It includes:
- Save and retrieve memory entries with categorization and tagging
- Search memories by query, type, tags, and time range
- Consolidate and summarize agent memories
- Delete memories by various criteria
- Get memory statistics and tag information
- Helper functions to add memory tools to existing agents

The memory tools integrate with the Xerxes memory store system and support
multiple memory types: short_term, long_term, working, episodic, semantic,
and procedural.

Example:
    >>> from xerxes.tools.memory_tool import save_memory, search_memory
    >>> result = save_memory(
    ...     content="User prefers dark mode",
    ...     tags=["preference", "ui"],
    ...     context_variables={"memory_store": store}
    ... )
    >>> memories = search_memory(
    ...     query="preferences",
    ...     context_variables={"memory_store": store}
    ... )
"""

from collections.abc import Callable
from datetime import datetime
from enum import Enum
from typing import Any

from ..memory import MemoryType
from ..types import Agent


class MemoryOperation(Enum):
    """Types of memory operations agents can perform.

    Enumeration of available memory operations for tracking
    and categorizing memory-related actions.

    Attributes:
        SAVE: Store a new memory entry.
        SEARCH: Search for memories by query and filters.
        RETRIEVE: Retrieve specific memories.
        DELETE: Remove memories from the store.
        SUMMARIZE: Generate summaries of memories.
        CONSOLIDATE: Consolidate related memories.
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

    Stores a piece of content as a memory entry with optional type
    classification, tagging, and metadata. Automatically timestamps the
    entry and associates it with the creating agent.

    Args:
        content: The text content to remember. This is the primary payload
            of the memory entry.
        memory_type: Classification of the memory. Determines how the
            memory is stored and retrieved. Options:
            - "short_term": Temporary, session-scoped memories.
            - "long_term": Persistent, cross-session memories.
            - "working": Active task-related memories.
            - "episodic": Event or experience memories.
            - "semantic": Factual knowledge memories.
            - "procedural": How-to and process memories.
        tags: List of string tags for categorization and filtering.
            Used for organizing and searching memories.
        metadata: Additional key-value metadata to store alongside the
            memory content. A timestamp and creator agent ID are
            automatically added.
        agent_id: Identifier of the agent creating this memory. If not
            provided, falls back to context_variables["agent_id"] or
            "default".
        context_variables: Runtime context dictionary from the agent. Must
            contain a "memory_store" key with an initialized memory store
            instance. May also contain "agent_id".

    Returns:
        A dictionary containing:
            - status (str): "success" or "error".
            - memory_id (str): Unique identifier of the saved memory
              (on success).
            - message (str): Human-readable status message.

    Example:
        >>> result = save_memory(
        ...     content="User prefers dark mode",
        ...     tags=["preference", "ui"],
        ...     context_variables={"memory_store": store}
        ... )
        >>> print(result["status"])
        'success'
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
    """Search for memories based on query and filters.

    Retrieves memories from the store that match the given query string
    and optional filters. Performs keyword-based matching against memory
    content and tags, with support for type filtering, tag filtering,
    and time range constraints.

    Args:
        query: Search query string. Words in the query are matched against
            memory content and tags using case-insensitive substring matching.
        memory_types: List of memory type names to restrict the search to.
            Options: "short_term", "long_term", "working", "episodic",
            "semantic", "procedural". If None, searches all types.
        tags: Filter results to only include memories that have at least
            one of the specified tags.
        limit: Maximum number of results to return. Defaults to 10.
        agent_id: Filter results to memories created by a specific agent.
            If not provided, falls back to context_variables["agent_id"]
            or "default".
        time_range: Optional time range filter as a dictionary with "start"
            and/or "end" keys containing ISO 8601 timestamp strings.
            Only memories within the range are returned.
        context_variables: Runtime context dictionary from the agent. Must
            contain a "memory_store" key with an initialized memory store
            instance. May also contain "agent_id".

    Returns:
        A dictionary containing:
            - status (str): "success" or "error".
            - count (int): Number of matching memories found.
            - memories (list[dict]): List of memory dicts, each with
              content, tags, timestamp, and metadata.
            - query (str): The original search query.
            - message (str): Error message (on failure).

    Example:
        >>> result = search_memory(
        ...     query="preferences",
        ...     tags=["ui"],
        ...     limit=5,
        ...     context_variables={"memory_store": store}
        ... )
        >>> print(result["count"])
        2
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
    """Get a consolidated summary of memories for a specific agent.

    Retrieves up to ``max_items`` memories for the given agent and groups
    them by tag into a human-readable summary. Also returns overall memory
    store statistics.

    Args:
        agent_id: Identifier of the agent whose memories to consolidate.
        max_items: Maximum number of memories to include in the
            consolidation. Defaults to 20.
        context_variables: Runtime context dictionary from the agent. Must
            contain a "memory_store" key with an initialized memory store
            instance.

    Returns:
        A dictionary containing:
            - status (str): "success" or "error".
            - summary (str): Human-readable summary of memories organized
              by tag category. Shows up to 3 items per tag with overflow
              counts.
            - statistics (dict): Overall memory store statistics from
              the underlying store's get_statistics() method.
            - message (str): Error message (on failure).

    Example:
        >>> result = consolidate_agent_memories(
        ...     agent_id="research_agent",
        ...     context_variables={"memory_store": store}
        ... )
        >>> print(result["summary"])
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
    """Delete memories based on one or more criteria.

    Removes memories from the store matching the specified criteria.
    At least one criterion (memory_id, tags, agent_id, or older_than)
    must be provided. When ``memory_id`` is given, it takes precedence
    and deletes that specific memory. Otherwise, the remaining criteria
    are combined as filters.

    Args:
        memory_id: Specific memory ID to delete. If provided, deletes
            exactly this memory regardless of other criteria.
        tags: Delete all memories that have any of these tags.
        agent_id: Delete all memories created by this agent.
        older_than: Delete memories created before this ISO 8601
            timestamp string (e.g., "2024-01-01T00:00:00").
        context_variables: Runtime context dictionary from the agent. Must
            contain a "memory_store" key with an initialized memory store
            instance.

    Returns:
        A dictionary containing:
            - status (str): "success" or "error".
            - deleted_count (int): Number of memories deleted (on success).
            - message (str): Human-readable status or error message.

    Example:
        >>> result = delete_memory(
        ...     tags=["temporary"],
        ...     context_variables={"memory_store": store}
        ... )
        >>> print(result["deleted_count"])
        3
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
    """Get all available memory tags organized by memory type.

    Retrieves all tags from the memory store, grouped by memory type
    (short_term, long_term, working, episodic, semantic, procedural).
    Also provides tag frequency counts and overall statistics.

    Args:
        agent_id: Filter tags to those associated with a specific agent.
            If not provided, falls back to context_variables["agent_id"]
            or "default".
        context_variables: Runtime context dictionary from the agent. Must
            contain a "memory_store" key with an initialized memory store
            instance. May also contain "agent_id".

    Returns:
        A dictionary containing:
            - status (str): "success" or "error".
            - tags_by_type (dict[str, list[str]]): Tags grouped by memory
              type. Only types with tags are included.
            - all_tags (list[str]): Sorted list of all unique tags.
            - tag_frequency (dict[str, int]): Tag-to-count mapping sorted
              by frequency (descending).
            - total_unique_tags (int): Total count of unique tags.
            - agent_id (str): The agent ID used for filtering.
            - message (str): Error message (on failure).

    Example:
        >>> result = get_memory_tags_and_terms(
        ...     context_variables={"memory_store": store}
        ... )
        >>> print(result["all_tags"])
        ['preference', 'ui', 'workflow']
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
    """Get statistics about memory usage.

    Retrieves overall memory store statistics and optionally adds
    agent-specific memory counts.

    Args:
        agent_id: If provided, includes the count of memories belonging
            to this specific agent in the returned statistics.
        context_variables: Runtime context dictionary from the agent. Must
            contain a "memory_store" key with an initialized memory store
            instance.

    Returns:
        A dictionary containing:
            - status (str): "success" or "error".
            - statistics (dict): Memory store statistics from the
              underlying store's get_statistics() method. When
              ``agent_id`` is provided, also includes:
              - agent_memory_count (int): Number of memories for that agent.
              - agent_id (str): The agent ID queried.
            - message (str): Error message (on failure).

    Example:
        >>> result = get_memory_statistics(
        ...     agent_id="main_agent",
        ...     context_variables={"memory_store": store}
        ... )
        >>> print(result["statistics"]["agent_memory_count"])
        42
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
    """Get descriptions of all memory tools for agent documentation.

    Iterates over the MEMORY_TOOLS list and extracts the name,
    first-line description, and category for each tool function.

    Returns:
        A list of dictionaries, each containing:
            - name (str): The function name of the tool.
            - description (str): First line of the tool's docstring.
            - category (str): Always "Memory Management".
    """
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
    """Add memory management tools to an agent's function list.

    Appends the selected memory tool functions to the agent's functions
    list, avoiding duplicates. Optionally sets the memory store on the
    agent instance if the agent has a ``_memory_store`` attribute.

    Args:
        agent: The Agent instance to add memory tools to. Its ``functions``
            list will be modified in-place.
        memory_store: The memory store instance to associate with the agent.
            If the agent has a ``_memory_store`` attribute, it will be set
            to this value.
        include_tools: List of specific tool function names to include.
            If None, all tools from MEMORY_TOOLS are added. Valid names:
            "save_memory", "search_memory", "consolidate_agent_memories",
            "delete_memory", "get_memory_statistics", "get_memory_tags_and_terms".

    Returns:
        The same Agent instance with memory tools appended to its
        functions list.

    Example:
        >>> from xerxes import Agent
        >>> agent = Agent(id="assistant", instructions="You help users", functions=[])
        >>> agent = add_memory_tools_to_agent(agent, memory_store=my_store)
        >>> agent = add_memory_tools_to_agent(
        ...     agent, include_tools=["save_memory", "search_memory"]
        ... )
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

    Convenience factory that creates an Agent instance and immediately
    adds memory management tools to it via ``add_memory_tools_to_agent``.

    Args:
        agent_id: Unique identifier for the new agent.
        instructions: System prompt / instructions for the agent.
        memory_store: Memory store instance to associate with the agent.
            Passed through to ``add_memory_tools_to_agent``.
        memory_tools: List of memory tool function names to include.
            If None, all memory tools are added. See
            ``add_memory_tools_to_agent`` for valid names.
        **agent_kwargs: Additional keyword arguments forwarded to the
            Agent constructor (e.g., model, temperature).

    Returns:
        A new Agent instance with the specified memory tools already
        added to its functions list.

    Example:
        >>> agent = create_memory_enabled_agent(
        ...     agent_id="assistant",
        ...     instructions="You remember past conversations",
        ...     memory_store=my_store,
        ...     memory_tools=["save_memory", "search_memory"]
        ... )
    """
    agent = Agent(id=agent_id, instructions=instructions, functions=[], **agent_kwargs)
    agent = add_memory_tools_to_agent(agent, memory_store=memory_store, include_tools=memory_tools)
    return agent


class MemoryToolContext:
    """Context manager for memory tools in function execution.

    Automatically provides memory_store in context_variables for
    memory tool functions. Use this to wrap function calls and
    ensure the memory store is always available.

    Attributes:
        memory_store: The memory store instance to inject into context.

    Example:
        >>> context = MemoryToolContext(memory_store)
        >>> result = context.wrap_function_call(save_memory, content="test")
    """

    def __init__(self, memory_store):
        """Initialize the memory tool context.

        Args:
            memory_store: The memory store instance to use for operations.
        """
        self.memory_store = memory_store

    def wrap_function_call(self, func, *args, **kwargs):
        """Wrap a function call to inject memory_store into context_variables.

        Args:
            func: The memory tool function to call.
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            The result of calling the wrapped function.
        """
        if "context_variables" not in kwargs:
            kwargs["context_variables"] = {}

        kwargs["context_variables"]["memory_store"] = self.memory_store
        return func(*args, **kwargs)
