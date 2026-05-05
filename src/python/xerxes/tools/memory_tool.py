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
"""Memory tool module for Xerxes.

Exports:
    - MemoryOperation
    - save_memory
    - search_memory
    - consolidate_agent_memories
    - delete_memory
    - get_memory_tags_and_terms
    - get_memory_statistics
    - get_memory_tool_descriptions
    - add_memory_tools_to_agent
    - create_memory_enabled_agent
    - ... and 1 more."""

from collections.abc import Callable
from datetime import datetime
from enum import Enum
from typing import Any

from ..memory import MemoryType
from ..types import Agent


class MemoryOperation(Enum):
    """Memory operation.

    Inherits from: Enum
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
    """Save memory.

    Args:
        content (str): IN: content. OUT: Consumed during execution.
        memory_type (str, optional): IN: memory type. Defaults to 'short_term'. OUT: Consumed during execution.
        tags (list[str] | None, optional): IN: tags. Defaults to None. OUT: Consumed during execution.
        metadata (dict[str, Any] | None, optional): IN: metadata. Defaults to None. OUT: Consumed during execution.
        agent_id (str | None, optional): IN: agent id. Defaults to None. OUT: Consumed during execution.
        context_variables (dict[str, Any] | None, optional): IN: context variables. Defaults to None. OUT: Consumed during execution.
    Returns:
        dict[str, Any]: OUT: Result of the operation."""

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
    """Search for memory.

    Args:
        query (str): IN: query. OUT: Consumed during execution.
        memory_types (list[str] | None, optional): IN: memory types. Defaults to None. OUT: Consumed during execution.
        tags (list[str] | None, optional): IN: tags. Defaults to None. OUT: Consumed during execution.
        limit (int, optional): IN: limit. Defaults to 10. OUT: Consumed during execution.
        agent_id (str | None, optional): IN: agent id. Defaults to None. OUT: Consumed during execution.
        time_range (dict[str, str] | None, optional): IN: time range. Defaults to None. OUT: Consumed during execution.
        context_variables (dict[str, Any] | None, optional): IN: context variables. Defaults to None. OUT: Consumed during execution.
    Returns:
        dict[str, Any]: OUT: Result of the operation."""

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
    """Consolidate agent memories.

    Args:
        agent_id (str): IN: agent id. OUT: Consumed during execution.
        max_items (int, optional): IN: max items. Defaults to 20. OUT: Consumed during execution.
        context_variables (dict[str, Any] | None, optional): IN: context variables. Defaults to None. OUT: Consumed during execution.
    Returns:
        dict[str, Any]: OUT: Result of the operation."""

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
    """Delete memory.

    Args:
        memory_id (str | None, optional): IN: memory id. Defaults to None. OUT: Consumed during execution.
        tags (list[str] | None, optional): IN: tags. Defaults to None. OUT: Consumed during execution.
        agent_id (str | None, optional): IN: agent id. Defaults to None. OUT: Consumed during execution.
        older_than (str | None, optional): IN: older than. Defaults to None. OUT: Consumed during execution.
        context_variables (dict[str, Any] | None, optional): IN: context variables. Defaults to None. OUT: Consumed during execution.
    Returns:
        dict[str, Any]: OUT: Result of the operation."""

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
    """Retrieve the memory tags and terms.

    Args:
        agent_id (str | None, optional): IN: agent id. Defaults to None. OUT: Consumed during execution.
        context_variables (dict[str, Any] | None, optional): IN: context variables. Defaults to None. OUT: Consumed during execution.
    Returns:
        dict[str, Any]: OUT: Result of the operation."""

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
    """Retrieve the memory statistics.

    Args:
        agent_id (str | None, optional): IN: agent id. Defaults to None. OUT: Consumed during execution.
        context_variables (dict[str, Any] | None, optional): IN: context variables. Defaults to None. OUT: Consumed during execution.
    Returns:
        dict[str, Any]: OUT: Result of the operation."""

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
    """Retrieve the memory tool descriptions.

    Returns:
        list[dict[str, str]]: OUT: Result of the operation."""

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
    """Add memory tools to agent.

    Args:
        agent (Agent): IN: agent. OUT: Consumed during execution.
        memory_store (Any, optional): IN: memory store. Defaults to None. OUT: Consumed during execution.
        include_tools (list[str] | None, optional): IN: include tools. Defaults to None. OUT: Consumed during execution.
    Returns:
        Agent: OUT: Result of the operation."""

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
    """Create memory enabled agent.

    Args:
        agent_id (str): IN: agent id. OUT: Consumed during execution.
        instructions (str): IN: instructions. OUT: Consumed during execution.
        memory_store (Any, optional): IN: memory store. Defaults to None. OUT: Consumed during execution.
        memory_tools (list[str] | None, optional): IN: memory tools. Defaults to None. OUT: Consumed during execution.
        **agent_kwargs: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
    Returns:
        Agent: OUT: Result of the operation."""

    agent = Agent(id=agent_id, instructions=instructions, functions=[], **agent_kwargs)
    agent = add_memory_tools_to_agent(agent, memory_store=memory_store, include_tools=memory_tools)
    return agent


class MemoryToolContext:
    """Memory tool context."""

    def __init__(self, memory_store):
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            memory_store (Any): IN: memory store. OUT: Consumed during execution.
        Returns:
            Any: OUT: Result of the operation."""

        self.memory_store = memory_store

    def wrap_function_call(self, func, *args, **kwargs):
        """Wrap function call.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            func (Any): IN: func. OUT: Consumed during execution.
            *args: IN: Additional positional arguments. OUT: Passed through to downstream calls.
            **kwargs: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            Any: OUT: Result of the operation."""

        if "context_variables" not in kwargs:
            kwargs["context_variables"] = {}

        kwargs["context_variables"]["memory_store"] = self.memory_store
        return func(*args, **kwargs)
