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
"""Per-turn memory indexer hook.

Builds a hook callable that the runtime registers under
``on_turn_end``. Every completed agent turn is condensed into a
:class:`MemoryItem` (with an embedding) and persisted to the supplied
memory store, enabling cross-session semantic recall.

Usage::

    from xerxes_agent.memory import LongTermMemory
    from xerxes_agent.memory.turn_indexer import make_turn_indexer_hook

    memory = LongTermMemory()
    hook = make_turn_indexer_hook(memory)
    runtime_features.hook_runner.register("on_turn_end", hook)
"""

from __future__ import annotations

import logging
import typing as tp

from .base import Memory, MemoryItem

logger = logging.getLogger(__name__)


def _coerce_text(response: tp.Any) -> str:
    """Best-effort string extraction from a response payload.

    The runtime delivers many different response shapes — raw strings,
    ``ChatMessage``-like objects, dicts with ``content`` keys, lists of
    chunks, etc. This helper accepts any of those and returns a plain
    string suitable for embedding.

    Args:
        response: The runtime response in any supported shape.

    Returns:
        A plain string. Empty when nothing extractable is found.
    """
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        content = response.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for c in content:
                if isinstance(c, str):
                    parts.append(c)
                elif isinstance(c, dict) and isinstance(c.get("text"), str):
                    parts.append(c["text"])
            return "\n".join(parts)
    text_attr = getattr(response, "content", None)
    if isinstance(text_attr, str):
        return text_attr
    text_attr = getattr(response, "text", None)
    if isinstance(text_attr, str):
        return text_attr
    return str(response)


def make_turn_indexer_hook(
    memory: Memory,
    *,
    min_chars: int = 32,
    importance: float = 0.5,
    memory_type: str = "turn",
) -> tp.Callable[..., None]:
    """Build an ``on_turn_end`` hook that indexes completed turns.

    Each invocation extracts text from the agent response, skips empty
    or trivially short turns, and persists the content to ``memory``
    along with ``agent_id`` and the chosen ``importance``. Failures are
    swallowed and logged so a misbehaving memory store can never break
    the agent loop.

    Args:
        memory: A :class:`Memory` implementation (e.g. :class:`LongTermMemory`).
        min_chars: Minimum response length to index. Skips short
            acknowledgements like "ok" / "done".
        importance: Importance score attached to the saved memory.
        memory_type: ``memory_type`` field on the saved item.

    Returns:
        A callable suitable for :meth:`HookRunner.register("on_turn_end", ...)`.
    """

    def _hook(**kwargs: tp.Any) -> None:
        """Persist the turn's agent response to *memory* if it is long enough."""
        agent_id = kwargs.get("agent_id")
        response = kwargs.get("response")
        text = _coerce_text(response).strip()
        if len(text) < min_chars:
            return
        try:
            item = memory.save(
                content=text,
                metadata={"source": "turn_indexer"},
                agent_id=agent_id,
                importance=importance,
                memory_type=memory_type,
            )
            logger.debug("Indexed turn %s for agent=%s", item.memory_id, agent_id)
        except TypeError:
            try:
                memory.save(content=text, metadata={"source": "turn_indexer"})
            except Exception:
                logger.debug("Memory.save fallback failed", exc_info=True)
        except Exception:
            logger.warning("turn_indexer hook failed to save memory", exc_info=True)

    return _hook


def make_memory_provider(
    memory: Memory,
    *,
    use_semantic: bool = True,
) -> tp.Callable[[str | None, int], list[str]]:
    """Build a ``memory_provider`` callable for :class:`PromptContextBuilder`.

    The returned callable performs ``memory.search(query, limit=k)``
    using a *recent context* hint as the query (the agent_id helps scope
    when multiple agents share a memory). Results are formatted into
    short snippets suitable for ``[Relevant Memories]`` injection.

    For now ``query`` is derived from the agent_id (a coarse signal).
    A future iteration will pull the recent user message instead.

    Args:
        memory: A :class:`Memory` instance.
        use_semantic: If supported by the storage backend, perform
            semantic search instead of keyword matching.

    Returns:
        A callable ``(agent_id, k) -> list[str]``.
    """

    def _provider(agent_id: str | None, k: int) -> list[str]:
        """Search *memory* for up to *k* snippets scoped by *agent_id*."""
        query = agent_id or "context"
        try:
            items = memory.search(query, limit=k, use_semantic=use_semantic)  # type: ignore[call-arg]
        except TypeError:
            try:
                items = memory.search(query, limit=k)
            except Exception:
                return []
        except Exception:
            return []
        out: list[str] = []
        for it in items[:k]:
            content = getattr(it, "content", None)
            if isinstance(content, str) and content:
                out.append(content)
        return out

    return _provider


__all__ = ["MemoryItem", "make_memory_provider", "make_turn_indexer_hook"]
