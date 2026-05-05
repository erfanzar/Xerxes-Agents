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
"""Turn indexer module for Xerxes.

Exports:
    - logger
    - make_turn_indexer_hook
    - make_memory_provider"""

from __future__ import annotations

import logging
import typing as tp

from .base import Memory, MemoryItem

logger = logging.getLogger(__name__)


def _coerce_text(response: tp.Any) -> str:
    """Internal helper to coerce text.

    Args:
        response (tp.Any): IN: response. OUT: Consumed during execution.
    Returns:
        str: OUT: Result of the operation."""

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
    """Make turn indexer hook.

    Args:
        memory (Memory): IN: memory. OUT: Consumed during execution.
        min_chars (int, optional): IN: min chars. Defaults to 32. OUT: Consumed during execution.
        importance (float, optional): IN: importance. Defaults to 0.5. OUT: Consumed during execution.
        memory_type (str, optional): IN: memory type. Defaults to 'turn'. OUT: Consumed during execution.
    Returns:
        tp.Callable[..., None]: OUT: Result of the operation."""

    def _hook(**kwargs: tp.Any) -> None:
        """Internal helper to hook.

        Args:
            **kwargs: IN: Additional keyword arguments. OUT: Passed through to downstream calls."""

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
    """Make memory provider.

    Args:
        memory (Memory): IN: memory. OUT: Consumed during execution.
        use_semantic (bool, optional): IN: use semantic. Defaults to True. OUT: Consumed during execution.
    Returns:
        tp.Callable[[str | None, int], list[str]]: OUT: Result of the operation."""

    def _provider(agent_id: str | None, k: int) -> list[str]:
        """Internal helper to provider.

        Args:
            agent_id (str | None): IN: agent id. OUT: Consumed during execution.
            k (int): IN: k. OUT: Consumed during execution.
        Returns:
            list[str]: OUT: Result of the operation."""

        query = agent_id or "context"
        try:
            items = memory.search(query, limit=k, use_semantic=use_semantic)
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
