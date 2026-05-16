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
"""Streaming-loop adapters that bridge assistant turns into memory.

Provides two factories used by the runtime:

* ``make_turn_indexer_hook`` — returns a hook the streaming loop fires
  at the end of every assistant turn to capture the response in memory.
* ``make_memory_provider`` — returns a callable the loop uses to pull
  back context strings for the next turn.

Both are written to tolerate the various ``Memory.save``/``search``
signatures across tiers."""

from __future__ import annotations

import logging
import typing as tp

from .base import Memory, MemoryItem

logger = logging.getLogger(__name__)


def _coerce_text(response: tp.Any) -> str:
    """Normalise an assistant response (string, dict, object) into a plain text string.

    Handles the most common shapes: raw strings, ``{"content": ...}``
    blocks where content is itself a string or list of text/dict
    fragments, and objects exposing ``content`` or ``text`` attributes.
    Falls back to ``str(response)``."""

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
    """Build a ``post_turn`` hook that indexes each assistant turn into ``memory``.

    The returned hook is keyword-only and tolerant of unrecognised
    kwargs: it pulls ``agent_id`` and ``response`` from the call,
    coerces the response to text, and saves it when long enough. If
    the configured tier's ``save`` signature is narrower than expected
    a fallback path retries with just ``content``/``metadata``.

    Args:
        memory: The memory tier to index into.
        min_chars: Minimum response length (in characters) to record;
            shorter turns are silently skipped.
        importance: Importance score attached to the saved item.
        memory_type: Tier label written to ``MemoryItem.memory_type``."""

    def _hook(**kwargs: tp.Any) -> None:
        """Indexer hook fired after each assistant turn."""

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
    """Build a ``(agent_id, k) -> list[str]`` callable the loop uses to fetch context.

    The provider runs ``memory.search`` against either the agent id
    (when present) or the literal string ``"context"``, taking at most
    ``k`` items and returning their ``content`` strings. Tiers that
    don't accept ``use_semantic`` are retried without it.

    Args:
        memory: The memory tier to query.
        use_semantic: Pass through to the tier's ``search`` when
            supported; silently dropped otherwise."""

    def _provider(agent_id: str | None, k: int) -> list[str]:
        """Memory provider invoked once per turn to fetch context strings."""

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
