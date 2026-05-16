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
"""Fork an existing session into a new one with the same history.

The child's ``parent_session_id`` points to the source so the lineage
is queryable. The metadata gets a ``forked_from`` annotation.

Exports:
    - branch_session
    - lineage"""

from __future__ import annotations

import copy
import uuid
from datetime import UTC, datetime

from .models import SessionRecord
from .store import SessionStore


def branch_session(
    store: SessionStore,
    *,
    source_session_id: str,
    title: str = "",
    new_session_id: str | None = None,
) -> SessionRecord:
    """Create a child session that shares the source's history.

    The new session inherits ``turns``, ``agent_transitions``, and
    ``metadata``; ``parent_session_id`` and timestamps are fresh."""
    source = store.load_session(source_session_id)
    if source is None:
        raise KeyError(f"unknown source session: {source_session_id}")
    now = datetime.now(UTC).isoformat()
    new = SessionRecord(
        session_id=new_session_id or uuid.uuid4().hex,
        workspace_id=source.workspace_id,
        created_at=now,
        updated_at=now,
        agent_id=source.agent_id,
        turns=copy.deepcopy(source.turns),
        agent_transitions=copy.deepcopy(source.agent_transitions),
        metadata={
            **source.metadata,
            "forked_from": source_session_id,
            "title": title or source.metadata.get("title", ""),
        },
        parent_session_id=source_session_id,
    )
    store.save_session(new)
    return new


def lineage(store: SessionStore, session_id: str) -> list[str]:
    """Return the chain of ``parent_session_id``s from this session to root.

    The first element is ``session_id``; the last has no parent."""
    chain: list[str] = []
    current = session_id
    while True:
        sess = store.load_session(current)
        if sess is None:
            break
        chain.append(sess.session_id)
        if not sess.parent_session_id:
            break
        current = sess.parent_session_id
    return chain


__all__ = ["branch_session", "lineage"]
