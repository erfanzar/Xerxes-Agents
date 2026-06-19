# Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Live context-window accounting for UI and daemon status."""

from __future__ import annotations

from typing import Any

from .token_counter import SmartTokenCounter


def estimate_context_tokens(messages: list[dict[str, Any]], *, model: str) -> int:
    """Return the estimated token count for messages currently in the prompt window."""
    if not messages:
        return 0
    try:
        return max(0, SmartTokenCounter(model=model).count_tokens(messages))
    except Exception:
        text = "\n".join(f"{message.get('role', '')}: {message.get('content', '')}" for message in messages)
        return max(0, len(text) // 4)
