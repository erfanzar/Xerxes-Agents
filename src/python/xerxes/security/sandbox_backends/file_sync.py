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
"""Bidirectional file-sync helpers for remote sandbox backends.

Used by the Modal / Daytona / SSH backends to push local files into
the remote workspace before exec and pull artifacts back after.

The push/pull functions are protocol-agnostic — they take a callable
that performs one round-trip per file. Tests pass a fake to verify
ordering and bounding.

Exports:
    - FileSyncSpec
    - sync_push
    - sync_pull"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CopyFn = Callable[[Path, str], None]


@dataclass
class FileSyncSpec:
    """One file to sync.

    Attributes:
        local_path: source on the local filesystem.
        remote_path: destination on the remote.
        bytes_estimate: optional pre-stat byte count; populated by ``stat``."""

    local_path: Path
    remote_path: str
    bytes_estimate: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


def sync_push(
    specs: list[FileSyncSpec],
    copy_fn: CopyFn,
    *,
    max_bytes: int | None = None,
) -> list[FileSyncSpec]:
    """Push each spec via ``copy_fn``. Skips files larger than ``max_bytes``."""
    out: list[FileSyncSpec] = []
    for spec in specs:
        if not spec.local_path.exists():
            continue
        size = spec.local_path.stat().st_size
        spec.bytes_estimate = size
        if max_bytes is not None and size > max_bytes:
            continue
        copy_fn(spec.local_path, spec.remote_path)
        out.append(spec)
    return out


def sync_pull(
    specs: list[FileSyncSpec],
    pull_fn: CopyFn,
) -> list[FileSyncSpec]:
    """Pull each remote_path back to the local_path via ``pull_fn``.

    The ``pull_fn`` semantics mirror ``copy_fn`` (signature is
    ``(local_path, remote_path)``) but transfer direction is reversed."""
    out: list[FileSyncSpec] = []
    for spec in specs:
        try:
            pull_fn(spec.local_path, spec.remote_path)
            out.append(spec)
        except Exception:
            continue
    return out


__all__ = ["FileSyncSpec", "sync_pull", "sync_push"]
