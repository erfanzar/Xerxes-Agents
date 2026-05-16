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
"""Session record migrations.

Migrations transform a session record dict from version N to version N+1.
The runner chains migrations so a v1 record can be loaded after the schema
advances to v3 without manual conversion.

Adding a migration:
    1. Bump CURRENT_SCHEMA_VERSION in xerxes/session/models.py.
    2. Add a file vNNN_short_description.py exporting a migrate(record) -> dict.
    3. Register it in MIGRATIONS below.
    4. Add a test in tests/session/test_migrations.py.

Forward-only — there are no downgrades. Each migration receives a dict whose
schema_version equals (target - 1) and returns a dict whose schema_version
equals (target)."""

from __future__ import annotations

import typing as tp

MigrationFn = tp.Callable[[dict[str, tp.Any]], dict[str, tp.Any]]

MIGRATIONS: dict[int, MigrationFn] = {}


def register(target_version: int) -> tp.Callable[[MigrationFn], MigrationFn]:
    """Decorator: register a migration that produces ``target_version``."""

    def deco(fn: MigrationFn) -> MigrationFn:
        if target_version in MIGRATIONS:
            raise ValueError(f"Duplicate migration registered for version {target_version}")
        MIGRATIONS[target_version] = fn
        return fn

    return deco


def migrate_record(record: dict[str, tp.Any], target_version: int) -> dict[str, tp.Any]:
    """Run migrations until record's schema_version reaches ``target_version``.

    Records without a schema_version field are assumed to be at version 1.
    Records at or above the target are returned unchanged."""

    current = int(record.get("schema_version", 1))
    if current > target_version:
        # Forward-only — we cannot downgrade. Leave the record alone and let
        # the caller decide what to do; commonly this means a newer client
        # wrote the file and an older client is reading it.
        return record
    while current < target_version:
        next_version = current + 1
        migration = MIGRATIONS.get(next_version)
        if migration is None:
            raise RuntimeError(
                f"No migration registered to produce schema_version={next_version}; "
                f"current={current}, target={target_version}"
            )
        record = migration(record)
        record["schema_version"] = next_version
        current = next_version
    return record


# Future migration files import and register against this module, e.g.:
#
# from . import register
#
# @register(2)
# def _v002_add_cache_tokens(record):
#     # add cache_read_tokens / cache_write_tokens fields to every turn
#     ...
#     return record
#
# As long as migration modules are imported below, they get registered. We
# import them lazily to avoid circulars and to keep the registry deterministic.
