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
"""Tests for xerxes.session.migrations."""

from __future__ import annotations

import json

import pytest
from xerxes.session import migrations as migmod
from xerxes.session.migrations import MIGRATIONS, migrate_record, register
from xerxes.session.models import CURRENT_SCHEMA_VERSION, SessionRecord
from xerxes.session.store import FileSessionStore


class TestMigrationRegistry:
    def teardown_method(self) -> None:
        # Strip any test-registered migrations to avoid polluting later tests.
        for v in [99, 100, 101, 102]:
            MIGRATIONS.pop(v, None)

    def test_register_and_run(self) -> None:
        @register(100)
        def _upgrade(record):
            record["added_in_v100"] = True
            return record

        out = migrate_record({"schema_version": 99}, 100)
        assert out["schema_version"] == 100
        assert out["added_in_v100"] is True

    def test_chain_runs_in_order(self) -> None:
        @register(100)
        def _up_a(record):
            record["a"] = 1
            return record

        @register(101)
        def _up_b(record):
            record["b"] = record["a"] + 1
            return record

        out = migrate_record({"schema_version": 99}, 101)
        assert out["schema_version"] == 101
        assert out["a"] == 1
        assert out["b"] == 2

    def test_duplicate_registration_raises(self) -> None:
        @register(102)
        def _first(record):
            return record

        with pytest.raises(ValueError, match="Duplicate migration"):

            @register(102)
            def _second(record):
                return record

    def test_missing_migration_raises(self) -> None:
        with pytest.raises(RuntimeError, match="No migration registered"):
            migrate_record({"schema_version": 99}, 101)

    def test_records_at_or_above_target_are_unchanged(self) -> None:
        record = {"schema_version": 5, "foo": "bar"}
        out = migrate_record(record, 3)
        assert out is record

    def test_missing_schema_version_defaults_to_one(self) -> None:
        @register(100)
        def _from_one(record):
            assert int(record.get("schema_version", 1)) == 1
            return record

        out = migrate_record({"foo": "bar"}, 1)
        assert out["foo"] == "bar"


class TestSessionRecordRoundtrip:
    def test_to_dict_includes_schema_version(self) -> None:
        rec = SessionRecord(session_id="abc")
        d = rec.to_dict()
        assert d["schema_version"] == CURRENT_SCHEMA_VERSION

    def test_from_dict_preserves_schema_version(self) -> None:
        rec = SessionRecord.from_dict({"session_id": "x", "schema_version": 1})
        assert rec.schema_version == 1

    def test_from_dict_defaults_schema_version(self) -> None:
        rec = SessionRecord.from_dict({"session_id": "x"})
        assert rec.schema_version == CURRENT_SCHEMA_VERSION

    def test_parent_session_id_roundtrips(self) -> None:
        rec = SessionRecord(session_id="child", parent_session_id="parent")
        d = rec.to_dict()
        again = SessionRecord.from_dict(d)
        assert again.parent_session_id == "parent"


class TestFileStoreMigrationOnLoad:
    def teardown_method(self) -> None:
        for v in [100]:
            MIGRATIONS.pop(v, None)
        migmod.MIGRATIONS.pop(100, None)

    def test_load_runs_pending_migrations(self, tmp_path, monkeypatch) -> None:
        # Pretend the codebase now wants version 100 and ship a migration.
        @register(100)
        def _upgrade(record):
            record["upgraded"] = True
            return record

        # Force CURRENT to 100 just for this test by monkeypatching.
        from xerxes.session import models as mmod
        from xerxes.session import store as smod

        monkeypatch.setattr(mmod, "CURRENT_SCHEMA_VERSION", 100)
        monkeypatch.setattr(smod, "CURRENT_SCHEMA_VERSION", 100)

        # Write a stale v1 file.
        store = FileSessionStore(tmp_path)
        path = tmp_path / "s1.json"
        path.write_text(json.dumps({"session_id": "s1", "schema_version": 99}))
        # Pre-register migrations for the gap (99 -> 100 only — start from 99 manually).
        rec = store.load_session("s1")
        assert rec is not None
        # to_dict echoes schema_version; load_session ran the migration and rewrote the file.
        on_disk = json.loads(path.read_text())
        assert on_disk["schema_version"] == 100
        assert on_disk["upgraded"] is True

    def test_load_atomic_write_doesnt_leave_temp_files(self, tmp_path) -> None:
        store = FileSessionStore(tmp_path)
        rec = SessionRecord(session_id="s2")
        store.save_session(rec)
        # No stray .tmpXXX files in the directory.
        leftovers = [p.name for p in tmp_path.iterdir() if p.name.startswith(".s2.")]
        assert leftovers == []
