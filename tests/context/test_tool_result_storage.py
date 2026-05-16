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
"""Tests for xerxes.context.tool_result_storage."""

from __future__ import annotations

from xerxes.context.tool_result_storage import ToolResultStorage


class TestToolResultStorage:
    def test_short_content_returned_inline(self, tmp_path) -> None:
        store = ToolResultStorage(tmp_path, inline_limit=1_000)
        out = store.maybe_store("read_file", "hello")
        assert out == "hello"

    def test_long_string_overflows_to_disk(self, tmp_path) -> None:
        store = ToolResultStorage(tmp_path, inline_limit=10)
        out = store.maybe_store("read_file", "x" * 1000)
        assert isinstance(out, str)
        assert ToolResultStorage.is_ref(out)
        ref_id = ToolResultStorage.parse_ref(out)
        assert ref_id is not None
        assert (tmp_path / "default" / f"{ref_id}.json").exists()

    def test_fetch_returns_original_string(self, tmp_path) -> None:
        store = ToolResultStorage(tmp_path, inline_limit=10)
        big = "y" * 1000
        ref = store.maybe_store("web_fetch", big)
        # Fetch from cache (just inserted).
        assert store.fetch(ref) == big

    def test_fetch_from_disk_after_cache_eviction(self, tmp_path) -> None:
        store = ToolResultStorage(tmp_path, inline_limit=10, lru_size=2)
        ref_a = store.maybe_store("a", "a" * 1000)
        store.maybe_store("b", "b" * 1000)
        store.maybe_store("c", "c" * 1000)
        # Cache only holds 2 entries; ref_a should have been evicted.
        # Fetch must still succeed by reading the on-disk file.
        out = store.fetch(ref_a)
        assert out == "a" * 1000

    def test_dict_content_serialized(self, tmp_path) -> None:
        store = ToolResultStorage(tmp_path, inline_limit=10)
        data = {"key": "value" * 1000}
        ref = store.maybe_store("structured", data)
        # Stored under a ref, fetch returns the original dict.
        assert isinstance(ref, str)
        assert ToolResultStorage.is_ref(ref)
        # Cache still has the original; fetch returns it.
        assert store.fetch(ref) == data

    def test_same_payload_dedupes_on_disk(self, tmp_path) -> None:
        store = ToolResultStorage(tmp_path, inline_limit=10)
        ref1 = store.maybe_store("read", "x" * 1000)
        ref2 = store.maybe_store("read", "x" * 1000)
        assert ref1 == ref2  # same digest → same ref id

    def test_list_refs(self, tmp_path) -> None:
        store = ToolResultStorage(tmp_path, inline_limit=10)
        store.maybe_store("a", "x" * 1000)
        store.maybe_store("b", "y" * 1000)
        refs = store.list_refs()
        assert len(refs) == 2

    def test_prune_keeps_most_recent(self, tmp_path) -> None:
        store = ToolResultStorage(tmp_path, inline_limit=10)
        for i in range(5):
            store.maybe_store(f"t{i}", "z" * 1000)
        removed = store.prune(keep=2)
        assert removed == 3
        assert len(store.list_refs()) == 2

    def test_is_ref_negative_cases(self) -> None:
        assert not ToolResultStorage.is_ref("plain string")
        assert not ToolResultStorage.is_ref(None)
        assert not ToolResultStorage.is_ref(42)

    def test_parse_ref_invalid(self) -> None:
        assert ToolResultStorage.parse_ref("nope") is None

    def test_fetch_missing_returns_none(self, tmp_path) -> None:
        store = ToolResultStorage(tmp_path)
        assert store.fetch("does_not_exist") is None
