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
"""Tests for send_message_tool, clarify_tool, memory_crud."""

from __future__ import annotations

from xerxes.tools import memory_crud, send_message_tool
from xerxes.tools.clarify_tool import StaticAsker, clarify

# ---------------------------- send_message ---------------------------------


class TestSendMessage:
    def test_missing_platform(self):
        out = send_message_tool.send_message(platform="", recipient="r", text="x")
        assert out["ok"] is False

    def test_missing_recipient(self):
        out = send_message_tool.send_message(platform="x", recipient="", text="x")
        assert out["ok"] is False

    def test_missing_payload(self):
        out = send_message_tool.send_message(platform="telegram", recipient="123")
        assert out["ok"] is False

    def test_unknown_platform(self):
        out = send_message_tool.send_message(platform="bogus", recipient="r", text="x")
        assert out["ok"] is False
        assert "unknown platform" in out["error"]

    def test_registered_platform_invoked(self):
        seen = {}

        def handler(platform, recipient, payload):
            seen.update(dict(platform=platform, recipient=recipient, payload=payload))
            return {"ok": True, "id": "msg-1"}

        send_message_tool.register_platform("telegram", handler)
        out = send_message_tool.send_message(platform="telegram", recipient="123", text="hi")
        assert out == {"ok": True, "id": "msg-1"}
        assert seen["platform"] == "telegram"
        assert seen["recipient"] == "123"
        assert seen["payload"]["text"] == "hi"

    def test_reply_to_propagates(self):
        captured = {}

        def handler(p, r, payload):
            captured["payload"] = payload
            return {"ok": True}

        send_message_tool.register_platform("slack", handler)
        send_message_tool.send_message(platform="slack", recipient="C123", text="hi", reply_to="m456")
        assert captured["payload"]["reply_to"] == "m456"


# ---------------------------- clarify --------------------------------------


class TestClarify:
    def test_empty_question(self):
        out = clarify(question="")
        assert out["ok"] is False

    def test_no_options_no_freeform(self):
        out = clarify(question="?", options=None, allow_freeform=False)
        assert out["ok"] is False

    def test_no_asker_returns_needs_ui(self):
        out = clarify(question="Pick one", options=["a", "b"])
        assert out["ok"] is True
        assert out["needs_ui"] is True

    def test_with_static_asker_freetext(self):
        asker = StaticAsker(answer="my answer")
        out = clarify(question="?", asker=asker)
        assert out["answer"] == "my answer"
        assert out["answered"] is True

    def test_with_static_asker_index(self):
        asker = StaticAsker(index=1)
        out = clarify(question="?", options=["red", "blue", "green"], asker=asker)
        assert out["answer"] == "blue"
        assert out["selected_index"] == 1

    def test_skip(self):
        asker = StaticAsker(skip=True)
        out = clarify(question="?", asker=asker)
        assert out["skipped"] is True
        assert out["answered"] is False


# ---------------------------- memory CRUD ----------------------------------


class TestMemoryCrud:
    def test_memory_add_creates_file(self, tmp_path):
        out = memory_crud.memory_add(tmp_path, "User prefers Rust over Go")
        assert out["ok"] is True
        assert (tmp_path / "MEMORY.md").exists()

    def test_memory_list_returns_items(self, tmp_path):
        memory_crud.memory_add(tmp_path, "fact A")
        memory_crud.memory_add(tmp_path, "fact B")
        out = memory_crud.memory_list(tmp_path)
        assert [it["content"] for it in out["items"]] == ["fact A", "fact B"]
        assert [it["id"] for it in out["items"]] == [1, 2]

    def test_memory_replace(self, tmp_path):
        memory_crud.memory_add(tmp_path, "old fact")
        out = memory_crud.memory_replace(tmp_path, 1, "new fact")
        assert out["ok"] is True
        assert memory_crud.memory_list(tmp_path)["items"][0]["content"] == "new fact"

    def test_memory_replace_unknown_id(self, tmp_path):
        out = memory_crud.memory_replace(tmp_path, 99, "x")
        assert out["ok"] is False

    def test_memory_remove(self, tmp_path):
        memory_crud.memory_add(tmp_path, "a")
        memory_crud.memory_add(tmp_path, "b")
        memory_crud.memory_remove(tmp_path, 1)
        out = memory_crud.memory_list(tmp_path)
        assert [it["content"] for it in out["items"]] == ["b"]

    def test_user_add_writes_user_md(self, tmp_path):
        memory_crud.user_add(tmp_path, "Likes terse responses")
        assert (tmp_path / "USER.md").exists()
        out = memory_crud.user_list(tmp_path)
        assert out["items"][0]["content"] == "Likes terse responses"

    def test_user_crud_mirror_memory(self, tmp_path):
        memory_crud.user_add(tmp_path, "x")
        memory_crud.user_add(tmp_path, "y")
        memory_crud.user_replace(tmp_path, 2, "Y")
        memory_crud.user_remove(tmp_path, 1)
        out = memory_crud.user_list(tmp_path)
        assert [it["content"] for it in out["items"]] == ["Y"]

    def test_add_empty_content(self, tmp_path):
        out = memory_crud.memory_add(tmp_path, "  ")
        assert out["ok"] is False

    def test_list_limit(self, tmp_path):
        for s in ("a", "b", "c", "d"):
            memory_crud.memory_add(tmp_path, s)
        out = memory_crud.memory_list(tmp_path, limit=2)
        assert [it["content"] for it in out["items"]] == ["c", "d"]
