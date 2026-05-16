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
"""Tests for xerxes.tools.skill_manage_tool."""

from __future__ import annotations

import pytest
from xerxes.tools import skill_manage_tool


@pytest.fixture(autouse=True)
def isolated_dir(tmp_path, monkeypatch):
    skills = tmp_path / "skills"
    skills.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(skill_manage_tool, "AUTHORED_DIR", skills)

    def fake_path(name):
        skills.mkdir(parents=True, exist_ok=True)
        return skills / f"{name}.md"

    monkeypatch.setattr(skill_manage_tool, "_skill_path", fake_path)
    return skills


def test_list_empty(isolated_dir):
    out = skill_manage_tool.skill_manage(intent="list")
    assert out == {"ok": True, "intent": "list", "skills": []}


def test_create_writes_file(isolated_dir):
    out = skill_manage_tool.skill_manage(
        intent="create",
        name="my-skill",
        body="# Hi\nDo a thing.",
        description="my desc",
    )
    assert out["ok"] is True
    assert (isolated_dir / "my-skill.md").exists()
    text = (isolated_dir / "my-skill.md").read_text()
    assert "name: my-skill" in text
    assert "description: my desc" in text
    assert "Do a thing." in text


def test_create_then_create_fails(isolated_dir):
    skill_manage_tool.skill_manage(intent="create", name="x", body="content")
    out = skill_manage_tool.skill_manage(intent="create", name="x", body="content")
    assert out["ok"] is False
    assert "already exists" in out["error"]


def test_edit_overwrites(isolated_dir):
    skill_manage_tool.skill_manage(intent="create", name="x", body="v1")
    out = skill_manage_tool.skill_manage(intent="edit", name="x", body="v2")
    assert out["ok"] is True
    assert "v2" in (isolated_dir / "x.md").read_text()


def test_view_missing(isolated_dir):
    out = skill_manage_tool.skill_manage(intent="view", name="ghost")
    assert out["ok"] is False


def test_view_existing(isolated_dir):
    skill_manage_tool.skill_manage(intent="create", name="x", body="content body")
    out = skill_manage_tool.skill_manage(intent="view", name="x")
    assert out["ok"] is True
    assert "content body" in out["body"]


def test_delete(isolated_dir):
    skill_manage_tool.skill_manage(intent="create", name="x", body="b")
    out = skill_manage_tool.skill_manage(intent="delete", name="x")
    assert out["ok"] is True
    assert not (isolated_dir / "x.md").exists()


def test_delete_missing(isolated_dir):
    out = skill_manage_tool.skill_manage(intent="delete", name="ghost")
    assert out["ok"] is False


def test_create_empty_body_fails(isolated_dir):
    out = skill_manage_tool.skill_manage(intent="create", name="x", body="   ")
    assert out["ok"] is False
    assert "non-empty" in out["error"]


def test_invalid_name(isolated_dir):
    with pytest.raises(ValueError):
        skill_manage_tool.skill_manage(intent="create", name="../escape", body="b")


def test_unknown_intent(isolated_dir):
    out = skill_manage_tool.skill_manage(intent="bogus", name="x")
    assert out["ok"] is False


def test_list_returns_existing(isolated_dir):
    skill_manage_tool.skill_manage(intent="create", name="a", body="b")
    skill_manage_tool.skill_manage(intent="create", name="b", body="b")
    out = skill_manage_tool.skill_manage(intent="list")
    assert sorted(out["skills"]) == ["a", "b"]
