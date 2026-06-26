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
"""Tests for the daemon ``complete`` RPC helpers used by ui-tui completion."""

from __future__ import annotations

from types import SimpleNamespace

from xerxes.daemon.server import DaemonServer


def _daemon_with_skills(
    skills: list[str] | None = None,
    *,
    descriptions: dict[str, str] | None = None,
    subcommands: dict[str, list[str]] | None = None,
) -> DaemonServer:
    descriptions = descriptions or {}
    subcommands = subcommands or {}
    roots = sorted({skill.partition(":")[0] for skill in skills or []})
    skill_objects = [
        SimpleNamespace(
            name=root,
            metadata=SimpleNamespace(
                description=descriptions.get(root, ""),
                subcommands=subcommands.get(root, []),
            ),
        )
        for root in roots
    ]
    server = DaemonServer.__new__(DaemonServer)
    server.runtime = SimpleNamespace(
        discover_skills=lambda: skills or [],
        skill_registry=SimpleNamespace(get_all=lambda: skill_objects),
    )
    return server


def test_complete_slash_matches_command_names() -> None:
    out = _daemon_with_skills()._complete_slash("/prov")
    labels = [c["label"] for c in out]
    assert "provider" in labels
    # Every entry is a ready-to-insert /value with a description meta.
    assert all(c["value"].startswith("/") for c in out)
    assert all("meta" in c for c in out)


def test_complete_slash_empty_prefix_lists_many() -> None:
    out = _daemon_with_skills()._complete_slash("/")
    assert len(out) > 5
    assert len(out) <= 50


def test_complete_slash_no_match_is_empty() -> None:
    assert _daemon_with_skills()._complete_slash("/zzzznotacommand") == []


def test_complete_slash_includes_skills() -> None:
    out = _daemon_with_skills(["deepscan", "eternal-army"])._complete_slash("/dee")
    assert out == [{"value": "/deepscan", "label": "deepscan", "meta": "skill"}]


def test_complete_slash_empty_prefix_reserves_skill_slots() -> None:
    out = _daemon_with_skills(["deepscan", "eternal-army"])._complete_slash("/")
    labels = [c["label"] for c in out]
    assert "deepscan" in labels
    assert "eternal-army" in labels


def test_commands_catalog_includes_commands_aliases_and_skills() -> None:
    out = _daemon_with_skills(
        ["deepscan", "eternal-army", "eternal-army:round"],
        descriptions={"deepscan": "deep scan", "eternal-army": "swarm"},
        subcommands={"eternal-army": ["round"]},
    )._commands_catalog()

    assert out["canon"]["/provider"] == "/provider"
    assert out["canon"]["/thinking"] == "/reasoning"
    assert out["skill_count"] == 3
    assert out["sub"] == {"eternal-army": ["round"]}
    assert ["/deepscan", "deep scan"] in out["pairs"]
    assert any(category["name"] == "project skills" for category in out["categories"])


def test_complete_path_lists_directory_entries(tmp_path) -> None:
    (tmp_path / "alpha.txt").write_text("x")
    (tmp_path / "alphabeta").mkdir()
    (tmp_path / "other.txt").write_text("y")

    out = DaemonServer._complete_path(f"{tmp_path}/alph")
    labels = {c["label"] for c in out}
    assert "alpha.txt" in labels
    assert "alphabeta/" in labels  # directories get a trailing slash
    assert "other.txt" not in labels
    # value preserves the typed directory prefix
    assert any(c["value"] == f"{tmp_path}/alphabeta/" for c in out)


def test_complete_path_honors_at_mention_prefix(tmp_path) -> None:
    (tmp_path / "notes.md").write_text("x")
    out = DaemonServer._complete_path(f"@{tmp_path}/no")
    assert any(c["value"] == f"@{tmp_path}/notes.md" for c in out)


def test_complete_path_ignores_non_path_tokens() -> None:
    # A bare word with no slash isn't a path completion.
    assert DaemonServer._complete_path("hello wor") == []


def test_complete_path_hides_dotfiles_unless_requested(tmp_path) -> None:
    (tmp_path / ".hidden").write_text("x")
    (tmp_path / "visible").write_text("y")
    assert {c["label"] for c in DaemonServer._complete_path(f"{tmp_path}/")} == {"visible"}
    assert any(c["label"] == ".hidden" for c in DaemonServer._complete_path(f"{tmp_path}/."))
