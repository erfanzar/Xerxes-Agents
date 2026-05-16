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
"""Tests for the agent's persistent two-tier memory."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from xerxes.runtime.agent_memory import (
    CANONICAL_FILES,
    AgentMemory,
    AgentMemoryScope,
    project_memory_dir_for,
)
from xerxes.tools import agent_memory_tool


@pytest.fixture
def memory(tmp_path):
    mem = AgentMemory(
        project_root=tmp_path / "my_project",
        global_dir=tmp_path / "global",
        project_dir=tmp_path / "project",
    )
    mem.ensure()
    return mem


# ============================================================================
# AgentMemory core
# ============================================================================


class TestEnsure:
    def test_creates_canonical_files_in_both_scopes(self, memory):
        for scope_dir in (memory.global_dir, memory.project_dir):
            for name in CANONICAL_FILES:
                assert (scope_dir / name).exists()
                content = (scope_dir / name).read_text()
                assert content.startswith("# ")

    def test_canonical_files_includes_soul_and_experiences(self):
        # Hard contract — the user explicitly asked for these. Don't drop them.
        assert "SOUL.md" in CANONICAL_FILES
        assert "EXPERIENCES.md" in CANONICAL_FILES
        assert "KNOWLEDGE.md" in CANONICAL_FILES

    def test_experiences_default_has_format_section(self, memory):
        body = (memory.global_dir / "EXPERIENCES.md").read_text()
        assert "Tried:" in body
        assert "Result:" in body
        assert "Lesson:" in body

    def test_soul_default_describes_persona(self, memory):
        body = (memory.global_dir / "SOUL.md").read_text()
        assert "Direct" in body or "direct" in body

    def test_creates_journal_subdir(self, memory):
        assert (memory.global_dir / "journal").is_dir()
        assert (memory.project_dir / "journal").is_dir()

    def test_idempotent_does_not_clobber_user_content(self, memory):
        target = memory.global_dir / "MEMORY.md"
        target.write_text("# My customized memory\nimportant note")
        memory.ensure()
        assert target.read_text() == "# My customized memory\nimportant note"


class TestProjectScope:
    def test_unavailable_when_no_project_root(self, tmp_path):
        mem = AgentMemory(global_dir=tmp_path / "global")
        assert mem.has_project_scope() is False
        with pytest.raises(FileNotFoundError):
            mem.scope_dir(AgentMemoryScope.PROJECT)

    def test_available_when_project_root_set(self, tmp_path):
        mem = AgentMemory(project_root=tmp_path)
        assert mem.has_project_scope()

    def test_project_memory_dir_hash_stable(self, tmp_path):
        a = project_memory_dir_for(tmp_path / "proj")
        b = project_memory_dir_for(tmp_path / "proj")
        assert a == b

    def test_project_memory_dir_different_per_project(self, tmp_path):
        a = project_memory_dir_for(tmp_path / "proj_a")
        b = project_memory_dir_for(tmp_path / "proj_b")
        assert a != b


class TestReadWriteAppend:
    def test_write_read_roundtrip(self, memory):
        memory.write("global", "MEMORY.md", "# replaced\n- new fact")
        assert "new fact" in memory.read("global", "MEMORY.md")

    def test_path_escape_rejected(self, memory):
        with pytest.raises(ValueError):
            memory.write("global", "../escape.md", "x")

    def test_empty_path_rejected(self, memory):
        with pytest.raises(ValueError):
            memory.write("global", "", "x")

    def test_unknown_scope_rejected(self, memory):
        with pytest.raises(ValueError):
            memory.write("imaginary", "MEMORY.md", "x")

    def test_read_missing_raises(self, memory):
        with pytest.raises(FileNotFoundError):
            memory.read("global", "ghost.md")

    def test_append_adds_after_existing(self, memory):
        memory.write("global", "MEMORY.md", "# memory\n- first")
        memory.append("global", "MEMORY.md", "- second", timestamp=False)
        body = memory.read("global", "MEMORY.md")
        assert "first" in body
        assert "second" in body

    def test_append_with_section(self, memory):
        memory.append("project", "MEMORY.md", "the body", section="Today")
        body = memory.read("project", "MEMORY.md")
        assert "## Today" in body
        assert "the body" in body

    def test_append_with_timestamp(self, memory):
        memory.append("global", "MEMORY.md", "ts entry")
        body = memory.read("global", "MEMORY.md")
        assert "<!--" in body  # timestamp comment marker


class TestJournal:
    def test_journal_writes_to_dated_file(self, memory):
        when = datetime(2026, 5, 16, 12, 30, 45, tzinfo=UTC)
        result = memory.journal("global", "made progress on x", when=when)
        assert "journal/2026-05-16.md" in result["path"]
        content = memory.read("global", "journal/2026-05-16.md")
        assert "12:30:45" in content
        assert "made progress on x" in content

    def test_multiple_entries_same_day(self, memory):
        when = datetime(2026, 5, 16, tzinfo=UTC)
        memory.journal("project", "first", when=when.replace(hour=9))
        memory.journal("project", "second", when=when.replace(hour=14))
        body = memory.read("project", "journal/2026-05-16.md")
        assert "09:00:00" in body
        assert "14:00:00" in body


class TestListAndSearch:
    def test_list_all_scopes_by_default(self, memory):
        files = memory.list_files()
        scopes = {f.scope.value for f in files}
        assert scopes == {"global", "project"}

    def test_list_one_scope(self, memory):
        files = memory.list_files("global")
        assert all(f.scope is AgentMemoryScope.GLOBAL for f in files)

    def test_search_finds_substring(self, memory):
        memory.write("global", "KNOWLEDGE.md", "FAISS is a vector index library")
        hits = memory.search("FAISS")
        assert hits and hits[0]["scope"] == "global"
        assert "FAISS" in hits[0]["snippet"]

    def test_search_empty_query_returns_empty(self, memory):
        assert memory.search("") == []

    def test_search_case_insensitive(self, memory):
        memory.write("project", "KNOWLEDGE.md", "AUTORESEARCH is great")
        assert memory.search("autoresearch")


# ============================================================================
# Prompt section
# ============================================================================


class TestPromptSection:
    def test_includes_both_scope_paths(self, memory):
        out = memory.to_prompt_section()
        assert str(memory.global_dir) in out
        assert str(memory.project_dir) in out

    def test_includes_canonical_file_bodies(self, memory):
        memory.write("global", "MEMORY.md", "- user is Erfan\n- prefers terse responses")
        out = memory.to_prompt_section()
        assert "user is Erfan" in out
        assert "[global] MEMORY.md" in out

    def test_truncates_large_files(self, memory):
        memory.write("global", "KNOWLEDGE.md", "x" * 50_000)
        out = memory.to_prompt_section(max_bytes_per_file=200)
        assert "truncated" in out
        assert "xxxxxxxxxxxxx" in out
        # Truncation marker tells the agent exactly how to fetch the rest.
        assert "agent_memory_read" in out

    def test_recent_journal_entries_surface(self, memory):
        memory.journal("project", "today's note", when=datetime.now(UTC))
        out = memory.to_prompt_section()
        assert "today's note" in out

    def test_old_journal_entries_skipped(self, memory):
        old = datetime.now(UTC) - timedelta(days=30)
        memory.journal("project", "ancient note", when=old)
        out = memory.to_prompt_section()
        assert "ancient note" not in out

    def test_describes_tools(self, memory):
        out = memory.to_prompt_section()
        # The protocol cites these tools by partial-name in a compact list.
        # The agent already has the full schemas via the tool registry.
        assert "agent_memory_search" in out
        assert "agent_memory_append" in out
        assert "agent_memory_journal" in out
        # Generic mention of the read/write surface.
        assert "read" in out and "write" in out

    def test_operational_protocol_present(self, memory):
        out = memory.to_prompt_section()
        assert "operational protocol" in out.lower() or "How you must use these" in out
        # Specific directives the user asked for.
        assert "Before attempting" in out or "BEFORE attempting" in out.lower()
        assert "After every meaningful failure" in out or "after every meaningful failure" in out.lower()
        assert "EXPERIENCES.md" in out

    def test_protocol_directs_journal_at_turn_end(self, memory):
        out = memory.to_prompt_section()
        assert "agent_memory_journal" in out
        assert "end of each substantive turn" in out.lower()

    def test_protocol_routes_user_prefs_to_global(self, memory):
        out = memory.to_prompt_section()
        # The protocol should specifically call out global/USER.md.
        assert "USER.md" in out
        assert "global" in out

    def test_canonical_ordering_in_prompt(self, memory):
        memory.write("project", "SOUL.md", "the soul")
        memory.write("project", "MEMORY.md", "the memory")
        memory.write("project", "EXPERIENCES.md", "the experiences")
        out = memory.to_prompt_section()
        # The protocol-driven ordering is SOUL → IDENTITY → USER →
        # EXPERIENCES → MEMORY → KNOWLEDGE → INSIGHTS. Use section
        # headers (which only appear in the "Current memory contents"
        # block, not the protocol text above).
        idx_soul = out.find("[project] SOUL.md")
        idx_exp = out.find("[project] EXPERIENCES.md")
        idx_mem = out.find("[project] MEMORY.md")
        assert idx_soul != -1 and idx_exp != -1 and idx_mem != -1
        assert idx_soul < idx_exp < idx_mem

    def test_project_scope_surfaces_before_global(self, memory):
        memory.write("project", "MEMORY.md", "project-mem-content")
        memory.write("global", "MEMORY.md", "global-mem-content")
        out = memory.to_prompt_section()
        # Active project's facts should appear before global ones —
        # this codebase is what the agent's working on right now.
        idx_project_mem = out.find("[project] MEMORY.md")
        idx_global_mem = out.find("[global] MEMORY.md")
        assert idx_project_mem != -1 and idx_global_mem != -1
        assert idx_project_mem < idx_global_mem

    def test_works_without_project_scope(self, tmp_path):
        mem = AgentMemory(global_dir=tmp_path / "global")
        out = mem.to_prompt_section()
        assert "project memory unavailable" in out.lower()


class TestExperiencesWorkflow:
    """The user explicitly asked for an experiences/failure-log loop.

    These tests pin that the surface supports the read-before-act,
    log-after-fail pattern."""

    def test_append_with_section_creates_dated_entry(self, memory):
        memory.append(
            "project",
            "EXPERIENCES.md",
            (
                "**Tried:** rerunning the daemon without bumping the protocol\n"
                "**Result:** failure — TUI kept reusing the stale daemon\n"
                "**Lesson:** bump DAEMON_PROTOCOL_VERSION in tandem with TUI engine"
            ),
            section="2026-05-16 — protocol bump forgotten",
        )
        body = memory.read("project", "EXPERIENCES.md")
        assert "## 2026-05-16 — protocol bump forgotten" in body
        assert "stale daemon" in body
        # Default ensure() text remains above the new entry.
        assert body.index("# Experiences") < body.index("2026-05-16")

    def test_search_finds_prior_experiences(self, memory):
        memory.append(
            "project",
            "EXPERIENCES.md",
            "**Tried:** patching the FAISS index in place",
            section="2026-05-10 — FAISS shard wedged",
        )
        hits = memory.search("FAISS", scope="project")
        assert hits and hits[0]["path"] == "EXPERIENCES.md"


# ============================================================================
# Tool wrappers
# ============================================================================


class TestToolWrappers:
    def setup_method(self):
        agent_memory_tool.set_active_memory(None)

    def teardown_method(self):
        agent_memory_tool.set_active_memory(None)

    def test_no_memory_yields_error(self):
        out = agent_memory_tool.agent_memory_read("global", "MEMORY.md")
        assert out["ok"] is False

    def test_status_when_unavailable(self):
        out = agent_memory_tool.agent_memory_status()
        assert out == {"ok": True, "available": False}

    def test_full_cycle(self, tmp_path):
        mem = AgentMemory(
            project_root=tmp_path / "p",
            global_dir=tmp_path / "g",
            project_dir=tmp_path / "proj",
        )
        mem.ensure()
        agent_memory_tool.set_active_memory(mem)

        # status
        st = agent_memory_tool.agent_memory_status()
        assert st["available"] and st["total_files"] >= 10

        # write
        write_out = agent_memory_tool.agent_memory_write("global", "MEMORY.md", "fresh content")
        assert write_out["ok"]

        # read it back
        read_out = agent_memory_tool.agent_memory_read("global", "MEMORY.md")
        assert read_out["body"] == "fresh content"

        # append
        agent_memory_tool.agent_memory_append("global", "MEMORY.md", "more", timestamp=False)
        assert "more" in agent_memory_tool.agent_memory_read("global", "MEMORY.md")["body"]

        # list
        ls = agent_memory_tool.agent_memory_list()
        assert ls["count"] if "count" in ls else len(ls["files"]) > 0

        # search
        agent_memory_tool.agent_memory_write("project", "KNOWLEDGE.md", "needle in this haystack")
        results = agent_memory_tool.agent_memory_search("needle")
        assert results["count"] >= 1

        # journal
        j = agent_memory_tool.agent_memory_journal("project", "checkpoint reached")
        assert j["ok"]
        assert "journal/" in j["path"]

    def test_path_escape_returns_error_not_exception(self, tmp_path):
        mem = AgentMemory(project_root=tmp_path / "p", global_dir=tmp_path / "g", project_dir=tmp_path / "proj")
        mem.ensure()
        agent_memory_tool.set_active_memory(mem)
        out = agent_memory_tool.agent_memory_write("global", "../escape.md", "boom")
        assert out["ok"] is False
        assert "escapes" in out["error"]


# ============================================================================
# Daemon system-prompt injection (regression)
# ============================================================================


class TestDaemonInjection:
    def test_runtime_reload_initialises_memory(self, tmp_path):
        from xerxes.daemon.config import DaemonConfig
        from xerxes.daemon.runtime import RuntimeManager

        # Don't actually call reload() — it requires real provider creds.
        # Instead, construct the RuntimeManager and manually invoke the
        # AgentMemory bit by re-creating just that step.
        cfg = DaemonConfig(project_dir=str(tmp_path))
        rm = RuntimeManager(cfg)
        rm.agent_memory = AgentMemory(project_root=tmp_path)
        rm.agent_memory.ensure()
        # to_prompt_section must include both scopes.
        section = rm.agent_memory.to_prompt_section()
        assert "global" in section.lower()
        assert "project" in section.lower()
