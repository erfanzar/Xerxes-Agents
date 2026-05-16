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
"""Tests for the 18 TUI parity gaps closed."""

from __future__ import annotations

import time

import pytest
from prompt_toolkit.document import Document
from xerxes.extensions.slash_plugins import (
    SlashPluginRegistry,
    register_slash,
    registered_slashes,
    resolve_slash,
)
from xerxes.extensions.slash_plugins import (
    registry as slash_registry,
)
from xerxes.runtime.background_sessions import (
    BackgroundSession,
    BackgroundSessionManager,
    BackgroundStatus,
)
from xerxes.tui.at_mentions import (
    AT_TRIGGERS,
    AtMentionCompleter,
    expand_mention,
    expand_mentions_in_text,
)
from xerxes.tui.banner import COMPACT_LOGO, FULL_LOGO, BannerData, render_banner
from xerxes.tui.clipboard_attach import AttachmentBuffer
from xerxes.tui.context_bar import context_bar, context_bar_with_pct
from xerxes.tui.input_buffer import (
    InputBufferConfig,
    build_input_buffer,
    build_multiline_key_bindings,
    history_file_path,
)
from xerxes.tui.panel_state import (
    DEFAULT_APPROVAL_OPTIONS,
    ApprovalChoice,
    ApprovalCountdown,
    ApprovalPanelState,
    PanelSelection,
)
from xerxes.tui.reasoning_filter import ReasoningFilter
from xerxes.tui.skin_engine import Skin, SkinEngine
from xerxes.tui.status_bar import StatusSnapshot, format_status
from xerxes.tui.tips import TIPS, random_tip, tip_of_the_day
from xerxes.tui.voice_keys import VoiceKeyHandler, VoiceState

# ============================================================================
# Gap 1 — Multiline input
# ============================================================================


class TestMultilineInput:
    def test_default_config_multiline(self):
        cfg = InputBufferConfig()
        assert cfg.multiline is True

    def test_buffer_built(self):
        buf = build_input_buffer(InputBufferConfig(history_path=None, auto_suggest=False))
        assert buf is not None
        assert buf.multiline() is True  # multiline is a Filter callable

    def test_accept_handler_called(self):
        seen = {}

        def cb(text: str) -> None:
            seen["text"] = text

        buf = build_input_buffer(InputBufferConfig(on_accept=cb, history_path=None))
        buf.text = "hello world"
        buf.validate_and_handle()
        assert seen["text"] == "hello world"

    def test_keybindings_enter_alt_enter(self):
        kb = build_multiline_key_bindings()
        # Three bindings: Enter (submit), Escape+Enter (Alt+Enter newline), Ctrl+J (newline).
        assert len(kb.bindings) == 3


# ============================================================================
# Gap 2 — Auto-suggest + Gap 3 — FileHistory persistence
# ============================================================================


class TestAutoSuggestAndHistory:
    def test_auto_suggest_enabled(self):
        buf = build_input_buffer(InputBufferConfig(auto_suggest=True, history_path=None))
        # AutoSuggestFromHistory is the only auto-suggest we attach.
        from prompt_toolkit.auto_suggest import AutoSuggestFromHistory

        assert isinstance(buf.auto_suggest, AutoSuggestFromHistory)

    def test_auto_suggest_off(self):
        buf = build_input_buffer(InputBufferConfig(auto_suggest=False, history_path=None))
        assert buf.auto_suggest is None

    def test_history_path_default(self):
        path = history_file_path()
        assert path.parent.is_dir()
        assert path.name.endswith(".txt")

    def test_file_history_persists(self, tmp_path):
        hist_path = tmp_path / "history.txt"
        buf = build_input_buffer(InputBufferConfig(history_path=hist_path))
        # FileHistory writes on store_string.
        buf.history.store_string("first prompt")
        # `load_history_strings()` is the sync iterator over persisted entries.
        items = list(buf.history.load_history_strings())
        assert "first prompt" in items
        assert hist_path.exists()


# ============================================================================
# Gap 4 — Reasoning tag suppression
# ============================================================================


class TestReasoningFilter:
    def test_no_tags_passthrough(self):
        rf = ReasoningFilter()
        vis, think = rf.feed("hello world")
        # Buffered tail; flush to drain.
        rest_vis, rest_think = rf.flush()
        assert (vis + rest_vis) == "hello world"
        assert (think + rest_think) == ""

    def test_strips_reasoning_block(self):
        rf = ReasoningFilter()
        v1, _ = rf.feed("before<reasoning>thinking aloud</reasoning>after")
        v2, _ = rf.flush()
        # Reasoning is gone from the visible stream.
        visible = v1 + v2
        assert "thinking aloud" not in visible
        assert visible.startswith("before")
        assert visible.endswith("after")

    def test_captures_thinking_log(self):
        rf = ReasoningFilter()
        rf.feed("a <think>secret</think> b")
        rf.flush()
        assert "secret" in rf.thinking_log

    def test_split_across_chunks(self):
        rf = ReasoningFilter()
        a, _ = rf.feed("a <reaso")
        b, _ = rf.feed("ning>hidden</reasoning> c")
        c, _ = rf.flush()
        visible = a + b + c
        assert "hidden" not in visible
        assert visible.startswith("a ")
        assert visible.endswith(" c")

    def test_case_insensitive(self):
        rf = ReasoningFilter()
        v1, _ = rf.feed("X <THINK>secret</THINK> Z")
        v2, _ = rf.flush()
        visible = v1 + v2
        # Secret content never appears in the visible stream.
        assert "secret" not in visible
        # …but it's captured in the thinking log for the side pane.
        assert "secret" in rf.thinking_log

    def test_unclosed_block_preserved_on_flush(self):
        rf = ReasoningFilter()
        rf.feed("before <think>incomplete")
        _, t = rf.flush()
        assert "incomplete" in t


# ============================================================================
# Gap 5 — Status bar
# ============================================================================


class TestStatusBar:
    def test_default_snapshot(self):
        s = StatusSnapshot()
        assert s.context_percent == 0.0
        assert s.cache_hit_rate == 0.0

    def test_context_percent(self):
        s = StatusSnapshot(input_tokens=50_000, context_window=200_000)
        assert abs(s.context_percent - 25.0) < 1e-6

    def test_context_percent_with_cache(self):
        s = StatusSnapshot(input_tokens=20_000, cache_read_tokens=80_000, context_window=200_000)
        assert abs(s.context_percent - 50.0) < 1e-6

    def test_cache_hit_rate(self):
        s = StatusSnapshot(input_tokens=200, cache_read_tokens=800)
        assert abs(s.cache_hit_rate - 0.8) < 1e-9

    def test_format_status(self):
        s = StatusSnapshot(
            model="claude-opus-4-7",
            input_tokens=1000,
            output_tokens=500,
            cost_usd=0.05,
            duration_sec=42,
            context_window=200_000,
        )
        out = format_status(s)
        assert "claude-opus-4-7" in out
        assert "$0.0500" in out
        assert "00:42" in out

    def test_format_includes_cache(self):
        s = StatusSnapshot(model="m", input_tokens=100, cache_read_tokens=500)
        out = format_status(s)
        assert "500c" in out

    def test_format_includes_skill_and_mode(self):
        s = StatusSnapshot(model="m", permission_mode="manual", active_skill="plan", queue_depth=2)
        out = format_status(s)
        assert "skill=plan" in out
        assert "queued=2" in out
        assert "manual" in out


# ============================================================================
# Gap 6 — Welcome tips
# ============================================================================


class TestTips:
    def test_at_least_150_tips(self):
        assert len(TIPS) >= 150

    def test_random_tip_with_seed_deterministic(self):
        assert random_tip(seed=0) == random_tip(seed=0)
        assert random_tip(seed=0) == TIPS[0]

    def test_random_tip_without_seed_in_pool(self):
        assert random_tip() in TIPS

    def test_tip_of_the_day_stable(self):
        import datetime as _dt

        a = tip_of_the_day(today=_dt.date(2026, 5, 16))
        b = tip_of_the_day(today=_dt.date(2026, 5, 16))
        assert a == b

    def test_tips_are_strings(self):
        assert all(isinstance(t, str) and t for t in TIPS)


# ============================================================================
# Gap 7 — Skin branding
# ============================================================================


class TestSkinBranding:
    def test_default_branding_present(self):
        engine = SkinEngine()
        skin = engine.load("default")
        assert skin.label("agent_name") == "Xerxes"
        assert skin.label("prompt_symbol") == "›"  # noqa: RUF001 — designed glyph (U+203A)

    def test_ares_brand_overrides(self):
        engine = SkinEngine()
        skin = engine.load("ares")
        assert skin.label("agent_name") == "Ares"
        assert "striking" in skin.spinner_verbs()

    def test_save_and_load_branding(self, tmp_path):
        engine = SkinEngine(base_dir=tmp_path)
        sk = Skin(name="custom", roles={"primary": "#aabbcc"}, branding={"agent_name": "Foo", "prompt_symbol": "$"})
        engine.save(sk)
        loaded = engine.load("custom")
        assert loaded.label("agent_name") == "Foo"
        assert loaded.label("prompt_symbol") == "$"

    def test_six_builtin_skins(self):
        engine = SkinEngine()
        names = set(engine.available())
        assert {"default", "high-contrast", "dim", "ares", "mono", "slate", "daylight"}.issubset(names)


# ============================================================================
# Gap 8 — Voice mode key binding
# ============================================================================


class TestVoiceKeyHandler:
    def _make_handler(self, *, transcript: str = "hello voice"):
        submitted: list[str] = []

        h = VoiceKeyHandler(
            transcribe=lambda path: transcript,
            submit=lambda text: submitted.append(text),
        )
        return h, submitted

    def test_initial_state(self):
        h, _ = self._make_handler()
        assert h.state is VoiceState.IDLE

    def test_start_then_stop(self):
        h, submitted = self._make_handler()
        h.start_recording()
        assert h.state is VoiceState.RECORDING
        text = h.stop_recording()
        assert text == "hello voice"
        assert submitted == ["hello voice"]
        assert h.state is VoiceState.IDLE

    def test_double_start_is_noop(self):
        h, _ = self._make_handler()
        h.start_recording()
        h.start_recording()  # No-op
        assert h.state is VoiceState.RECORDING

    def test_stop_without_start_returns_empty(self):
        h, _ = self._make_handler()
        assert h.stop_recording() == ""

    def test_toggle_round_trip(self):
        h, submitted = self._make_handler(transcript="ping")
        h.toggle()  # starts recording
        assert h.state is VoiceState.RECORDING
        h.toggle()  # stops + transcribes
        assert submitted == ["ping"]

    def test_continuous_mode_restarts(self):
        h, _submitted = self._make_handler(transcript="msg")
        h.set_continuous(True)
        h.start_recording()
        h.stop_recording()
        # After auto-restart it's recording again.
        assert h.state is VoiceState.RECORDING
        h.set_continuous(False)  # Prevent infinite restart in test runtime.
        h.stop_recording()

    def test_transcribe_failure_does_not_break(self):
        def fail(path):
            raise RuntimeError("synthetic stt failure")

        h = VoiceKeyHandler(transcribe=fail)
        h.start_recording()
        out = h.stop_recording()
        assert out == ""
        assert h.state is VoiceState.IDLE


# ============================================================================
# Gap 9 — Clipboard image attach
# ============================================================================


class TestAttachmentBuffer:
    def test_capture_no_image(self, tmp_path):
        buf = AttachmentBuffer(capture_image=lambda d: None, store_dir=tmp_path)
        assert buf.capture_clipboard_image() is None
        assert buf.pending() == []

    def test_capture_creates_attachment(self, tmp_path):
        img_path = tmp_path / "img.png"
        img_path.write_bytes(b"FAKE-PNG-DATA")

        buf = AttachmentBuffer(capture_image=lambda d: img_path, store_dir=tmp_path)
        att = buf.capture_clipboard_image()
        assert att is not None
        assert att.kind == "image"
        assert att.bytes == len(b"FAKE-PNG-DATA")
        assert buf.pending() == [att]

    def test_attach_path(self, tmp_path):
        p = tmp_path / "data.txt"
        p.write_text("hello")
        buf = AttachmentBuffer(store_dir=tmp_path)
        att = buf.attach_path(p)
        assert att is not None
        assert att.kind == "file"

    def test_attach_missing_path(self, tmp_path):
        buf = AttachmentBuffer(store_dir=tmp_path)
        assert buf.attach_path(tmp_path / "nope.txt") is None

    def test_drain_clears(self, tmp_path):
        buf = AttachmentBuffer(store_dir=tmp_path)
        p = tmp_path / "a.txt"
        p.write_text("x")
        buf.attach_path(p)
        drained = buf.drain()
        assert len(drained) == 1
        assert buf.pending() == []


# ============================================================================
# Gap 10 — Modal panel navigation + Gap 11 — Approval timer + Gap 12 — 5-option
# ============================================================================


class TestPanelSelection:
    def test_up_down_wrap(self):
        s = PanelSelection(options=["a", "b", "c"])
        assert s.down() == 1
        assert s.down() == 2
        assert s.down() == 0  # wraps
        assert s.up() == 2  # wraps backwards

    def test_set_modulus(self):
        s = PanelSelection(options=["a", "b"])
        s.set(5)
        assert s.index == 1

    def test_empty_options(self):
        s = PanelSelection(options=[])
        assert s.down() == 0
        assert s.selected() == ""


class TestApprovalPanel:
    def test_default_five_options(self):
        assert len(DEFAULT_APPROVAL_OPTIONS) == 5
        assert ApprovalChoice.APPROVE_ONCE in DEFAULT_APPROVAL_OPTIONS
        assert ApprovalChoice.APPROVE_ALWAYS in DEFAULT_APPROVAL_OPTIONS
        assert ApprovalChoice.DENY in DEFAULT_APPROVAL_OPTIONS

    def test_navigation(self):
        state = ApprovalPanelState()
        first = state.current()
        state.down()
        assert state.current() != first

    def test_current_is_choice(self):
        state = ApprovalPanelState()
        assert isinstance(state.current(), ApprovalChoice)


class TestApprovalCountdown:
    def test_default_timeout(self):
        c = ApprovalCountdown()
        assert c.timeout_seconds == 60.0
        assert c.is_active() is False

    def test_start_and_cancel(self):
        c = ApprovalCountdown(timeout_seconds=10.0)
        c.start(lambda: None)
        assert c.is_active() is True
        c.cancel()
        assert c.is_active() is False

    def test_fires_on_timeout(self):
        called = []
        c = ApprovalCountdown(timeout_seconds=0.05)
        c.start(lambda: called.append(1))
        time.sleep(0.15)
        assert called == [1]

    def test_remaining_decreases(self):
        c = ApprovalCountdown(timeout_seconds=5.0)
        c.start(lambda: None)
        assert 4.0 < c.remaining() <= 5.0
        c.cancel()


# ============================================================================
# Gap 13 — @-mentions
# ============================================================================


class TestAtMentions:
    def test_all_triggers_listed(self):
        for trig in ("@file:", "@folder:", "@diff", "@staged", "@git:", "@url:"):
            assert trig in AT_TRIGGERS

    def test_file_completer_lists_files(self, tmp_path):
        (tmp_path / "alpha.py").write_text("x")
        (tmp_path / "beta.md").write_text("x")
        comp = AtMentionCompleter(tmp_path)
        doc = Document("@file:")
        out = list(comp.get_completions(doc, None))
        names = [c.display[0][1] if isinstance(c.display, list) else c.display for c in out]
        joined = " ".join(names)
        assert "alpha.py" in joined
        assert "beta.md" in joined

    def test_folder_completer_lists_dirs(self, tmp_path):
        (tmp_path / "sub").mkdir()
        (tmp_path / "f.txt").write_text("x")
        comp = AtMentionCompleter(tmp_path)
        doc = Document("@folder:")
        out = list(comp.get_completions(doc, None))
        names = " ".join(str(c.display) for c in out)
        assert "sub/" in names
        # Files should be filtered out of @folder: results.
        assert "f.txt" not in names

    def test_bare_at_suggests_triggers(self, tmp_path):
        comp = AtMentionCompleter(tmp_path)
        doc = Document("@")
        out = list(comp.get_completions(doc, None))
        suggested = " ".join(c.text for c in out)
        assert "@file:" in suggested
        assert "@diff" in suggested

    def test_no_completion_after_space(self, tmp_path):
        comp = AtMentionCompleter(tmp_path)
        doc = Document("hello @file:foo and ")
        out = list(comp.get_completions(doc, None))
        assert out == []

    def test_expand_file(self, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("the body")
        out = expand_mention("@file:x.txt", workspace_root=tmp_path)
        assert out.kind == "file"
        assert out.payload == "the body"

    def test_expand_folder(self, tmp_path):
        (tmp_path / "a.txt").write_text("x")
        (tmp_path / "b.txt").write_text("x")
        out = expand_mention("@folder:.", workspace_root=tmp_path)
        assert "a.txt" in out.payload
        assert "b.txt" in out.payload

    def test_expand_url(self, tmp_path):
        out = expand_mention("@url:https://example.com/x", workspace_root=tmp_path)
        assert out.kind == "url"
        assert out.payload == "https://example.com/x"

    def test_expand_missing(self, tmp_path):
        out = expand_mention("@file:nope.txt", workspace_root=tmp_path)
        assert out.error

    def test_extract_multiple(self, tmp_path):
        (tmp_path / "a.txt").write_text("alpha")
        out = expand_mentions_in_text("read @file:a.txt and @url:http://x", workspace_root=tmp_path)
        kinds = [m.kind for m in out]
        assert "file" in kinds
        assert "url" in kinds


# ============================================================================
# Gap 14 — Context bar
# ============================================================================


class TestContextBar:
    def test_empty_when_zero(self):
        assert context_bar(used=0, window=200_000, width=10) == "░" * 10

    def test_full_when_at_window(self):
        assert context_bar(used=200_000, window=200_000, width=10) == "█" * 10

    def test_half_fill(self):
        bar = context_bar(used=100_000, window=200_000, width=10)
        # 50% of 10 chars = 5 full blocks (no half needed exactly at .0).
        assert bar.count("█") == 5

    def test_with_pct(self):
        out = context_bar_with_pct(used=50_000, window=200_000, width=20)
        assert "25.0%" in out


# ============================================================================
# Gap 15 — /rollback diff [n]
# ============================================================================


class TestSnapshotDiff:
    def test_diff_against_snapshot(self, tmp_path):
        import shutil

        from xerxes.session.snapshot_diff import diff_against_snapshot
        from xerxes.session.snapshots import SnapshotManager

        if shutil.which("git") is None:
            pytest.skip("git missing")

        ws = tmp_path / "work"
        ws.mkdir()
        (ws / "a.txt").write_text("v1\n")
        shadow = tmp_path / "shadow"
        shadow.mkdir()
        mgr = SnapshotManager(ws, shadow_root=shadow)
        snap = mgr.snapshot("baseline")
        (ws / "a.txt").write_text("v2 CHANGED\n")
        diff = diff_against_snapshot(mgr, snap.id)
        assert diff.file_count >= 1
        assert "CHANGED" in diff.diff_text

    def test_diff_missing_snapshot_raises(self, tmp_path):
        import shutil

        from xerxes.session.snapshot_diff import diff_against_snapshot
        from xerxes.session.snapshots import SnapshotManager

        if shutil.which("git") is None:
            pytest.skip("git missing")

        ws = tmp_path / "work"
        ws.mkdir()
        shadow = tmp_path / "shadow"
        shadow.mkdir()
        mgr = SnapshotManager(ws, shadow_root=shadow)
        with pytest.raises(KeyError):
            diff_against_snapshot(mgr, "ghost-id")


# ============================================================================
# Gap 16 — Background sessions
# ============================================================================


class TestBackgroundSessions:
    def test_submit_runs_to_completion(self):
        def runner(sess: BackgroundSession) -> str:
            return f"answered:{sess.prompt}"

        mgr = BackgroundSessionManager(runner)
        sess = mgr.submit("hello")
        # Wait for the background thread to finish.
        for _ in range(50):
            if sess.status in (BackgroundStatus.SUCCEEDED, BackgroundStatus.FAILED):
                break
            time.sleep(0.05)
        assert sess.status is BackgroundStatus.SUCCEEDED
        assert sess.result == "answered:hello"
        mgr.shutdown()

    def test_failure_recorded(self):
        def runner(sess):
            raise RuntimeError("nope")

        mgr = BackgroundSessionManager(runner)
        sess = mgr.submit("x")
        for _ in range(50):
            if sess.status in (BackgroundStatus.SUCCEEDED, BackgroundStatus.FAILED):
                break
            time.sleep(0.05)
        assert sess.status is BackgroundStatus.FAILED
        assert "nope" in sess.error
        mgr.shutdown()

    def test_concurrency_cap_queues_pending(self):
        # Block the runner so we can verify queueing.
        block = []

        def runner(sess):
            block.append(sess.id)
            while sess.id in block:
                time.sleep(0.01)
            return "ok"

        mgr = BackgroundSessionManager(runner, max_concurrent=1)
        first = mgr.submit("a")
        second = mgr.submit("b")
        # `second` should be PENDING until `first` finishes.
        time.sleep(0.05)
        assert second.status is BackgroundStatus.PENDING
        # Release first.
        block.remove(first.id)
        # Wait for drain.
        for _ in range(50):
            if second.status is BackgroundStatus.RUNNING:
                break
            time.sleep(0.05)
        assert second.status is BackgroundStatus.RUNNING
        block.remove(second.id)
        mgr.shutdown()

    def test_cancel_pending(self):
        def runner(sess):
            time.sleep(1.0)
            return "ok"

        mgr = BackgroundSessionManager(runner, max_concurrent=0 if False else 1)
        mgr.submit("a")
        second = mgr.submit("b")
        assert mgr.cancel(second.id) is True
        assert second.status is BackgroundStatus.CANCELLED
        mgr.shutdown(timeout=2.0)


# ============================================================================
# Gap 17 — Compact banner
# ============================================================================


class TestBanner:
    def test_full_banner_includes_logo(self):
        data = BannerData(version="0.2.0", model="claude-opus-4-7", session_id="abcd1234ef", workspace="/proj")
        out = render_banner(data, terminal_width=120)
        assert FULL_LOGO.strip().splitlines()[0] in out
        assert "claude-opus-4-7" in out

    def test_compact_banner_under_64_cols(self):
        data = BannerData(model="m", session_id="abcd1234", tip="tip text")
        out = render_banner(data, terminal_width=50)
        assert COMPACT_LOGO in out
        # No full-logo box.
        assert "╭" not in out

    def test_compact_tip_truncated(self):
        long_tip = "x" * 200
        data = BannerData(tip=long_tip)
        out = render_banner(data, terminal_width=50)
        assert "…" in out
        # Banner stays within reasonable bounds.
        for line in out.splitlines():
            assert len(line) <= 100


# ============================================================================
# Gap 18 — Plugin-registered slash commands
# ============================================================================


class TestSlashPlugins:
    def setup_method(self):
        # Make sure each test starts with a clean default registry.
        slash_registry()._plugins.clear()

    def teardown_method(self):
        slash_registry()._plugins.clear()

    def test_register_and_resolve(self):
        plugin = register_slash("hello", lambda: "hi there", description="say hi")
        out = resolve_slash("/hello arg1 arg2")
        assert out is plugin
        assert plugin.command.name == "hello"

    def test_alias_resolves(self):
        register_slash("hello", lambda: "hi", aliases=("hi",))
        assert resolve_slash("/hi") is not None

    def test_collision_with_core_rejected(self):
        with pytest.raises(ValueError):
            register_slash("help", lambda: "nope")

    def test_empty_name_rejected(self):
        with pytest.raises(ValueError):
            register_slash("", lambda: None)

    def test_list_alphabetical(self):
        register_slash("zeta", lambda: None)
        register_slash("alpha", lambda: None)
        names = [p.command.name for p in registered_slashes()]
        assert names == ["alpha", "zeta"]

    def test_merged_all_commands_includes_core_and_plugin(self):
        from xerxes.bridge.commands import COMMAND_REGISTRY

        register_slash("zzz_my_plugin", lambda: None)
        merged = slash_registry().all_commands()
        names = [c.name for c in merged]
        # Includes both core and plugin entries.
        assert "help" in names
        assert "zzz_my_plugin" in names
        # Core entries come first; plugin commands appear after sorted.
        assert names.index("zzz_my_plugin") > names.index("help")
        # Length sanity.
        assert len(merged) == len(COMMAND_REGISTRY) + 1

    def test_separate_registry_instance(self):
        local = SlashPluginRegistry()
        local.register("solo", lambda: None)
        assert local.resolve("/solo") is not None
        # Default registry unaffected.
        assert resolve_slash("/solo") is None
