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
"""Tests for tui.skin_engine + tui.clipboard."""

from __future__ import annotations

import pytest
from xerxes.tui.clipboard import PromptQueue, save_clipboard_image
from xerxes.tui.skin_engine import Skin, SkinEngine, hex_to_ansi_fg, hex_to_rgb


class TestSkin:
    def test_hex_to_rgb(self):
        assert hex_to_rgb("#ff0080") == (255, 0, 128)

    def test_hex_to_rgb_no_hash(self):
        assert hex_to_rgb("00ff00") == (0, 255, 0)

    def test_hex_to_rgb_invalid_raises(self):
        with pytest.raises(ValueError):
            hex_to_rgb("not a color")

    def test_hex_to_ansi_fg_starts_with_csi(self):
        out = hex_to_ansi_fg("#ffffff")
        assert out.startswith("\033[38;2;")
        assert "255;255;255" in out

    def test_builtin_skins_available(self, tmp_path):
        engine = SkinEngine(base_dir=tmp_path)
        names = engine.available()
        assert "default" in names
        assert "high-contrast" in names

    def test_load_builtin(self, tmp_path):
        engine = SkinEngine(base_dir=tmp_path)
        skin = engine.load("default")
        assert skin.fg("primary").startswith("\033[")

    def test_load_unknown_raises(self, tmp_path):
        engine = SkinEngine(base_dir=tmp_path)
        with pytest.raises(KeyError):
            engine.load("ghost")

    def test_save_and_load_roundtrip(self, tmp_path):
        engine = SkinEngine(base_dir=tmp_path)
        skin = Skin(name="custom", roles={"primary": "#123456", "accent": "#abcdef"})
        engine.save(skin)
        loaded = engine.load("custom")
        assert loaded.roles["primary"] == "#123456"
        assert loaded.roles["accent"] == "#abcdef"

    def test_set_active(self, tmp_path):
        engine = SkinEngine(base_dir=tmp_path)
        engine.set_active("high-contrast")
        assert engine.active().name == "high-contrast"


class TestPromptQueue:
    def test_initial_empty(self):
        q = PromptQueue()
        assert q.size() == 0
        assert q.pop() is None

    def test_push_pop_fifo(self):
        q = PromptQueue()
        q.push("first")
        q.push("second")
        assert q.pop() == "first"
        assert q.pop() == "second"
        assert q.pop() is None

    def test_empty_push_no_op(self):
        q = PromptQueue()
        assert q.push("") == 0
        assert q.size() == 0

    def test_drain(self):
        q = PromptQueue()
        for s in ("a", "b", "c"):
            q.push(s)
        assert q.drain() == ["a", "b", "c"]
        assert q.size() == 0

    def test_clear_returns_count(self):
        q = PromptQueue()
        q.push("a")
        q.push("b")
        assert q.clear() == 2


class TestClipboardImage:
    def test_save_returns_none_when_clipboard_empty(self, tmp_path, monkeypatch):
        # Force ImageGrab.grabclipboard to return None.
        import xerxes.tui.clipboard as mod

        monkeypatch.setattr(mod, "grab_clipboard_image", lambda: None)
        assert save_clipboard_image(tmp_path) is None
