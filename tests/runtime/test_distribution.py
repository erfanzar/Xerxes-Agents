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
"""Tests for runtime.distribution."""

from __future__ import annotations

from xerxes.runtime.distribution import (
    POWERSHELL_INSTALL_SNIPPET,
    SHELL_INSTALL_SNIPPET,
    detect_platform,
    render_homebrew_formula,
    termux_filter_extras,
)


class TestPlatformInfo:
    def test_detect_returns_info(self):
        info = detect_platform()
        assert info.system in {"Darwin", "Linux", "Windows"}
        assert info.python.startswith("3.")

    def test_termux_detection_env(self, monkeypatch):
        monkeypatch.setenv("PREFIX", "/data/data/com.termux/files/usr")
        info = detect_platform()
        assert info.is_termux is True

    def test_wsl_detection_env(self, monkeypatch):
        monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu-22.04")
        info = detect_platform()
        assert info.is_wsl is True


class TestHomebrewFormula:
    def test_renders(self):
        out = render_homebrew_formula(
            version="0.2.0",
            tarball_url="https://example.test/xerxes-agent-0.2.0.tar.gz",
            sha256="deadbeef",
        )
        assert "class XerxesAgent < Formula" in out
        assert "Apache-2.0" in out
        assert "deadbeef" in out
        assert 'depends_on "python@3.11"' in out


class TestTermuxFilter:
    def test_filters_excluded(self):
        out = termux_filter_extras({"voice": ["faster-whisper>=1.0", "sounddevice>=0.4", "numpy>=1.24"]})
        assert "numpy>=1.24" in out["voice"]
        assert not any("faster" in d.lower() for d in out["voice"])
        assert not any("sounddevice" in d.lower() for d in out["voice"])

    def test_keeps_pure_python(self):
        out = termux_filter_extras({"web": ["fastapi>=0.104", "uvicorn>=0.24"]})
        assert out["web"] == ["fastapi>=0.104", "uvicorn>=0.24"]


class TestInstallSnippets:
    def test_shell_snippet_has_uv(self):
        assert "uv tool install xerxes-agent" in SHELL_INSTALL_SNIPPET

    def test_ps_snippet_has_uv(self):
        assert "uv tool install xerxes-agent" in POWERSHELL_INSTALL_SNIPPET
