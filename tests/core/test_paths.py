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
"""Tests for XERXES_HOME env override in xerxes.core.paths."""

from __future__ import annotations

from pathlib import Path

import pytest
from xerxes.core.paths import XERXES_HOME_ENV, xerxes_home, xerxes_subdir


class TestXerxesHome:
    def test_default_is_user_home_dot_xerxes(self, monkeypatch):
        monkeypatch.delenv(XERXES_HOME_ENV, raising=False)
        assert xerxes_home() == Path.home() / ".xerxes"

    def test_env_var_overrides(self, monkeypatch, tmp_path):
        monkeypatch.setenv(XERXES_HOME_ENV, str(tmp_path))
        assert xerxes_home() == tmp_path

    def test_env_var_expands_user(self, monkeypatch):
        monkeypatch.setenv(XERXES_HOME_ENV, "~/some-other-xerxes")
        assert xerxes_home() == Path.home() / "some-other-xerxes"

    def test_empty_env_var_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv(XERXES_HOME_ENV, "")
        assert xerxes_home() == Path.home() / ".xerxes"

    def test_whitespace_env_var_falls_back(self, monkeypatch):
        monkeypatch.setenv(XERXES_HOME_ENV, "   ")
        assert xerxes_home() == Path.home() / ".xerxes"

    def test_subdir_joins_under_home(self, monkeypatch, tmp_path):
        monkeypatch.setenv(XERXES_HOME_ENV, str(tmp_path))
        assert xerxes_subdir("skills") == tmp_path / "skills"
        assert xerxes_subdir("daemon", "logs") == tmp_path / "daemon" / "logs"

    def test_subdir_with_no_args_returns_home(self, monkeypatch, tmp_path):
        monkeypatch.setenv(XERXES_HOME_ENV, str(tmp_path))
        assert xerxes_subdir() == tmp_path

    def test_xerxes_home_is_not_created_on_disk(self, monkeypatch, tmp_path):
        target = tmp_path / "fresh"
        monkeypatch.setenv(XERXES_HOME_ENV, str(target))
        result = xerxes_home()
        assert result == target
        assert not target.exists()


@pytest.mark.parametrize(
    "module_attr",
    [
        ("xerxes.daemon.config", "DAEMON_DIR"),
        ("xerxes.bridge.profiles", "PROFILES_DIR"),
    ],
)
def test_module_constants_respect_xerxes_home(module_attr, monkeypatch, tmp_path):
    """Module-level constants resolve against XERXES_HOME at import time.

    These constants are computed once at import; this test reloads the
    module so the env override takes effect.
    """
    import importlib

    module_name, attr = module_attr
    monkeypatch.setenv(XERXES_HOME_ENV, str(tmp_path))
    mod = importlib.import_module(module_name)
    importlib.reload(mod)
    value = getattr(mod, attr)
    assert str(value).startswith(str(tmp_path))
