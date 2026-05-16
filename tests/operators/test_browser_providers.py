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
"""Tests for browser provider plugins."""

from __future__ import annotations

import pytest
from xerxes.operators.browser_providers import (
    SUPPORTED_PROVIDERS,
    BrowserProvider,
    BrowserSession,
    get,
    register,
    registry,
)


def test_supported_provider_names():
    assert set(SUPPORTED_PROVIDERS) == {"local", "camofox", "browserbase", "browser_use", "firecrawl"}


def test_registry_includes_all_builtins():
    reg = registry()
    for name in SUPPORTED_PROVIDERS:
        assert name in reg


def test_get_returns_instance():
    p = get("local")
    assert p is not None
    assert isinstance(p, BrowserProvider)


def test_register_custom_provider():
    class Stub(BrowserProvider):
        name = "stub"

        def open(self, **kw):
            return BrowserSession(provider=self.name)

        def close(self, s):
            pass

    register(Stub())
    assert get("stub") is not None


def test_camofox_missing_env_raises(monkeypatch):
    monkeypatch.delenv("XERXES_CAMOFOX_SCRIPT", raising=False)
    # Provide a fake node binary so we get to the env check.
    monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/node" if cmd == "node" else None)
    with pytest.raises(RuntimeError, match="XERXES_CAMOFOX_SCRIPT"):
        get("camofox").open()


def test_browserbase_missing_credentials_raises(monkeypatch):
    monkeypatch.delenv("BROWSERBASE_API_KEY", raising=False)
    monkeypatch.delenv("BROWSERBASE_PROJECT_ID", raising=False)
    with pytest.raises(RuntimeError, match="BROWSERBASE_API_KEY"):
        get("browserbase").open()


def test_browser_use_missing_creds_raises(monkeypatch):
    monkeypatch.delenv("BROWSER_USE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="BROWSER_USE_API_KEY"):
        get("browser_use").open()


def test_firecrawl_missing_creds_raises(monkeypatch):
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="FIRECRAWL_API_KEY"):
        get("firecrawl").open()


def test_firecrawl_session_when_env_set(monkeypatch):
    monkeypatch.setenv("FIRECRAWL_API_KEY", "abc")
    sess = get("firecrawl").open()
    assert sess.provider == "firecrawl"
    assert sess.metadata["api_key_present"] is True
