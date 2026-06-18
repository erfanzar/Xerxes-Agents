# Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License").
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
"""Pytest glue for the Xerxes playground eval harnesses.

The eval scripts (``eval.py`` warm-up suite, ``eval_hard.py`` hard battery) are
standalone CLIs that mutate process-global state: they ``os.chdir`` into a
sandbox, set ``XERXES_HOME`` at import time (via ``_harness``), and spawn a real
daemon that bills tokens against the active provider profile.

To expose them under ``pytest -m eval`` WITHOUT dragging any of that global state
into the pytest process, every scenario runs as an isolated **subprocess** that
invokes the existing script with ``-k <name>``. The script's own exit code
(0 = pass, 1 = fail) becomes the test verdict. As a result the standalone CLIs
(``uv run python playground/eval.py`` / ``eval_hard.py``) are completely
unchanged, and the pytest process never imports ``_harness`` or ``xerxes``.

Tests are marked ``@pytest.mark.eval``. When no provider API key is available
they are skipped automatically (see :func:`has_eval_credentials`), so CI without
secrets is never blocked by them.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

PLAYGROUND = Path(__file__).resolve().parent
REPO_ROOT = PLAYGROUND.parent

# Provider API-key env vars that are enough to run the evals even when no xerxes
# provider profile has been persisted.
_PROVIDER_KEY_ENV = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "MINIMAX_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "DEEPSEEK_API_KEY",
    "XAI_API_KEY",
    "TOGETHER_API_KEY",
    "GROQ_API_KEY",
    "MISTRAL_API_KEY",
)


def _xerxes_home() -> Path:
    raw = os.environ.get("XERXES_HOME", "").strip()
    return Path(raw).expanduser() if raw else Path.home() / ".xerxes"


def _active_profile() -> dict | None:
    """Read the active provider profile from ``profiles.json`` WITHOUT importing xerxes.

    Mirrors :func:`xerxes.bridge.profiles.get_active_profile` using only the
    stdlib, so this conftest never triggers a xerxes import (which would freeze
    paths / spawn daemons we explicitly want to avoid in the pytest process).
    """
    profiles_file = _xerxes_home() / "profiles.json"
    if not profiles_file.exists():
        return None
    try:
        store = json.loads(profiles_file.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    active = store.get("active")
    profiles = store.get("profiles") or {}
    if active and active in profiles:
        return profiles[active]
    return None


def has_eval_credentials() -> bool:
    """True if the evals can actually reach a provider.

    A persisted active profile with an ``api_key`` is enough, as is any of the
    common ``*_API_KEY`` env vars. ``False`` => eval tests are auto-skipped.
    """
    if (_active_profile() or {}).get("api_key"):
        return True
    return any(os.environ.get(name) for name in _PROVIDER_KEY_ENV)


def run_eval(script: str, keyword: str | None = None, *, timeout: float = 1800.0) -> subprocess.CompletedProcess:
    """Run a playground eval script as an isolated subprocess.

    Uses the same interpreter/virtualenv pytest runs under and invokes the
    script by path (Python prepends the script's own directory to ``sys.path``,
    so the script's top-level ``import _harness`` still resolves correctly).
    The script's exit code is the verdict: ``0`` = all selected tasks passed,
    ``1`` = at least one failed.
    """
    cmd = [sys.executable, str(PLAYGROUND / script)]
    if keyword:
        cmd += ["-k", keyword]
    # argv is fully controlled (no shell), so this is safe despite subprocess.
    return subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout)


@pytest.fixture
def eval_runner() -> Callable[..., subprocess.CompletedProcess]:
    """Return the :func:`run_eval` subprocess driver for eval-marked tests."""
    return run_eval


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip ``eval``-marked tests when no provider credentials are configured."""
    if has_eval_credentials():
        return
    skip_eval = pytest.mark.skip(reason="no provider API key configured (set a xerxes profile or a *_API_KEY env var)")
    for item in items:
        if "eval" in item.keywords:
            item.add_marker(skip_eval)
