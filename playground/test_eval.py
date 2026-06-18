# Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License").
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
"""Pytest wrappers for the playground eval harnesses (``@pytest.mark.eval``).

Each scenario is a separate, subprocess-isolated test, so
``uv run pytest -m eval`` reports exactly which evals pass or fail. The driver
lives in ``playground/conftest.py`` (see :func:`conftest.run_eval`); the
standalone CLIs (``uv run python playground/eval.py`` / ``eval_hard.py``) are
unchanged and remain the primary way to run a whole suite interactively.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

PLAYGROUND = Path(__file__).resolve().parent

# Warm-up suite task names (see ``playground/eval.py::task_suite``). Listed here
# rather than imported, because importing eval.py runs ``import _harness`` which
# mutates XERXES_HOME / sys.path — exactly the global state we isolate against by
# running scenarios as subprocesses. Add a name here when you add a warm-up task.
WARMUP_TASKS = [
    "reasoning",
    "file_read",
    "file_edit",
    "bug_fix",
    "tool_search",
    "shell",
    "multiturn",
    "memory_recall",
]


def _hard_task_names() -> list[str]:
    """Scenario names from the hard battery's source-of-truth JSON."""
    tasks_file = PLAYGROUND / "hard_tasks.json"
    try:
        return [t["name"] for t in json.loads(tasks_file.read_text())]
    except (OSError, json.JSONDecodeError, KeyError):
        return []


def _failed_message(name: str, res: subprocess.CompletedProcess) -> str:
    tail = (res.stdout or "").strip().splitlines()[-12:]
    detail = "\n".join(tail) if tail else "(no stdout)"
    return f"eval scenario {name!r} failed (exit {res.returncode})\n--- stdout tail ---\n{detail}"


@pytest.mark.eval
@pytest.mark.parametrize("name", WARMUP_TASKS, ids=lambda n: f"warmup-{n}")
def test_eval_warmup(eval_runner, name: str) -> None:
    """Run one warm-up-suite scenario: ``playground/eval.py -k <name>``."""
    res = eval_runner("eval.py", keyword=name)
    assert res.returncode == 0, _failed_message(name, res)


@pytest.mark.eval
@pytest.mark.parametrize("name", _hard_task_names(), ids=lambda n: f"hard-{n}")
def test_eval_hard(eval_runner, name: str) -> None:
    """Run one hard-battery scenario: ``playground/eval_hard.py -k <name>``."""
    res = eval_runner("eval_hard.py", keyword=name)
    assert res.returncode == 0, _failed_message(name, res)
