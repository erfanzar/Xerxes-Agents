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
"""Tests for the RL training scaffolding (Plan 15)."""

from __future__ import annotations

import pytest
from xerxes.training.rl import (
    RLEnvironment,
    RLEnvironmentRegistry,
    RLRunState,
    RLRunStatus,
    TinkerClient,
    TinkerRunConfig,
    WandBHook,
    builtin_envs,
)
from xerxes.training.rl.status import can_transition


class TestEnvRegistry:
    def test_builtin_envs_present(self):
        reg = builtin_envs()
        names = [e.name for e in reg.list_envs()]
        assert "xerxes-terminal-test" in names
        assert "xerxes-swe-bench" in names

    def test_register_and_get(self):
        reg = RLEnvironmentRegistry()
        env = RLEnvironment(name="custom", description="d")
        reg.register(env)
        assert reg.get("custom") is env

    def test_reject_unnamed_env(self):
        reg = RLEnvironmentRegistry()
        with pytest.raises(ValueError):
            reg.register(RLEnvironment(name="", description="x"))

    def test_reward_fn_invocation(self):
        reg = builtin_envs()
        env = reg.get("xerxes-swe-bench")
        assert env is not None and env.reward_fn is not None
        r = env.reward_fn({"tests_passed": 4, "tests_total": 10})
        assert abs(r - 0.4) < 1e-9


class TestStatusStateMachine:
    def test_can_transition_to_running(self):
        assert can_transition(RLRunStatus.PENDING, RLRunStatus.RUNNING)

    def test_cannot_run_after_terminal(self):
        assert not can_transition(RLRunStatus.SUCCEEDED, RLRunStatus.RUNNING)
        assert not can_transition(RLRunStatus.FAILED, RLRunStatus.RUNNING)

    def test_cancellation_allowed_from_pending(self):
        assert can_transition(RLRunStatus.PENDING, RLRunStatus.CANCELLED)


class TestTinkerClient:
    def test_start_uses_backend(self):
        seen = {}

        def fake_start(payload):
            seen["payload"] = payload
            return "run-1"

        client = TinkerClient(backend_start=fake_start)
        rid = client.start(TinkerRunConfig(model="m", env="e"))
        assert rid == "run-1"
        assert seen["payload"]["model"] == "m"

    def test_status_maps_running(self):
        client = TinkerClient(backend_status=lambda rid: {"status": "running", "iteration": 5, "tokens_seen": 1000})
        state = client.status("run-1")
        assert state.status is RLRunStatus.RUNNING
        assert state.iteration == 5
        assert state.tokens_seen == 1000

    def test_status_failed(self):
        client = TinkerClient(backend_status=lambda rid: {"status": "failed", "error": "oom"})
        state = client.status("r")
        assert state.status is RLRunStatus.FAILED
        assert state.error == "oom"

    def test_status_unconfigured_returns_failed(self):
        client = TinkerClient()
        state = client.status("r")
        assert state.status is RLRunStatus.FAILED

    def test_cancel_routes_to_backend(self):
        called = []
        client = TinkerClient(backend_cancel=lambda rid: called.append(rid) or True)
        assert client.cancel("rid") is True
        assert called == ["rid"]

    def test_cancel_no_backend_returns_false(self):
        assert TinkerClient().cancel("rid") is False

    def test_run_state_defaults(self):
        s = RLRunState(run_id="x")
        assert s.status is RLRunStatus.PENDING
        assert s.iteration == 0


class TestWandBHook:
    def test_unavailable_when_no_env(self, monkeypatch):
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        hook = WandBHook()
        assert hook.is_available() is False

    def test_start_noop_when_unavailable(self, monkeypatch):
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        hook = WandBHook()
        assert hook.start({}) == ""

    def test_log_noop_when_no_run(self):
        WandBHook().log({"x": 1})  # must not raise
