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
"""Tests for RL training-control tools."""

from __future__ import annotations

import pytest
from xerxes.tools.rl_tools import (
    InMemoryRLBackend,
    reset_rl_backend,
    rl_check_status,
    rl_edit_config,
    rl_get_current_config,
    rl_get_results,
    rl_list_environments,
    rl_list_runs,
    rl_select_environment,
    rl_start_training,
    rl_stop_training,
    rl_test_inference,
    set_rl_backend,
)


@pytest.fixture
def backend():
    b = InMemoryRLBackend()
    b.register("cartpole-v1", {"lr": 1e-3, "steps": 100}, description="classic control")
    b.register("lunarlander-v2", {"lr": 5e-4, "steps": 200})
    set_rl_backend(b)
    yield b
    reset_rl_backend()


class TestEnvironments:
    def test_list(self, backend):
        out = rl_list_environments.static_call()
        assert out["count"] == 2
        names = {e["name"] for e in out["environments"]}
        assert "cartpole-v1" in names

    def test_select_existing(self, backend):
        out = rl_select_environment.static_call(name="cartpole-v1")
        assert out["name"] == "cartpole-v1"
        assert out["config"]["lr"] == 1e-3

    def test_select_missing(self, backend):
        out = rl_select_environment.static_call(name="nope")
        assert out["error"] == "not_found"

    def test_get_current_config_after_select(self, backend):
        rl_select_environment.static_call(name="cartpole-v1")
        out = rl_get_current_config.static_call()
        assert out["environment"] == "cartpole-v1"
        assert out["config"]["lr"] == 1e-3

    def test_edit_config_merges(self, backend):
        rl_select_environment.static_call(name="cartpole-v1")
        out = rl_edit_config.static_call(updates={"lr": 5e-3, "gamma": 0.99})
        assert out["config"]["lr"] == 5e-3
        assert out["config"]["gamma"] == 0.99
        assert out["config"]["steps"] == 100  # untouched


class TestRunLifecycle:
    def test_start_requires_environment(self, backend):
        out = rl_start_training.static_call()
        assert "error" in out

    def test_start_status_stop(self, backend):
        rl_select_environment.static_call(name="cartpole-v1")
        run = rl_start_training.static_call()
        assert run["status"] == "running"
        rid = run["run_id"]
        live = rl_check_status.static_call(run_id=rid)
        assert live["run_id"] == rid
        stopped = rl_stop_training.static_call(run_id=rid)
        assert stopped["status"] == "stopped"

    def test_get_results_for_completed(self, backend):
        rl_select_environment.static_call(name="cartpole-v1")
        run = rl_start_training.static_call()
        rl_stop_training.static_call(run_id=run["run_id"])
        out = rl_get_results.static_call(run_id=run["run_id"])
        assert out["status"] == "stopped"
        assert "duration_s" in out

    def test_get_results_partial_for_running(self, backend):
        rl_select_environment.static_call(name="cartpole-v1")
        run = rl_start_training.static_call()
        out = rl_get_results.static_call(run_id=run["run_id"])
        assert out["status"] == "running"
        assert "partial_metrics" in out

    def test_list_runs_sorted_desc(self, backend):
        rl_select_environment.static_call(name="cartpole-v1")
        a = rl_start_training.static_call()
        b = rl_start_training.static_call()
        out = rl_list_runs.static_call()
        assert out["count"] == 2
        ids = [r["run_id"] for r in out["runs"]]
        assert ids[0] in {a["run_id"], b["run_id"]}

    def test_stop_unknown(self, backend):
        out = rl_stop_training.static_call(run_id="missing")
        assert out["error"] == "not_found"

    def test_check_status_unknown(self, backend):
        out = rl_check_status.static_call(run_id="missing")
        assert out["error"] == "not_found"

    def test_test_inference(self, backend):
        rl_select_environment.static_call(name="cartpole-v1")
        out = rl_test_inference.static_call(prompt="hello world")
        assert "completion" in out
        assert "hello world" in out["completion"] or out["prompt"] == "hello world"
