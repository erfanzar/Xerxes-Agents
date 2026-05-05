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
"""Tests for xerxes.sandbox -- sandbox routing, config, and backend integration."""

from __future__ import annotations

import logging
from unittest import mock

import pytest
from xerxes.security.sandbox import (
    ExecutionContext,
    SandboxBackendConfig,
    SandboxConfig,
    SandboxExecutionUnavailableError,
    SandboxMode,
    SandboxRouter,
)


class TestSandboxRouter:
    def test_off_mode_always_host(self):
        router = SandboxRouter(SandboxConfig(mode=SandboxMode.OFF))
        decision = router.decide("execute_shell")
        assert decision.context == ExecutionContext.HOST

    def test_elevated_tool_always_host(self):
        config = SandboxConfig(
            mode=SandboxMode.STRICT,
            sandboxed_tools={"execute_shell"},
            elevated_tools={"execute_shell"},
        )
        router = SandboxRouter(config)
        decision = router.decide("execute_shell")
        assert decision.context == ExecutionContext.HOST
        assert "elevated" in decision.reason.lower()

    def test_warn_mode_sandboxed_tool(self):
        config = SandboxConfig(mode=SandboxMode.WARN, sandboxed_tools={"execute_shell"})
        router = SandboxRouter(config)
        decision = router.decide("execute_shell")
        assert decision.context == ExecutionContext.HOST
        assert "warn mode" in decision.reason.lower()

    def test_strict_mode_sandboxed_tool(self):
        config = SandboxConfig(mode=SandboxMode.STRICT, sandboxed_tools={"execute_shell"})
        router = SandboxRouter(config)
        decision = router.decide("execute_shell")
        assert decision.context == ExecutionContext.SANDBOX

    def test_non_sandboxed_tool_on_host(self):
        config = SandboxConfig(mode=SandboxMode.STRICT, sandboxed_tools={"execute_shell"})
        router = SandboxRouter(config)
        decision = router.decide("search")
        assert decision.context == ExecutionContext.HOST

    def test_default_config_all_host(self):
        router = SandboxRouter()
        decision = router.decide("anything")
        assert decision.context == ExecutionContext.HOST

    def test_execute_in_sandbox_raises_without_backend(self):
        router = SandboxRouter(SandboxConfig(mode=SandboxMode.STRICT, sandboxed_tools={"execute_shell"}))
        with pytest.raises(SandboxExecutionUnavailableError):
            router.execute_in_sandbox("execute_shell", lambda **kwargs: kwargs, {"x": 1})


class TestSandboxConfig:
    def test_backend_type_defaults_none(self):
        config = SandboxConfig()
        assert config.backend_type is None

    def test_backend_type_set(self):
        config = SandboxConfig(backend_type="docker")
        assert config.backend_type == "docker"

    def test_backend_config_defaults(self):
        config = SandboxConfig()
        assert isinstance(config.backend_config, SandboxBackendConfig)
        assert config.backend_config.image == "python:3.12-slim"
        assert config.backend_config.mount_readonly is True

    def test_backend_config_custom(self):
        bc = SandboxBackendConfig(image="myimage:latest", mount_readonly=False)
        config = SandboxConfig(backend_config=bc)
        assert config.backend_config.image == "myimage:latest"
        assert config.backend_config.mount_readonly is False


class TestSandboxBackendConfig:
    def test_defaults(self):
        bc = SandboxBackendConfig()
        assert bc.image == "python:3.12-slim"
        assert bc.mount_paths == {}
        assert bc.mount_readonly is True
        assert bc.env_vars == {}
        assert bc.extra_args == {}

    def test_custom_values(self):
        bc = SandboxBackendConfig(
            image="ubuntu:22.04",
            mount_paths={"/host/data": "/data"},
            env_vars={"FOO": "bar"},
        )
        assert bc.image == "ubuntu:22.04"
        assert bc.mount_paths == {"/host/data": "/data"}
        assert bc.env_vars["FOO"] == "bar"


class TestDockerSandboxBackendMocked:
    """Test DockerSandboxBackend using mocked subprocess calls."""

    def _make_backend(self, **config_overrides):
        from xerxes.security.sandbox_backends.docker_backend import DockerSandboxBackend

        config = SandboxConfig(
            mode=SandboxMode.STRICT,
            sandboxed_tools={"test_tool"},
            sandbox_timeout=10.0,
            sandbox_memory_limit_mb=256,
            sandbox_network_access=False,
            **config_overrides,
        )
        return DockerSandboxBackend(sandbox_config=config)

    def test_is_available_true(self):
        backend = self._make_backend()
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0)
            assert backend.is_available() is True
            mock_run.assert_called_once()
            args = mock_run.call_args
            assert args[0][0] == ["docker", "info"]

    def test_is_available_false_when_docker_missing(self):
        backend = self._make_backend()
        with mock.patch("subprocess.run", side_effect=FileNotFoundError):
            assert backend.is_available() is False

    def test_is_available_false_when_daemon_down(self):
        backend = self._make_backend()
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=1)
            assert backend.is_available() is False

    def test_get_capabilities(self):
        backend = self._make_backend()
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0)
            caps = backend.get_capabilities()
        assert caps["backend"] == "docker"
        assert caps["available"] is True
        assert caps["memory_limit_mb"] == 256
        assert caps["network_access"] is False

    def test_execute_success(self):
        import base64
        import json

        backend = self._make_backend()
        result_payload = json.dumps({"ok": True, "value": 42}).encode()
        encoded_result = base64.b64encode(result_payload).decode()

        with mock.patch("xerxes.security.sandbox_backends.docker_backend.subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(
                returncode=0,
                stdout=encoded_result,
                stderr="",
            )
            result = backend.execute("test_tool", _add, {"a": 41, "b": 1})
        assert result == 42

    def test_execute_container_failure(self):
        backend = self._make_backend()
        with mock.patch("xerxes.security.sandbox_backends.docker_backend.subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(
                returncode=1,
                stdout="",
                stderr="container error",
            )
            with pytest.raises(RuntimeError, match=r"failed.*exit 1"):
                backend.execute("test_tool", _add, {})

    def test_execute_timeout(self):
        import subprocess as sp

        backend = self._make_backend()
        with mock.patch(
            "xerxes.security.sandbox_backends.docker_backend.subprocess.run",
            side_effect=sp.TimeoutExpired("docker", 10),
        ):
            with pytest.raises(RuntimeError, match="timed out"):
                backend.execute("test_tool", _add, {})

    def test_docker_command_includes_memory_and_network(self):
        backend = self._make_backend()
        cmd = backend._build_docker_command("test_tool")
        assert "--memory" in cmd
        assert "256m" in cmd
        assert "--network" in cmd
        assert "none" in cmd

    def test_docker_command_with_working_directory(self):
        backend = self._make_backend(working_directory="/tmp/workdir")
        cmd = backend._build_docker_command("test_tool")

        assert "-v" in cmd
        assert any("/tmp/workdir:/workspace" in arg for arg in cmd)


class TestSubprocessSandboxBackend:
    def _make_backend(self, **config_overrides):
        from xerxes.security.sandbox_backends.subprocess_backend import SubprocessSandboxBackend

        config = SandboxConfig(
            mode=SandboxMode.STRICT,
            sandboxed_tools={"test_tool"},
            sandbox_timeout=10.0,
            **config_overrides,
        )
        return SubprocessSandboxBackend(sandbox_config=config)

    def test_is_available_always_true(self):
        backend = self._make_backend()
        assert backend.is_available() is True

    def test_get_capabilities(self):
        backend = self._make_backend()
        caps = backend.get_capabilities()
        assert caps["backend"] == "subprocess"
        assert caps["available"] is True
        assert caps["filesystem_isolation"] is False

    def test_execute_simple_function(self):
        """Run a real function in a subprocess and get the result back."""
        backend = self._make_backend()

        result = backend.execute("test_tool", _add, {"a": 3, "b": 4})
        assert result == 7

    def test_execute_function_raising(self):
        backend = self._make_backend()
        with pytest.raises(RuntimeError, match="inside subprocess sandbox"):
            backend.execute("test_tool", _raise_value_error, {})


class TestSandboxRouterWithBackend:
    def test_strict_mode_with_available_backend_succeeds(self):
        from xerxes.security.sandbox_backends.subprocess_backend import SubprocessSandboxBackend

        config = SandboxConfig(
            mode=SandboxMode.STRICT,
            sandboxed_tools={"test_tool"},
            sandbox_timeout=10.0,
        )
        backend = SubprocessSandboxBackend(sandbox_config=config)
        router = SandboxRouter(config=config, backend=backend)

        decision = router.decide("test_tool")
        assert decision.context == ExecutionContext.SANDBOX

        result = router.execute_in_sandbox("test_tool", _add, {"a": 10, "b": 20})
        assert result == 30

    def test_strict_mode_without_backend_fails_hard(self):
        config = SandboxConfig(
            mode=SandboxMode.STRICT,
            sandboxed_tools={"test_tool"},
        )
        router = SandboxRouter(config=config, backend=None)

        decision = router.decide("test_tool")
        assert decision.context == ExecutionContext.SANDBOX

        with pytest.raises(SandboxExecutionUnavailableError):
            router.execute_in_sandbox("test_tool", _add, {"a": 1, "b": 2})

    def test_warn_mode_runs_on_host_with_logging(self, caplog):
        config = SandboxConfig(
            mode=SandboxMode.WARN,
            sandboxed_tools={"test_tool"},
        )
        router = SandboxRouter(config=config, backend=None)

        with caplog.at_level(logging.WARNING, logger="xerxes.sandbox"):
            decision = router.decide("test_tool")

        assert decision.context == ExecutionContext.HOST
        assert "warn mode" in decision.reason.lower()
        assert any("would run in sandbox" in r.message for r in caplog.records)


class TestBackendRegistry:
    def test_list_backends(self):
        from xerxes.security.sandbox_backends import list_backends

        names = list_backends()
        assert "docker" in names
        assert "subprocess" in names

    def test_get_subprocess_backend(self):
        from xerxes.security.sandbox_backends import get_backend
        from xerxes.security.sandbox_backends.subprocess_backend import SubprocessSandboxBackend

        config = SandboxConfig()
        backend = get_backend("subprocess", config)
        assert isinstance(backend, SubprocessSandboxBackend)

    def test_get_docker_backend(self):
        from xerxes.security.sandbox_backends import get_backend
        from xerxes.security.sandbox_backends.docker_backend import DockerSandboxBackend

        config = SandboxConfig()
        backend = get_backend("docker", config)
        assert isinstance(backend, DockerSandboxBackend)

    def test_unknown_backend_raises(self):
        from xerxes.security.sandbox_backends import get_backend

        with pytest.raises(ValueError, match="Unknown sandbox backend"):
            get_backend("nonexistent", SandboxConfig())

    def test_register_custom_backend(self):
        from xerxes.security.sandbox_backends import get_backend, register_backend

        class _DummyBackend:
            def __init__(self, sandbox_config):
                self.config = sandbox_config

            def execute(self, tool_name, func, arguments):
                return func(**arguments)

            def is_available(self):
                return True

            def get_capabilities(self):
                return {"backend": "dummy"}

        register_backend("dummy_test", _DummyBackend)
        config = SandboxConfig()
        backend = get_backend("dummy_test", config)
        assert isinstance(backend, _DummyBackend)


def _add(a: int = 0, b: int = 0) -> int:
    return a + b


def _raise_value_error() -> None:
    raise ValueError("intentional test error")
