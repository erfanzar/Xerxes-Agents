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
"""Detailed tests for sandbox backend implementations."""

from __future__ import annotations

import base64
import json
import subprocess
from unittest import mock

import pytest
from xerxes.security.sandbox import SandboxBackendConfig, SandboxConfig, SandboxMode


def _multiply(a: int = 1, b: int = 1) -> int:
    return a * b


def _identity(value=None):
    return value


def _greet(name: str = "world") -> str:
    return f"hello {name}"


class TestDockerCommandConstruction:
    """Test that the Docker CLI commands are assembled correctly."""

    def _make_backend(self, **overrides):
        from xerxes.security.sandbox_backends.docker_backend import DockerSandboxBackend

        defaults = dict(
            mode=SandboxMode.STRICT,
            sandboxed_tools={"t"},
            sandbox_timeout=30.0,
            sandbox_memory_limit_mb=512,
            sandbox_network_access=False,
        )
        defaults.update(overrides)
        config = SandboxConfig(**defaults)
        return DockerSandboxBackend(sandbox_config=config)

    def test_basic_command_structure(self):
        backend = self._make_backend()
        cmd = backend._build_docker_command("test")
        assert cmd[0] == "docker"
        assert cmd[1] == "run"
        assert "--rm" in cmd
        assert "-i" in cmd
        # Image should be near the end.
        assert "python:3.12-slim" in cmd

    def test_network_disabled(self):
        backend = self._make_backend(sandbox_network_access=False)
        cmd = backend._build_docker_command("test")
        idx = cmd.index("--network")
        assert cmd[idx + 1] == "none"

    def test_network_enabled(self):
        backend = self._make_backend(sandbox_network_access=True)
        cmd = backend._build_docker_command("test")
        assert "--network" not in cmd

    def test_memory_limit(self):
        backend = self._make_backend(sandbox_memory_limit_mb=1024)
        cmd = backend._build_docker_command("test")
        idx = cmd.index("--memory")
        assert cmd[idx + 1] == "1024m"

    def test_custom_image(self):
        bc = SandboxBackendConfig(image="node:20-slim")
        backend = self._make_backend(backend_config=bc)
        cmd = backend._build_docker_command("test")
        assert "node:20-slim" in cmd

    def test_mount_paths(self):
        bc = SandboxBackendConfig(
            mount_paths={"/host/src": "/app/src", "/host/data": "/app/data"},
            mount_readonly=True,
        )
        backend = self._make_backend(backend_config=bc)
        cmd = backend._build_docker_command("test")
        # Should have two -v flags for custom mounts.
        v_indices = [i for i, x in enumerate(cmd) if x == "-v"]
        # May also have working directory mount, so check custom ones exist.
        vol_args = [cmd[i + 1] for i in v_indices]
        assert any("/host/src:/app/src:ro" in v for v in vol_args)
        assert any("/host/data:/app/data:ro" in v for v in vol_args)

    def test_env_vars(self):
        bc = SandboxBackendConfig(env_vars={"DEBUG": "1", "LANG": "C"})
        backend = self._make_backend(backend_config=bc)
        cmd = backend._build_docker_command("test")
        e_indices = [i for i, x in enumerate(cmd) if x == "-e"]
        env_args = [cmd[i + 1] for i in e_indices]
        assert "DEBUG=1" in env_args
        assert "LANG=C" in env_args


class TestDockerExecuteEdgeCases:
    def _make_backend(self):
        from xerxes.security.sandbox_backends.docker_backend import DockerSandboxBackend

        config = SandboxConfig(
            mode=SandboxMode.STRICT,
            sandboxed_tools={"t"},
            sandbox_timeout=5.0,
        )
        return DockerSandboxBackend(sandbox_config=config)

    def test_execute_returns_none(self):
        backend = self._make_backend()
        result_payload = json.dumps({"ok": True, "value": None}).encode("utf-8")
        encoded = base64.b64encode(result_payload).decode()

        with mock.patch("xerxes.security.sandbox_backends.docker_backend.subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0, stdout=encoded, stderr="")
            result = backend.execute("t", _identity, {})
        assert result is None

    def test_execute_bad_stdout(self):
        backend = self._make_backend()
        with mock.patch("xerxes.security.sandbox_backends.docker_backend.subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0, stdout="not-valid-b64!!!", stderr="")
            with pytest.raises(RuntimeError, match="deserialise"):
                backend.execute("t", _identity, {})

    def test_execute_tool_error_inside_container(self):
        backend = self._make_backend()
        result_payload = json.dumps({"ok": False, "error": "division by zero", "type": "ZeroDivisionError"}).encode(
            "utf-8"
        )
        encoded = base64.b64encode(result_payload).decode()

        with mock.patch("xerxes.security.sandbox_backends.docker_backend.subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0, stdout=encoded, stderr="")
            with pytest.raises(RuntimeError, match=r"ZeroDivisionError.*division by zero"):
                backend.execute("t", _identity, {})


class TestSubprocessBackendExecution:
    def _make_backend(self, **overrides):
        from xerxes.security.sandbox_backends.subprocess_backend import SubprocessSandboxBackend

        defaults = dict(
            mode=SandboxMode.STRICT,
            sandboxed_tools={"t"},
            sandbox_timeout=10.0,
        )
        defaults.update(overrides)
        config = SandboxConfig(**defaults)
        return SubprocessSandboxBackend(sandbox_config=config)

    def test_multiply(self):
        backend = self._make_backend()
        assert backend.execute("t", _multiply, {"a": 6, "b": 7}) == 42

    def test_identity_with_string(self):
        backend = self._make_backend()
        assert backend.execute("t", _identity, {"value": "hello"}) == "hello"

    def test_identity_with_list(self):
        backend = self._make_backend()
        assert backend.execute("t", _identity, {"value": [1, 2, 3]}) == [1, 2, 3]

    def test_greet(self):
        backend = self._make_backend()
        assert backend.execute("t", _greet, {"name": "xerxes"}) == "hello xerxes"

    def test_timeout_very_short(self):
        """A function that takes too long should raise RuntimeError."""
        backend = self._make_backend(sandbox_timeout=0.001)
        # _multiply is fast but the subprocess startup itself may exceed 1ms.
        # We just verify the timeout path exists -- it may or may not trigger.
        # Use a mock to ensure the timeout path is exercised.
        with mock.patch(
            "xerxes.security.sandbox_backends.subprocess_backend.subprocess.run",
            side_effect=subprocess.TimeoutExpired("python", 0.001),
        ):
            with pytest.raises(RuntimeError, match="timed out"):
                backend.execute("t", _multiply, {"a": 1, "b": 1})

    def test_capabilities_shape(self):
        backend = self._make_backend(sandbox_memory_limit_mb=256)
        caps = backend.get_capabilities()
        assert caps["backend"] == "subprocess"
        assert caps["available"] is True
        assert caps["memory_limit_mb"] == 256
        assert caps["isolation_level"] == "process"
