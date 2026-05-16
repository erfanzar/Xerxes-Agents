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
"""Tests for the new remote sandbox backends."""

from __future__ import annotations

import shutil

import pytest
from xerxes.security.sandbox_backends import credential_files
from xerxes.security.sandbox_backends.daytona_backend import DaytonaSandboxBackend
from xerxes.security.sandbox_backends.file_sync import FileSyncSpec, sync_pull, sync_push
from xerxes.security.sandbox_backends.modal_backend import ModalSandboxBackend
from xerxes.security.sandbox_backends.singularity_backend import SingularitySandboxBackend
from xerxes.security.sandbox_backends.ssh_backend import SshBackendConfig, SshSandboxBackend


class TestSshBackend:
    def test_missing_host_raises(self, monkeypatch):
        monkeypatch.delenv("XERXES_SSH_HOST", raising=False)
        b = SshSandboxBackend()
        with pytest.raises(RuntimeError, match="XERXES_SSH_HOST"):
            b.execute("echo hi")

    def test_executes_via_local_ssh_argv(self, monkeypatch):
        """Verify argv assembly by stubbing subprocess.run."""
        import subprocess as sp

        captured = {}

        def fake_run(argv, **kw):
            captured["argv"] = argv

            class R:
                returncode = 0
                stdout = "OK"
                stderr = ""

            return R()

        monkeypatch.setattr(sp, "run", fake_run)
        b = SshSandboxBackend(SshBackendConfig(host="build-host", user="ci", port=2222, identity_file="/k.pem"))
        out = b.execute("uname -a", cwd="/work", env={"FOO": "bar"})
        assert out["returncode"] == 0
        # First few argv components
        assert captured["argv"][0] == "ssh"
        assert "BatchMode=yes" in captured["argv"]
        assert "ci@build-host" in captured["argv"]
        assert "2222" in captured["argv"]
        # Remote command was constructed properly. shlex.quote produces unquoted
        # values for safe alphanumeric strings, so we check both possibilities.
        remote_cmd = captured["argv"][-1]
        assert remote_cmd.startswith("cd /work")
        assert "FOO=bar" in remote_cmd
        assert "uname -a" in remote_cmd


class TestModalBackend:
    def test_missing_sdk_raises(self, monkeypatch):
        import importlib.util

        original = importlib.util.find_spec
        monkeypatch.setattr(importlib.util, "find_spec", lambda n: None if n == "modal" else original(n))
        b = ModalSandboxBackend()
        with pytest.raises(RuntimeError, match="Modal SDK"):
            b.execute("echo hi")


class TestDaytonaBackend:
    def test_missing_sdk_raises(self, monkeypatch):
        import importlib.util

        original = importlib.util.find_spec
        monkeypatch.setattr(importlib.util, "find_spec", lambda n: None if n == "daytona" else original(n))
        b = DaytonaSandboxBackend()
        with pytest.raises(RuntimeError, match="Daytona SDK"):
            b.execute("echo hi")


class TestSingularityBackend:
    def test_missing_binary_raises(self, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda c: None)
        with pytest.raises(RuntimeError, match="singularity nor apptainer"):
            SingularitySandboxBackend().execute("echo hi")


class TestFileSync:
    def test_push_filters_oversized(self, tmp_path):
        big = tmp_path / "big.bin"
        big.write_bytes(b"x" * 5_000)
        small = tmp_path / "small.txt"
        small.write_text("hi")
        copied = []
        sync_push(
            [FileSyncSpec(big, "/r/big"), FileSyncSpec(small, "/r/small")],
            copy_fn=lambda lp, rp: copied.append((lp.name, rp)),
            max_bytes=1024,
        )
        assert copied == [("small.txt", "/r/small")]

    def test_push_skips_missing(self, tmp_path):
        out = sync_push(
            [FileSyncSpec(tmp_path / "nope.txt", "/r/x")],
            copy_fn=lambda lp, rp: None,
        )
        assert out == []

    def test_pull_continues_on_error(self, tmp_path):
        calls = []

        def pull(lp, rp):
            calls.append(rp)
            if rp == "/r/bad":
                raise OSError("nope")

        out = sync_pull(
            [
                FileSyncSpec(tmp_path / "a", "/r/ok"),
                FileSyncSpec(tmp_path / "b", "/r/bad"),
                FileSyncSpec(tmp_path / "c", "/r/also-ok"),
            ],
            pull,
        )
        assert calls == ["/r/ok", "/r/bad", "/r/also-ok"]
        assert [s.remote_path for s in out] == ["/r/ok", "/r/also-ok"]


class TestCredentialFiles:
    def setup_method(self):
        credential_files.clear()

    def teardown_method(self):
        credential_files.clear()

    def test_register_and_check(self, tmp_path):
        p = tmp_path / "k.json"
        p.write_text("secret")
        credential_files.register(p)
        assert credential_files.is_allowed(p) is True

    def test_unregister(self, tmp_path):
        p = tmp_path / "k.json"
        p.write_text("x")
        credential_files.register(p)
        assert credential_files.unregister(p) is True
        assert credential_files.is_allowed(p) is False

    def test_env_passthrough(self, monkeypatch):
        monkeypatch.setenv("FOO_X", "yes")
        monkeypatch.delenv("BAR_X", raising=False)
        out = credential_files.env_passthrough(["FOO_X", "BAR_X"])
        assert out == {"FOO_X": "yes"}
