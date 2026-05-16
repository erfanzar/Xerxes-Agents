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
"""Tests for runtime.doctor, runtime.update, runtime.setup_wizard."""

from __future__ import annotations

import httpx
import pytest
from xerxes.runtime import doctor, setup_wizard, update

# ---------------------------- doctor ---------------------------------------


class TestDoctor:
    def test_python_check_passes_for_supported(self):
        d = doctor.check_python_version()
        # We're running on whatever pyver the venv has; this should be 3.11+.
        assert d.severity == "ok"

    def test_required_imports_check(self):
        d = doctor.check_required_imports()
        # pydantic / httpx / openai / rich are all in core; should be ok.
        assert d.severity == "ok"

    def test_provider_keys_warn_when_missing(self, monkeypatch):
        for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY"):
            monkeypatch.delenv(k, raising=False)
        d = doctor.check_provider_keys()
        assert d.severity == "warn"

    def test_provider_keys_ok_when_present(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-stub")
        d = doctor.check_provider_keys()
        assert d.severity == "ok"

    def test_run_all_returns_one_per_check(self):
        report = doctor.run_all_checks()
        assert len(report) == len(doctor.DEFAULT_CHECKS)

    def test_format_report_has_severity_icon(self):
        report = doctor.run_minimal()
        out = doctor.format_report(report)
        assert any(icon in out for icon in ("✓", "!", "✗"))

    def test_has_failures(self):
        from xerxes.runtime.doctor import Diagnosis

        assert doctor.has_failures([Diagnosis("x", "ok", "")]) is False
        assert doctor.has_failures([Diagnosis("x", "fail", "")]) is True


# ---------------------------- update ---------------------------------------


class TestUpdate:
    def test_detect_install_mode_returns_enum(self):
        m = update.detect_install_mode()
        assert isinstance(m, update.InstallMode)

    def test_semver_gt_basic(self):
        assert update._semver_gt("0.2.1", "0.2.0") is True
        assert update._semver_gt("0.2.0", "0.2.0") is False
        assert update._semver_gt("0.1.99", "0.2.0") is False

    def test_latest_pypi_version_parses_response(self):
        def handler(req):
            return httpx.Response(200, json={"info": {"version": "9.9.9"}})

        c = httpx.Client(transport=httpx.MockTransport(handler))
        assert update.latest_pypi_version(client=c) == "9.9.9"
        c.close()

    def test_latest_pypi_version_network_error_returns_none(self):
        def handler(req):
            raise httpx.RequestError("boom")

        c = httpx.Client(transport=httpx.MockTransport(handler))
        assert update.latest_pypi_version(client=c) is None
        c.close()

    def test_check_for_update_returns_none_when_current_or_newer(self):
        def handler(req):
            return httpx.Response(200, json={"info": {"version": "0.0.1"}})

        c = httpx.Client(transport=httpx.MockTransport(handler))
        out = update.check_for_update(client=c)
        # Installed is 0.2.0 in pyproject; latest mocked to 0.0.1 → no update.
        assert out is None
        c.close()

    def test_check_for_update_returns_value_when_newer(self):
        def handler(req):
            return httpx.Response(200, json={"info": {"version": "99.0.0"}})

        c = httpx.Client(transport=httpx.MockTransport(handler))
        out = update.check_for_update(client=c)
        assert out is not None
        assert out.latest_version == "99.0.0"
        c.close()

    def test_apply_update_dry_run(self):
        out = update.apply_update(dry_run=True)
        assert out["dry_run"] is True
        assert "argv" in out


# ---------------------------- setup wizard ---------------------------------


class TestSetupWizard:
    def test_default_steps_with_no_input_uses_defaults(self):
        out = setup_wizard.run_wizard({})
        # api_key is optional; expect it skipped.
        assert "api_key" in out.skipped
        assert out.answers["provider"] == "anthropic"
        assert out.answers["model"] == "claude-opus-4-7"

    def test_explicit_answers_override_defaults(self):
        out = setup_wizard.run_wizard(
            {
                "provider": "openrouter",
                "model": "google/gemini-2.5-flash",
                "api_key": "sk-or-...",
                "messaging_platform": "telegram",
            }
        )
        assert out.answers["provider"] == "openrouter"
        assert out.answers["api_key"] == "sk-or-..."

    def test_validator_failure_raises(self):
        steps = (
            setup_wizard.SetupStep(
                key="n",
                prompt="number",
                default=0,
                validator=lambda v: isinstance(v, int) and v > 0,
            ),
        )
        with pytest.raises(ValueError):
            setup_wizard.run_wizard({"n": "not-an-int"}, steps=steps)

    def test_write_config_writes_keys(self, tmp_path):
        path = setup_wizard.write_config(
            {"provider": "anthropic", "model": "claude-haiku-4-5"},
            target=tmp_path / "config.yaml",
        )
        text = path.read_text()
        assert 'provider: "anthropic"' in text
        assert 'model: "claude-haiku-4-5"' in text
