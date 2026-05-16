# Copyright 2026 Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
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
"""Regression coverage for the daemon's ``/provider`` slash command.

The bridge had this; the daemon shipped without it, so ``/provider``
came back as ``Unknown command`` even though it was listed in /help.
"""

from __future__ import annotations

import asyncio

import pytest
from xerxes.daemon.config import DaemonConfig
from xerxes.daemon.runtime import RuntimeManager


class _Recorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    async def __call__(self, event_type: str, payload: dict) -> None:
        self.events.append((event_type, payload))

    def slash_outputs(self) -> list[str]:
        return [
            payload.get("body", "")
            for (etype, payload) in self.events
            if etype == "notification" and payload.get("category") == "slash"
        ]

    # Alias used by the interactive-panel tests.
    slash_bodies = slash_outputs


@pytest.fixture
def daemon(tmp_path):
    from xerxes.daemon.server import DaemonServer

    server = DaemonServer.__new__(DaemonServer)
    server.config = DaemonConfig(project_dir=str(tmp_path))
    server.runtime = RuntimeManager(server.config)
    server.runtime.runtime_config = {"permission_mode": "auto"}
    server._provider_flow = None
    server._pending_slash_arg = None
    server._pending_skill_create = None
    return server


def _drive(server, command: str) -> list[str]:
    rec = _Recorder()
    asyncio.new_event_loop().run_until_complete(server._handle_slash(command, rec))
    return rec.slash_outputs()


class TestSlashProvider:
    def test_no_profiles_returns_helpful_hint(self, daemon, monkeypatch):
        monkeypatch.setattr("xerxes.bridge.profiles.list_profiles", lambda: [])
        monkeypatch.setattr("xerxes.bridge.profiles.get_active_profile", lambda: None)
        out = _drive(daemon, "/provider")
        assert out
        assert "No provider profiles" in out[0]
        assert "XERXES_BASE_URL" in out[0]

    def test_bare_provider_emits_panel_with_active_marker(self, daemon, monkeypatch):
        """Bare ``/provider`` now opens an interactive panel — assert the
        emitted question_request includes the right options + an "active"
        marker on the current profile.
        """
        profiles_data = [
            {"name": "kimi", "model": "kimi-k2", "base_url": "https://api.moonshot.cn"},
            {"name": "openai", "model": "gpt-4o", "base_url": "https://api.openai.com/v1"},
        ]
        monkeypatch.setattr("xerxes.bridge.profiles.list_profiles", lambda: profiles_data)
        monkeypatch.setattr("xerxes.bridge.profiles.get_active_profile", lambda: profiles_data[0])
        rec = _Recorder()
        asyncio.new_event_loop().run_until_complete(daemon._handle_slash("/provider", rec))
        qrs = [p for (t, p) in rec.events if t == "question_request"]
        assert qrs, "no question_request emitted"
        options = qrs[0]["questions"][0]["options"]
        # Active marker should be present on kimi (current active) but not openai.
        kimi_row = next(o for o in options if o.startswith("kimi"))
        openai_row = next(o for o in options if o.startswith("openai"))
        assert "active" in kimi_row
        assert "active" not in openai_row
        # Models and base URLs are part of the row.
        assert "kimi-k2" in kimi_row and "gpt-4o" in openai_row

    def test_switch_to_unknown_profile_errors(self, daemon, monkeypatch):
        profiles_data = [
            {"name": "kimi", "model": "kimi-k2", "base_url": "https://api.moonshot.cn"},
        ]
        monkeypatch.setattr("xerxes.bridge.profiles.list_profiles", lambda: profiles_data)
        monkeypatch.setattr("xerxes.bridge.profiles.get_active_profile", lambda: profiles_data[0])
        out = _drive(daemon, "/provider does-not-exist")
        assert "No profile named" in out[0]
        assert "kimi" in out[0]

    def test_switch_calls_set_active_and_reloads(self, daemon, monkeypatch):
        switched_to: list[str] = []
        profiles_data = [
            {"name": "kimi", "model": "kimi-k2", "base_url": "https://api.moonshot.cn"},
            {"name": "openai", "model": "gpt-4o", "base_url": "https://api.openai.com/v1"},
        ]
        monkeypatch.setattr("xerxes.bridge.profiles.list_profiles", lambda: profiles_data)
        monkeypatch.setattr(
            "xerxes.bridge.profiles.get_active_profile",
            lambda: profiles_data[1 if switched_to else 0],
        )

        def _fake_set_active(name):
            switched_to.append(name)
            return True

        monkeypatch.setattr("xerxes.bridge.profiles.set_active", _fake_set_active)
        # Stub reload so the test doesn't try to bootstrap a real LLM.
        monkeypatch.setattr(daemon.runtime, "reload", lambda *_a, **_kw: None)

        out = _drive(daemon, "/provider openai")
        assert switched_to == ["openai"]
        assert "Switched to `openai`" in out[0]
        assert "gpt-4o" in out[0]


class TestSlashProviderInteractivePanel:
    """Bare ``/provider`` should pop a question_request panel; follow-ups
    advance the state machine through Add / Edit / Remove flows.
    """

    @pytest.fixture
    def profiles_data(self):
        return [
            {"name": "kimi", "model": "kimi-k2", "base_url": "https://api.moonshot.cn", "api_key": "K"},
            {"name": "openai", "model": "gpt-4o", "base_url": "https://api.openai.com/v1", "api_key": "O"},
        ]

    def _events_of_type(self, recorder, etype):
        return [p for (t, p) in recorder.events if t == etype]

    def _async_question_response(self, daemon, request_id, answers, rec):
        from xerxes.daemon.server import DaemonServer

        coro = DaemonServer._handle_rpc(
            daemon,
            "question_response",
            {"request_id": request_id, "answers": answers},
            rec,
        )
        asyncio.new_event_loop().run_until_complete(coro)

    def test_bare_provider_emits_question_request(self, daemon, monkeypatch, profiles_data):
        monkeypatch.setattr("xerxes.bridge.profiles.list_profiles", lambda: profiles_data)
        monkeypatch.setattr("xerxes.bridge.profiles.get_active_profile", lambda: profiles_data[0])

        rec = _drive(daemon, "/provider")  # noqa: F841 — drive returns slash bodies, we want events
        # Drive again capturing the full Recorder for inspection.
        full = _Recorder()
        asyncio.new_event_loop().run_until_complete(daemon._handle_slash("/provider", full))
        qrs = self._events_of_type(full, "question_request")
        assert qrs, "no question_request emitted"
        options = qrs[0]["questions"][0]["options"]
        # Existing profiles + Add + Edit + Remove + Cancel.
        assert any("kimi" in o for o in options)
        assert any("openai" in o for o in options)
        assert "+ Add new profile…" in options
        assert "✎ Edit existing profile…" in options
        assert "✗ Remove existing profile…" in options
        assert "Cancel" in options

    def test_picking_existing_profile_switches(self, daemon, monkeypatch, profiles_data):
        switched: list[str] = []
        monkeypatch.setattr("xerxes.bridge.profiles.list_profiles", lambda: profiles_data)
        monkeypatch.setattr("xerxes.bridge.profiles.get_active_profile", lambda: profiles_data[0])
        monkeypatch.setattr(
            "xerxes.bridge.profiles.set_active",
            lambda name: (switched.append(name), True)[1],
        )
        monkeypatch.setattr(daemon.runtime, "reload", lambda *_a, **_kw: None)

        rec = _Recorder()
        asyncio.new_event_loop().run_until_complete(daemon._handle_slash("/provider", rec))
        qrs = self._events_of_type(rec, "question_request")
        rid = qrs[0]["id"]
        # Pick the openai row (label format: "openai  (gpt-4o @ https://api.openai.com/v1)").
        picked = next(o for o in qrs[0]["questions"][0]["options"] if o.startswith("openai"))
        self._async_question_response(daemon, rid, {"action": picked}, rec)
        assert switched == ["openai"]
        bodies = rec.slash_bodies()
        assert any("Switched to `openai`" in b for b in bodies)

    def test_add_flow_emits_followup_then_saves(self, daemon, monkeypatch, profiles_data):
        saved: list[dict] = []
        monkeypatch.setattr("xerxes.bridge.profiles.list_profiles", lambda: profiles_data)
        monkeypatch.setattr("xerxes.bridge.profiles.get_active_profile", lambda: profiles_data[0])

        def _save(name, base_url, api_key, model, provider="", *_a, **_kw):
            saved.append(
                {
                    "name": name,
                    "base_url": base_url,
                    "api_key": api_key,
                    "model": model,
                    "provider": provider,
                }
            )
            return saved[-1]

        monkeypatch.setattr("xerxes.bridge.profiles.save_profile", _save)
        monkeypatch.setattr(daemon.runtime, "reload", lambda *_a, **_kw: None)

        rec = _Recorder()
        asyncio.new_event_loop().run_until_complete(daemon._handle_slash("/provider", rec))
        rid1 = self._events_of_type(rec, "question_request")[0]["id"]
        self._async_question_response(daemon, rid1, {"action": "+ Add new profile…"}, rec)
        # A second question_request should have been emitted with the four
        # field questions.
        qrs = self._events_of_type(rec, "question_request")
        assert len(qrs) == 2, "follow-up question_request not emitted"
        meta_panel = qrs[1]
        ids = [q["id"] for q in meta_panel["questions"]]
        assert ids == ["name", "provider_type"]
        # ``provider_type`` is a picker — must list `auto` plus known tags.
        ptype_q = next(q for q in meta_panel["questions"] if q["id"] == "provider_type")
        assert "auto" in ptype_q["options"]
        assert "anthropic" in ptype_q["options"]
        assert "ollama" in ptype_q["options"]
        assert "kimi-code" in ptype_q["options"]

        # Stub fetch_models so stage 3 (model picker) doesn't hit the network.
        monkeypatch.setattr(
            "xerxes.bridge.profiles.fetch_models",
            lambda base_url, api_key: ["gpt-4o", "gpt-4o-mini", "o1-preview"],
        )

        # Stage 1 answer → stage 2 panel pops with provider-aware defaults.
        # Use openai because its ProviderConfig has a concrete base_url set
        # (anthropic relies on the SDK default and has no URL we can suggest).
        self._async_question_response(
            daemon,
            meta_panel["id"],
            {"name": "myopenai", "provider_type": "openai"},
            rec,
        )
        all_qrs = [p for (t, p) in rec.events if t == "question_request"]
        creds_panel = all_qrs[-1]
        creds_ids = [q["id"] for q in creds_panel["questions"]]
        # Stage 2 only collects URL + key now; model moved to stage 3.
        assert creds_ids == ["base_url", "api_key"]
        # The base_url question must surface the openai default.
        url_q = next(q for q in creds_panel["questions"] if q["id"] == "base_url")
        assert "https://api.openai.com/v1" in url_q["question"]

        # Stage 2 answer → stage 3 (model picker) with fetched options.
        self._async_question_response(
            daemon,
            creds_panel["id"],
            {"base_url": "", "api_key": "sk-test"},
            rec,
        )
        model_panel = [p for (t, p) in rec.events if t == "question_request"][-1]
        model_q = model_panel["questions"][0]
        assert model_q["id"] == "model"
        # The /models response surfaced as options.
        assert "gpt-4o" in model_q["options"]
        # Registry default is hoisted to the top of the option list.
        assert model_q["options"][0] == "gpt-4o"
        # A "type custom" sentinel is appended so users can paste anything.
        assert any("Type a custom" in o for o in model_q["options"])

        # Stage 3 answer → save.
        self._async_question_response(
            daemon,
            model_panel["id"],
            {"model": "gpt-4o"},
            rec,
        )
        assert saved
        assert saved[0]["name"] == "myopenai"
        # Defaults flowed through despite being blank in the answer.
        assert saved[0]["base_url"] == "https://api.openai.com/v1"
        assert saved[0]["model"] == "gpt-4o"
        assert saved[0].get("provider") == "openai"
        bodies = rec.slash_bodies()
        assert any("Added profile `myopenai`" in b for b in bodies)

    def test_remove_flow_requires_confirm(self, daemon, monkeypatch, profiles_data):
        deleted: list[str] = []
        monkeypatch.setattr("xerxes.bridge.profiles.list_profiles", lambda: profiles_data)
        monkeypatch.setattr("xerxes.bridge.profiles.get_active_profile", lambda: profiles_data[0])
        monkeypatch.setattr(
            "xerxes.bridge.profiles.delete_profile",
            lambda name: (deleted.append(name), True)[1],
        )
        monkeypatch.setattr(daemon.runtime, "reload", lambda *_a, **_kw: None)

        rec = _Recorder()
        asyncio.new_event_loop().run_until_complete(daemon._handle_slash("/provider", rec))
        rid1 = self._events_of_type(rec, "question_request")[0]["id"]
        self._async_question_response(daemon, rid1, {"action": "✗ Remove existing profile…"}, rec)
        followup = self._events_of_type(rec, "question_request")[-1]
        # Decline confirmation — nothing is deleted.
        self._async_question_response(daemon, followup["id"], {"profile": "kimi", "confirm": "no"}, rec)
        assert deleted == []
        # Re-enter the flow and confirm.
        rec2 = _Recorder()
        asyncio.new_event_loop().run_until_complete(daemon._handle_slash("/provider", rec2))
        rid_main = self._events_of_type(rec2, "question_request")[0]["id"]
        self._async_question_response(daemon, rid_main, {"action": "✗ Remove existing profile…"}, rec2)
        followup2 = self._events_of_type(rec2, "question_request")[-1]
        self._async_question_response(daemon, followup2["id"], {"profile": "kimi", "confirm": "yes"}, rec2)
        assert deleted == ["kimi"]
        bodies = rec2.slash_bodies()
        assert any("Removed profile `kimi`" in b for b in bodies)

    def test_kimi_code_is_a_picker_option(self, daemon, monkeypatch, profiles_data):
        """The Add flow's provider_type picker lists ``kimi-code`` as a
        distinct choice from ``kimi`` — the two hit different endpoints.
        """
        monkeypatch.setattr("xerxes.bridge.profiles.list_profiles", lambda: profiles_data)
        monkeypatch.setattr("xerxes.bridge.profiles.get_active_profile", lambda: profiles_data[0])

        rec = _Recorder()
        asyncio.new_event_loop().run_until_complete(daemon._handle_slash("/provider", rec))
        rid1 = next(p for (t, p) in rec.events if t == "question_request")["id"]
        self._async_question_response(daemon, rid1, {"action": "+ Add new profile…"}, rec)
        # The stage-1 meta panel (name + provider_type) is the latest emit.
        meta = [p for (t, p) in rec.events if t == "question_request"][-1]
        ptype = next(q for q in meta["questions"] if q["id"] == "provider_type")
        assert "kimi" in ptype["options"]
        assert "kimi-code" in ptype["options"]
        # Order: general "kimi" before "kimi-code".
        assert ptype["options"].index("kimi") < ptype["options"].index("kimi-code")

    def test_kimi_code_defaults_auto_fill(self, daemon, monkeypatch, profiles_data):
        """Pick provider_type=kimi-code and the credentials panel must
        suggest the right base URL. After URL+key, /models is fetched (we
        stub a failure here to exercise the free-text fallback) and the
        user can press Enter to accept `kimi-for-coding`.
        """
        saved: list[dict] = []
        monkeypatch.setattr("xerxes.bridge.profiles.list_profiles", lambda: profiles_data)
        monkeypatch.setattr("xerxes.bridge.profiles.get_active_profile", lambda: profiles_data[0])

        def _save(name, base_url, api_key, model, provider="", *_a, **_kw):
            saved.append(
                {
                    "name": name,
                    "base_url": base_url,
                    "api_key": api_key,
                    "model": model,
                    "provider": provider,
                }
            )
            return saved[-1]

        monkeypatch.setattr("xerxes.bridge.profiles.save_profile", _save)
        monkeypatch.setattr(daemon.runtime, "reload", lambda *_a, **_kw: None)
        # Simulate /models being unreachable — the panel must still let the
        # user accept the registry default by pressing Enter.

        def _boom(*_a, **_kw):
            raise RuntimeError("no /models endpoint")

        monkeypatch.setattr("xerxes.bridge.profiles.fetch_models", _boom)

        rec = _Recorder()
        asyncio.new_event_loop().run_until_complete(daemon._handle_slash("/provider", rec))
        rid = next(p for (t, p) in rec.events if t == "question_request")["id"]
        self._async_question_response(daemon, rid, {"action": "+ Add new profile…"}, rec)
        meta = [p for (t, p) in rec.events if t == "question_request"][-1]
        # Pick kimi-code; stage 2 should propose the right URL.
        self._async_question_response(
            daemon,
            meta["id"],
            {"name": "kc", "provider_type": "kimi-code"},
            rec,
        )
        creds = [p for (t, p) in rec.events if t == "question_request"][-1]
        url_q = next(q for q in creds["questions"] if q["id"] == "base_url")
        assert "https://api.kimi.com/coding/v1" in url_q["question"]

        # Stage 2 answers → stage 3 (model picker).
        self._async_question_response(
            daemon,
            creds["id"],
            {"base_url": "", "api_key": "kk-test"},
            rec,
        )
        model_panel = [p for (t, p) in rec.events if t == "question_request"][-1]
        model_q = model_panel["questions"][0]
        # /models failed → fallback question text mentions it and offers
        # free-form (no real options).
        assert "couldn't reach" in model_q["question"].lower()

        # Press Enter on stage 3 → accepts registry default.
        self._async_question_response(daemon, model_panel["id"], {"model": ""}, rec)
        assert saved == [
            {
                "name": "kc",
                "base_url": "https://api.kimi.com/coding/v1",
                "api_key": "kk-test",
                "model": "kimi-for-coding",
                "provider": "kimi-code",
            }
        ]

    def test_cancel_aborts_flow(self, daemon, monkeypatch, profiles_data):
        monkeypatch.setattr("xerxes.bridge.profiles.list_profiles", lambda: profiles_data)
        monkeypatch.setattr("xerxes.bridge.profiles.get_active_profile", lambda: profiles_data[0])

        rec = _Recorder()
        asyncio.new_event_loop().run_until_complete(daemon._handle_slash("/provider", rec))
        rid = self._events_of_type(rec, "question_request")[0]["id"]
        self._async_question_response(daemon, rid, {"action": "Cancel"}, rec)
        bodies = rec.slash_bodies()
        assert any("Cancelled" in b for b in bodies)
        assert daemon._provider_flow is None
