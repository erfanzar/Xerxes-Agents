from __future__ import annotations

from xerxes.bridge import profiles
from xerxes.bridge.server import BridgeServer


def _isolated_profiles(monkeypatch, tmp_path):
    monkeypatch.setattr(profiles, "PROFILES_DIR", tmp_path)
    monkeypatch.setattr(profiles, "PROFILES_FILE", tmp_path / "profiles.json")


def test_model_switch_persists_to_active_profile(monkeypatch, tmp_path) -> None:
    _isolated_profiles(monkeypatch, tmp_path)
    profiles.save_profile(
        name="local",
        base_url="http://localhost:8000/v1",
        api_key="",
        model="old-model",
    )

    server = BridgeServer()
    server.config = {"model": "old-model", "base_url": "http://localhost:8000/v1"}

    output = server._run_slash("model", "new-model")

    assert output == "Model set to: new-model"
    assert server.config["model"] == "new-model"
    assert profiles.get_active_profile()["model"] == "new-model"


def test_model_list_auto_switches_single_available_model(monkeypatch, tmp_path) -> None:
    _isolated_profiles(monkeypatch, tmp_path)
    profiles.save_profile(
        name="local",
        base_url="http://localhost:8000/v1",
        api_key="",
        model="removed-model",
    )
    monkeypatch.setattr(profiles, "fetch_models", lambda base_url, api_key: ["qwen3_5-27.36b"])

    server = BridgeServer()
    server.config = {"model": "removed-model", "base_url": "http://localhost:8000/v1"}

    output = server._run_slash("model", "")

    assert server.config["model"] == "qwen3_5-27.36b"
    assert profiles.get_active_profile()["model"] == "qwen3_5-27.36b"
    assert "Switched from unavailable model 'removed-model'" in output
    assert "qwen3_5-27.36b (active)" in output


def test_cancel_cancels_subagents(monkeypatch) -> None:
    class Manager:
        def __init__(self) -> None:
            self.cancelled = False

        def cancel_all(self) -> int:
            self.cancelled = True
            return 1

    manager = Manager()
    monkeypatch.setattr("xerxes.tools.claude_tools._get_agent_manager", lambda: manager)

    server = BridgeServer()
    server.handle_cancel()

    assert server._cancel is True
    assert manager.cancelled is True
