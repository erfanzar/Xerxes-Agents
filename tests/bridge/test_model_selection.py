from __future__ import annotations

from types import SimpleNamespace

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


def test_openrouter_profile_provider_is_detected(monkeypatch, tmp_path) -> None:
    _isolated_profiles(monkeypatch, tmp_path)

    profile = profiles.save_profile(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        api_key="sk-or-test",
        model="anthropic/claude-sonnet-4.5",
    )

    assert profile["provider"] == "openrouter"


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


def test_generate_skill_uses_kimi_code_headers_for_saved_profile(monkeypatch, tmp_path) -> None:
    import openai

    captured: dict = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured["request"] = kwargs
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content='---\nname: deep-scan\ndescription: Scan deeply\nversion: "1.0"\n---\n'
                        )
                    )
                ]
            )

    class FakeOpenAI:
        def __init__(self, **kwargs) -> None:
            captured["client"] = kwargs
            self.chat = SimpleNamespace(completions=FakeCompletions())

    class FakeSkillRegistry:
        def __init__(self) -> None:
            self.skill_names = {"deep-scan"}

        def discover(self, *_paths: str) -> None:
            pass

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)

    server = BridgeServer()
    server.config = {
        "model": "kimi/kimi-for-coding",
        "base_url": "https://api.kimi.com/coding/v1",
        "api_key": "test-key",
    }
    server._skills_dir = tmp_path / "skills"
    server._skill_registry = FakeSkillRegistry()
    server._emit = lambda *_args, **_kwargs: None

    output = server._generate_skill("deep-scan", "Scan deeply")

    assert "Skill 'deep-scan' generated" in output
    assert captured["client"]["default_headers"]["User-Agent"] == "claude-code/1.0.0"
    assert captured["request"]["model"] == "kimi-for-coding"
