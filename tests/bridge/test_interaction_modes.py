from __future__ import annotations

import io

from xerxes.bridge import server as bridge_server
from xerxes.bridge.server import BridgeServer
from xerxes.runtime.bridge import populate_registry
from xerxes.streaming.events import TurnDone
from xerxes.tools.claude_tools import SetInteractionModeTool


def test_research_mode_is_injected_into_turn_system_prompt(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_agent_loop(**kwargs):
        captured["system_prompt"] = kwargs["system_prompt"]
        captured["mode"] = kwargs["config"]["mode"]
        captured["tools"] = [schema["name"] for schema in kwargs["tool_schemas"]]
        yield TurnDone(input_tokens=0, output_tokens=0, model="test-model")

    monkeypatch.setattr(bridge_server, "run_agent_loop", fake_run_agent_loop)

    srv = BridgeServer()
    srv._stdout = io.StringIO()
    srv._initialized = True
    srv.config = {"model": "test-model", "permission_mode": "accept-all"}
    srv.system_prompt = "base prompt"
    srv.tool_schemas = populate_registry().tool_schemas()

    srv.handle_query({"text": "inspect this", "mode": "researcher", "plan_mode": False})

    assert captured["mode"] == "researcher"
    assert "You are a research assistant focused on understanding codebases." in str(captured["system_prompt"])
    assert "You are now running as a subagent" not in str(captured["system_prompt"])
    assert set(captured["tools"]) == {"ReadFile", "GlobTool", "GrepTool", "DuckDuckGoSearch", "SetInteractionModeTool"}
    assert srv.config["plan_mode"] is False


def test_plan_mode_overrides_mode_and_uses_planner_agent_spec(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_agent_loop(**kwargs):
        captured["system_prompt"] = kwargs["system_prompt"]
        captured["mode"] = kwargs["config"]["mode"]
        captured["tools"] = [schema["name"] for schema in kwargs["tool_schemas"]]
        yield TurnDone(input_tokens=0, output_tokens=0, model="test-model")

    monkeypatch.setattr(bridge_server, "run_agent_loop", fake_run_agent_loop)

    srv = BridgeServer()
    srv._stdout = io.StringIO()
    srv._initialized = True
    srv.config = {"model": "test-model", "permission_mode": "accept-all"}
    srv.system_prompt = "base prompt"
    srv.tool_schemas = populate_registry().tool_schemas()

    srv.handle_query({"text": "make a plan", "mode": "code", "plan_mode": True})

    assert captured["mode"] == "plan"
    assert "You are an expert software architect and planner." in str(captured["system_prompt"])
    assert "You are now running as a subagent" not in str(captured["system_prompt"])
    assert set(captured["tools"]) == {"ReadFile", "GlobTool", "GrepTool", "DuckDuckGoSearch", "SetInteractionModeTool"}
    assert srv.config["plan_mode"] is True


def test_code_mode_uses_coder_agent_spec(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_agent_loop(**kwargs):
        captured["system_prompt"] = kwargs["system_prompt"]
        captured["tools"] = [schema["name"] for schema in kwargs["tool_schemas"]]
        yield TurnDone(input_tokens=0, output_tokens=0, model="test-model")

    monkeypatch.setattr(bridge_server, "run_agent_loop", fake_run_agent_loop)

    srv = BridgeServer()
    srv._stdout = io.StringIO()
    srv._initialized = True
    srv.config = {"model": "test-model", "permission_mode": "accept-all"}
    srv.system_prompt = "base prompt"
    srv.tool_schemas = populate_registry().tool_schemas()

    srv.handle_query({"text": "fix this", "mode": "code", "plan_mode": False})

    assert "You are a coding specialist focused on software engineering implementation." in str(
        captured["system_prompt"]
    )
    assert set(captured["tools"]) == {
        "ExecuteShell",
        "ReadFile",
        "WriteFile",
        "FileEditTool",
        "GlobTool",
        "GrepTool",
        "DuckDuckGoSearch",
        "SetInteractionModeTool",
    }


def test_model_tool_can_switch_interaction_mode_and_emit_status() -> None:
    srv = BridgeServer(wire_mode=True)
    srv._stdout = io.StringIO()
    srv._initialized = True
    srv.config = {"model": "test-model", "mode": "code", "plan_mode": False}
    bridge_server.set_global_config(srv.config)
    bridge_server.set_event_callback(srv._on_agent_event)

    result = SetInteractionModeTool.static_call(mode="research", reason="Need to inspect docs first")

    assert "researcher" in result
    assert srv.config["mode"] == "researcher"
    assert srv.config["plan_mode"] is False
    assert '"mode": "researcher"' in srv._stdout.getvalue()
