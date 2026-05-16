from __future__ import annotations

from xerxes.runtime.config_context import get_config, set_config, set_event_callback
from xerxes.tools.claude_tools import SetInteractionModeTool


def test_set_interaction_mode_tool_updates_global_config_and_emits_event() -> None:
    events: list[tuple[str, dict]] = []
    set_config({"mode": "code", "plan_mode": False})
    set_event_callback(lambda event_type, data: events.append((event_type, data)))

    result = SetInteractionModeTool.static_call("planner", reason="Need a plan")

    config = get_config()
    assert "plan" in result
    assert config["mode"] == "plan"
    assert config["plan_mode"] is True
    assert events == [
        (
            "interaction_mode_changed",
            {
                "mode": "plan",
                "plan_mode": True,
                "reason": "Need a plan",
                "source": "model",
            },
        )
    ]

    set_event_callback(None)


def test_set_interaction_mode_tool_rejects_unknown_mode() -> None:
    set_config({"mode": "code", "plan_mode": False})

    result = SetInteractionModeTool.static_call("banana")

    assert result.startswith("Error:")
    assert get_config()["mode"] == "code"
