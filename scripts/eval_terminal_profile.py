#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import re
from dataclasses import asdict

from calute import Calute, OperatorRuntimeConfig, RuntimeFeaturesConfig
from calute.llms import create_llm
from calute.runtime.profiles import PromptProfile
from calute.tui.cli import TerminalConfigStore, build_default_agent, reconcile_terminal_profile
from calute.types import Completion, FunctionExecutionStart

DEFAULT_PROFILE_PATH = "/Users/erfan/.config/calute/terminal_profiles.json"
MARKUP_RE = re.compile(r"<\s*(tool_call|response)\b|<function=|<parameter=", re.IGNORECASE)

DEFAULT_SUITE = [
    {"name": "greeting", "prompt": "hi", "expect_tool": False},
    {"name": "simple_math", "prompt": "What is 2+2? Answer in one word.", "expect_tool": False},
    {"name": "capabilities", "prompt": "What can you help with?", "expect_tool": False},
    {
        "name": "explain_python",
        "prompt": "Explain what a Python list comprehension is in two sentences.",
        "expect_tool": False,
    },
    {"name": "write_code", "prompt": "Write a Python function that adds two numbers.", "expect_tool": False},
    {"name": "current_time", "prompt": "What time is it right now in UTC+3?", "expect_tool": True},
    {"name": "weather", "prompt": "What's the weather in Istanbul right now?", "expect_tool": True},
    {"name": "list_dir", "prompt": "List the files in the current working directory.", "expect_tool": True},
    {
        "name": "project_name",
        "prompt": "Read pyproject.toml from the repo root and tell me the project name.",
        "expect_tool": True,
    },
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a saved Calute terminal profile against a small prompt suite.")
    parser.add_argument("--profile-name", default="default")
    parser.add_argument("--profile-path", default=DEFAULT_PROFILE_PATH)
    parser.add_argument("--timeout-seconds", type=float, default=45.0)
    parser.add_argument("--json", action="store_true", help="Emit compact JSON instead of a human summary.")
    return parser


async def _build_runtime(profile_name: str, profile_path: str):
    store = TerminalConfigStore(profile_path)
    profile = store.get_profile(profile_name)
    if profile is None:
        raise SystemExit(f"profile {profile_name!r} not found in {profile_path}")
    profile = reconcile_terminal_profile(profile)

    llm_kwargs: dict[str, str] = {}
    if profile.model:
        llm_kwargs["model"] = profile.model
    if profile.api_key:
        llm_kwargs["api_key"] = profile.api_key
    if profile.base_url:
        llm_kwargs["base_url"] = profile.base_url

    llm = create_llm(profile.provider, **llm_kwargs)
    calute = Calute(
        llm=llm,
        runtime_features=RuntimeFeaturesConfig(
            enabled=True,
            default_prompt_profile=PromptProfile(profile.prompt_profile),
            operator=OperatorRuntimeConfig(
                enabled=True,
                power_tools_enabled=profile.power_tools_enabled,
            ),
        ),
    )
    agent = build_default_agent(
        model=profile.model,
        agent_id=profile.agent_id,
        instructions=profile.instructions,
        include_tools=profile.include_tools,
        sampling_params=profile.sampling_params,
    )
    calute.register_agent(agent)
    return calute, agent, profile


async def _run_prompt(calute: Calute, agent, prompt: str, timeout_seconds: float) -> dict[str, object]:
    tool_names: list[str] = []
    final = {
        "content": "",
        "reasoning": "",
        "function_calls_executed": 0,
        "agent_id": None,
        "timed_out": False,
        "error": None,
    }

    async def _consume() -> None:
        nonlocal final
        stream = await calute.create_response(prompt=prompt, agent_id=agent, stream=True)
        async for item in stream:
            if isinstance(item, FunctionExecutionStart):
                tool_names.append(item.function_name)
            elif isinstance(item, Completion):
                final = {
                    "content": item.final_content or "",
                    "reasoning": item.reasoning_content or "",
                    "function_calls_executed": item.function_calls_executed,
                    "agent_id": item.agent_id,
                    "timed_out": False,
                    "error": None,
                }

    try:
        await asyncio.wait_for(_consume(), timeout=timeout_seconds)
    except TimeoutError:
        final["timed_out"] = True
    except Exception as exc:  # pragma: no cover - live endpoint/runtime behavior
        final["error"] = str(exc)

    final["tool_names"] = tool_names
    return final


def _score_case(case: dict[str, object], outcome: dict[str, object]) -> dict[str, object]:
    name = str(case["name"])
    expect_tool = bool(case["expect_tool"])
    content = str(outcome["content"])
    reasoning = str(outcome["reasoning"])
    tool_names = [str(tool) for tool in outcome["tool_names"]]
    used_tool = bool(tool_names)
    failures: list[str] = []

    if bool(outcome.get("timed_out")):
        failures.append("timeout")
    if outcome.get("error"):
        failures.append("runtime_error")
    if expect_tool and not used_tool:
        failures.append("missed_required_tool")
    if not expect_tool and used_tool:
        failures.append("unnecessary_tool")
    if MARKUP_RE.search(content):
        failures.append("markup_leak")
    if MARKUP_RE.search(reasoning):
        failures.append("reasoning_markup_leak")
    if not expect_tool and len(reasoning.strip()) > 120:
        failures.append("excessive_reasoning_simple")
    if name == "simple_math" and content.strip().lower() != "4":
        failures.append("wrong_direct_answer_shape")
    if name == "capabilities" and not content.strip():
        failures.append("empty_capabilities_answer")

    return {
        "name": name,
        "prompt": case["prompt"],
        "used_tool": used_tool,
        "tool_names": tool_names,
        "content": content,
        "reasoning_preview": reasoning[:240],
        "timed_out": bool(outcome.get("timed_out")),
        "error": outcome.get("error"),
        "has_markup": bool(MARKUP_RE.search(content)),
        "has_reasoning_markup": bool(MARKUP_RE.search(reasoning)),
        "has_reasoning": bool(reasoning.strip()),
        "failures": failures,
    }


async def _evaluate(profile_name: str, profile_path: str, timeout_seconds: float) -> dict[str, object]:
    calute, agent, profile = await _build_runtime(profile_name, profile_path)
    results: list[dict[str, object]] = []
    failure_count = 0

    for case in DEFAULT_SUITE:
        outcome = await _run_prompt(calute, agent, str(case["prompt"]), timeout_seconds)
        scored = _score_case(case, outcome)
        if scored["failures"]:
            failure_count += 1
        results.append(scored)

    return {
        "profile": {
            "name": profile.name,
            "provider": profile.provider,
            "model": profile.model,
            "base_url": profile.base_url,
            "prompt_profile": profile.prompt_profile,
            "agent_id": profile.agent_id,
            "include_tools": profile.include_tools,
            "power_tools_enabled": profile.power_tools_enabled,
            "instructions": profile.instructions,
            "sampling_params": dict(profile.sampling_params),
        },
        "suite_size": len(DEFAULT_SUITE),
        "failure_count": failure_count,
        "results": results,
    }


def _print_human_summary(report: dict[str, object]) -> None:
    profile = report["profile"]
    print(
        f"profile={profile['name']} provider={profile['provider']} model={profile['model']} "
        f"prompt_profile={profile['prompt_profile']} failures={report['failure_count']}/{report['suite_size']}"
    )
    for result in report["results"]:
        failures = result["failures"]
        status = "FAIL" if failures else "PASS"
        print(f"[{status}] {result['name']}")
        print(f"  tools: {', '.join(result['tool_names']) if result['tool_names'] else 'none'}")
        print(f"  content: {result['content'][:200]}")
        if failures:
            print(f"  failures: {', '.join(failures)}")


def main() -> int:
    args = build_parser().parse_args()
    report = asyncio.run(_evaluate(args.profile_name, args.profile_path, args.timeout_seconds))
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _print_human_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
