#!/usr/bin/env python3
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
"""Example demonstrating Xerxes's OpenClaw-class capabilities.

This example shows:
1. Registering a plugin tool
2. Loading skills from a directory
3. Tool policy allow/deny behavior
4. Loop detection behavior
5. Hook system (before/after tool)
6. Enriched system prompt with runtime context
7. Sandbox routing decisions

Run with:
    python examples/openclaw_capabilities_demo.py
"""

import tempfile
from pathlib import Path

from xerxes.extensions.plugins import PluginMeta, PluginRegistry, PluginType

print("=" * 60)
print("1. PLUGIN REGISTRATION")
print("=" * 60)

registry = PluginRegistry()

# Register a tool plugin
meta = PluginMeta(
    name="search_plugin",
    version="1.0.0",
    plugin_type=PluginType.TOOL,
    description="Web search tools",
)


def web_search(query: str) -> str:
    """Search the web for a query."""
    return f"Search results for: {query}"


registry.register_tool("web_search", web_search, meta=meta)
print(f"Registered plugins: {registry.plugin_names}")
print(f"Available tools: {list(registry.get_all_tools().keys())}")

from xerxes.extensions.skills import SkillRegistry  # noqa: E402

print("\n" + "=" * 60)
print("2. SKILL DISCOVERY")
print("=" * 60)

# Create a temporary skill directory
with tempfile.TemporaryDirectory() as tmpdir:
    skill_dir = Path(tmpdir) / "web_research"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        """---
name: web_research
description: Search the web and synthesize findings
version: "1.0"
tags: [research, web]
---

# Web Research Skill

When asked to research a topic:
1. Break the query into sub-questions
2. Search for each sub-question
3. Synthesize the findings into a coherent answer
"""
    )

    skill_registry = SkillRegistry()
    discovered = skill_registry.discover(tmpdir)
    print(f"Discovered skills: {discovered}")
    print(f"Skills index:\n{skill_registry.build_skills_index()}")

    skill = skill_registry.get("web_research")
    if skill:
        print(f"\nSkill prompt section:\n{skill.to_prompt_section()[:200]}...")

from xerxes.security.policy import PolicyEngine, ToolPolicy, ToolPolicyViolation  # noqa: E402

print("\n" + "=" * 60)
print("3. TOOL POLICY (Allow/Deny)")
print("=" * 60)

engine = PolicyEngine(
    global_policy=ToolPolicy(deny={"execute_shell", "delete_file"}),
    agent_policies={
        "admin_agent": ToolPolicy(allow={"execute_shell", "web_search", "read_file"}),
    },
)

# Global policy
print(f"Global: web_search -> {engine.check('web_search').value}")
print(f"Global: execute_shell -> {engine.check('execute_shell').value}")

# Agent-specific override
print(f"Admin:  execute_shell -> {engine.check('execute_shell', agent_id='admin_agent').value}")
print(f"Other:  execute_shell -> {engine.check('execute_shell', agent_id='reader_agent').value}")

# Enforce raises on deny
try:
    engine.enforce("delete_file", agent_id="reader_agent")
except ToolPolicyViolation as e:
    print(f"Policy violation: {e}")

from xerxes.runtime.loop_detection import LoopDetectionConfig, LoopDetector, LoopSeverity  # noqa: E402

print("\n" + "=" * 60)
print("4. LOOP DETECTION")
print("=" * 60)

config = LoopDetectionConfig(same_call_warning=3, same_call_critical=5)
detector = LoopDetector(config)

# Simulate repeated identical calls
for i in range(6):
    event = detector.record_call("web_search", {"query": "same query"})
    status = "OK" if event.severity == LoopSeverity.OK else event.severity.value.upper()
    print(f"  Call {i + 1}: {status}" + (f" - {event.details}" if event.details else ""))

print(f"\nTotal calls tracked: {detector.call_count}")

from xerxes.extensions.hooks import HookRunner  # noqa: E402

print("\n" + "=" * 60)
print("5. HOOK SYSTEM")
print("=" * 60)

runner = HookRunner()


# Before tool call hook — inject audit metadata
def audit_hook(tool_name, arguments, agent_id):
    print(f"  [HOOK] before_tool_call: tool={tool_name}, agent={agent_id}")
    args = arguments.copy()
    args["_audit_traced"] = True
    return args


# After tool call hook — sanitize results
def sanitize_hook(tool_name, result, agent_id):
    if isinstance(result, str) and "secret" in result.lower():
        print("  [HOOK] after_tool_call: REDACTING sensitive data")
        return "[REDACTED]"
    return result


# Bootstrap hook — inject extra context
def bootstrap_hook(agent_id):
    return f"[Bootstrap] Agent {agent_id} initialized at runtime"


runner.register("before_tool_call", audit_hook)
runner.register("after_tool_call", sanitize_hook)
runner.register("bootstrap_files", bootstrap_hook)

# Simulate tool execution with hooks
args = runner.run("before_tool_call", tool_name="search", arguments={"q": "hello"}, agent_id="a1")
print(f"  Modified args: {args}")

result = runner.run("after_tool_call", tool_name="search", result="Contains Secret Data", agent_id="a1")
print(f"  Sanitized result: {result}")

bootstrap = runner.run("bootstrap_files", agent_id="test_agent")
print(f"  Bootstrap content: {bootstrap}")

from xerxes.runtime.context import PromptContextBuilder  # noqa: E402
from xerxes.security.sandbox import SandboxConfig, SandboxMode  # noqa: E402

print("\n" + "=" * 60)
print("6. ENRICHED SYSTEM PROMPT")
print("=" * 60)

builder = PromptContextBuilder(
    hook_runner=runner,
    sandbox_config=SandboxConfig(mode=SandboxMode.WARN, sandboxed_tools={"execute_shell"}),
    guardrails=["Respect user privacy", "Do not execute destructive operations"],
)

prefix = builder.assemble_system_prompt_prefix(
    agent_id="demo_agent",
    tool_names=["web_search", "read_file", "execute_shell"],
)
print(prefix)

from xerxes.security.sandbox import SandboxRouter  # noqa: E402

print("=" * 60)
print("7. SANDBOX ROUTING")
print("=" * 60)

sandbox_config = SandboxConfig(
    mode=SandboxMode.STRICT,
    sandboxed_tools={"execute_shell", "execute_python"},
    elevated_tools={"read_file"},
)
router = SandboxRouter(sandbox_config)

for tool in ["execute_shell", "read_file", "web_search"]:
    decision = router.decide(tool)
    print(f"  {tool}: {decision.context.value} ({decision.reason})")

print("\n" + "=" * 60)
print("DEMO COMPLETE")
print("=" * 60)
print("\nAll OpenClaw-class capabilities demonstrated successfully.")
print("See docs/openclaw_parity.md for the full capability matrix.")
