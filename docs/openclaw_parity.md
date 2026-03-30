# OpenClaw Capability Parity — Design Document

> **Date**: 2026-03-28
> **Author**: Calute maintainers
> **Status**: Production-capable runtime implemented

## Overview

This document maps OpenClaw platform capabilities to their Calute-native equivalents. The goal is to bring OpenClaw-class runtime features to Calute while preserving its Python-first architecture and existing APIs.

## Source Docs Reviewed

The following OpenClaw concept areas were analyzed for parity:

- Agent loop architecture (structured execution, lifecycle events)
- System prompt assembly (tooling summary, guardrails, skills, runtime context)
- Tooling model (typed functions, allow/deny policy, optional tools, reinvocation)
- Plugin architecture (tool/hook/provider plugins, local discovery)
- Skills system (SKILL.md, metadata, indexing, prompt injection)
- Sandboxing (modes, elevated execution, host vs sandbox routing)
- Tool-loop detection (repetition, ping-pong, iteration caps)
- Hooks and automation (lifecycle hooks, mutation hooks, bootstrap injection)
- Session/workspace isolation
- Enterprise guardrails (audit events, policy enforcement)

## Capability Matrix

| OpenClaw Concept | Calute Module | Status | Notes |
|---|---|---|---|
| **Tool allow/deny policy** | `calute.security.policy` + `calute.runtime.features` | **Integrated** | Global + per-agent overrides enforced at live execution seam |
| **Tool-loop detection** | `calute.runtime.loop_detection` + `calute.calute.Calute` | **Integrated** | Same-call, ping-pong, max-calls, per-turn detector reused across reinvocation |
| **SKILL.md discovery** | `calute.extensions.skills` + `calute.runtime.features` | **Integrated** | YAML frontmatter, directory scanning, indexing, prompt injection |
| **Plugin registry** | `calute.extensions.plugins` + `calute.runtime.features` | **Integrated** | Tool/hook/provider plugins, local discovery, conflict detection |
| **Lifecycle hooks** | `calute.extensions.hooks` + `calute.calute.Calute` | **Integrated** | 7 hook points wired into live turn/tool lifecycle |
| **Sandbox config** | `calute.security.sandbox` + `calute.executors` | **Integrated** | Config model, routing layer, elevated escape hatch, honest strict-mode failure |
| **Runtime prompt enrichment** | `calute.runtime.context` + `calute.calute.Calute.manage_messages` | **Integrated** | Runtime/workspace/sandbox/skills/tools/guardrails/bootstrap sections |
| Structured agent loop | `calute.calute.Calute` | **Enhanced** | Streaming, retry, reinvocation, loop-aware reinvocation, lifecycle hooks |
| System prompt assembly | `calute.calute.Calute.manage_messages` | **Enhanced** | Runtime context, bootstrap content, skill index, enabled skill instructions |
| Typed tool schemas | `calute.utils.function_to_json` | **Existing** | JSON schema generation from Python functions |
| Tool result post-processing | `calute.extensions.hooks` (`after_tool_call`) | **Implemented** | Via hook system |
| Tool result persistence transform | `calute.extensions.hooks` (`tool_result_persist`) | **Implemented** | Via hook system |
| Bootstrap file injection | `calute.extensions.hooks` (`bootstrap_files`) | **Implemented** | Via hook system |
| Agent switching | `calute.executors.AgentOrchestrator` | **Existing** | Capability-based, error recovery triggers |
| MCP integration | `calute.mcp` | **Existing** | Full MCP client/manager/integration |
| Container sandbox execution | `calute.security.sandbox` + `calute.security.sandbox_backends` | **Implemented** | Docker and subprocess backends; strict mode enforced; backend auto-instantiation from config |
| Structured audit event export | `calute.audit` | **Implemented** | 12 typed events, in-memory/JSONL/composite collectors, emitter wired into executor and turn lifecycle |
| Session persistence & replay | `calute.session` | **Implemented** | Turn/tool-call/agent-transition recording; in-memory and file stores; replay with timeline and filtering |
| Plugin dependency/version system | `calute.extensions.plugins` + `calute.extensions.dependency` | **Implemented** | Version constraints, dependency resolution, topological sort, conflict detection |
| Skill dependency validation | `calute.extensions.skills` + `calute.extensions.dependency` | **Implemented** | Skill-to-skill and skill-to-tool dependency validation |
| Compact sub-agent prompt mode | `calute.runtime.profiles` + `calute.runtime.context` | **Implemented** | FULL/COMPACT/MINIMAL profiles; per-agent override; safety sections preserved |
| Channel/group semantics | — | **Deferred** | Future plugin type defined but not implemented |

## Design Rationale

### Python-Native Adaptation

1. **Typed config models**: All configuration uses Python dataclasses/Pydantic rather than YAML-first config. This enables IDE completion and runtime validation.

2. **Registry-based plugins**: Instead of a file-convention plugin system, Calute uses explicit `register(registry)` functions. This is more Pythonic and avoids magic import scanning.

3. **Hook chain semantics**: Mutation hooks (before/after tool) chain return values; observation hooks collect results. Errors in hooks are logged but never break the execution chain.

4. **Policy vs Sandbox separation**: Tool policy (allow/deny) is distinct from sandbox routing (host/sandbox/elevated). A tool can be allowed by policy but routed to sandbox for execution safety.

5. **Skills as markdown**: SKILL.md files with YAML frontmatter are a natural format for Python projects. No compilation step needed; skills are parsed at discovery time.

### Backwards Compatibility

The new runtime is opt-in through `RuntimeFeaturesConfig`. Existing `Calute`, `Agent`, `FunctionExecutor`, and `AgentOrchestrator` behavior remains unchanged when runtime features are disabled.

When enabled, Calute now:

- discovers plugins and skills from configured directories and conventional local paths
- merges plugin tools at agent registration time with conflict detection
- builds per-turn loop detectors and reuses them across reinvocation cycles
- enforces policy, hooks, and sandbox routing at the single tool execution seam
- enriches prompt assembly with runtime/workspace/sandbox/skills/guardrails/bootstrap context

### Runtime Integration Summary

The runtime feature layer is now wired into the live `Calute.create_response()` / `Calute.run()` flow:

1. `RuntimeFeaturesState` is created during `Calute` initialization when features are enabled.
2. `manage_messages()` prepends the enriched runtime context block.
3. `create_response()` creates one loop detector per turn and threads it through reinvocation.
4. `FunctionExecutor._execute_single_call()` enforces loop detection, policy, hooks, and sandbox routing.
5. Tool persistence goes through `tool_result_persist` before reinvocation messages are built.
6. Turn lifecycle hooks fire for start, end, and error events.

## What Remains for Complete Parity

### Completed (this cycle)
- Real sandbox backend execution (Docker + subprocess backends)
- Structured audit event export (12 typed events, JSONL + in-memory collectors)
- Session persistence and replay (file + in-memory stores, replay view)
- Plugin dependency/version resolution (version constraints, topological sort)
- Skill dependency validation (skill-to-skill, skill-to-tool)
- Compact sub-agent prompt mode (FULL/COMPACT/MINIMAL profiles)

### Remaining
- Channel plugin type with basic implementations
- Multi-workspace isolation (workspace identity layer exists but not fully integrated)
- Enterprise SSO/RBAC integration
- OpenTelemetry trace export

## File References

| File | Description |
|---|---|
| `calute/policy.py` | Tool policy engine with allow/deny/optional |
| `calute/loop_detection.py` | Loop detection with same-call, ping-pong, max-calls |
| `calute/skills.py` | SKILL.md parsing, discovery, indexing |
| `calute/plugins.py` | Plugin registry for tools/hooks/providers |
| `calute/hooks.py` | Lifecycle hook runner with mutation/observation semantics |
| `calute/sandbox.py` | Sandbox config, routing, and execution abstraction |
| `calute/runtime_context.py` | Enriched system prompt context builder |
| `calute/runtime_features.py` | Opt-in runtime config and integrated runtime state |
| `tests/test_policy.py` | Policy tests (14 tests) |
| `tests/test_loop_detection.py` | Loop detection tests (12 tests) |
| `tests/test_skills.py` | Skills tests (14 tests) |
| `tests/test_plugins.py` | Plugin tests (14 tests) |
| `tests/test_hooks.py` | Hooks tests (14 tests) |
| `tests/test_sandbox.py` | Sandbox tests (34 tests) |
| `tests/test_sandbox_backends.py` | Sandbox backend tests (16 tests) |
| `tests/test_runtime_context.py` | Runtime context tests (18 tests) |
| `tests/test_prompt_profiles.py` | Prompt profile tests (30 tests) |
| `tests/test_audit.py` | Audit event/collector/emitter tests (50 tests) |
| `tests/test_audit_events.py` | Audit event dataclass tests (13 tests) |
| `tests/test_session.py` | Session persistence tests (38 tests) |
| `tests/test_session_replay.py` | Session replay tests (23 tests) |
| `tests/test_dependency.py` | Dependency resolution tests (30 tests) |
| `tests/test_runtime_integration.py` | End-to-end runtime integration tests (12 tests) |
| `calute/audit/` | Structured audit event system (events, collectors, emitter) |
| `calute/session/` | Session persistence, replay, and workspace management |
| `calute/sandbox_backends/` | Docker and subprocess sandbox backends |
| `calute/dependency.py` | Version constraint and dependency resolution |
| `calute/prompt_profiles.py` | Prompt verbosity profiles for sub-agents |
| `examples/openclaw_capabilities_demo.py` | Example demonstrating all new capabilities |
| `docs/openclaw_parity.md` | This document |
