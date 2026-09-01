---
name: authoring-agent-presets
description: Create or repair Xerxes agent presets in Creator mode. Use when the user asks to build, duplicate, customize, validate, or troubleshoot an agent preset.
version: 1.0.0
author: Xerxes-Agents
license: Apache-2.0
metadata:
  tags: [xerxes, creator, agent, preset, authoring]
---

# Authoring Xerxes Agent Presets

An agent preset is a directory under `${XERXES_HOME:-$HOME/.xerxes}/agents/<id>/` containing:

- `agent.yaml` — the version-1 composition loaded by the runtime;
- `preset.json` — display metadata and copy provenance;
- optional `subagents/*.yaml` files referenced by the root composition.

The directory name is the preset id. It must match `[a-z0-9][a-z0-9-]*`. Built-in presets are shipped read-only. Project presets may be inspected but are not managed as user presets.

## Required authoring loop

1. Call `CreatorRuntimeTool` with `catalog: "tools"`. Use only installed tool names.
2. Call `AgentPresetInspectTool` with `action: "list"` and select a known-good source.
3. Read that source with `AgentPresetInspectTool` action `read` when you need its exact composition.
4. Call `AgentPresetTool` action `copy`; never rewrite a built-in or project preset.
5. Read the new copy. Make the smallest complete version-1 YAML change.
6. Replace the user composition with `AgentPresetTool` action `write`. The runtime validates it before the atomic replacement.
7. Call `AgentPresetInspectTool` action `validate`.
8. Tell the user to start a new session using the new preset. Never imply that an existing session changed.

## Version-1 shape

```yaml
version: 1
agent:
  name: my-agent
  when_to_use: |
    When this preset is appropriate.
  system_prompt: |
    The agent's system instructions.
  model: optional-model-override
  isolation: shared # or worktree; omit to inherit runtime behavior
  max_depth: 3
  tools:
    - ReadFile
    - GrepTool
  allowed_tools:
    - ReadFile
    - GrepTool
  exclude_tools:
    - exec_command
  subagents:
    reviewer:
      path: ./subagents/reviewer.yaml
      description: Reviews the result.
```

`system_prompt` is required unless `system_prompt_path` resolves to a real file. `allowed_tools` is a ceiling, not a capability installer. A name absent from the live tool catalog cannot be made real by placing it in YAML.

## Trust and lifecycle

Preset authoring is privileged because it changes the schemas and system prompt shown to future model sessions. The host still owns provider routes, filesystem policy, sandboxing, approvals, plugins, and credentials; a preset cannot bypass those boundaries.

A session's preset is fixed after its first transcript message. Changing the default affects only sessions created later. Removing a user preset does not alter sessions already running on it.
