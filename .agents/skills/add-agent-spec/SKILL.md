---
name: add-agent-spec
description: Add a Bun-native YAML agent definition with inheritance, tool policy, and subagent references.
version: 2.0.0
tags: [agents, yaml, typescript, bun, orchestration, xerxes]
required_tools: [ReadFile, WriteFile, FileEditTool, GlobTool]
---

# When to use

Use this skill when creating or updating a declarative Xerxes agent definition.
Agent definitions set a role prompt, default model, tool restrictions, and
subagent references; they do not add runtime logic.

Do not use this skill to add a provider, a tool implementation, or a channel
adapter.

# How to use

## 1. Choose the right definition directory

Read `src/typescript/src/agents/definitions.ts` and
`src/typescript/src/agents/agentSpec.ts` before editing.

- Bundled definitions live in `src/typescript/src/agents/default/`.
- A user override lives in `$XERXES_HOME/agents/`.
- A project override lives in `.xerxes/agents/`.
- A project may also provide `agent.yaml` or `agents.yaml` at its root.

Later sources override an earlier definition with the same name. Use a project
directory for project-specific behavior; change the bundled directory only when
the framework itself needs a new built-in agent.

## 2. Start from the native base definition

Read `src/typescript/src/agents/default/agent.yaml`, its `system.md`, and one
specialist such as `coder.yaml` or `planner.yaml`. The loader supports YAML
specification version `1`, relative prompt paths, and `extend: default`.

Create `.xerxes/agents/reviewer.yaml`:

```yaml
version: 1
agent:
  name: reviewer
  extend: default
  when_to_use: Review a proposed code change for correctness and safety.
  system_prompt_args:
    ROLE_ADDITIONAL: >
      Review the change for correctness, security, tests, and maintainability.
      State concrete findings before suggesting edits.
  model: claude-sonnet-4-6
  allowed_tools:
    - ReadFile
    - GrepTool
    - GlobTool
  exclude_tools:
    - exec_command
```

`extend: default` resolves to the native bundled `agent.yaml`. An explicit
relative path such as `extend: ./base.yaml` resolves relative to the current
definition.

## 3. Configure tool policy deliberately

- `tools` replaces the inherited tool list when it is present.
- `allowed_tools` is an exclusive allow-list; use `null` to remove an inherited
  allow-list.
- `exclude_tools` removes named tools from the resolved list.
- Tool names must match native `ToolDefinition.function.name` values exactly.
  Inspect `src/typescript/src/tools/index.ts` and the relevant tool module
  instead of guessing a class or file name.

Use an allow-list for an agent that should be read-only or otherwise narrowly
scoped. Do not put credentials or policy bypasses in a YAML definition.

## 4. Add subagent references only when they are needed

Subagents are a mapping of name to a path, with an optional description:

```yaml
agent:
  subagents:
    researcher:
      path: ./researcher.yaml
      description: Investigates code and returns evidence without editing files.
```

The path is resolved relative to the YAML file. Keep the delegation graph small
and set `max_depth` only when the default depth of five is not appropriate.

## 5. Add an observable Bun test

Add or extend `src/typescript/test/agents.test.ts`. Test inheritance, prompt
substitution, tool restrictions, and the source-precedence behavior that your
definition changes. Use a temporary directory in the test rather than writing
to a real user agent directory.

Run:

```bash
bun test src/typescript/test/agents.test.ts
bun run --cwd src/typescript check
```

## Common pitfalls

- A duplicate name silently overrides an earlier source by design; choose a
  distinctive project agent name when that is not intended.
- `allowed_tools` and `exclude_tools` use public tool names, not TypeScript
  export names or file names.
- Missing `system_prompt` and `system_prompt_path` after inheritance resolution
  is a load error.
- Keep `ROLE_ADDITIONAL` concise. Long prompts consume context on every turn.
- Do not use a subprocess or hidden host capability to implement isolation;
  provide an explicit native integration when one is required.
