---
name: add-tool-module
description: Add a Bun-native TypeScript tool with schema, registry wiring, policy boundaries, and Bun tests.
version: 2.0.0
tags: [tools, registry, schema, typescript, bun, xerxes]
required_tools: [ReadFile, WriteFile, FileEditTool, GlobTool]
---

# When to use

Use this skill when adding an LLM-callable Xerxes tool or a native tool module.
A tool supplies a JSON-schema-compatible `ToolDefinition` and a typed handler
registered with `ToolRegistry`.

Do not use this skill for an agent YAML definition, a channel adapter, or a
frontend command that has no tool-call contract.

# How to use

## 1. Choose a native module and inspect the contract

Create a descriptive camel-case module under `xerxes/src/tools/`, such
as `financeTools.ts`. Read:

- `xerxes/src/executors/toolRegistry.ts`
- `xerxes/src/types/toolCalls.ts`
- a small nearby module such as `clarify.ts` or `mathTools.ts`

Every tool definition uses the OpenAI-style native shape:

```ts
import { ToolRegistry } from '../executors/toolRegistry.js'
import type { JsonObject, ToolDefinition } from '../types/toolCalls.js'

export const MY_TOOL_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: 'my_tool',
    description: 'Describe the observable operation.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: { input: { type: 'string' } },
      required: ['input'],
    },
  },
}

export async function myTool(inputs: JsonObject): Promise<JsonObject> {
  const input = inputs.input
  if (typeof input !== 'string' || !input.trim()) {
    throw new TypeError('input must be a non-empty string')
  }
  return { value: input.trim() }
}

export function registerMyTools(registry: ToolRegistry): void {
  registry.register(MY_TOOL_DEFINITION, myTool)
}
```

Use `JsonObject`/`JsonValue` and validate all untrusted inputs at the boundary.
Use `AbortSignal` when the handler performs a cancellable asynchronous
operation. Do not swallow failures; let `ToolRegistry` surface a typed tool
execution error.

## 2. Register the tool at the correct surface

Export the module from `xerxes/src/tools/index.ts`. Add its registration
to `registerCoreTools()` only when it belongs to the baseline runtime surface.
Otherwise keep it opt-in behind an explicit option or host port.

Tools that access a browser, operating system, credential, remote service, or
hardware must receive an explicit native port. Do not automatically discover a
credential, start a child process, or report success when the port is absent.

## 3. Define schemas and policy implications

- Give the public tool name a stable, unique string.
- Use `additionalProperties: false` unless arbitrary extension fields are part
  of the documented protocol.
- Make destructive operations clear in the schema description and register
  them only in an explicitly gated surface.
- Keep user workspace paths inside `WorkspacePathResolver` or the equivalent
  native path-safety boundary.

## 4. Add observable Bun tests

Create or extend a focused file in `xerxes/test/`. Test schema
validation through `ToolRegistry.execute()`, the happy path, an invalid input,
and failure/cancellation behavior for external or asynchronous work.

```bash
bun test xerxes/test/<matching-tool>.test.ts
bun run --cwd xerxes check
```

Use injected ports, temporary directories, and deterministic fake fetchers;
never call a live service from the normal test suite.

## Common pitfalls

- The schema function name and registered definition name must be identical.
- `ToolRegistry` picks an agent-specific registration before the default one;
  add an agent ID only when that distinction is intentional.
- Return JSON-serializable values. Circular data fails at result serialization.
- Do not use unsafe string evaluation or shell interpolation for a convenience
  implementation.
- An optional dependency belongs behind a native port or explicit Bun package
  dependency, not an implicit runtime install.
