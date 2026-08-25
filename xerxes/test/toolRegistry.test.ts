// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  DEFAULT_TOOL_CAPABILITIES,
  ToolRegistry,
  revealedToolNames,
} from '../src/executors/toolRegistry.js'
import { SkillRegistry } from '../src/extensions/skills.js'
import { UserPromptManager } from '../src/operators/userPrompt.js'
import { registerInteractionModeTool } from '../src/runtime/interactionModeTool.js'
import {
  CLAUDE_WORKFLOW_TOOL_CAPABILITIES,
  CLAUDE_WORKFLOW_TOOL_DEFINITIONS,
  registerClaudeWorkflowTools,
} from '../src/tools/claudeTools/workflow.js'
import type { ChatMessage } from '../src/types/messages.js'
import type { JsonObject, ToolCall, ToolDefinition } from '../src/types/toolCalls.js'

function definition(name: string): ToolDefinition {
  return {
    type: 'function',
    function: { name, description: name + ' test double', parameters: { properties: {}, type: 'object' } },
  }
}

function toolMessage(content: string, isError = false): ChatMessage {
  return { role: 'tool', content, tool_call_id: 'call-1', name: 'ToolSearchTool', is_error: isError }
}

function names(definitions: readonly ToolDefinition[]): string[] {
  return definitions.map(entry => entry.function.name).sort()
}

function call(name: string, arguments_: JsonObject = {}): ToolCall {
  return { id: crypto.randomUUID(), type: 'function', function: { name, arguments: arguments_ } }
}

test('agent-specific tool variants stay isolated from other agents and anonymous callers', async () => {
  const registry = new ToolRegistry()
  registry.register(definition('shared'), () => 'default-handler')
  registry.register(definition('variant'), () => 'agent-a-handler', 'agent-a')

  // The matching agent gets its own variant.
  expect(registry.get('variant', 'agent-a')?.({}, { metadata: {} })).toBe('agent-a-handler')
  // Other agents and anonymous callers cannot see the agent-only variant at all.
  expect(registry.get('variant', 'agent-b')).toBeUndefined()
  expect(registry.get('variant')).toBeUndefined()
  expect(registry.definitions('agent-b').map(entry => entry.function.name)).toEqual(['shared'])
  expect(registry.definitions().map(entry => entry.function.name)).toEqual(['shared'])
  await expect(registry.execute(call('variant'), { agentId: 'agent-b', metadata: {} }))
    .rejects.toThrow('is not registered')

  // A default entry remains the fallback for every other agent.
  registry.register(definition('mixed'), () => 'mixed-default')
  registry.register(definition('mixed'), () => 'mixed-a', 'agent-a')
  expect(registry.get('mixed', 'agent-a')?.({}, { metadata: {} })).toBe('mixed-a')
  expect(registry.get('mixed', 'agent-b')?.({}, { metadata: {} })).toBe('mixed-default')
  expect(registry.get('mixed')?.({}, { metadata: {} })).toBe('mixed-default')
  expect(await registry.execute(call('mixed'), { agentId: 'agent-b', metadata: {} })).toBe('mixed-default')
})

test('capabilities fail closed, follow agent-first lookup, and seed the always-loaded core by name', () => {
  const registry = new ToolRegistry()
  registry.register(definition('undeclared'), () => 'x')
  registry.register(definition('ReadFile'), () => 'x')
  registry.register(definition('shell'), () => 'x', 'default', { destructive: true, openWorld: true })
  registry.register(definition('shell'), () => 'x', 'agent-a', {
    concurrencySafe: true,
    defer: false,
    destructive: false,
    readOnly: true,
  })

  expect(registry.capabilities('undeclared')).toEqual(DEFAULT_TOOL_CAPABILITIES)
  expect(registry.hasDeclaredCapabilities('undeclared')).toBe(false)
  // An unregistered name must resolve to the same fail-closed record, not throw or open up.
  expect(registry.capabilities('never-registered')).toEqual(DEFAULT_TOOL_CAPABILITIES)
  // The always-loaded seed applies without a declaration; everything else defers.
  expect(registry.capabilities('ReadFile').defer).toBe(false)
  expect(registry.capabilities('undeclared').defer).toBe(true)

  expect(registry.hasDeclaredCapabilities('shell')).toBe(true)
  expect(registry.capabilities('shell')).toMatchObject({ destructive: true, readOnly: false })
  expect(registry.capabilities('shell', 'agent-a')).toMatchObject({ destructive: false, readOnly: true })
  // Agents without their own variant fall back to the default registration's record.
  expect(registry.capabilities('shell', 'agent-b').destructive).toBe(true)

  registry.replace(definition('shell'), () => 'x', 'default', { destructive: false, readOnly: true })
  expect(registry.capabilities('shell')).toMatchObject({ destructive: false, readOnly: true })
  registry.replace(definition('shell'), () => 'x')
  expect(registry.hasDeclaredCapabilities('shell')).toBe(false)
})

test('every tool registered by the Claude workflow module declares an explicit capability record', () => {
  const registry = new ToolRegistry()
  // No host ports attached: the port-backed tools are filtered out of both the
  // registration loop and the returned definitions.
  const registered = registerClaudeWorkflowTools(registry)

  expect(names(registered)).not.toContain('AskUserQuestionTool')
  expect(names(registered)).not.toContain('SkillTool')
  expect(names(registered)).not.toContain('PlanTool')
  for (const tool of registered) {
    const name = tool.function.name
    // A new workflow tool without a record would silently inherit the fail-closed
    // defaults, including defer: true, and quietly vanish from deferred requests.
    expect(CLAUDE_WORKFLOW_TOOL_CAPABILITIES[name]).toBeDefined()
    expect(registry.hasDeclaredCapabilities(name)).toBe(true)
  }
  expect(registry.hasDeclaredCapabilities('AskUserQuestionTool')).toBe(false)
  expect(registry.capabilities('TodoWriteTool').defer).toBe(false)
  expect(registry.capabilities('ToolSearchTool')).toMatchObject({ defer: false, readOnly: true })
  expect(registry.capabilities('ExitWorktreeTool').destructive).toBe(true)
})

test('Claude workflow registration preserves a host-owned interaction-mode handler', async () => {
  const registry = new ToolRegistry()
  let mode = 'plan'
  registerInteractionModeTool(registry, {
    setMode(request) {
      mode = request.mode
      return { mode: request.mode, planMode: request.mode === 'plan' }
    },
  })

  const registered = registerClaudeWorkflowTools(registry)
  const result = JSON.parse(await registry.execute({
    id: 'mode-code',
    type: 'function',
    function: { name: 'SetInteractionModeTool', arguments: { mode: 'code' } },
  }, { metadata: {}, sessionId: 'main-session' })) as { message: string; mode: string }

  expect(registered.map(tool => tool.function.name)).not.toContain('SetInteractionModeTool')
  expect(mode).toBe('code')
  // The host-owned handler answered, not the workflow adapter's inert copy.
  // Its message states the applied mode, since the host commits immediately.
  expect(result.message).toContain('Interaction mode is now code')

  // Same guard covers the plan-mode pair, which used to reach the workflow
  // adapter and flip a WorkflowState nothing reads.
  expect(registered.map(tool => tool.function.name)).not.toContain('ExitPlanModeTool')
  expect(registered.map(tool => tool.function.name)).not.toContain('EnterPlanModeTool')
})

test('port-backed workflow tools are advertised exactly when their host port is attached', () => {
  const bare = new ToolRegistry()
  const bareRegistered = names(registerClaudeWorkflowTools(bare))
  expect(bareRegistered).not.toContain('AskUserQuestionTool')
  expect(bareRegistered).not.toContain('PlanTool')
  expect(bareRegistered).not.toContain('SkillTool')

  const ports: Parameters<typeof registerClaudeWorkflowTools>[1] = {
    planGenerator: { generate: async () => [] },
    skillRegistry: new SkillRegistry(),
    userPromptManager: new UserPromptManager(),
  }
  const full = new ToolRegistry()
  const fullRegistered = names(registerClaudeWorkflowTools(full, ports))
  expect(fullRegistered).toContain('AskUserQuestionTool')
  expect(fullRegistered).toContain('PlanTool')
  expect(fullRegistered).toContain('SkillTool')
  expect(full.get('AskUserQuestionTool')).toBeDefined()
  expect(full.get('PlanTool')).toBeDefined()
  expect(full.get('SkillTool')).toBeDefined()

  // Each port backs only its own tool.
  const partial = new ToolRegistry()
  const partialRegistered = names(registerClaudeWorkflowTools(partial, {
    planGenerator: { generate: async () => [] },
  }))
  expect(partialRegistered).toContain('PlanTool')
  expect(partialRegistered).not.toContain('AskUserQuestionTool')
  expect(partialRegistered).not.toContain('SkillTool')
})

test('deferred loading is opt-in and its live schema set is derived from the transcript', async () => {
  const eager = new ToolRegistry()
  registerClaudeWorkflowTools(eager)
  // Default registry behavior is unchanged: every schema goes out on every request.
  expect(names(eager.definitionsForTranscript([]))).toEqual(names(eager.definitions()))
  expect(eager.deferredToolLoading).toBe(false)

  const registry = new ToolRegistry({ deferredToolLoading: true })
  registerClaudeWorkflowTools(registry)
  expect(names(registry.definitionsForTranscript([]))).toEqual(['TodoWriteTool', 'ToolSearchTool'])
  expect(registry.deferredCatalog().map(entry => entry.name)).toContain('EnterWorktreeTool')
  expect(registry.deferredCatalog().every(entry => entry.description && !entry.description.includes('\n'))).toBe(true)

  const result = await registry.execute(
    { id: 'call-1', type: 'function', function: { name: 'ToolSearchTool', arguments: { query: 'worktree' } } },
    { metadata: {} },
  )
  const matches = JSON.parse(result) as Array<{ loaded: boolean; name: string; parameters?: JsonObject }>
  expect(matches.map(match => match.name).sort()).toEqual(['EnterWorktreeTool', 'ExitWorktreeTool'])
  // The full schema is what makes the tool callable; a name list would be a no-op.
  expect(matches[0]?.parameters).toEqual(
    CLAUDE_WORKFLOW_TOOL_DEFINITIONS.find(tool => tool.function.name === 'EnterWorktreeTool')?.function
      .parameters as JsonObject,
  )

  const transcript = [toolMessage(result)]
  expect(names(registry.definitionsForTranscript(transcript)))
    .toEqual(['EnterWorktreeTool', 'ExitWorktreeTool', 'TodoWriteTool', 'ToolSearchTool'])
  // Dropping the result (compaction, resume, rewind) drops the schemas with it,
  // so the request can never advertise a tool the model can no longer see.
  expect(names(registry.definitionsForTranscript([]))).toEqual(['TodoWriteTool', 'ToolSearchTool'])
})

test('only successful tool results reveal schemas', () => {
  const payload = JSON.stringify([{ loaded: true, loaded_tool: 'PlanTool', name: 'PlanTool' }])
  expect([...revealedToolNames([toolMessage(payload)])]).toEqual(['PlanTool'])
  expect([...revealedToolNames([toolMessage(payload, true)])]).toEqual([])
  // User or assistant text carrying the marker must not conjure a tool into the request.
  expect([...revealedToolNames([{ role: 'user', content: payload }])]).toEqual([])
})
