// Licensed under the Apache License, Version 2.0.

import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { expect, test } from 'bun:test'

import type { AgentDefinition } from '../src/agents/definitions.js'
import { AgentTurnRunner } from '../src/daemon/turnRunner.js'
import type { DaemonEvent, DaemonSession } from '../src/daemon/runtime.js'
import { ToolRegistry } from '../src/executors/toolRegistry.js'
import type { CompletionRequest, LlmClient } from '../src/llms/client.js'
import type { JsonObject, ToolDefinition } from '../src/types/toolCalls.js'

function sessionFor(cwd: string, interactionMode: string, permissionMode = 'auto'): DaemonSession {
  const id = `permission-${interactionMode}`
  return {
    activeTurnId: '', agentId: 'default', cancelRequested: false, cwd, extra: {}, id,
    interactionMode, sessionKey: id, lastActive: 0, messages: [], metadata: {},
    model: 'test-model', planMode: interactionMode === 'plan', status: 'working', thinkingContent: [],
    toolExecutions: [], totalInputTokens: 0, totalOutputTokens: 0, turnCount: 0,
    workspace: cwd, permissionMode,
  }
}

const toolDefinition = (name: string): ToolDefinition => ({ type: 'function', function: { name, description: name, parameters: {} } })

const execCommand: ToolDefinition = {
  type: 'function',
  function: { name: 'exec_command', description: 'Run a command.', parameters: {} },
}

const PLAN_AGENT: AgentDefinition = {
  name: 'plan', description: 'Read-only planning.', allowedTools: null, excludeTools: [],
  isolation: 'none', maxDepth: 0, model: '', source: 'test', systemPrompt: 'Plan only.',
  tools: ['exec_command', 'SpawnAgents', 'SkillTool', 'WriteFile'],
}
const DEFAULT_AGENT: AgentDefinition = { ...PLAN_AGENT, name: 'default', description: 'Default coding.' }

const spawnAgents: ToolDefinition = {
  type: 'function',
  function: { name: 'SpawnAgents', description: 'Start managed agents.', parameters: {} },
}

type ScenarioCall = { readonly name: string; readonly arguments: JsonObject }
const remoteRead: ScenarioCall = { name: 'exec_command', arguments: { cmd: 'ssh', args: ['host', 'ls'] } }
const startResearchers: ScenarioCall = { name: 'SpawnAgents', arguments: { agents: [{ subagent_type: 'researcher', title: 'Inspect', prompt: 'Read only.' }] } }
const loadSkill: ScenarioCall = { name: 'SkillTool', arguments: { skill_name: 'systematic-debugging' } }
const writeFile: ScenarioCall = { name: 'WriteFile', arguments: { file_path: 'x.txt', content: 'x' } }

function commandClient(session: DaemonSession, call: ScenarioCall, advertisedTools: string[][], liveMode = 'accept-all'): LlmClient {
  return {
    async *stream(request: CompletionRequest) {
      advertisedTools.push(request.tools?.map(tool => tool.function.name) ?? [])
      if (request.messages.some(message => message.role === 'tool')) {
        yield { content: 'done' }
        return
      }
      // Reproduce /permissions accept-all arriving while the model round is
      // still in flight, after the turn captured its initial auto mode.
      session.permissionMode = liveMode
      yield {
        toolCalls: [{
          id: 'policy-test-tool', type: 'function',
          function: call,
        }],
      }
    },
  }
}

async function runScenario(interactionMode: string, call: ScenarioCall = remoteRead, startMode = 'auto', liveMode = 'accept-all'): Promise<{
  advertisedTools: string[][]; brokerCalls: number; events: DaemonEvent[]; session: DaemonSession
}> {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-live-permission-'))
  try {
    const session = sessionFor(root, interactionMode, startMode)
    let brokerCalls = 0
    const advertisedTools: string[][] = []
    const runner = new AgentTurnRunner({
      llm: commandClient(session, call, advertisedTools, liveMode),
      model: 'test-model',
      permissionMode: 'auto',
      permissionBroker: { request: async () => { brokerCalls += 1; return 'approve' } },
      tools: [execCommand, spawnAgents, toolDefinition('SkillTool'), toolDefinition('WriteFile')],
      agentDefinitions: new Map([['default', DEFAULT_AGENT], ['planner', PLAN_AGENT], ['researcher', { ...PLAN_AGENT, name: 'researcher', description: 'Read-only research.' }]]),
      toolExecutor: new ToolRegistry(),
    })
    const events: DaemonEvent[] = []
    for await (const event of runner.run(session, 'Inspect the remote host.', new AbortController().signal)) {
      events.push(event)
    }
    return { advertisedTools, brokerCalls, events, session }
  } finally {
    await rm(root, { recursive: true, force: true })
  }
}

test('switching a running turn from auto to accept-all approves an ordinary command immediately', async () => {
  const { brokerCalls, events, session } = await runScenario('code')
  expect(brokerCalls).toBe(0)
  expect(events.some(event => event.type === 'approval_request')).toBeFalse()
  expect(events.some(event => event.type === 'tool_call'
    && (event.payload as Record<string, unknown>).name === 'exec_command')).toBeTrue()
  // AgentTool reads this value when it spawns a child: default children now
  // inherit the main runtime's live accept-all policy too.
  expect(session.metadata.permission_mode).toBe('accept-all')
})

test('a live accept-all selection never widens plan mode: an unclassified command is refused without a card', async () => {
  const { brokerCalls, events } = await runScenario('plan')
  expect(brokerCalls).toBe(0)
  expect(events.some(event => event.type === 'approval_request' || event.type === 'tool_call')).toBeFalse()
  expect(events.filter(event => event.type === 'tool_result').map(event => event.payload)).toEqual([
    expect.objectContaining({ name: 'exec_command', permitted: false, result: expect.stringContaining('Plan first is read-only') }),
  ])
})

test('plan first with accept-all runs non-mutating tools such as SkillTool without asking', async () => {
  const { brokerCalls, events } = await runScenario('plan', loadSkill)
  expect(brokerCalls).toBe(0)
  expect(events.some(event => event.type === 'approval_request')).toBeFalse()
  expect(events.find(event => event.type === 'tool_call')?.payload).toEqual(expect.objectContaining({ name: 'SkillTool' }))
})

test('researcher mode with accept-all also runs SkillTool without asking', async () => {
  const { brokerCalls, events } = await runScenario('researcher', loadSkill)
  expect(brokerCalls).toBe(0)
  expect(events.some(event => event.type === 'approval_request')).toBeFalse()
  expect(events.find(event => event.type === 'tool_call')?.payload).toEqual(expect.objectContaining({ name: 'SkillTool' }))
})

test('plan first with accept-all refuses workspace writes without asking', async () => {
  const { brokerCalls, events } = await runScenario('plan', writeFile)
  expect(brokerCalls).toBe(0)
  expect(events.some(event => event.type === 'approval_request' || event.type === 'tool_call')).toBeFalse()
  expect(events.filter(event => event.type === 'tool_result').map(event => event.payload)).toEqual([
    expect.objectContaining({ name: 'WriteFile', permitted: false }),
  ])
})

test('plan first under auto still asks before an unclassified command', async () => {
  const { brokerCalls, events } = await runScenario('plan', remoteRead, 'auto', 'auto')
  expect(brokerCalls).toBe(1)
  expect(events.filter(event => event.type === 'approval_request').map(event => event.payload))
    .toEqual([expect.objectContaining({ permission_mode: 'plan' })])
})

test('plan first with accept-all refuses managed delegation without asking for permission', async () => {
  const { advertisedTools, brokerCalls, events } = await runScenario('plan', startResearchers)
  expect(advertisedTools[0]).toContain('SpawnAgents') // Auto when the first round started.
  expect(advertisedTools[1]).not.toContain('SpawnAgents') // Accept all was selected in flight.
  expect(advertisedTools[1]).toContain('exec_command')
  expect(brokerCalls).toBe(0)
  expect(events.some(event => event.type === 'approval_request' || event.type === 'tool_call')).toBeFalse()
  expect(events.filter(event => event.type === 'tool_result').map(event => event.payload)).toEqual([
    expect.objectContaining({ name: 'SpawnAgents', permitted: false, result: expect.stringContaining('Plan first') }),
  ])
})

test('accept-all in code mode admits the same managed delegation call without a prompt', async () => {
  const { brokerCalls, events } = await runScenario('code', startResearchers)
  expect(brokerCalls).toBe(0)
  expect(events.some(event => event.type === 'approval_request')).toBeFalse()
  expect(events.find(event => event.type === 'tool_call')?.payload).toEqual(expect.objectContaining({ name: 'SpawnAgents' }))
})
