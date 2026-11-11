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
import type { ToolDefinition } from '../src/types/toolCalls.js'

function sessionFor(cwd: string, interactionMode: string): DaemonSession {
  const id = `permission-${interactionMode}`
  return {
    activeTurnId: '', agentId: 'default', cancelRequested: false, cwd, extra: {}, id,
    interactionMode, sessionKey: id, lastActive: 0, messages: [], metadata: {},
    model: 'test-model', planMode: interactionMode === 'plan', status: 'working', thinkingContent: [],
    toolExecutions: [], totalInputTokens: 0, totalOutputTokens: 0, turnCount: 0,
    workspace: cwd, permissionMode: 'auto',
  }
}

const execCommand: ToolDefinition = {
  type: 'function',
  function: { name: 'exec_command', description: 'Run a command.', parameters: {} },
}

const PLAN_AGENT: AgentDefinition = {
  name: 'plan', description: 'Read-only planning.', allowedTools: null, excludeTools: [],
  isolation: 'none', maxDepth: 0, model: '', source: 'test', systemPrompt: 'Plan only.',
  tools: ['exec_command'],
}
const DEFAULT_AGENT: AgentDefinition = { ...PLAN_AGENT, name: 'default', description: 'Default coding.' }

function commandClient(session: DaemonSession): LlmClient {
  return {
    async *stream(request: CompletionRequest) {
      if (request.messages.some(message => message.role === 'tool')) {
        yield { content: 'done' }
        return
      }
      // Reproduce /permissions accept-all arriving while the model round is
      // still in flight, after the turn captured its initial auto mode.
      session.permissionMode = 'accept-all'
      yield {
        toolCalls: [{
          id: 'remote-read', type: 'function',
          function: { name: 'exec_command', arguments: { cmd: 'ssh', args: ['host', 'ls'] } },
        }],
      }
    },
  }
}

async function runScenario(interactionMode: string): Promise<{ brokerCalls: number; events: DaemonEvent[]; session: DaemonSession }> {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-live-permission-'))
  try {
    const session = sessionFor(root, interactionMode)
    let brokerCalls = 0
    const runner = new AgentTurnRunner({
      llm: commandClient(session),
      model: 'test-model',
      permissionMode: 'auto',
      permissionBroker: { request: async () => { brokerCalls += 1; return 'approve' } },
      tools: [execCommand],
      agentDefinitions: new Map([['default', DEFAULT_AGENT], ['planner', PLAN_AGENT]]),
      toolExecutor: new ToolRegistry(),
    })
    const events: DaemonEvent[] = []
    for await (const event of runner.run(session, 'Inspect the remote host.', new AbortController().signal)) {
      events.push(event)
    }
    return { brokerCalls, events, session }
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

test('a live accept-all selection never widens plan mode', async () => {
  const { brokerCalls, events } = await runScenario('plan')
  expect(brokerCalls).toBe(1)
  expect(events.filter(event => event.type === 'approval_request')).toHaveLength(1)
})
