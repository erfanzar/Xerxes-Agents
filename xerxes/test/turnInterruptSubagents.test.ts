// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { type AgentDefinition } from '../src/agents/definitions.js'
import { DaemonSubagentEventBus } from '../src/daemon/subagentEvents.js'
import { createNativeSubagentHost } from '../src/daemon/subagentHost.js'
import { InMemoryDaemonRuntime, type DaemonEvent } from '../src/daemon/runtime.js'
import { AgentTurnRunner } from '../src/daemon/turnRunner.js'
import { ToolRegistry } from '../src/executors/toolRegistry.js'
import type { CompletionRequest, LlmClient, LlmDelta } from '../src/llms/client.js'
import { BackgroundCommandManager } from '../src/tools/backgroundCommands.js'
import { registerClaudeAgentTools } from '../src/tools/claudeTools/agentOps.js'
import { WorkspacePathResolver } from '../src/tools/pathSafety.js'
import { registerProcessTools } from '../src/tools/processTools.js'
import type { JsonObject, ToolCall } from '../src/types/toolCalls.js'

function agentDefinition(name: string): AgentDefinition {
  return {
    allowedTools: null,
    description: `${name} test agent`,
    excludeTools: [],
    isolation: '',
    maxDepth: 3,
    model: '',
    name,
    source: 'test',
    systemPrompt: `You are the ${name} test agent.`,
    tools: [],
  }
}

function creatorDefinition(...children: readonly string[]): AgentDefinition {
  return {
    ...agentDefinition('default'),
    subagents: Object.freeze(Object.fromEntries(children.map(name => [
      name,
      Object.freeze({ path: `${name}.yaml`, description: `${name} child` }),
    ]))),
  }
}

async function waitFor(predicate: () => boolean, timeoutMs = 3_000): Promise<void> {
  const deadline = Date.now() + timeoutMs
  while (!predicate()) {
    if (Date.now() >= deadline) throw new Error(`condition was not met within ${timeoutMs}ms`)
    await Bun.sleep(2)
  }
}

const CHILD_PROMPT = 'long running child task'

function toolCall(id: string, name: string, args: JsonObject): ToolCall {
  return { id, type: 'function', function: { name, arguments: args } }
}

function lastToolName(request: CompletionRequest): string | undefined {
  return request.messages.findLast(message => message.role === 'tool')?.name
}

function lastToolContent(request: CompletionRequest): string {
  const content = request.messages.findLast(message => message.role === 'tool')?.content
  return typeof content === 'string' ? content : JSON.stringify(content ?? '')
}

/** What the parent model does next, given the request it was sent. */
type ParentScript = (request: CompletionRequest) => LlmDelta

/** Spawn one child (in the background or awaited), then answer in text. */
function spawnThenAnswer(wait: boolean): ParentScript {
  return request => {
    if (lastToolName(request) === 'SpawnAgents') return { content: 'The delegated review is under way.' }
    return {
      toolCalls: [toolCall(`spawn-${crypto.randomUUID()}`, 'SpawnAgents', {
        agents: [{ name: 'child', prompt: CHILD_PROMPT, subagent_type: 'researcher', title: 'Child review' }],
        wait,
        ...(wait ? { timeout: 60 } : {}),
      })],
    }
  }
}

interface Harness {
  readonly childRelease: PromiseWithResolvers<void>
  readonly childStarted: PromiseWithResolvers<void>
  readonly cleanup: () => Promise<void>
  readonly host: ReturnType<typeof createNativeSubagentHost>
  readonly parentRequests: CompletionRequest[]
  /** Resolves with each parent request as it is made. */
  readonly onParentRequest: (predicate: (request: CompletionRequest) => boolean) => Promise<CompletionRequest>
  readonly runtime: InMemoryDaemonRuntime
  readonly background: BackgroundCommandManager
}

/**
 * The production wiring (cli.ts) in miniature: a native subagent host, the
 * Claude agent tools and the process tools, behind the daemon runtime.
 */
async function harness(sessionKey: string, script: ParentScript): Promise<Harness> {
  const sessionDirectory = await mkdtemp(join(tmpdir(), `xerxes-${sessionKey}-`))
  const childRelease = Promise.withResolvers<void>()
  const childStarted = Promise.withResolvers<void>()
  const parentRequests: CompletionRequest[] = []
  const watchers: Array<{ readonly predicate: (request: CompletionRequest) => boolean; readonly resolve: (request: CompletionRequest) => void }> = []
  const definitions = new Map<string, AgentDefinition>([
    ['default', creatorDefinition('researcher')],
    ['researcher', agentDefinition('researcher')],
  ])
  const client: LlmClient = {
    async *stream(request, signal): AsyncGenerator<LlmDelta> {
      const first = request.messages.find(message => message.role === 'user')?.content
      if (typeof first === 'string' && first.includes(CHILD_PROMPT)) {
        childStarted.resolve()
        await new Promise<void>((resolve, reject) => {
          if (signal?.aborted) return reject(signal.reason)
          signal?.addEventListener('abort', () => reject(signal.reason ?? new Error('child cancelled')), { once: true })
          void childRelease.promise.then(resolve)
        })
        yield { content: 'child final report' }
        return
      }
      parentRequests.push(request)
      for (const watcher of [...watchers]) {
        if (!watcher.predicate(request)) continue
        watchers.splice(watchers.indexOf(watcher), 1)
        watcher.resolve(request)
      }
      yield script(request)
    },
  }
  const eventBus = new DaemonSubagentEventBus()
  const registry = new ToolRegistry()
  const background = new BackgroundCommandManager()
  registerProcessTools(registry, new WorkspacePathResolver(sessionDirectory), background)
  const host = createNativeSubagentHost({
    agentDefinitions: definitions,
    cwd: process.cwd(),
    eventBus,
    llm: client,
    model: 'test-model',
    permissionMode: 'accept-all',
    toolExecutor: registry,
    tools: [],
  })
  registerClaudeAgentTools(registry, {
    backgroundAgents: host.turnCoordinator,
    manager: host.managerPort,
  })
  const runner = new AgentTurnRunner({
    agentDefinitions: definitions,
    llm: client,
    model: 'test-model',
    permissionMode: 'accept-all',
    subagentCoordinator: host.turnCoordinator,
    subagentEvents: eventBus,
    toolExecutor: registry,
    tools: registry.definitions(),
  })
  const runtime = new InMemoryDaemonRuntime(runner, {
    currentProjectDirectory: process.cwd(),
    model: 'test-model',
    onSessionEvict: sessionId => {
      host.cancelSource(sessionId)
    },
    runningSubagents: sessionId => host.managerPort.listHandles()
      .filter(snapshot => snapshot.sourceAgentId === sessionId && snapshot.status === 'running').length,
    shutdown: () => host.manager.shutdown(),
    sessionDirectory,
  })

  return {
    background,
    childRelease,
    childStarted,
    cleanup: async () => {
      childRelease.resolve()
      background.disposeAll()
      await host.manager.shutdown()
      await rm(sessionDirectory, { force: true, recursive: true })
    },
    host,
    onParentRequest: predicate => new Promise(resolve => {
      const seen = parentRequests.find(predicate)
      if (seen) resolve(seen)
      else watchers.push({ predicate, resolve })
    }),
    parentRequests,
    runtime,
  }
}

function childStatus(kit: Harness): string | undefined {
  return kit.host.managerPort.listHandles()[0]?.status
}

function notices(events: readonly DaemonEvent[]): unknown[] {
  return events.filter(event => event.type === 'notification').map(event => event.payload.message)
}

function mentions(request: CompletionRequest, text: string): boolean {
  return JSON.stringify(request.messages).includes(text)
}

test('interrupting a turn leaves its background agents running and their results reach the next turn', async () => {
  const kit = await harness('interrupt-detached', spawnThenAnswer(false))
  const events: DaemonEvent[] = []

  try {
    const turn = kit.runtime.submitTurn('interrupt-detached', 'delegate the review', event => events.push(event))
    await kit.onParentRequest(request => lastToolName(request) === 'SpawnAgents')
    await kit.childStarted.promise

    expect(kit.runtime.cancelTurn('interrupt-detached')).toBe(true)
    await turn
    expect(events.at(-1)).toMatchObject({ type: 'turn_end', payload: { cancelled: true } })
    expect(notices(events)).toContain('Stopped the turn. 1 agent is still running.')
    expect(notices(events).some(message => String(message).startsWith('Interrupt stopped'))).toBe(false)
    await Bun.sleep(50)
    expect(childStatus(kit)).toBe('running')

    kit.childRelease.resolve()
    await waitFor(() => childStatus(kit) === 'completed')
    const next: DaemonEvent[] = []
    await kit.runtime.submitTurn('interrupt-detached', 'what did it find?', event => next.push(event))
    expect(next.at(-1)).toMatchObject({ type: 'turn_end', payload: { cancelled: false } })
    expect(kit.parentRequests.at(-1) && mentions(kit.parentRequests.at(-1)!, 'child final report')).toBe(true)
  } finally {
    await kit.cleanup()
  }
}, 20_000)

test('an agent awaited inside the interrupted tool call keeps running as a background agent', async () => {
  const kit = await harness('interrupt-awaited', spawnThenAnswer(true))
  const events: DaemonEvent[] = []

  try {
    const turn = kit.runtime.submitTurn('interrupt-awaited', 'delegate the review', event => events.push(event))
    await kit.childStarted.promise
    await Bun.sleep(30)

    kit.runtime.cancelTurn('interrupt-awaited')
    await turn
    await Bun.sleep(50)
    expect(kit.host.managerPort.listHandles()[0]).toMatchObject({ status: 'running', closed: false })

    kit.childRelease.resolve()
    await waitFor(() => childStatus(kit) === 'completed')
    await kit.runtime.submitTurn('interrupt-awaited', 'what did it find?', () => {})
    expect(kit.parentRequests.at(-1) && mentions(kit.parentRequests.at(-1)!, 'child final report')).toBe(true)
  } finally {
    await kit.cleanup()
  }
}, 20_000)

test('closing the session still stops its agents', async () => {
  const kit = await harness('evict-stops', spawnThenAnswer(false))

  try {
    const turn = kit.runtime.submitTurn('evict-stops', 'delegate the review', () => {})
    await kit.childStarted.promise
    kit.runtime.cancelTurn('evict-stops')
    await turn
    expect(childStatus(kit)).toBe('running')

    kit.runtime.evictSession('evict-stops')
    await waitFor(() => kit.host.manager.listTasks()[0]?.status === 'cancelled')
  } finally {
    await kit.cleanup()
  }
}, 20_000)

test('stopping the daemon still stops its agents', async () => {
  const kit = await harness('shutdown-stops', spawnThenAnswer(false))

  try {
    const turn = kit.runtime.submitTurn('shutdown-stops', 'delegate the review', () => {})
    await kit.childStarted.promise
    kit.runtime.cancelTurn('shutdown-stops')
    await turn
    expect(childStatus(kit)).toBe('running')

    await kit.runtime.shutdown()
    expect(kit.host.manager.listTasks()[0]?.status).toBe('cancelled')
  } finally {
    await kit.cleanup()
  }
}, 20_000)

test('a steer sent while AwaitAgents blocks reaches the next model round of the same turn', async () => {
  const kit = await harness('steer-await', request => {
    const last = lastToolName(request)
    if (last === 'SpawnAgents') return { toolCalls: [toolCall('await-1', 'AwaitAgents', { timeout_seconds: 60 })] }
    if (last === 'AwaitAgents') return { content: 'Noted, focusing on the summary.' }
    return spawnThenAnswer(false)(request)
  })
  const events: DaemonEvent[] = []

  try {
    const turn = kit.runtime.submitTurn('steer-await', 'delegate the review', event => events.push(event))
    await kit.onParentRequest(request => lastToolName(request) === 'SpawnAgents')
    await Bun.sleep(100)
    const steeredAt = Date.now()
    expect(kit.runtime.steerTurn('steer-await', 'focus on the summary')).toBe(true)

    const after = await kit.onParentRequest(request => lastToolName(request) === 'AwaitAgents')
    // Woke on the steer, well before the 60 s timeout, and the agent kept going.
    expect(Date.now() - steeredAt).toBeLessThan(2_000)
    expect(lastToolContent(after)).toMatch(/"wake_reason":\s*"steer"/)
    expect(after.messages).toContainEqual({ role: 'user', content: '[steer from user]\nfocus on the summary' })
    expect(childStatus(kit)).toBe('running')
    expect(events.some(event => event.type === 'steer_applied')).toBe(true)
    expect(events.some(event => event.type === 'turn_end')).toBe(false)

    kit.childRelease.resolve()
    await turn
    expect(notices(events).some(message => String(message).includes('for the next turn'))).toBe(false)
  } finally {
    await kit.cleanup()
  }
}, 20_000)

test('a steer sent while check_command waits reaches the next model round and the command keeps running', async () => {
  const kit = await harness('steer-check', request => {
    const last = lastToolName(request)
    if (last === 'exec_command') {
      const procId = /"procId":\s*"([^"]+)"/.exec(lastToolContent(request))?.[1] ?? 'missing'
      return { toolCalls: [toolCall('check-1', 'check_command', { proc_id: procId, wait_ms: 60_000 })] }
    }
    if (last === 'check_command') return { content: 'Adjusted to the new instruction.' }
    return { toolCalls: [toolCall('exec-1', 'exec_command', { cmd: 'sleep', args: ['30'], run_in_background: true })] }
  })
  const events: DaemonEvent[] = []

  try {
    const turn = kit.runtime.submitTurn('steer-check', 'start the server', event => events.push(event))
    await kit.onParentRequest(request => lastToolName(request) === 'exec_command')
    await Bun.sleep(100)
    const steeredAt = Date.now()
    expect(kit.runtime.steerTurn('steer-check', 'use port 8080 instead')).toBe(true)

    const after = await kit.onParentRequest(request => lastToolName(request) === 'check_command')
    expect(Date.now() - steeredAt).toBeLessThan(2_000)
    expect(lastToolContent(after)).toMatch(/"running":\s*true/)
    expect(after.messages).toContainEqual({ role: 'user', content: '[steer from user]\nuse port 8080 instead' })
    await turn
    expect(events.at(-1)).toMatchObject({ type: 'turn_end', payload: { cancelled: false } })
    expect(notices(events).some(message => String(message).includes('for the next turn'))).toBe(false)
  } finally {
    await kit.cleanup()
  }
}, 20_000)

test('a steer sent while the turn waits on background agents runs one more round in that turn', async () => {
  const kit = await harness('steer-cohort', spawnThenAnswer(false))
  const events: DaemonEvent[] = []

  try {
    const turn = kit.runtime.submitTurn('steer-cohort', 'delegate the review', event => events.push(event))
    await kit.onParentRequest(request => lastToolName(request) === 'SpawnAgents')
    await Bun.sleep(100)
    const before = kit.parentRequests.length
    expect(kit.runtime.steerTurn('steer-cohort', 'also check the tests')).toBe(true)

    await waitFor(() => kit.parentRequests.length > before)
    expect(kit.parentRequests.at(-1)?.messages).toContainEqual({ role: 'user', content: '[steer from user]\nalso check the tests' })
    expect(events.some(event => event.type === 'turn_end')).toBe(false)

    kit.childRelease.resolve()
    await turn
    expect(notices(events).some(message => String(message).includes('for the next turn'))).toBe(false)
  } finally {
    await kit.cleanup()
  }
}, 20_000)
