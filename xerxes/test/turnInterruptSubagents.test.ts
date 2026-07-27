// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { type AgentDefinition } from '../src/agents/definitions.js'
import { DaemonSubagentEventBus } from '../src/daemon/subagentEvents.js'
import { createNativeSubagentHost } from '../src/daemon/subagentHost.js'
import { InMemoryDaemonRuntime, type DaemonEvent, type DaemonSession } from '../src/daemon/runtime.js'
import { AgentTurnRunner } from '../src/daemon/turnRunner.js'
import { ToolRegistry } from '../src/executors/toolRegistry.js'
import type { CompletionRequest, LlmClient, LlmDelta } from '../src/llms/client.js'
import { registerClaudeAgentTools } from '../src/tools/claudeTools/agentOps.js'

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

async function waitFor(predicate: () => boolean, timeoutMs = 2_000): Promise<void> {
  const deadline = Date.now() + timeoutMs
  while (!predicate()) {
    if (Date.now() >= deadline) throw new Error(`condition was not met within ${timeoutMs}ms`)
    await Bun.sleep(2)
  }
}

interface Harness {
  /** Resolves true when the child's provider stream observed an abort within the window. */
  readonly childAbortedWithin: (timeoutMs: number) => Promise<boolean>
  readonly childRelease: PromiseWithResolvers<void>
  readonly cleanup: () => Promise<void>
  readonly events: DaemonEvent[]
  readonly host: ReturnType<typeof createNativeSubagentHost>
  readonly parentWaiting: PromiseWithResolvers<void>
  readonly runtime: InMemoryDaemonRuntime
}

/**
 * Parent turn spawns one detached (`wait: false`) child, then parks in the
 * cohort join while the child keeps working. This is the exact shape the user
 * interrupt has to reason about: no in-flight blocking wait owns the child, so
 * only an explicit cancellation port can reach it.
 */
async function harness(sessionKey: string): Promise<Harness> {
  const sessionDirectory = await mkdtemp(join(tmpdir(), `xerxes-${sessionKey}-`))
  const childRelease = Promise.withResolvers<void>()
  const parentWaiting = Promise.withResolvers<void>()
  const childAborted = Promise.withResolvers<boolean>()
  const parentRequests: CompletionRequest[] = []
  const definitions = new Map<string, AgentDefinition>([
    ['default', creatorDefinition('researcher')],
    ['researcher', agentDefinition('researcher')],
  ])
  const client: LlmClient = {
    async *stream(request, signal): AsyncGenerator<LlmDelta> {
      const userText = request.messages
        .filter(message => message.role === 'user')
        .map(message => String(message.content))
        .join('\n')
      if (userText.includes('long running child task')) {
        const abortObserved = new Promise<boolean>(resolve => {
          if (signal?.aborted) return resolve(true)
          signal?.addEventListener('abort', () => resolve(true), { once: true })
        })
        const released = childRelease.promise.then(() => false)
        childAborted.resolve(await Promise.race([abortObserved, released]))
        if (signal?.aborted) throw signal.reason ?? new Error('child cancelled')
        yield { content: 'child final report' }

        return
      }

      parentRequests.push(request)
      if (request.messages.some(message => message.role === 'tool' && message.name === 'SpawnAgents')) {
        parentWaiting.resolve()
        yield { content: 'The delegated review is still running.' }

        return
      }
      yield {
        toolCalls: [{
          id: 'spawn-detached-child',
          type: 'function',
          function: {
            name: 'SpawnAgents',
            arguments: {
              agents: [
                {
                  name: 'child',
                  prompt: 'long running child task',
                  subagent_type: 'researcher',
                  title: 'Child review',
                },
              ],
              wait: false,
            },
          },
        }],
      }
    },
  }
  const eventBus = new DaemonSubagentEventBus()
  const registry = new ToolRegistry()
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
  // Mirrors the production cli.ts wiring.
  const runtime = new InMemoryDaemonRuntime(runner, {
    currentProjectDirectory: process.cwd(),
    model: 'test-model',
    onSessionEvict: sessionId => {
      host.cancelSource(sessionId)
    },
    onTurnCancel: sessionId => host.interruptSource(sessionId),
    sessionDirectory,
  })

  return {
    childAbortedWithin: timeoutMs => Promise.race([
      childAborted.promise,
      Bun.sleep(timeoutMs).then(() => false),
    ]),
    childRelease,
    cleanup: async () => {
      childRelease.resolve()
      await host.manager.shutdown()
      await rm(sessionDirectory, { force: true, recursive: true })
    },
    events: [],
    host,
    parentWaiting,
    runtime,
  }
}

function nestedEvent(event: DaemonEvent): { payload?: Record<string, unknown>; type?: string } {
  const nested = event.payload.event

  return typeof nested === 'object' && nested !== null ? nested as { payload?: Record<string, unknown>; type?: string } : {}
}

function runningChildId(host: Harness['host']): string {
  const task = host.manager.listTasks().find(candidate => candidate.status === 'running')
  if (!task) throw new Error('expected a running background child')

  return task.id
}

test('interrupting a turn cancels its detached background subagents', async () => {
  const kit = await harness('interrupt-detached')
  const events: DaemonEvent[] = []

  try {
    const turn = kit.runtime.submitTurn('interrupt-detached', 'delegate the review', event => events.push(event))
    await kit.parentWaiting.promise
    await waitFor(() => kit.host.manager.listTasks().some(task => task.status === 'running'))
    const childId = runningChildId(kit.host)

    expect(kit.runtime.cancelTurn('interrupt-detached')).toBe(true)
    await turn

    expect(await kit.childAbortedWithin(3_000)).toBe(true)
    expect(kit.host.manager.listTasks().find(task => task.id === childId)?.status).toBe('cancelled')
    expect(events.filter(event => event.type === 'turn_end')).toHaveLength(1)
    expect(events.at(-1)?.payload.cancelled).toBe(true)
    // The daemon, not the TUI's own turn-boundary guess, is what declares the
    // child stopped: the terminal transition must reach the parent's stream
    // while it is still subscribed.
    expect(events.filter(event => event.type === 'notification').map(event => event.payload.message))
      .toContain('Interrupt stopped 1 delegated agent.')
    const childTerminal = events.filter(event => (
      event.type === 'subagent_event'
      && event.payload.agent_id === childId
      && nestedEvent(event).type === 'turn_end'
    ))
    expect(childTerminal.length).toBeGreaterThan(0)
    expect(childTerminal.at(-1) && nestedEvent(childTerminal.at(-1)!).payload?.status).toBe('cancelled')
  } finally {
    await kit.cleanup()
  }
}, 20_000)

test('a cancelled child stays retryable rather than permanently invalidated', async () => {
  const kit = await harness('interrupt-retryable')
  const events: DaemonEvent[] = []

  try {
    const turn = kit.runtime.submitTurn('interrupt-retryable', 'delegate the review', event => events.push(event))
    await kit.parentWaiting.promise
    await waitFor(() => kit.host.manager.listTasks().some(task => task.status === 'running'))
    const childId = runningChildId(kit.host)

    kit.runtime.cancelTurn('interrupt-retryable')
    await turn
    await waitFor(() => kit.host.manager.listTasks().find(task => task.id === childId)?.status === 'cancelled')

    const handle = kit.host.managerPort.listHandles().find(candidate => candidate.id === childId)
    expect(handle).toMatchObject({ closed: false, status: 'cancelled' })
    // Interruption is a user pause, not a policy revocation: the handle must
    // still accept an explicit retry.
    kit.childRelease.resolve()
    const retried = await kit.host.retry(childId)
    expect(retried.id).toBe(childId)
  } finally {
    await kit.cleanup()
  }
}, 20_000)

test('steering a live turn never cancels its background subagents', async () => {
  const kit = await harness('steer-keeps-children')
  const events: DaemonEvent[] = []

  try {
    const turn = kit.runtime.submitTurn('steer-keeps-children', 'delegate the review', event => events.push(event))
    await kit.parentWaiting.promise
    await waitFor(() => kit.host.manager.listTasks().some(task => task.status === 'running'))
    const childId = runningChildId(kit.host)

    expect(kit.runtime.steerTurn('steer-keeps-children', 'focus on the summary')).toBe(true)
    await Bun.sleep(30)
    expect(kit.host.manager.listTasks().find(task => task.id === childId)?.status).toBe('running')

    kit.childRelease.resolve()
    await turn
    expect(kit.host.manager.listTasks().find(task => task.id === childId)?.status).toBe('completed')
  } finally {
    await kit.cleanup()
  }
}, 20_000)
