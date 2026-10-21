// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { InMemoryDaemonRuntime, type DaemonEvent } from '../src/daemon/runtime.js'
import { AgentTurnRunner, type AgentTurnRunnerOptions } from '../src/daemon/turnRunner.js'
import { ToolRegistry } from '../src/executors/toolRegistry.js'
import { blockGoal, completeGoal, createGoal, getGoal } from '../src/runtime/goalDomain.js'
import type { LlmDelta } from '../src/llms/client.js'

async function scenario(options: {
  mode?: 'goals' | 'all' | 'off'; phase?: 'complete' | 'blocked'; calls?: number
  abort?: boolean; error?: boolean; subagent?: boolean; reflectionError?: boolean; alreadyComplete?: boolean
}) {
  const root = await mkdtemp(join(tmpdir(), 'reflection-turn-'))
  const events: DaemonEvent[] = []
  const seen: string[] = []
  const finished = Promise.withResolvers<void>()
  const release = Promise.withResolvers<void>()
  const controller = new AbortController()
  const tools = new ToolRegistry()
  tools.register({ type: 'function', function: { name: 'Finish', description: 'Finish work', parameters: { type: 'object', properties: {} } } }, (_input, context) => {
    if (options.phase) {
      const id = String(context.sessionId)
      const goal = getGoal(context.metadata, id)!
      if (options.phase === 'complete') completeGoal(context.metadata, id, goal, Date.now())
      else blockGoal(context.metadata, id, goal, { code: 'needs-input', message: 'Need input' }, Date.now())
    }
    if (options.abort) controller.abort()
    return { ok: true }
  })
  let round = 0
  const runnerOptions: AgentTurnRunnerOptions = {
    model: 'gpt-4o', permissionMode: 'accept-all', tools: tools.definitions(), toolExecutor: tools,
    reflectionMode: () => options.mode ?? 'goals',
    reflection: async input => {
      expect(events.some(event => event.type === 'turn_end')).toBe(true)
      expect(input.messages.some(message => message.role === 'tool')).toBe(true)
      seen.push(input.trigger)
      finished.resolve()
      if (options.reflectionError) throw new Error('reflection port failed')
      await release.promise
      events.push({ type: 'notification', payload: { message: 'Reflection proposal fake: skill_create fixture. Nothing was changed.' } })
      return { id: 'fake', summary: 'skill_create fixture' }
    },
    llm: { async *stream(): AsyncGenerator<LlmDelta> {
      if (++round === 1) yield { toolCalls: Array.from({ length: options.calls ?? 1 }, (_, index) => ({ id: `finish-${index}`, type: 'function' as const, function: { name: 'Finish', arguments: {} } })) }
      else if (options.error) throw new Error('invalid request: provider failed')
      else yield { content: 'Done' }
    } },
  }
  const runtime = new InMemoryDaemonRuntime(new AgentTurnRunner(runnerOptions), { model: 'gpt-4o', currentProjectDirectory: root, sessionDirectory: join(root, 'sessions') })
  const session = await runtime.openSession('reflection')
  if (options.phase || options.alreadyComplete) createGoal(session.metadata, session.id, { objective: 'Finish work' }, Date.now())
  if (options.alreadyComplete) completeGoal(session.metadata, session.id, getGoal(session.metadata, session.id)!, Date.now())
  if (options.subagent) session.metadata.session_kind = 'subagent'
  try {
    await runtime.submitTurn(session.sessionKey, 'Do the work', event => events.push(event), { signal: controller.signal })
    // The unresolved reflection must not keep submitTurn pending.
    if (seen.length) await finished.promise
    release.resolve()
    await new Promise<void>(resolve => setTimeout(resolve, 0))
    return { seen, events }
  } finally { release.resolve(); await runtime.shutdown(); await rm(root, { recursive: true, force: true }) }
}

for (const phase of ['complete', 'blocked'] as const) test(`reflection runs once after terminal work for goal_${phase}`, async () => {
  const { seen, events } = await scenario({ phase })
  expect(seen).toEqual([`goal_${phase}`])
  expect(events.findIndex(event => event.type === 'notification' && String(event.payload.message).startsWith('Reflection proposal'))).toBeGreaterThan(events.findIndex(event => event.type === 'turn_end'))
})

test('reflection never runs for aborted, errored, disabled or subagent turns', async () => {
  for (const options of [{ abort: true }, { error: true }, { mode: 'off' as const }, { subagent: true }]) {
    expect((await scenario({ phase: 'complete', ...options })).seen).toEqual([])
  }
})

test('an already complete goal does not request another reflection on an ordinary turn', async () => {
  expect((await scenario({ alreadyComplete: true })).seen).toEqual([])
})

test('long turns reflect only when all is enabled', async () => {
  expect((await scenario({ calls: 25 })).seen).toEqual([])
  expect((await scenario({ calls: 25, mode: 'all' })).seen).toEqual(['long_turn'])
})

test('a reflection port rejection cannot fail the completed turn', async () => {
  const { seen, events } = await scenario({ phase: 'complete', reflectionError: true })
  expect(seen).toEqual(['goal_complete'])
  expect(events.find(event => event.type === 'turn_end')?.payload.cancelled).toBe(false)
  expect(events.some(event => event.payload.level === 'error' && String(event.payload.message).includes('reflection port failed'))).toBe(false)
})
