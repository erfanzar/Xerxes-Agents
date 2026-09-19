// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { InMemoryDaemonRuntime } from '../src/daemon/runtime.js'
import { AgentTurnRunner } from '../src/daemon/turnRunner.js'
import type { CompletionRequest, LlmClient } from '../src/llms/client.js'

for (const boundary of ['cancel', 'failure'] as const) test(`native ${boundary} preserves partial output across disk resume and the next provider request`, async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xr-partial-'))
  const requests: CompletionRequest[] = []
  const provider: LlmClient = { async *stream(request, signal) {
    requests.push(structuredClone(request))
    if (requests.length > 1) { yield { content: 'Continued reply.' }; return }
    yield { thinking: 'Reasoning already received. ' }
    yield { content: 'function partial() {\n  return ' }
    if (boundary === 'failure') throw new Error('401 private provider diagnostic')
    if (!signal?.aborted) await new Promise<void>(resolve => signal?.addEventListener('abort', () => resolve(), { once: true }))
    throw new DOMException('Aborted', 'AbortError')
  } }
  const makeRuntime = () => new InMemoryDaemonRuntime(new AgentTurnRunner({ model: 'gpt-4o', llm: provider, tools: [] }), {
    model: 'gpt-4o', currentProjectDirectory: directory, sessionDirectory: join(directory, 'sessions'),
  })
  const runtime = makeRuntime()
  let resumedRuntime: InMemoryDaemonRuntime | undefined
  try {
    const session = await runtime.openSession('partial')
    await runtime.submitTurn(session.sessionKey, 'Start the function.', event => {
      if (boundary === 'cancel' && event.type === 'text_part') runtime.cancelTurn(session.sessionKey)
    })
    await runtime.flushSessions()
    const saved = await Bun.file(join(directory, 'sessions', session.id + '.json')).json()
    expect(saved.messages.filter((message: { role: string }) => message.role === 'assistant')).toEqual([
      { role: 'assistant', content: 'function partial() {\n  return ', thinking: 'Reasoning already received. ',
        turn_outcome: { version: 1, reason: boundary === 'cancel' ? 'aborted' : 'provider_failed', turn_id: expect.any(String) } },
    ])
    expect(JSON.stringify(saved.messages)).not.toContain('private provider diagnostic')
    await runtime.shutdown()
    resumedRuntime = makeRuntime()
    const resumed = await resumedRuntime.openSession(session.id, undefined, { resume: true, cwd: directory })
    expect(resumed.messages.at(-1)?.content).toBe('function partial() {\n  return ')
    await resumedRuntime.submitTurn(resumed.sessionKey, 'Continue the interrupted function.', () => {})
    const history = requests[1]!.messages
    expect(history.filter(message => message.role === 'assistant')).toEqual([
      { role: 'assistant', content: 'function partial() {\n  return ', thinking: 'Reasoning already received. ' },
    ])
    expect(JSON.stringify(history)).not.toContain('private provider diagnostic')
    expect(JSON.stringify(history)).not.toContain('turn_outcome')
    expect(resumed.messages.find(message => message.role === 'assistant')?.turn_outcome).toEqual(saved.messages.find((message: {role: string}) => message.role === 'assistant').turn_outcome)
    expect(resumed.messages.at(-1)?.turn_outcome).toMatchObject({ version: 1, reason: 'completed' })
    expect(resumed.messages.at(-1)?.content).toBe('Continued reply.')
  } finally {
    await resumedRuntime?.shutdown(); await runtime.shutdown(); await rm(directory, { recursive: true, force: true })
  }
})

for (const boundary of ['no-output', 'setup'] as const) test(`native ${boundary} failure persists one prompt and does not relabel the preceding turn`, async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xr-outcome-'))
  let fail = false
  const llm: LlmClient = { async *stream() {
    if (fail) throw new Error('401 provider failure')
    yield { content: 'First answer.' }
  } }
  const runtime = new InMemoryDaemonRuntime(new AgentTurnRunner({ model: 'gpt-4o', llm, tools: [],
    resolveSessionProvider: () => {
      if (fail && boundary === 'setup') throw new Error('Provider configuration unavailable')
      return { llm }
    },
  }), { model: 'gpt-4o', currentProjectDirectory: directory, sessionDirectory: join(directory, 'sessions') })
  try {
    const session = await runtime.openSession('outcome')
    await runtime.submitTurn(session.sessionKey, 'First prompt.', () => {})
    const firstOutcome = structuredClone(session.messages.at(-1)?.turn_outcome)
    fail = true
    const ends: unknown[] = []
    await runtime.submitTurn(session.sessionKey, 'Failed prompt.', event => { if (event.type === 'turn_end') ends.push(event.payload) })
    expect(session.messages.map(message => message.content)).toEqual(['First prompt.', 'First answer.', 'Failed prompt.'])
    expect(session.messages[1]?.turn_outcome).toEqual(firstOutcome)
    expect(session.messages.at(-1)?.turn_outcome).toMatchObject({ version: 1, reason: boundary === 'setup' ? 'turn_failed' : 'provider_failed' })
    expect(ends).toHaveLength(1)
    expect(ends[0]).toMatchObject({ stop_reason: boundary === 'setup' ? 'turn_failed' : 'provider_failed' })
    expect(session.turnCount).toBe(2)
    await runtime.flushSessions()
    const saved = await Bun.file(join(directory, 'sessions', session.id + '.json')).json()
    expect(saved.messages).toEqual(session.messages)
  } finally { await runtime.shutdown(); await rm(directory, { recursive: true, force: true }) }
})

test('cancellation after native completion cannot relabel completed output', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xr-late-cancel-'))
  const runtime = new InMemoryDaemonRuntime(new AgentTurnRunner({model: 'gpt-4o', tools: [], llm: {async *stream() {yield {content: 'Complete answer.'}}}}), {
    model: 'gpt-4o', currentProjectDirectory: directory, sessionDirectory: join(directory, 'sessions'),
  })
  try {
    const session = await runtime.openSession('late-cancel')
    const ends: unknown[] = []
    let cancelled = false
    await runtime.submitTurn(session.sessionKey, 'A short request.', event => {
      if (event.type === 'status_update' && event.payload.stop_reason === 'completed') cancelled = runtime.cancelTurn(session.sessionKey)
      if (event.type === 'turn_end') ends.push(event.payload)
    })
    expect(cancelled).toBe(true)
    expect(ends).toHaveLength(1)
    expect(ends[0]).toMatchObject({stop_reason: 'completed', cancelled: false})
    expect(session.messages.at(-1)?.turn_outcome).toMatchObject({reason: 'completed'})
    expect(session.messages.at(-1)?.content).toBe('Complete answer.')
    await runtime.flushSessions()
    const saved = await Bun.file(join(directory, 'sessions', session.id + '.json')).json()
    expect(saved.messages.at(-1).turn_outcome).toEqual(session.messages.at(-1)?.turn_outcome)
  } finally {await runtime.shutdown(); await rm(directory, {recursive: true, force: true})}
})

test('a first turn failing before output remains discoverable after restart with its outcome', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xr-first-failure-'))
  const makeRuntime = () => new InMemoryDaemonRuntime(new AgentTurnRunner({model: 'gpt-4o', tools: [], llm: {async *stream() {throw new Error('401 unavailable')}}}), {
    model: 'gpt-4o', currentProjectDirectory: directory, sessionDirectory: join(directory, 'sessions'),
  })
  const runtime = makeRuntime()
  let restarted: InMemoryDaemonRuntime | undefined
  try {
    const session = await runtime.openSession('first-failure')
    await runtime.submitTurn(session.sessionKey, 'Keep this failed request.', () => {})
    await runtime.shutdown()
    restarted = makeRuntime()
    expect((await restarted.listSavedSessions(10)).some(saved => saved.id === session.id)).toBe(true)
    const resumed = await restarted.openSession(session.id, undefined, {resume: true, cwd: directory})
    expect(resumed.messages).toEqual([{role: 'user', content: 'Keep this failed request.', turn_outcome: {version: 1, reason: 'provider_failed', turn_id: expect.any(String)}}])
  } finally {await restarted?.shutdown(); await runtime.shutdown(); await rm(directory, {recursive: true, force: true})}
})
