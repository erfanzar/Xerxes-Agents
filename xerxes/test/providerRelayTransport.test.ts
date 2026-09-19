// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { LocalProviderRelay } from '../src/security/localProviderRelay.js'
import { LocalProviderEndpoint } from '../src/security/localProviderEndpoint.js'
import { decodeRelayCompletion } from '../src/security/providerRelayProtocol.js'
import { LocalRelayClient } from '../src/llms/localRelayClient.js'
import type { CompletionRequest, LlmClient } from '../src/llms/client.js'
import { AgentTurnRunner } from '../src/daemon/turnRunner.js'
import { InMemoryDaemonRuntime } from '../src/daemon/runtime.js'
import { ToolRegistry } from '../src/executors/toolRegistry.js'

const request: CompletionRequest = { model: 'gpt-4o', messages: [{ role: 'user', content: 'Write the file.' }] }
function endpoint(provider: LlmClient, timeout = 60_000, batchSize = 32) {
  const relay = new LocalProviderRelay(), peer = Object.freeze({ destination: 'verified-loopback-fixture', workspace: '/remote-fixture' })
  const grant = relay.authorize(peer, { profile: 'local-fixture', model: request.model, expiresAt: Date.now() + 60_000, maxRequests: 20, maxOutputTokens: 2048, maxConcurrent: 1 }, () => ({ client: provider, routeIdentity: 'fixed-route' }), 'fixed-route')
  const host = new LocalProviderEndpoint(relay, peer, grant.token, timeout, batchSize)
  return { relay, peer, grant, host, close() { host.close(); relay.close() } }
}

test('native completion codec preserves reasoning, tool history, cache segments, effort and grammar', () => {
  const full: CompletionRequest = { ...request, querySource: 'compaction', sessionId: 'session', serviceTier: 'default', thinking: { effort: 'high', budgetTokens: 1024 }, systemSegments: [{ name: 'system', text: 'rules', volatile: false }], topK: 5, minP: 0.1,
    messages: [{ role: 'assistant', content: 'code\n', thinking: 'reasoning\n', thinking_signature: 'signed', tool_calls: [{ id: 'call-1', type: 'function', function: { name: 'ReadFile', arguments: { path: 'a.ts' } } }] }, { role: 'tool', content: 'output\n', tool_call_id: 'call-1', is_error: false, added_tool_names: ['WriteFile'] }],
    tools: [{ type: 'function', constrainedSampling: { type: 'grammar', variants: { openai_regex: '.*' } }, function: { name: 'ReadFile', description: 'Read remote file', parameters: { type: 'object' } } }],
  }
  expect(decodeRelayCompletion(JSON.parse(JSON.stringify(full)))).toEqual(full)
  for (const bad of [{ ...full, api_key: 'private-sentinel' }, { ...full, extraBody: { headers: { authorization: 'private-sentinel' } } }, { ...full, thinking: { effort: 4 } }]) {
    try { decodeRelayCompletion(bad); throw new Error('accepted') } catch (error) { expect(String(error)).not.toContain('private-sentinel'); expect(error).toHaveProperty('code', 'invalid_request') }
  }
})

test('pull transport does not request another provider delta until the consumer advances', async () => {
  let pulled = 0
  const fixture = endpoint({ async *stream() { for (const content of ['one', 'two']) { pulled++; yield { content } } } }, 60_000, 1)
  const client = new LocalRelayClient(frame => fixture.host.handle(JSON.parse(JSON.stringify(frame))))
  try {
    const stream = client.stream(request)
    expect(await stream.next()).toMatchObject({ value: { content: 'one' } })
    await Bun.sleep(20)
    expect(pulled).toBe(1)
    expect(await stream.next()).toMatchObject({ value: { content: 'two' } })
    expect(await stream.next()).toMatchObject({ done: true })
    expect(fixture.relay.inspect(fixture.peer, fixture.grant.token)).toMatchObject({ requestsUsed: 1, activeRequests: 0 })
  } finally { fixture.close() }
})

test('closing a channel settles a pending pull even when the provider ignores cancellation', async () => {
  const started = Promise.withResolvers<void>(), release = Promise.withResolvers<void>()
  const fixture = endpoint({ async *stream() { started.resolve(); await release.promise; yield { content: 'late' } } })
  try {
    const pending = fixture.host.handle({ op: 'next', id: 'pending', request })
    await started.promise
    fixture.host.close()
    expect(await pending).toEqual({ error: 'cancelled' })
    // The actual backend is still occupied, so cancellation cannot manufacture
    // spare grant capacity for a second expensive request.
    expect(fixture.relay.inspect(fixture.peer, fixture.grant.token).activeRequests).toBe(1)
    release.resolve(); await Bun.sleep(5)
    expect(fixture.relay.inspect(fixture.peer, fixture.grant.token).activeRequests).toBe(0)
  } finally { release.resolve(); fixture.close() }
})

test('stalled pulls expire, malformed requests stay safe and no provider exception text crosses the endpoint', async () => {
  const fixture = endpoint({ async *stream() { throw new Error('Authorization: private-sentinel') } }, 20)
  try {
    expect(await fixture.host.handle({ op: 'next', id: 'bad', request: { ...request, model: 42 } })).toEqual({ error: 'invalid_request' })
    expect(await fixture.host.handle({ op: 'next', id: 'failed', request })).toEqual({ error: 'provider_failed' })
    expect(await fixture.host.handle({ op: 'next', id: 'failed' })).toEqual({ error: 'invalid_request' })
  } finally { fixture.close() }
  const slow = endpoint({ async *stream(_request, signal) { await new Promise<void>(resolve => signal!.addEventListener('abort', () => resolve(), { once: true })); yield { content: 'late' } } }, 20)
  try { expect(await slow.host.handle({ op: 'next', id: 'slow', request })).toEqual({ error: 'cancelled' }) }
  finally { slow.close() }
})

test('real Unix-socket relay runs the native agent loop and executes coding tools only on its workspace side', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xr-relay-'))
  const file = join(directory, 'remote-result.ts')
  const frames: string[] = [], calls: CompletionRequest[] = []
  const fixture = endpoint({ async *stream(input) {
    calls.push(input)
    if (calls.length === 1) yield { toolCalls: [{ id: 'write-1', type: 'function', function: { name: 'WriteFixture', arguments: {} } }] }
    else yield { content: 'Saved the code.', usage: { inputTokens: 12, outputTokens: 4 } }
  } })
  const socket = join(directory, 'relay.sock')
  const server = Bun.serve({ unix: socket, maxRequestBodySize: 16 * 1024 * 1024, async fetch(incoming) {
    const text = await incoming.text(); frames.push(text)
    return Response.json(await fixture.host.handle(JSON.parse(text)))
  } })
  const client = new LocalRelayClient(async (frame, signal) => {
    const response = await fetch('http://localhost/relay', { unix: socket, method: 'POST', body: JSON.stringify(frame), ...(signal ? { signal } : {}) })
    return response.json()
  })
  const registry = new ToolRegistry()
  let executed = 0
  registry.register({ type: 'function', function: { name: 'WriteFixture', description: 'Write the isolated remote fixture file', parameters: { type: 'object' } } }, async () => { executed++; await Bun.write(file, 'export const result = 42;\n'); return 'saved' })
  const runtime = new InMemoryDaemonRuntime(new AgentTurnRunner({ llm: client, model: request.model, permissionMode: 'accept-all', maxTokens: 1024, toolExecutor: registry, tools: registry.definitions() }), { model: request.model, currentProjectDirectory: directory, sessionDirectory: join(directory, 'sessions') })
  try {
    const session = await runtime.openSession('remote-turn')
    await runtime.submitTurn(session.sessionKey, 'Write the fixture code.', () => {})
    expect(executed).toBe(1)
    expect(await Bun.file(file).text()).toBe('export const result = 42;\n')
    expect(JSON.stringify(session.messages)).toContain('Saved the code.')
    expect(calls).toHaveLength(2)
    expect(calls[1]!.messages.some(message => message.role === 'tool' && message.content.includes('saved'))).toBe(true)
    expect(frames.join('\n')).not.toContain(fixture.grant.token)
    expect(calls.every(call => call.sessionId?.startsWith('relay:'))).toBe(true)
  } finally { await runtime.shutdown(); fixture.close(); await server.stop(true); await rm(directory, { recursive: true, force: true }) }
})

test('remote client cancellation does not wait for an unresponsive transport and sends an explicit cancel', async () => {
  const signal = new AbortController(), operations: string[] = []
  const client = new LocalRelayClient(async frame => { operations.push(frame.op); if (frame.op === 'cancel') return { done: true }; return new Promise(() => {}) })
  const next = client.stream(request, signal.signal).next().catch(error => error)
  signal.abort()
  expect(await next).toHaveProperty('code', 'cancelled')
  expect(operations).toEqual(['next', 'cancel'])
})

test('bounded batches retain partial output before a terminal provider error', async () => {
  const fixture = endpoint({ async *stream() { yield { content: 'retained ' }; yield { content: 'text' }; throw new Error('secret diagnostic') } })
  const stream = new LocalRelayClient(frame => fixture.host.handle(frame)).stream(request)
  try {
    expect(await stream.next()).toMatchObject({ value: { content: 'retained ' } })
    expect(await stream.next()).toMatchObject({ value: { content: 'text' } })
    await expect(stream.next()).rejects.toHaveProperty('code', 'provider_failed')
  } finally { fixture.close() }
})

test('batches never prefetch beyond the configured window and retain a delayed delta exactly once', async () => {
  const release = Promise.withResolvers<void>()
  let produced = 0
  const fixture = endpoint({ async *stream() { for (let i = 0; i < 5; i++) { produced++; yield { content: String(i) } }; await release.promise; produced++; yield { content: 'delayed' } } }, 60_000, 4)
  const stream = new LocalRelayClient(frame => fixture.host.handle(frame)).stream(request)
  try {
    expect(await stream.next()).toMatchObject({ value: { content: '0' } })
    expect(produced).toBeLessThanOrEqual(4)
    const values = ['0']
    for (let i = 0; i < 4; i++) values.push((await stream.next()).value!.content!)
    expect(values).toEqual(['0', '1', '2', '3', '4'])
    release.resolve()
    expect(await stream.next()).toMatchObject({ value: { content: 'delayed' } })
    expect(await stream.next()).toMatchObject({ done: true })
    expect(produced).toBe(6)
  } finally { release.resolve(); fixture.close() }
})

test('local context overflow survives the transport as a fixed safe recovery signal', async () => {
  const fixture = endpoint({ async *stream() { throw new Error('maximum context length exceeded; Authorization: Bearer SECRET_FIXTURE') } })
  const client = new LocalRelayClient(frame => fixture.host.handle(frame))
  try {
    const error = await client.stream(request)[Symbol.asyncIterator]().next().catch(error => error)
    expect(error).toHaveProperty('code', 'context_overflow')
    expect(error.message).toContain('context window')
    expect(error.message).not.toContain('SECRET_FIXTURE')
    expect(error.message).not.toContain('Authorization')
  } finally { fixture.close() }
})

test('compacted native history crosses the relay without serializing internal summary provenance', () => {
  const input = { ...request, messages: [{ role: 'user', content: 'Saved summary', xerxes_compaction_summary: true }] }
  expect(decodeRelayCompletion(input).messages).toEqual([{ role: 'user', content: 'Saved summary' }])
  expect(() => decodeRelayCompletion({ ...input, messages: [{ ...input.messages[0], xerxes_compaction_summary: 'invalid' }] })).toThrow('unsupported')
  expect(() => decodeRelayCompletion({ ...input, messages: [{ ...input.messages[0], arbitrary_metadata: 'unknown' }] })).toThrow('unsupported')
})
