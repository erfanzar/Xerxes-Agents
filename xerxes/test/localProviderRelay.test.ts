// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { OUTPUT_TOKEN_LIMIT } from '../src/llms/outputTokenLimit.js'
import { LocalProviderRelay, type LocalProviderGrant } from '../src/security/localProviderRelay.js'
import type { CompletionRequest, LlmClient } from '../src/llms/client.js'

const peer = () => Object.freeze({ destination: 'verified-ssh-host', workspace: '/workspace/project' })
const policy = (): LocalProviderGrant => ({ profile: 'local-selected-profile', model: 'fixture-model', expiresAt: Date.now() + 60_000, maxRequests: 3, maxOutputTokens: 256, maxConcurrent: 1 })
const request: CompletionRequest = { model: 'fixture-model', messages: [{ role: 'user', content: 'write code' }] }
const collect = async (stream: AsyncIterable<unknown>) => { const output = []; for await (const part of stream) output.push(part); return output }

test('relay uses the exact local route, passes coding tools and effort, and exposes no token in its view', async () => {
  const relay = new LocalProviderRelay(), link = peer()
  let received: CompletionRequest | undefined
  const client: LlmClient = { async *stream(value) { received = value; yield { content: 'const x = 1;\n' }; yield { thinking: 'done' } } }
  const grant = relay.authorize(link, policy(), () => ({ client, routeIdentity: 'original-route' }), 'original-route')
  try {
    const input = { ...request, tools: [{ type: 'function' as const, function: { name: 'ReadFile', description: 'Read remote workspace', parameters: { type: 'object' } } }], thinking: { effort: 'high' } }
    expect(await collect(relay.stream(link, grant.token, input))).toEqual([{ content: 'const x = 1;\n' }, { thinking: 'done' }])
    expect(received).toEqual({ ...input, maxTokens: 256, sessionId: expect.stringMatching(/^relay:/), [OUTPUT_TOKEN_LIMIT]: 256 })
    expect(JSON.stringify(received)).not.toContain('outputTokenLimit')
    expect(relay.inspect(link, grant.token)).toMatchObject({ requestsUsed: 1, activeRequests: 0, execution: 'local provider; remote tools', persistence: 'memory only' })
    expect(JSON.stringify(grant.view)).not.toContain(grant.token)
    await expect(collect(relay.stream(peer(), grant.token, request))).rejects.toMatchObject({ code: 'grant_unavailable' })
    await expect(collect(relay.stream(link, grant.token, { ...request, model: 'another-model' }))).rejects.toMatchObject({ code: 'route_mismatch' })
    expect(relay.inspect(link, grant.token).requestsUsed).toBe(1)
  } finally { relay.close() }
})

test('relay rejects routing overrides, oversized budgets and local-network image fetches before using a provider', async () => {
  const relay = new LocalProviderRelay(), link = peer()
  let calls = 0
  const grant = relay.authorize(link, policy(), () => { calls++; throw new Error('must not resolve') }, 'original-route')
  try {
    for (const input of [
      { ...request, extraBody: { base_url: 'http://localhost/private' } },
      { ...request, maxTokens: 257 },
      { ...request, messages: [{ role: 'user' as const, content: [{ type: 'image_url' as const, image_url: { url: 'http://127.0.0.1/private' } }] }] },
    ]) await expect(collect(relay.stream(link, grant.token, input))).rejects.toMatchObject({ code: 'invalid_request' })
    expect(calls).toBe(0)
  } finally { relay.close() }
})

test('relay counts failed requests, never relays provider diagnostics, and never falls back after exhaustion', async () => {
  const relay = new LocalProviderRelay(), link = peer()
  let calls = 0
  const grant = relay.authorize(link, { ...policy(), maxRequests: 1 }, () => ({ routeIdentity: 'original-route', client: { async *stream() { calls++; throw new Error('Bearer synthetic-private-credential') } } }), 'original-route')
  try {
    const error = await collect(relay.stream(link, grant.token, request)).catch(error => error)
    expect(error.code).toBe('provider_failed')
    expect(String(error)).not.toContain('synthetic-private-credential')
    await expect(collect(relay.stream(link, grant.token, request))).rejects.toMatchObject({ code: 'request_limit' })
    expect(calls).toBe(1)
  } finally { relay.close() }
})

for (const cap of [256, null]) for (const action of ['cancel', 'disconnect', 'revoke', 'expire', 'close'] as const) test(`relay ${action} aborts active provider work with ${cap === null ? 'provider-controlled' : 'bounded'} output`, async () => {
  let now = Date.now(), aborted = false, started!: () => void
  const ready = new Promise<void>(resolve => { started = resolve })
  const relay = new LocalProviderRelay(() => now), link = peer(), external = new AbortController()
  const grant = relay.authorize(link, { ...policy(), maxOutputTokens: cap, expiresAt: now + 60_000 }, () => ({ routeIdentity: 'original-route', client: { async *stream(_input, signal) {
    started()
    await new Promise<void>(resolve => signal!.addEventListener('abort', () => { aborted = true; resolve() }, { once: true }))
    yield { content: 'late output must not escape' }
  } } }), 'original-route')
  try {
    const output = collect(relay.stream(link, grant.token, request, external.signal)).catch(error => error)
    await ready
    await expect(collect(relay.stream(link, grant.token, request))).rejects.toMatchObject({ code: 'concurrency_limit' })
    if (action === 'cancel') external.abort()
    else if (action === 'disconnect') relay.disconnect(link)
    else if (action === 'revoke') relay.revoke(link, grant.token)
    else if (action === 'close') relay.close()
    else { now += 60_001; expect(relay.inspect(link, grant.token).status).toBe('expired') }
    const result = await output
    expect(aborted).toBe(true)
    expect(result.code).toBe(action === 'expire' ? 'grant_expired' : action === 'revoke' || action === 'close' ? 'grant_revoked' : 'cancelled')
    if (action === 'cancel' || action === 'disconnect') expect(relay.inspect(link, grant.token)).toMatchObject({ activeRequests: 0, requestsUsed: 1, status: 'active' })
    else await expect(collect(relay.stream(link, grant.token, request))).rejects.toMatchObject({ code: action === 'expire' ? 'grant_expired' : action === 'revoke' ? 'grant_revoked' : 'grant_unavailable' })
  } finally { relay.close() }
})

test('relay rejects changed provider routes and lost authority without silently selecting a replacement', async () => {
  const relay = new LocalProviderRelay(), link = peer()
  let routeIdentity = 'approved-route', calls = 0
  const grant = relay.authorize(link, policy(), () => ({ routeIdentity, client: { async *stream() { calls++; yield { content: 'authorized' } } } }), routeIdentity)
  try {
    routeIdentity = 'different-provider'
    await expect(collect(relay.stream(link, grant.token, request))).rejects.toMatchObject({ code: 'route_changed' })
    expect(calls).toBe(0)
    routeIdentity = 'approved-route'
    expect(await collect(relay.stream(link, grant.token, request))).toEqual([{ content: 'authorized' }])
    const restarted = new LocalProviderRelay()
    try { await expect(collect(restarted.stream(link, grant.token, request))).rejects.toMatchObject({ code: 'grant_unavailable' }) }
    finally { restarted.close() }
  } finally { relay.close() }
})

test('already-cancelled requests consume no authority and reconnect never replays cancelled work', async () => {
  const relay = new LocalProviderRelay(), link = peer()
  let calls = 0
  const grant = relay.authorize(link, policy(), () => ({ routeIdentity: 'route', client: { async *stream() { calls++; yield { content: 'one request' } } } }), 'route')
  try {
    const controller = new AbortController(); controller.abort()
    await expect(collect(relay.stream(link, grant.token, request, controller.signal))).rejects.toMatchObject({ code: 'cancelled' })
    expect(relay.inspect(link, grant.token).requestsUsed).toBe(0)
    relay.disconnect(link)
    expect(await collect(relay.stream(link, grant.token, request))).toEqual([{ content: 'one request' }])
    expect(calls).toBe(1)
  } finally { relay.close() }
})
