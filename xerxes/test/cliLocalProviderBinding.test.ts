// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { existsSync } from 'node:fs'
import { mkdtemp, mkdir, rm } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { GatewayClient } from '../src/ui/gatewayClient.js'
import { LocalProviderRelay } from '../src/security/localProviderRelay.js'
import { LocalProviderEndpoint } from '../src/security/localProviderEndpoint.js'
import {localCapabilitySnapshot} from '../src/daemon/localReasoningCapabilities.js'
import {catalogReasoningLevels} from '../src/llms/reasoningLevels.js'
import type { CompletionRequest, LlmClient } from '../src/llms/client.js'

async function until(check: () => boolean | Promise<boolean>) {
  for (let i = 0; i < 500; i++) { if (await check()) return; await Bun.sleep(20) }
  throw new Error('Timed out waiting for isolated production daemon')
}

for (const configured of [false, true]) test(`production daemon ${configured ? 'with conflicting remote defaults' : 'without credentials'} runs local turns and delegation; release fails closed`, async () => {
  const root = await mkdtemp(join(tmpdir(), 'xr-cli-local-'))
  const home = join(root, 'home'), socket = join(root, 'rpc.sock')
  await mkdir(join(home, 'daemon'), { recursive: true })
  let remoteCalls = 0
  const remote = Bun.serve({ hostname: '127.0.0.1', port: 0, fetch() { remoteCalls++; return new Response('Remote fixture must not be called', { status: 503 }) } })
  await Bun.write(join(home, 'daemon', 'config.json'), JSON.stringify({ project_directory: root, runtime: { permission_mode: 'accept-all', ...(configured ? {
    model: 'gpt-5.1', provider: 'openai', api_key: 'synthetic-remote-key', base_url: remote.url.href + 'v1',
    max_tokens: 65536, temperature: 1.9, top_k: 999, top_p: 0.99, reasoning_effort: 'high', thinking: true, thinking_budget: 32000, service_tier: 'priority',
  } : {}) } }))
  const calls: string[] = [], models: string[] = []
  const received: CompletionRequest[] = []
  let overflowed = false
  const provider: LlmClient = { async *stream(request, signal) {
    received.push(request)
    models.push(request.model)
    const lastUser = request.messages.findLastIndex(message => message.role === 'user')
    const text = String(request.messages[lastUser]?.content ?? '')
    if (text.startsWith('You are the context-compaction engine')) { calls.push('compaction'); yield { content: '## User requests\nVerify local routing.\n## Progress\nThe historical checks passed.' }; return }
    if (text === 'Trigger overflow recovery' && !overflowed) { overflowed = true; calls.push('overflow'); throw new Error('maximum context length exceeded: PRIVATE_FIXTURE_DIAGNOSTIC') }
    if (text.startsWith('Historical data')) { calls.push('history'); yield { content: 'Historical detail to summarize. '.repeat(1500) }; return }
    if (text === 'Wait for cancellation') { calls.push('waiting'); await new Promise<void>(resolve => { if (signal?.aborted) resolve(); else signal?.addEventListener('abort', () => resolve(), { once: true }) }); calls.push('cancelled'); return }
    if (text.startsWith('Objective:')) { calls.push('planner'); yield { content: '<step id="s1" agent="coder" depends=""><description>LOCAL_CHILD_TASK</description></step>' }; return }
    if (text === 'LOCAL_CHILD_TASK') { calls.push('child'); yield { content: 'Local production child completed.' }; return }
    if (String(request.messages[0]?.content).startsWith('Write a very short title')) { calls.push('title'); yield { content: 'Local production audit' }; return }
    const afterTool = request.messages.slice(lastUser + 1).some(message => message.role === 'tool')
    if (!afterTool && text.includes('delegate')) {
      calls.push('delegation'); yield { toolCalls: [{ id: 'delegate', type: 'function', function: { name: 'AgentTool', arguments: { prompt: 'LOCAL_CHILD_TASK', subagent_type: 'coder', wait: true } } }] }; return
    }
    if (!afterTool && text.includes('plan')) {
      calls.push('plan-tool'); yield { toolCalls: [{ id: 'plan', type: 'function', function: { name: 'PlanTool', arguments: { objective: 'Verify local production planner', execute: false } } }] }; return
    }
    calls.push('parent'); yield { content: 'Local production route completed.' }
  } }
  const authority = new LocalProviderRelay(), peer = { destination: 'isolated-production-daemon', workspace: root }
  const grant = authority.authorize(peer, { profile: 'local-test', model: 'gpt-5.1', expiresAt: Date.now() + 60000, maxRequests: 40, maxOutputTokens: 8192, maxConcurrent: 4 }, () => ({ client: provider, routeIdentity: 'fixed-test-route', defaults: {
    maxTokens: 4096, temperature: 0.2, topK: 20, topP: 0.8, serviceTier: 'flex', thinking: { effort: 'low', budgetTokens: 128 },
  } }), 'fixed-test-route')
  const endpoint = new LocalProviderEndpoint(authority, peer, grant.token)
  const child = Bun.spawn([process.execPath, join(import.meta.dir, '../src/cli.ts'), 'daemon', '--project-dir', root, '--socket', socket], {
    cwd: root, env: { PATH: process.env.PATH, HOME: home, XERXES_HOME: home, XERXES_DEFERRED_TOOL_LOADING: '0' }, stdin: 'ignore', stdout: 'pipe', stderr: 'pipe',
  })
  const stderr = new Response(child.stderr).text(), stdout = new Response(child.stdout).text()
  const client = new GatewayClient({ externalSocketPath: socket, projectDir: root, providerRelay: (_binding, frame) => endpoint.handle(frame) })
  const events: unknown[] = []; let ended = 0
  client.on('event', event => { events.push(event); if (event.type === 'message.complete') ended++ })
  try {
    await until(async () => {
      if (child.exitCode !== null) throw new Error(await stderr)
      return existsSync(socket)
    })
    await client.start()
    const session = await client.request<{ session_id: string }>('session.create')
    expect(await client.request('provider.remote.bind', { consent: true, source: 'local test', profile: 'local-test', model: 'gpt-5.1', capabilities:localCapabilitySnapshot('gpt-5.1',catalogReasoningLevels('gpt-5.1','openai')!) })).toMatchObject({ ok: true })
    expect(await client.request('reasoning_levels')).toMatchObject({ current: 'local default' })
    for (const text of ['Use local provider.', ...Array.from({ length: 8 }, (_, i) => 'Historical data ' + i), 'delegate to the local child', 'plan using the local provider', 'Trigger overflow recovery']) {
      const before = ended
      await client.request('turn.submit', { text })
      await until(() => ended > before)
    }
    expect(calls).toContain('parent')
    expect(calls).toContain('child')
    if (!calls.includes('planner')) throw new Error(JSON.stringify(events))
    expect(calls).toContain('planner')
    expect(calls).toContain('compaction')
    if (calls.at(-1) !== 'parent') throw new Error(JSON.stringify(events.slice(-10)))
    expect(calls.at(-1)).toBe('parent')
    expect(JSON.stringify(events)).not.toContain('PRIVATE_FIXTURE_DIAGNOSTIC')
    expect(models.every(model => model === 'gpt-5.1')).toBe(true)
    const mainRequests = received.filter(request => request.querySource === 'main')
    expect(mainRequests.length).toBeGreaterThan(10)
    for (const request of mainRequests) expect(request).toMatchObject({ maxTokens: 4096, temperature: 0.2, topK: 20, topP: 0.8, serviceTier: 'flex', thinking: { effort: 'low', budgetTokens: 128 } })
    expect(received.find(request => request.messages.at(-1)?.content === 'LOCAL_CHILD_TASK')).toMatchObject({ maxTokens: 4096, temperature: 0.2, topK: 20, topP: 0.8, thinking: { effort: 'low', budgetTokens: 128 } })
    expect(await client.request('context_breakdown')).toMatchObject({ context_limit: 0 })
    const saved = await Bun.file(join(home, 'sessions', session.session_id + '.json')).text()
    expect(saved).toContain('Local production child completed.')
    expect(saved).toContain('local_provider_binding')
    expect(saved).not.toContain(grant.token)
    expect(await client.request('set_reasoning', { effort: 'off' })).toMatchObject({ ok: true })
    expect(await client.request('reasoning_levels')).toMatchObject({ current: 'off' })
    const explicitBefore = ended
    await client.request('turn.submit', { text: 'Use my explicit off effort.' })
    await until(() => ended > explicitBefore)
    expect(received.find(request => request.messages.at(-1)?.content === 'Use my explicit off effort.')?.thinking).toEqual({ effort: 'none' })
    expect(await Bun.file(join(home, 'sessions', session.session_id + '.json')).json()).toMatchObject({ metadata: { reasoning_effort: 'off' } })
    const cancelledBefore = ended
    await client.request('turn.submit', { text: 'Wait for cancellation' })
    await until(() => calls.includes('waiting'))
    await client.request('turn.cancel')
    await until(() => ended > cancelledBefore && calls.includes('cancelled'))
    const before = ended, callCount = calls.length
    expect(await client.request('provider.remote.release')).toMatchObject({ ok: true })
    await client.request('turn.submit', { text: 'Continue after release.' })
    await until(() => ended > before)
    expect(calls).toHaveLength(callCount)
    expect(JSON.stringify(events)).toContain('Local provider grant is unavailable')
    expect(remoteCalls).toBe(0)
  } finally {
    client.close(); endpoint.close(); authority.close(); child.kill('SIGTERM')
    await until(() => child.exitCode !== null)
    await Promise.all([stderr, stdout])
    remote.stop(true)
    await rm(root, { recursive: true, force: true })
  }
}, 30000)
