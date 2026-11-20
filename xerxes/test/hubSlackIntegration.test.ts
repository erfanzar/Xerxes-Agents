// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, expect, test } from 'bun:test'
import { join } from 'node:path'
import { createHubSlackIntegration, SLACK_CALLBACK_PATH } from '../src/hub/slack/integration.js'
import { SupervisorRunner } from '../src/hub/supervisor/runner.js'
import { SLACK_LIST_TOOL, SLACK_PREPARE_TOOL, type SupervisorSlackPort } from '../src/hub/supervisor/slackTools.js'
import { disposeAll, FakeIdentity, registerCleanup, standardUsers, startFixture, tempDir } from './hubHarness.js'
import { makeWorld, scriptedModel, toolCall } from './hubSupervisorHarness.js'

afterEach(disposeAll)
const origin = 'https://hub.example.test'
const peer = { team_id: 'TEAM1', user_id: 'UPEER' }
const account = { team_id: 'TEAM1', user_id: 'UOWNER' }
const redirectUri = `${origin}${SLACK_CALLBACK_PATH}`
const untrusted = 'DM input must not reach the supervisor model'

async function setup(configured = true) {
  const dir = await tempDir()
  const identity = standardUsers(new FakeIdentity())
  const live = { owner: true, now: Date.now(), takeSend: true }
  const calls: string[] = []
  const integration = createHubSlackIntegration({
    databasePath: join(dir, 'slack-test.sqlite'), stateDir: dir, identity,
    publicOrigins: [origin], isHubOwner: id => id === 'owner' && live.owner, now: () => live.now,
    ...(configured ? { oauth: { clientId: 'fake.id', clientSecret: 'fake.secret', redirectUri } } : {}),
    fetch: async (url, init) => {
      const method = new URL(url).pathname.split('/').at(-1)!
      calls.push(method)
      const json = (body: Record<string, unknown>) => Response.json({ ok: true, ...body })
      switch (method) {
        case 'oauth.v2.access': return json({ team: { id: 'TEAM1' }, authed_user: { id: 'UOWNER', token_type: 'user',
          access_token: 'xoxp-test-only', scope: 'users:read,users:read.email,im:write,im:read,im:history,chat:write' } })
        case 'auth.test': return json({ team_id: 'TEAM1', user_id: 'UOWNER' })
        case 'users.info': return json({ user: { id: new URL(url).searchParams.get('user'), team_id: 'TEAM1', profile: { display_name: untrusted } } })
        case 'conversations.open': case 'conversations.info': return json({ channel: { id: 'DDIRECT', user: 'UPEER', is_im: true } })
        case 'conversations.history': return json({ messages: [{ type: 'message', user: 'UPEER', ts: '1700000000.000001', text: untrusted }] })
        case 'chat.postMessage': {
          expect(new Headers(init.headers).get('authorization')).toEqual('Bearer xoxp-test-only')
          if (!live.takeSend) throw new Error('upstream request body, token or error must remain private')
          return json({ channel: 'DDIRECT', ts: '1700000000.000002' })
        }
        default: throw new Error('unexpected mock Slack endpoint')
      }
    },
  })
  registerCleanup(() => integration.close())
  const f = await startFixture({ server: { slack: integration } })
  const client = await f.client('owner')
  const initiate = async () => new URL(String((await client.ok('supervisor.slack.connect')).authorize_url)).searchParams.get('state')!
  const callback = (state: string, query = 'code=fake-code', host = 'hub.example.test') =>
    integration.callback(new Request(`${redirectUri}?${query}&state=${encodeURIComponent(state)}`, { headers: { host } }))
  const connect = async () => expect((await callback(await initiate())).status).toBe(200)
  return { f, client, identity, live, integration, calls, initiate, callback, connect }
}

test('only a live human Hub owner may use every Slack RPC; unconfigured and old Hub replies differ', async () => {
  const { f, client, live, calls } = await setup(false)
  expect(await client.ok('supervisor.slack.status')).toEqual({ configured: false, connection: null, contacts: [] })
  expect(await client.fails('supervisor.slack.connect')).toBe('unconfigured')
  for (const who of ['alice', 'robot']) {
    const denied = await f.client(who)
    expect(await denied.fails('supervisor.slack.status')).toBe('forbidden')
    expect(await denied.fails('supervisor.slack.draft.send', { ...peer, draft_id: 'draft', text: 'secret' })).toBe('forbidden')
  }
  live.owner = false
  expect(await client.fails('supervisor.slack.status')).toBe('forbidden')
  expect(calls).toEqual([])
  const old = await startFixture()
  expect(await (await old.client('owner')).fails('supervisor.slack.status')).toBe('unknown_method')
})

test('OAuth callback is fixed-host, expiring, one-use, owner/device-bound, and not cached', async () => {
  const s = await setup()
  const state = await s.initiate()
  expect((await s.callback(state, undefined, 'attacker.example.test')).status).toBe(404)
  expect((await s.callback('not-the-state')).status).toBe(400)
  expect((await s.callback(state, 'code=fake-code&state=duplicate')).status).toBe(400)
  expect(s.calls).toEqual([])
  const completed = await fetch(s.f.url(`${SLACK_CALLBACK_PATH}?code=fake-code&state=${encodeURIComponent(state)}`), {
    headers: { host: 'hub.example.test' },
  })
  expect(completed.status).toBe(200)
  expect(completed.headers.get('cache-control')).toBe('no-store')
  expect(completed.headers.get('referrer-policy')).toBe('no-referrer')
  expect(await completed.text()).not.toContain('fake-code')
  expect((await s.callback(state)).status).toBe(400)
  expect(s.calls).toEqual(['oauth.v2.access', 'auth.test'])
  expect(await s.client.ok('supervisor.slack.status')).toEqual({ configured: true, connection: account, contacts: [] })
  expect(await s.client.ok('supervisor.slack.disconnect', account)).toEqual({ configured: true, connection: null, contacts: [] })
  const denied = await s.initiate()
  expect((await s.callback(denied, 'error=access_denied')).status).toBe(400)
  expect((await s.callback(denied)).status).toBe(400)
  const revoked = await s.initiate()
  s.identity.revoke({ principalId: 'owner', browserSessionId: 'bs-owner', reason: 'test' })
  expect((await s.callback(revoked)).status).toBe(400)
  const expired = await s.initiate()
  s.live.now += 10 * 60_000
  expect((await s.callback(expired)).status).toBe(400)
  expect(s.calls).toEqual(['oauth.v2.access', 'auth.test'])
})

test('confirmed exact contacts are draft-first; reviewed draft retries never cause repeat sends', async () => {
  const s = await setup()
  await s.connect()
  expect(await s.client.fails('supervisor.slack.dm.open', peer)).toBe('forbidden')
  expect(await s.client.fails('supervisor.slack.contact.lookup', { ...peer, email: '@a.test' })).toBe('invalid_request')
  expect(await s.client.fails('supervisor.slack.contact.lookup', { ...peer, team_id: 'TOTHER' })).toBe('not_found')
  expect(await s.client.ok('supervisor.slack.contact.lookup', peer)).toEqual({ identity: { ...peer, display_name: untrusted } })
  expect(await s.client.ok('supervisor.slack.contact.confirm', peer)).toMatchObject({ contacts: [{ ...peer, mode: 'draft' }] })
  const tools = s.integration.tools!
  const signal = new AbortController().signal
  expect(await tools.prepare('owner', { teamId: 'TEAM1', peerId: 'UPEER' }, 'Reviewed text', signal, 'request-1')).toBe('drafted')
  const dm = await s.client.ok('supervisor.slack.dm.open', peer)
  expect(dm.messages).toMatchObject([{ id: '1700000000.000001', direction: 'inbound', text: untrusted }])
  const drafts = dm.drafts as { id: string; text: string }[]
  expect(drafts).toHaveLength(1)
  const send = { ...peer, draft_id: drafts[0]!.id, text: drafts[0]!.text }
  expect(await s.client.fails('supervisor.slack.draft.send', { ...send, text: 'edited' })).toBe('conflict')
  expect(s.calls).not.toContain('chat.postMessage')
  const raced = await Promise.allSettled([s.client.ok('supervisor.slack.draft.send', send), s.client.ok('supervisor.slack.draft.send', send)])
  expect(raced.some(result => result.status === 'fulfilled' && Array.isArray(result.value.drafts) && result.value.drafts.length === 0)).toBe(true)
  expect((await s.client.ok('supervisor.slack.draft.send', send)).drafts).toEqual([])
  expect(s.calls.filter(call => call === 'chat.postMessage')).toHaveLength(1)
  expect(await s.client.ok('supervisor.slack.contact.mode', { ...peer, mode: 'auto' })).toMatchObject({ contacts: [{ ...peer, mode: 'auto' }] })
  expect(await s.client.ok('supervisor.slack.contact.confirm', peer)).toMatchObject({ contacts: [{ ...peer, mode: 'auto' }] })
  s.live.takeSend = false
  for (let attempt = 0; attempt < 2; attempt += 1) {
    await expect(tools.prepare('owner', { teamId: 'TEAM1', peerId: 'UPEER' }, 'other message', signal, 'request-2')).rejects.toThrow('uncertain')
  }
  expect(s.calls.filter(call => call === 'chat.postMessage')).toHaveLength(2)
  await s.client.ok('supervisor.slack.contact.remove', peer)
  expect(await s.client.fails('supervisor.slack.dm.open', peer)).toBe('forbidden')
  expect(JSON.stringify(s.identity.audits)).not.toContain('Reviewed text')
})

test('only direct private owner turns see or execute Slack tools, with allowlisted metadata and sanitized errors', async () => {
  const world = makeWorld()
  const requester = world.user('owner')
  const privateThread = world.threads.privateFor(requester, true).id
  const shared = world.threads.ensureShared().id
  const causal = (threadId: string) => world.threads.append({ threadId, role: 'user', text: 'Draft the specified Slack message.', author: { id: requester.id, name: requester.displayName } }).message.id
  let prepared = 0
  let live = true
  const contacts = [{ teamId: 'TEAM1', peerId: 'UPEER', mode: 'draft' as const, displayName: untrusted, text: untrusted }]
  const slack: SupervisorSlackPort = { isOwner: () => live, contacts: () => contacts,
    prepare: async () => { prepared += 1; throw new Error(untrusted) } }
  const list = { text: '', toolCalls: [toolCall(SLACK_LIST_TOOL)] }
  const prepare = { text: '', toolCalls: [toolCall(SLACK_PREPARE_TOOL, { ...peer, text: 'Hello' })] }
  const scripted = scriptedModel([prepare, prepare, prepare, list, prepare, { text: 'See Slack settings', toolCalls: [] }])
  const runner = new SupervisorRunner({ threads: world.threads, registry: world.registry, slack, model: scripted.model, persona: 'test',
    scopeReader: (id, who) => world.threads.scopeReader(world.threads.get(id)!, who) })
  registerCleanup(() => runner.shutdown())
  expect((await runner.enqueue({ threadId: shared, requester, causalMessageId: causal(shared) })).stop).toBe('denied')
  expect((await runner.enqueue({ threadId: privateThread, requester, trigger: { kind: 'watch', text: 'automated' } })).stop).toBe('denied')
  live = false
  expect((await runner.enqueue({ threadId: privateThread, requester, causalMessageId: causal(privateThread) })).stop).toBe('denied')
  live = true
  expect((await runner.enqueue({ threadId: privateThread, requester, causalMessageId: causal(privateThread) })).stop).toBe('completed')
  expect(prepared).toBe(1)
  for (const request of scripted.requests.slice(0, 3)) expect(request.tools.map(tool => tool.function.name)).not.toContain(SLACK_PREPARE_TOOL)
  expect(scripted.requests[3]!.tools.map(tool => tool.function.name)).toContain(SLACK_PREPARE_TOOL)
  expect(JSON.stringify(scripted.requests)).not.toContain(untrusted)
  expect(JSON.stringify(world.events)).not.toContain(untrusted)
})
