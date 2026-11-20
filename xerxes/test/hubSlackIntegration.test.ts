// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, expect, test } from 'bun:test'
import { createHash, randomBytes } from 'node:crypto'
import { readdir } from 'node:fs/promises'
import { join } from 'node:path'
import { createHubSlackIntegration, SLACK_CALLBACK_PATH, slackBrokerFromEnvironment } from '../src/hub/slack/integration.js'
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
  const connect = async () => {
    expect((await callback(await initiate())).status).toBe(200)
    expect(await client.ok('supervisor.slack.status')).toMatchObject({ connection: null, pending_identity: account })
    expect(await client.ok('supervisor.slack.confirm_connection', account)).toMatchObject({ connection: account, pending_identity: null })
  }
  return { f, client, identity, live, integration, calls, initiate, callback, connect }
}

test('only a live human Hub owner may use every Slack RPC; unconfigured and old Hub replies differ', async () => {
  const { f, client, live, calls } = await setup(false)
  expect(await client.ok('supervisor.slack.status')).toEqual({ configured: true, browser_sign_in: false, connection: null, pending_identity: null, contacts: [] })
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
  expect(await s.client.ok('supervisor.slack.status')).toEqual({ configured: true, browser_sign_in: true, connection: null, pending_identity: account, contacts: [] })
  expect(await s.client.ok('supervisor.slack.confirm_connection', account)).toEqual({ configured: true, browser_sign_in: true, connection: account, pending_identity: null, contacts: [] })
  expect(await s.client.ok('supervisor.slack.disconnect', account)).toEqual({ configured: true, browser_sign_in: true, connection: null, pending_identity: null, contacts: [] })
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

const brokerOrigin = 'https://broker.example.test'
const scope = 'users:read,users:read.email,im:write,im:read,im:history,chat:write'
const ready = { status: 'ready', access_token: 'xoxp-managed-test-only', ...account, scope }

async function setupManaged(enrollmentToken: string | null = randomBytes(32).toString('base64url')) {
  const dir = await tempDir()
  const identity = standardUsers(new FakeIdentity())
  const live = { now: Date.now(), result: Response.json({ status: 'pending' }), verifiedUser: account.user_id,
    browserUrl: `${brokerOrigin}/v1/slack/authorize?id=test-flow`, pairingCode: undefined as string | undefined }
  let challenge: string | undefined
  const calls: string[] = []
  const integration = createHubSlackIntegration({ databasePath: join(dir, 'hub.sqlite'), stateDir: dir, identity,
    publicOrigins: [origin], brokerOrigin, ...(enrollmentToken === null
      ? { oauth: { clientId: 'fake.id', clientSecret: 'fake.secret', redirectUri } }
      : { brokerEnrollmentToken: enrollmentToken }),
    isHubOwner: id => id === 'owner', now: () => live.now,
    fetch: async (href, init) => {
      calls.push(href)
      expect(init.redirect).toBe('error')
      expect(init.method).toBe('POST')
      const body = init.body ? JSON.parse(String(init.body)) : undefined
      switch (href) {
        case `${brokerOrigin}/v1/slack/start`:
          expect(new Headers(init.headers).get('authorization')).toBe(`Bearer ${enrollmentToken}`)
          expect(body).toEqual({ challenge: expect.stringMatching(/^[A-Za-z0-9_-]{43}$/u) })
          challenge = body.challenge
          return Response.json({ id: 'test-flow', browser_url: live.browserUrl, pairing_code: live.pairingCode })
        case `${brokerOrigin}/v1/slack/redeem`:
          expect(new Headers(init.headers).get('authorization')).toBe(`Bearer ${enrollmentToken}`)
          expect(body).toEqual({ id: 'test-flow', verifier: expect.stringMatching(/^[A-Za-z0-9_-]{43}$/u) })
          expect(body.verifier).not.toEqual(challenge)
          if (challenge === undefined) throw new Error('redeem before start')
          expect(createHash('sha256').update(body.verifier).digest('base64url')).toBe(challenge)
          return live.result.clone()
        case 'https://slack.com/api/auth.test':
          expect(new Headers(init.headers).get('authorization')).toEqual(`Bearer ${ready.access_token}`)
          return Response.json({ ok: true, team_id: account.team_id, user_id: live.verifiedUser })
        default: throw new Error('Unexpected outbound request')
      }
    },
  })
  registerCleanup(() => integration.close())
  const f = await startFixture({ server: { slack: integration } })
  return { dir, identity, f, integration, client: await f.client('owner'), live, calls }
}

test('managed sign-in without enrollment is unavailable and makes no requests or self-hosted fallback', async () => {
  const s = await setupManaged(null)
  expect(await s.client.ok('supervisor.slack.status')).toMatchObject({ configured: true, browser_sign_in: false, pending_identity: null })
  expect(await s.client.fails('supervisor.slack.connect')).toBe('unavailable')
  expect(s.calls).toEqual([])
})

test('managed OAuth binds redemption/review to the initiating device and confirms exact verified identity before persisting', async () => {
  const enrollmentToken = randomBytes(32).toString('base64url')
  const s = await setupManaged(enrollmentToken)
  s.live.pairingCode = 'a'.repeat(22)
  const authorization = await s.client.ok('supervisor.slack.connect')
  expect(authorization).toEqual({ authorize_url: s.live.browserUrl, expires_at: s.live.now + 600_000, pairing_code: s.live.pairingCode })
  expect(await s.client.fails('supervisor.slack.connect')).toBe('conflict')
  expect(await s.client.ok('supervisor.slack.status')).toMatchObject({ configured: true, connection: null, pending_identity: null })
  expect(s.calls).toEqual([`${brokerOrigin}/v1/slack/start`, `${brokerOrigin}/v1/slack/redeem`])
  const original = s.identity.context('owner')
  if (original.principal.kind !== 'user') throw new Error('expected owner context')
  const otherDevice = { ...original, principal: { ...original.principal, deviceId: 'other-device' } }
  const otherSession = { ...original, browserSessionId: 'other-session' }
  s.live.now += 2000
  s.live.result = Response.json(ready)
  const status = s.integration.handlers['supervisor.slack.status']
  for (const context of [otherDevice, otherSession]) {
    expect(await status({ context, grants: [] }, {})).toMatchObject({ pending_identity: null })
  }
  expect(s.calls).toHaveLength(2)
  expect(await s.client.ok('supervisor.slack.status')).toEqual({ configured: true, browser_sign_in: true, connection: null, pending_identity: account, contacts: [] })
  expect(s.calls.slice(2)).toEqual([`${brokerOrigin}/v1/slack/redeem`, 'https://slack.com/api/auth.test'])
  expect(await readdir(s.dir)).not.toContainEqual(expect.stringMatching(/^slack-owner-/))
  expect(s.integration.tools?.isOwner('owner')).toBe(false)
  expect(await s.client.fails('supervisor.slack.confirm_connection', { ...account, user_id: 'UOTHER' })).toBe('not_found')
  for (const context of [otherDevice, otherSession]) {
    await expect(Promise.resolve(s.integration.handlers['supervisor.slack.confirm_connection']({ context, grants: [] }, account)))
      .rejects.toMatchObject({ code: 'not_found' })
  }
  expect(await (await s.f.client('alice')).fails('supervisor.slack.confirm_connection', account)).toBe('forbidden')
  expect(await s.client.ok('supervisor.slack.confirm_connection', account)).toEqual({ configured: true, browser_sign_in: true, connection: account, pending_identity: null, contacts: [] })
  expect((await readdir(s.dir)).some(name => name.startsWith('slack-owner-'))).toBe(true)
  expect(await s.client.fails('supervisor.slack.confirm_connection', account)).toBe('not_found')
  expect(await s.client.fails('supervisor.slack.dm.open', peer)).toBe('forbidden')
  expect(JSON.stringify(s.identity.audits)).not.toContain(ready.access_token)
  expect(JSON.stringify(s.identity.audits)).not.toContain(enrollmentToken)
  expect(JSON.stringify(s.identity.audits)).not.toContain(s.live.pairingCode)
})

test('managed broker origin, grant scope, auth.test and revocation fail closed', async () => {
  expect(slackBrokerFromEnvironment({})).toBeUndefined()
  for (const bad of ['http://bad.test', 'https://bad.test/path', 'https://a@bad.test', 'https://bad.test/?']) {
    expect(() => slackBrokerFromEnvironment({ XERXES_HUB_SLACK_BROKER_ORIGIN: bad })).toThrow('trusted HTTPS origin')
  }
  const s = await setupManaged()
  for (const bad of ['https://other.example.test/v1/slack/authorize', `${brokerOrigin}/untrusted`, `${brokerOrigin}/v1/slack/authorize#fragment`]) {
    s.live.browserUrl = bad
    expect(await s.client.fails('supervisor.slack.connect')).toBe('unavailable')
  }
  s.live.browserUrl = `${brokerOrigin}/v1/slack/authorize?id=test-flow`
  await s.client.ok('supervisor.slack.connect')
  s.live.result = Response.json({ ...ready, scope: `${scope},files:read` })
  expect(await s.client.fails('supervisor.slack.status')).toBe('unconfigured')
  expect(s.calls).not.toContain('https://slack.com/api/auth.test')
  await s.client.ok('supervisor.slack.connect')
  s.live.result = Response.json(ready)
  s.live.verifiedUser = 'UOTHER'
  expect(await s.client.fails('supervisor.slack.status')).toBe('unavailable')
  expect(await s.client.fails('supervisor.slack.confirm_connection', account)).toBe('not_found')
  s.live.verifiedUser = account.user_id
  await s.client.ok('supervisor.slack.connect')
  expect(await s.client.ok('supervisor.slack.status')).toMatchObject({ connection: null, pending_identity: account })
  s.identity.revoke({ principalId: 'owner', browserSessionId: 'bs-owner', reason: 'test' })
  expect(await s.client.fails('supervisor.slack.confirm_connection', account)).toBe('not_found')
  expect((await readdir(s.dir)).some(name => name.startsWith('slack-owner-'))).toBe(false)
})

test('managed redeem stops for denial and expiry, and respects broker Retry-After', async () => {
  const s = await setupManaged()
  for (const status of ['denied', 'expired']) {
    await s.client.ok('supervisor.slack.connect')
    s.live.result = Response.json({ status })
    expect(await s.client.ok('supervisor.slack.status')).toMatchObject({ connection: null, pending_identity: null })
  }
  await s.client.ok('supervisor.slack.connect')
  s.live.result = new Response(null, { status: 429, headers: { 'retry-after': '4' } })
  await s.client.ok('supervisor.slack.status')
  const count = s.calls.length
  s.live.result = Response.json(ready)
  s.live.now += 3999
  await s.client.ok('supervisor.slack.status')
  expect(s.calls).toHaveLength(count)
  s.live.now += 1
  expect(await s.client.ok('supervisor.slack.status')).toMatchObject({ pending_identity: account })
  s.live.now += 600_000
  expect(await s.client.ok('supervisor.slack.status')).toMatchObject({ pending_identity: null })
  expect(await s.client.fails('supervisor.slack.confirm_connection', account)).toBe('not_found')
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
