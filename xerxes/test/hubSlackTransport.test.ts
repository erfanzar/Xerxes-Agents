// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, describe, expect, test } from 'bun:test'
import { chmod, lstat, mkdtemp, readFile, readdir, rm, symlink, unlink, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { HubSlackTransport, type HubSlackTransportOptions } from '../src/hub/slack/transport.js'

const account = { teamId: 'T12345678', userId: 'U12345678' }
const peer = { ...account, peerId: 'W87654321' }
const dm = { ...peer, channelId: 'D012AB345' }
const channel = { id: dm.channelId, is_im: true, user: peer.peerId }
const oauth = { clientId: '123.456', clientSecret: 'fixture-secret', redirectUri: 'https://hub.example.test/slack/callback' }
const token = 'xoxp-fake-fixture'
const ts = '1710000000.000001'
const scope = 'users:read,users:read.email,im:write,im:read,im:history,chat:write'
const grant = { ok: true, team: { id: account.teamId }, authed_user: { id: account.userId, access_token: token, token_type: 'user', scope } }
const auth = { ok: true, team_id: account.teamId, user_id: account.userId, url: 'https://workspace.example.test/' }
const dirs: string[] = []
afterEach(async () => {
  await Promise.all(dirs.splice(0).map(dir => rm(dir, { recursive: true, force: true })))
})

async function fixture(options: Partial<Omit<HubSlackTransportOptions, 'stateDir' | 'fetch'>> = {}) {
  const stateDir = await mkdtemp(join(tmpdir(), 'xerxes-slack-'))
  dirs.push(stateDir)
  const calls: { url: URL; init: RequestInit }[] = []
  const queue: { method: string; respond: () => Response | Promise<Response> }[] = []
  const transport = new HubSlackTransport({ stateDir, oauth, ...options, fetch: async (url, init) => {
    calls.push({ url: new URL(url), init })
    const next = queue.shift()
    if (new URL(url).origin !== 'https://slack.com' || new URL(url).pathname !== `/api/${next?.method}` || !next) {
      throw new Error('Unexpected offline fixture request')
    }
    return next.respond()
  } })
  const reply = (method: string, json: unknown): void => { queue.push({ method, respond: () => Response.json(json) }) }
  const start = (expected = account): string => new URL(transport.beginOAuth('owner/session', expected).authorizeUrl).searchParams.get('state') ?? ''
  const complete = (state: string) => transport.completeOAuth('owner/session', { state, code: 'fixture-code' })
  const connect = async () => { reply('oauth.v2.access', grant); reply('auth.test', auth); return complete(start()) }
  return { stateDir, transport, calls, reply, start, complete, connect, queue, credential: join(stateDir, 'slack-credentials', `${account.teamId}-${account.userId}.json`) }
}

describe('Hub Slack OAuth and protected credentials', () => {
  test('requests only user scopes and exchanges once using fixed URLs, Basic and Bearer', async () => {
    const f = await fixture()
    const start = f.transport.beginOAuth('owner/session', account)
    const url = new URL(start.authorizeUrl)
    const state = url.searchParams.get('state') ?? ''
    expect(url.origin + url.pathname).toBe('https://slack.com/oauth/v2/authorize')
    expect(url.searchParams.get('client_id')).toBe(oauth.clientId)
    expect(url.searchParams.get('redirect_uri')).toBe(oauth.redirectUri)
    expect(url.searchParams.get('user_scope')).toBe(scope)
    expect(url.searchParams.has('scope')).toBe(false)
    expect(state).toMatch(/^[A-Za-z0-9_-]{43}$/u)
    expect(f.start()).not.toBe(state)
    await expect(f.transport.completeOAuth('another/session', { state, code: 'fixture-code' })).rejects.toMatchObject({ code: 'invalid_state' })
    expect(f.calls).toHaveLength(0)
    f.reply('oauth.v2.access', grant)
    f.reply('auth.test', auth)
    const attempts = await Promise.allSettled([f.complete(state), f.complete(state)])
    expect(attempts[0]).toEqual({ status: 'fulfilled', value: account })
    expect(attempts[1]).toMatchObject({ status: 'rejected', reason: { code: 'invalid_state' } })
    const [exchange, verify] = f.calls
    expect(f.calls).toHaveLength(2)
    expect(exchange?.init).toMatchObject({ method: 'POST', redirect: 'error' })
    expect(exchange?.url.href).toBe('https://slack.com/api/oauth.v2.access')
    expect(new Headers(exchange?.init.headers).get('authorization')).toBe(`Basic ${Buffer.from(`${oauth.clientId}:${oauth.clientSecret}`).toString('base64')}`)
    expect(Object.fromEntries(new URLSearchParams(String(exchange?.init.body)))).toEqual({ code: 'fixture-code', redirect_uri: oauth.redirectUri })
    expect(new Headers(verify?.init.headers).get('authorization')).toBe(`Bearer ${token}`)
    expect(verify?.url.href).toBe('https://slack.com/api/auth.test')
    expect((await lstat(f.credential)).mode & 0o777).toBe(0o600)
    expect((await lstat(join(f.stateDir, 'slack-credentials'))).mode & 0o777).toBe(0o700)
    expect(await readdir(join(f.stateDir, 'slack-credentials'))).toEqual([`${account.teamId}-${account.userId}.json`])
    const reopened = new HubSlackTransport({ stateDir: f.stateDir, fetch: async () => Response.json(auth) })
    await expect(reopened.authTest(account)).resolves.toEqual(account)
    await expect(reopened.authTest({ ...account, teamId: 'TOTHER' })).rejects.toMatchObject({ code: 'not_connected' })
    await reopened.disconnect(account)
    await expect(reopened.authTest(account)).rejects.toMatchObject({ code: 'not_connected' })
  })

  test('expires state, consumes denial and malformed code, and refuses insecure redirects before any I/O', async () => {
    let now = 1
    const f = await fixture({ now: () => now })
    const expired = f.start()
    now += 600_000
    await expect(f.complete(expired)).rejects.toMatchObject({ code: 'invalid_state' })
    const state = f.start()
    await expect(f.transport.completeOAuth('owner/session', { state, error: 'private Slack detail' })).rejects.toMatchObject({ code: 'oauth_denied' })
    await expect(f.complete(state)).rejects.toMatchObject({ code: 'invalid_state' })
    const malformed = f.start()
    await expect(f.transport.completeOAuth('owner/session', { state: malformed, code: 'code\n' })).rejects.toMatchObject({ code: 'invalid_request' })
    await expect(f.complete(malformed)).rejects.toMatchObject({ code: 'invalid_state' })
    expect(f.calls).toHaveLength(0)
    expect(() => new HubSlackTransport({ stateDir: f.stateDir, oauth: { ...oauth, redirectUri: 'http://hub.example.test' } })).toThrow('Invalid Slack request.')
  })

  test('does not save mismatched identities, bot or over-scoped grants, or rotating tokens', async () => {
    const f = await fixture()
    for (const data of [
      { ...grant, access_token: 'xoxb-fake' },
      { ...grant, is_enterprise_install: true },
      { ...grant, authed_user: { ...grant.authed_user, scope: `${scope},channels:history` } },
      { ...grant, authed_user: { ...grant.authed_user, scope: 'chat:write' } },
      { ...grant, authed_user: { ...grant.authed_user, refresh_token: 'fake' } },
      { ...grant, authed_user: { ...grant.authed_user, access_token: 'xoxb-fake' } },
    ]) {
      f.reply('oauth.v2.access', data)
      await expect(f.complete(f.start())).rejects.toMatchObject({ code: 'oauth_unsupported' })
    }
    f.reply('oauth.v2.access', grant)
    await expect(f.complete(f.start({ ...account, teamId: 'TOTHER' }))).rejects.toMatchObject({ code: 'identity_mismatch' })
    f.reply('oauth.v2.access', grant)
    f.reply('auth.test', { ...auth, user_id: peer.peerId })
    await expect(f.complete(f.start())).rejects.toMatchObject({ code: 'identity_mismatch' })
    await expect(f.transport.authTest(account)).rejects.toMatchObject({ code: 'not_connected' })
  })

  test('refuses permissive credential files and symlinks without reading or overwriting their target', async () => {
    const f = await fixture()
    await f.connect()
    const original = await readFile(f.credential, 'utf8')
    await chmod(f.credential, 0o644)
    await expect(f.transport.authTest(account)).rejects.toMatchObject({ code: 'storage_error' })
    await unlink(f.credential)
    const target = join(f.stateDir, 'unrelated')
    await writeFile(target, original, { mode: 0o600 })
    await symlink(target, f.credential)
    await expect(f.transport.authTest(account)).rejects.toMatchObject({ code: 'storage_error' })
    await expect(f.connect()).rejects.toMatchObject({ code: 'storage_error' })
    expect(await readFile(target, 'utf8')).toBe(original)
    expect((await lstat(f.credential)).isSymbolicLink()).toBe(true)
    expect(f.calls).toHaveLength(4)
  })
})

describe('Hub Slack approved-peer HTTP transport', () => {
  test('looks up exact IDs and email, projects only ID/name, and rejects mismatches', async () => {
    const f = await fixture()
    await f.connect()
    const user = { id: peer.peerId, team_id: account.teamId, profile: { display_name: '  Example Peer ', email: 'peer@example.test', phone: 'private' } }
    f.reply('users.info', { ok: true, user })
    await expect(f.transport.lookupProfile(account, { userId: peer.peerId })).resolves.toEqual({ userId: peer.peerId, displayName: 'Example Peer' })
    f.reply('users.lookupByEmail', { ok: true, user })
    await expect(f.transport.lookupProfile(account, { email: 'Peer@Example.TEST' })).resolves.toEqual({ userId: peer.peerId, displayName: 'Example Peer' })
    expect(f.calls.at(-1)?.url.href).toBe('https://slack.com/api/users.lookupByEmail?email=peer%40example.test')
    expect(f.calls.at(-1)?.init.method).toBe('GET')
    f.reply('users.info', { ok: true, user: { ...user, id: account.userId } })
    await expect(f.transport.lookupProfile(account, { userId: peer.peerId })).rejects.toMatchObject({ code: 'not_found' })
    f.reply('users.lookupByEmail', { ok: true, user })
    await expect(f.transport.lookupProfile(account, { email: 'other@example.test' })).rejects.toMatchObject({ code: 'not_found' })
    f.reply('users.info', { ok: false, error: 'user_not_found' })
    await expect(f.transport.lookupProfile(account, { userId: 'UMISSING' })).rejects.toMatchObject({ code: 'not_found' })
    f.reply('users.lookupByEmail', { ok: false, error: 'users_not_found' })
    await expect(f.transport.lookupProfile(account, { email: 'missing@example.test' })).rejects.toMatchObject({ code: 'not_found' })
    await expect(f.transport.lookupProfile(account, { userId: 'example peer' })).rejects.toMatchObject({ code: 'invalid_request' })
    expect(f.calls).toHaveLength(8)
  })

  test('opens exactly one peer, verifies matching DM, reads bounded human text, and sends non-unfurling text', async () => {
    const approvals: unknown[] = []
    const f = await fixture({ approvePeer: (value, op) => { approvals.push([value, op]); return true } })
    await f.connect()
    f.reply('conversations.open', { ok: true, channel })
    await expect(f.transport.openPeer(peer)).resolves.toEqual(dm)
    expect(JSON.parse(String(f.calls.at(-1)?.init.body))).toEqual({ users: peer.peerId, return_im: true })
    f.reply('conversations.info', { ok: true, channel })
    await expect(f.transport.verifyDm(dm)).resolves.toEqual(dm)
    f.reply('conversations.info', { ok: true, channel })
    f.reply('conversations.history', { ok: true, has_more: true, messages: [
      { type: 'message', user: peer.peerId, ts, text: 'Hello', files: ['not exposed'] },
      { type: 'message', user: 'UOTHER', ts, text: 'not exposed' },
      { type: 'message', user: peer.peerId, ts, text: 'not exposed', subtype: 'bot_message' },
      { type: 'message', user: account.userId, ts, text: 'a'.repeat(4_001) },
    ] })
    await expect(f.transport.readDm(dm, 4)).resolves.toEqual([{ ts, userId: peer.peerId, text: 'Hello' }, { ts, userId: account.userId, text: 'a'.repeat(4_000) }])
    expect(f.calls.at(-1)?.url.href).toBe(`https://slack.com/api/conversations.history?channel=${dm.channelId}&limit=4`)
    f.reply('conversations.info', { ok: true, channel })
    f.reply('chat.postMessage', { ok: true, channel: dm.channelId, ts, message: { text: 'not exposed' } })
    await expect(f.transport.sendDm(dm, '<!here> & hi')).resolves.toEqual({ channelId: dm.channelId, ts })
    expect(JSON.parse(String(f.calls.at(-1)?.init.body))).toEqual({
      channel: dm.channelId, text: '&lt;!here&gt; &amp; hi', mrkdwn: false, parse: 'none', unfurl_links: false, unfurl_media: false,
    })
    expect(approvals).toEqual(['open', 'verify', 'read', 'read', 'send', 'send'].map(op => [peer, op]))
    for (const call of f.calls.slice(1)) {
      expect(call.init.redirect).toBe('error')
      expect(new Headers(call.init.headers).get('authorization')).toBe(`Bearer ${token}`)
      expect(call.url.href).not.toContain(token)
    }
  })

  test('fails closed on unapproved or substituted peers, channel types and IDs, before content I/O', async () => {
    const noPolicy = await fixture()
    await expect(noPolicy.transport.openPeer(peer)).rejects.toMatchObject({ code: 'not_approved' })
    expect(noPolicy.calls).toHaveLength(0)
    const f = await fixture({ approvePeer: candidate => candidate.peerId === peer.peerId })
    await f.connect()
    await expect(f.transport.sendDm({ ...dm, peerId: 'UOTHER' }, 'hello')).rejects.toMatchObject({ code: 'not_approved' })
    await expect(f.transport.openPeer({ ...peer, peerId: account.userId })).rejects.toMatchObject({ code: 'invalid_request' })
    await expect(f.transport.readDm(dm, 16)).rejects.toMatchObject({ code: 'invalid_request' })
    for (const wrong of [
      { ...channel, user: 'UOTHER' }, { ...channel, is_im: false }, { ...channel, is_mpim: true },
      { ...channel, is_ext_shared: true }, { ...channel, is_channel: true }, { ...channel, id: 'DOTHER' },
    ]) {
      f.reply('conversations.info', { ok: true, channel: wrong })
      await expect(f.transport.sendDm(dm, 'hello')).rejects.toMatchObject({ code: 'not_dm' })
    }
    f.reply('conversations.open', { ok: true, channel: { id: dm.channelId } })
    await expect(f.transport.openPeer(peer)).rejects.toMatchObject({ code: 'not_dm' })
    expect(f.calls.some(call => /history|postMessage/u.test(call.url.pathname))).toBe(false)
  })

  test('stops after verification if approval is revoked or the account is disconnected', async () => {
    let approved = true
    const f = await fixture({ approvePeer: () => approved })
    await f.connect()
    f.queue.push({ method: 'conversations.info', respond: () => { approved = false; return Response.json({ ok: true, channel }) } })
    await expect(f.transport.readDm(dm)).rejects.toMatchObject({ code: 'not_approved' })
    approved = true
    f.queue.push({ method: 'conversations.info', respond: async () => { await f.transport.disconnect(account); return Response.json({ ok: true, channel }) } })
    await expect(f.transport.sendDm(dm, 'hello')).rejects.toMatchObject({ code: 'not_connected' })
    expect(f.calls.map(call => call.url.pathname)).toEqual(['/api/oauth.v2.access', '/api/auth.test', '/api/conversations.info', '/api/conversations.info'])
  })

  test('sanitizes provider failures, blocks redirects, and never retries an uncertain send', async () => {
    const f = await fixture({ approvePeer: () => true })
    await f.connect()
    f.queue.push({ method: 'auth.test', respond: () => Response.redirect('https://not-slack.example.test') })
    await expect(f.transport.authTest(account)).rejects.toMatchObject({ code: 'slack_rejected' })
    f.queue.push({ method: 'auth.test', respond: () => new Response(token, { status: 429, headers: { 'retry-after': '30' } }) })
    await expect(f.transport.authTest(account)).rejects.toMatchObject({ code: 'rate_limited', retryAfterSeconds: 30 })
    f.queue.push({ method: 'auth.test', respond: () => new Response(token) })
    await expect(f.transport.authTest(account)).rejects.toMatchObject({ code: 'invalid_response' })
    f.reply('conversations.info', { ok: true, channel })
    f.queue.push({ method: 'chat.postMessage', respond: () => { throw new Error(`sensitive ${token}`) } })
    await expect(f.transport.sendDm(dm, 'hello')).rejects.toMatchObject({ code: 'network_error', message: 'Slack could not be reached.' })
    expect(f.calls.filter(call => call.url.pathname.endsWith('/chat.postMessage'))).toHaveLength(1)
  })

  test('propagates cancellation and bounds a stalled response', async () => {
    const f = await fixture({ timeoutMs: 100 })
    await f.connect()
    const aborter = new AbortController()
    const pending = f.transport.authTest(account, aborter.signal)
    aborter.abort(new Error(token))
    await expect(pending).rejects.toMatchObject({ code: 'aborted', message: 'Slack request was cancelled.' })
    expect(f.calls).toHaveLength(2)
    f.queue.push({ method: 'auth.test', respond: () => new Response(new ReadableStream({ start() {} })) })
    await expect(f.transport.authTest(account)).rejects.toMatchObject({ code: 'timeout' })
    expect(f.calls.at(-1)?.init.signal?.aborted).toBe(true)
  })
})
