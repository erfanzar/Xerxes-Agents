// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, expect, test } from 'bun:test'
import { readdir, stat } from 'node:fs/promises'
import { join } from 'node:path'
import { SLACK_USER_SCOPES, slackAppManifest } from '../src/client/supervisorSlack.js'
import { SLACK_CLIENT_CHECK_CODE } from '../src/hub/slack/http.js'
import { createHubSlackIntegration, SLACK_CALLBACK_PATH } from '../src/hub/slack/integration.js'
import { USER_SCOPES } from '../src/hub/slack/oauth.js'
import { disposeAll, FakeIdentity, registerCleanup, standardUsers, startFixture, tempDir } from './hubHarness.js'

afterEach(disposeAll)
const origin = 'https://hub.example.test'
const redirect = `${origin}${SLACK_CALLBACK_PATH}`
const account = { team_id: 'TEAM1', user_id: 'UOWNER' }
const app = { client_id: '1234.5678', client_secret: 'abcdef0123456789' }
const userToken = 'xoxp-signed-in-test-only'
const scope = 'users:read,users:read.email,im:write,im:read,im:history,chat:write'

// A Hub with no Slack settings at all: the owner sets everything up from the app.
async function setup(dir?: string, publicOrigins: readonly string[] = [origin]) {
  const stateDir = dir ?? await tempDir()
  const identity = standardUsers(new FakeIdentity())
  const live = { secret: app.client_secret }
  const calls: string[] = []
  const integration = createHubSlackIntegration({
    databasePath: join(stateDir, 'hub.sqlite'), stateDir, identity, publicOrigins, isHubOwner: id => id === 'owner',
    fetch: async (url, init) => {
      const method = new URL(url).pathname.split('/').at(-1)!
      expect(init.redirect).toBe('error')
      if (method === 'oauth.v2.access') {
        const body = new URLSearchParams(String(init.body))
        calls.push(body.get('code') === SLACK_CLIENT_CHECK_CODE ? 'check' : 'exchange')
        expect(body.get('redirect_uri')).toBe(redirect)
        const expected = `Basic ${Buffer.from(`${app.client_id}:${live.secret}`).toString('base64')}`
        if (new Headers(init.headers).get('authorization') !== expected) return Response.json({ ok: false, error: 'bad_client_secret' })
        if (body.get('code') === SLACK_CLIENT_CHECK_CODE) return Response.json({ ok: false, error: 'invalid_code' })
        return Response.json({ ok: true, team: { id: 'TEAM1' }, authed_user: { id: 'UOWNER', token_type: 'user', access_token: userToken, scope } })
      }
      calls.push(method)
      if (method === 'auth.test') return Response.json({ ok: true, team_id: 'TEAM1', user_id: 'UOWNER' })
      throw new Error('Unexpected outbound request')
    },
  })
  registerCleanup(() => integration.close())
  const f = await startFixture({ server: { slack: integration } })
  const client = await f.client('owner')
  const signIn = async () => {
    const authorize = new URL(String((await client.ok('supervisor.slack.connect')).authorize_url))
    expect(authorize.origin + authorize.pathname).toBe('https://slack.com/oauth/v2/authorize')
    expect(authorize.searchParams.get('client_id')).toBe(app.client_id)
    expect(authorize.searchParams.get('redirect_uri')).toBe(redirect)
    expect(authorize.searchParams.get('user_scope')).toBe(USER_SCOPES.join(','))
    expect(authorize.searchParams.get('scope')).toBeNull()
    const state = authorize.searchParams.get('state')!
    const back = await integration.callback(new Request(`${redirect}?code=real-code&state=${encodeURIComponent(state)}`, { headers: { host: 'hub.example.test' } }))
    expect(back.status).toBe(200)
  }
  return { stateDir, identity, f, integration, live, calls, client, signIn }
}

test('the manifest asks for exactly the user scopes the Hub requests, with no bot', () => {
  expect([...SLACK_USER_SCOPES]).toEqual([...USER_SCOPES])
  const manifest = JSON.parse(slackAppManifest(redirect))
  expect(manifest.oauth_config).toEqual({ redirect_urls: [redirect], scopes: { user: [...USER_SCOPES] } })
  expect(manifest.features).toBeUndefined()
})

test('an unconfigured Hub: save the app once, then Sign in with Slack and confirm, all as the owner', async () => {
  const s = await setup()
  expect(await s.client.ok('supervisor.slack.status')).toEqual({ configured: true, browser_sign_in: false, app: 'none', redirect_url: redirect,
    connection: null, pending_identity: null, contacts: [] })
  expect(await s.client.fails('supervisor.slack.connect')).toBe('unconfigured')
  const saved = await s.client.ok('supervisor.slack.app.save', { client_id: ` ${app.client_id} `, client_secret: `${app.client_secret}\n` })
  expect(saved).toMatchObject({ browser_sign_in: true, app: 'saved', connection: null })
  expect(JSON.stringify(saved)).not.toContain(app.client_secret)
  expect(s.calls).toEqual(['check'])
  expect(((await stat(join(s.stateDir, 'slack-app.json'))).mode & 0o777)).toBe(0o600)
  await s.signIn()
  expect(s.calls).toEqual(['check', 'exchange', 'auth.test'])
  expect(await s.client.ok('supervisor.slack.status')).toMatchObject({ connection: null, pending_identity: account })
  expect(await s.client.fails('supervisor.slack.app.remove')).toBe('conflict')
  expect(await s.client.ok('supervisor.slack.confirm_connection', account)).toMatchObject({ connection: account, pending_identity: null })
  expect(s.integration.tools?.isOwner('owner')).toBe(true)
  expect(await s.client.fails('supervisor.slack.app.remove')).toBe('conflict')
  // Reconnecting after a disconnect is the same one click; nothing is copied again.
  await s.client.ok('supervisor.slack.disconnect', account)
  await s.signIn()
  expect(await s.client.ok('supervisor.slack.confirm_connection', account)).toMatchObject({ connection: account })
  const audits = JSON.stringify(s.identity.audits)
  expect(audits).not.toContain(app.client_secret)
  expect(audits).not.toContain(userToken)
})

test('the saved app survives a Hub restart and can be removed while disconnected', async () => {
  const first = await setup()
  await first.client.ok('supervisor.slack.app.save', app)
  await first.integration.close()
  const second = await setup(first.stateDir)
  expect(await second.client.ok('supervisor.slack.status')).toMatchObject({ browser_sign_in: true, app: 'saved' })
  expect(await second.client.ok('supervisor.slack.app.remove')).toMatchObject({ browser_sign_in: false, app: 'none' })
  expect(await readdir(second.stateDir)).not.toContain('slack-app.json')
  expect(await second.client.fails('supervisor.slack.connect')).toBe('unconfigured')
})

test('a wrong secret, bad input, non-owners and a Hub without HTTPS all fail closed and save nothing', async () => {
  const s = await setup()
  s.live.secret = 'the-real-secret'
  expect(await s.client.fails('supervisor.slack.app.save', app)).toBe('invalid_request')
  for (const bad of [{ client_id: '', client_secret: 'x' }, { client_id: '1.2', client_secret: 'has space' }, { ...app, extra: 1 }]) {
    expect(await s.client.fails('supervisor.slack.app.save', bad)).toBe('invalid_request')
  }
  for (const who of ['alice', 'robot']) expect(await (await s.f.client(who)).fails('supervisor.slack.app.save', app)).toBe('forbidden')
  expect(s.calls).toEqual(['check'])
  expect(await readdir(s.stateDir)).not.toContain('slack-app.json')
  expect(await s.client.ok('supervisor.slack.status')).toMatchObject({ browser_sign_in: false, app: 'none' })
  const plain = await setup(undefined, [])
  expect(await plain.client.ok('supervisor.slack.status')).toMatchObject({ browser_sign_in: false, app: 'none', redirect_url: null })
  expect(await plain.client.fails('supervisor.slack.app.save', app)).toBe('unconfigured')
  expect(plain.calls).toEqual([])
})
