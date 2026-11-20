// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, expect, test } from 'bun:test'
import { readdir } from 'node:fs/promises'
import { join } from 'node:path'
import { SLACK_USER_SCOPES, slackAppManifest } from '../src/client/supervisorSlack.js'
import { createHubSlackIntegration } from '../src/hub/slack/integration.js'
import { USER_SCOPES } from '../src/hub/slack/oauth.js'
import { disposeAll, FakeIdentity, registerCleanup, standardUsers, startFixture, tempDir } from './hubHarness.js'

afterEach(disposeAll)
const account = { team_id: 'TEAM1', user_id: 'UOWNER' }
const scope = 'users:read,users:read.email,im:write,im:read,im:history,chat:write'
const token = 'xoxp-own-app-test-only'

// A Hub with no Slack settings at all: the owner connects entirely from the app.
async function setup() {
  const dir = await tempDir()
  const identity = standardUsers(new FakeIdentity())
  const live = { scopes: scope, bot: false, ok: true }
  const calls: string[] = []
  const integration = createHubSlackIntegration({
    databasePath: join(dir, 'hub.sqlite'), stateDir: dir, identity, publicOrigins: ['https://hub.example.test'],
    isHubOwner: id => id === 'owner',
    fetch: async (url, init) => {
      calls.push(url)
      expect(init.redirect).toBe('error')
      if (url !== 'https://slack.com/api/auth.test') throw new Error('Unexpected outbound request')
      expect(new Headers(init.headers).get('authorization')).toBe(`Bearer ${token}`)
      if (!live.ok) return Response.json({ ok: false, error: 'invalid_auth' })
      return Response.json({ ok: true, team_id: account.team_id, user_id: account.user_id, ...(live.bot ? { bot_id: 'B01' } : {}) },
        { headers: { 'x-oauth-scopes': live.scopes } })
    },
  })
  registerCleanup(() => integration.close())
  const f = await startFixture({ server: { slack: integration } })
  return { dir, identity, f, integration, live, calls, client: await f.client('owner') }
}

const stored = async (dir: string): Promise<boolean> => (await readdir(dir)).some(name => name.startsWith('slack-owner-'))

test('the in-app manifest asks for exactly the scopes the Hub verifies, as a user app with no bot', () => {
  expect([...SLACK_USER_SCOPES]).toEqual([...USER_SCOPES])
  const manifest = JSON.parse(slackAppManifest())
  expect(manifest.oauth_config.scopes).toEqual({ user: [...USER_SCOPES] })
  expect(manifest.features).toBeUndefined()
  expect(manifest.settings.token_rotation_enabled).toBe(false)
})

test('an unconfigured Hub connects a pasted user token only after the owner confirms the verified account', async () => {
  const s = await setup()
  expect(await s.client.ok('supervisor.slack.status')).toEqual({ configured: true, browser_sign_in: false, connection: null, pending_identity: null, contacts: [] })
  expect(await s.client.fails('supervisor.slack.connect')).toBe('unconfigured')
  const pending = await s.client.ok('supervisor.slack.connect_token', { token: `  ${token}\n` })
  expect(pending).toEqual({ configured: true, browser_sign_in: false, connection: null, pending_identity: account, contacts: [] })
  expect(JSON.stringify(pending)).not.toContain(token)
  expect(s.calls).toEqual(['https://slack.com/api/auth.test'])
  expect(await stored(s.dir)).toBe(false)
  expect(s.integration.tools?.isOwner('owner')).toBe(false)
  expect(await s.client.fails('supervisor.slack.connect_token', { token })).toBe('conflict')
  expect(await s.client.fails('supervisor.slack.confirm_connection', { ...account, user_id: 'UOTHER' })).toBe('not_found')
  expect(await s.client.ok('supervisor.slack.confirm_connection', account)).toMatchObject({ connection: account, pending_identity: null })
  expect(await stored(s.dir)).toBe(true)
  expect(s.integration.tools?.isOwner('owner')).toBe(true)
  expect(await s.client.fails('supervisor.slack.connect_token', { token })).toBe('conflict')
  expect(JSON.stringify(s.identity.audits)).not.toContain(token)
  expect(await s.client.ok('supervisor.slack.disconnect', account)).toMatchObject({ connection: null })
  expect(await stored(s.dir)).toBe(true)
})

test('bot tokens, wrong scopes, rejected tokens and non-owners fail closed and store nothing', async () => {
  const s = await setup()
  for (const bad of ['xoxb-bot-token', 'not-a-token', '', 'xoxp-']) {
    expect(await s.client.fails('supervisor.slack.connect_token', { token: bad })).toBe('invalid_request')
  }
  expect(await s.client.fails('supervisor.slack.connect_token', { token, extra: true })).toBe('invalid_request')
  expect(s.calls).toEqual([])
  for (const who of ['alice', 'robot']) expect(await (await s.f.client(who)).fails('supervisor.slack.connect_token', { token })).toBe('forbidden')
  expect(s.calls).toEqual([])
  s.live.bot = true
  expect(await s.client.fails('supervisor.slack.connect_token', { token })).toBe('invalid_request')
  s.live.bot = false
  for (const scopes of ['users:read,im:write,im:read,im:history,chat:write', `${scope},files:read`, '']) {
    s.live.scopes = scopes
    expect(await s.client.fails('supervisor.slack.connect_token', { token })).toBe('invalid_request')
  }
  s.live.scopes = scope
  s.live.ok = false
  expect(await s.client.fails('supervisor.slack.connect_token', { token })).toBe('invalid_request')
  expect(await s.client.ok('supervisor.slack.status')).toMatchObject({ connection: null, pending_identity: null })
  expect(await stored(s.dir)).toBe(false)
  // A failed attempt never leaves a stuck pending flow: a good token still connects.
  s.live.ok = true
  expect(await s.client.ok('supervisor.slack.connect_token', { token })).toMatchObject({ pending_identity: account })
})
