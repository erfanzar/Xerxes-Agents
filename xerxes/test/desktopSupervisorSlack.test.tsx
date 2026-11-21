// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { renderToStaticMarkup } from 'react-dom/server'

import { HubRequestError } from '../src/client/hubConnection.js'
import { initialHubState, type HubState } from '../src/client/hubStore.js'
import { createSupervisorSlackClient, slackCreateAppUrl, slackUnavailable, SlackBrokerUnavailable, type SlackContact } from '../src/client/supervisorSlack.js'
import { SupervisorView } from '../src/desktop/renderer/SupervisorPage.js'
import { SlackAppSetup, SlackDirectMessage } from '../src/desktop/renderer/SupervisorSlack.js'
import type { HubSlackMethod } from '../src/hub/protocol.js'

const contact = { team_id: 'T01TEST', user_id: 'U01PEER', display_name: 'Jordan', mode: 'draft' } satisfies SlackContact
const account = { team_id: contact.team_id, user_id: 'U01OWNER' }
const status = { configured: true, connection: account, contacts: [contact] }
const dest = { team_id: contact.team_id, user_id: contact.user_id }
const draft = { id: 'draft-1', text: 'Hello, Jordan.\n  Reviewed text.' }
const dm = { ...dest, messages: [], drafts: [draft] }

function fake(replies: Partial<Record<HubSlackMethod, unknown>> = {}, trustedBrokerOrigin?: string) {
  const calls: [HubSlackMethod, Readonly<Record<string, unknown>>][] = []
  const client = createSupervisorSlackClient({ ...(trustedBrokerOrigin === undefined ? {} : { trustedBrokerOrigin }), allowed: () => true, request: async (method, params) => {
    calls.push([method, params])
    if (!Object.hasOwn(replies, method)) throw new Error(`Unexpected RPC: ${method}`)
    return replies[method]
  } })
  return { client, calls }
}

test('Slack controls use exact Slack IDs for confirmation, removal, mode and account disconnect', async () => {
  const { client, calls } = fake({
    'supervisor.slack.status': status, 'supervisor.slack.contact.lookup': { identity: contact },
    'supervisor.slack.contact.confirm': status, 'supervisor.slack.contact.mode': { ...status, contacts: [{ ...contact, mode: 'auto' }] },
    'supervisor.slack.contact.remove': { ...status, contacts: [] },
    'supervisor.slack.disconnect': { ...status, connection: null, contacts: [] },
  })
  expect(await client.status()).toEqual(status)
  const identity = await client.lookup(dest)
  await client.confirm(identity)
  expect((await client.mode(contact, 'auto')).contacts[0]?.mode).toBe('auto')
  expect((await client.remove(contact)).contacts).toEqual([])
  expect((await client.disconnect(account)).connection).toBeNull()
  expect(calls).toEqual([
    ['supervisor.slack.status', {}], ['supervisor.slack.contact.lookup', dest], ['supervisor.slack.contact.confirm', dest],
    ['supervisor.slack.contact.mode', { ...dest, mode: 'auto' }], ['supervisor.slack.contact.remove', dest], ['supervisor.slack.disconnect', account],
  ])
})

test('Slack rejects mismatched lookup and DM identities, including send acknowledgements', async () => {
  const other = { ...dest, user_id: 'U01OTHER' }
  const { client, calls } = fake({
    'supervisor.slack.contact.lookup': { identity: other },
    'supervisor.slack.dm.open': { ...dm, ...other },
    'supervisor.slack.draft.send': { ...dm, ...other, drafts: [] },
  })
  await expect(client.lookup(dest)).rejects.toThrow('Invalid Slack response')
  await expect(client.dm(contact)).rejects.toThrow('Invalid Slack response')
  await expect(client.send(contact, draft)).rejects.toThrow('Invalid Slack response')
  expect(calls.at(-1)).toEqual(['supervisor.slack.draft.send', { ...dest, draft_id: draft.id, text: draft.text }])
  expect(calls).toHaveLength(3)
})

test('review sends exactly the selected pending draft; a response that still lists it is ambiguous', async () => {
  const responses = { 'supervisor.slack.dm.open': dm, 'supervisor.slack.draft.send': { ...dm, drafts: [] as typeof dm.drafts } }
  const { client, calls } = fake(responses)
  const selected = (await client.dm(contact)).drafts[0]!
  expect((await client.send(contact, selected)).drafts).toEqual([])
  responses['supervisor.slack.draft.send'] = dm
  await expect(client.send(contact, selected)).rejects.toThrow('Invalid Slack response')
  expect(calls.filter(([method]) => method === 'supervisor.slack.draft.send')).toHaveLength(2)
})

test('OAuth accepts only the Slack authorize endpoint with an OAuth state', async () => {
  const replies = { 'supervisor.slack.connect': { authorize_url: 'https://slack.com/oauth/v2/authorize?state=fake' } }
  const { client } = fake(replies)
  expect(await client.connect()).toEqual({ url: replies['supervisor.slack.connect'].authorize_url })
  for (const url of ['https://slack.com.evil.test/oauth/v2/authorize?state=fake', 'https://slack.com/redirect?state=fake',
    'https://slack.com/oauth/v2/authorize', 'https://fake@slack.com/oauth/v2/authorize?state=fake',
    'https://slack.com/oauth/v2/authorize?state=fake&state=other', 'http://slack.com/oauth/v2/authorize?state=fake']) {
    replies['supervisor.slack.connect'].authorize_url = url
    await expect(client.connect()).rejects.toThrow()
  }
})

test('managed Slack requires locally pinned HTTPS broker and keeps the pairing code separate', async () => {
  const trusted = 'https://broker.example.test'
  const reply = { authorize_url: `${trusted}/v1/slack/authorize?id=fixture`, pairing_code: 'ABCDEFGHIJKLMNOPQRSTUV',
    broker_origin: trusted, expires_at: 1_800_000_000_000, access_token: 'never-returned' }
  const replies = { 'supervisor.slack.connect': reply }
  await expect(fake(replies).client.connect()).rejects.toBeInstanceOf(SlackBrokerUnavailable)
  const { client } = fake(replies, trusted)
  expect(await client.connect()).toEqual({ url: reply.authorize_url, pairingCode: reply.pairing_code, expiresAt: reply.expires_at })
  for (const url of [`${trusted}.evil.test/v1/slack/authorize?id=fixture`, `${trusted}/redirect?id=fixture`,
    `${trusted}/v1/slack/authorize?id=fixture&redirect=https://evil.test`,
    `${trusted}/v1/slack/authorize?id=fixture&id=extra`, `${trusted}/v1/slack/authorize?id=${reply.pairing_code}`]) {
    reply.authorize_url = url
    await expect(client.connect()).rejects.toThrow()
  }
})

test('browser authorization stays pending and confirmation sends only the two reviewed IDs', async () => {
  const pending = { configured: true, connection: null, contacts: [], pending_identity: { ...account, access_token: 'not-for-desktop' } }
  const replies = { 'supervisor.slack.status': pending, 'supervisor.slack.confirm_connection': status }
  const { client, calls } = fake(replies)
  const afterBrowser = await client.status()
  expect(afterBrowser.connection).toBeNull()
  expect(afterBrowser.pending_identity).toEqual(account)
  expect(calls).toEqual([['supervisor.slack.status', {}]])
  if (!afterBrowser.pending_identity) throw new Error('Missing pending identity')
  expect(await client.confirmConnection(afterBrowser.pending_identity)).toEqual(status)
  expect(calls.at(-1)).toEqual(['supervisor.slack.confirm_connection', account])
  replies['supervisor.slack.confirm_connection'] = { ...status, connection: dest }
  await expect(client.confirmConnection(account)).rejects.toThrow('Invalid Slack response')
  pending.pending_identity.team_id = 'https://example.test'
  await expect(client.status()).rejects.toThrow('Invalid Slack response')
})

test('Slack denial and expiry do not expose broker details', async () => {
  for (const oauth_status of ['denied', 'expired']) {
    const { client } = fake({ 'supervisor.slack.status': {
      configured: true, connection: null, contacts: [], pending_identity: null, oauth_status, error: 'private broker details',
    } })
    expect(await client.status()).toEqual({ configured: true, connection: null, contacts: [], oauth_status })
  }
})

test('non-owners do not call Slack and owner access loss discards an in-flight private result', async () => {
  let allowed = false
  const response = Promise.withResolvers<unknown>()
  let calls = 0
  const client = createSupervisorSlackClient({ allowed: () => allowed, request: () => { calls++; return response.promise } })
  await expect(client.dm(contact)).rejects.toThrow('Hub owner')
  expect(calls).toBe(0)
  allowed = true
  const request = client.dm(contact)
  allowed = false
  response.resolve(dm)
  await expect(request).rejects.toThrow('Owner access or connection changed')
  expect(calls).toBe(1)
  expect(slackUnavailable(new HubRequestError('invalid_request', 'unknown hub method "supervisor.slack.status"'))).toBe(true)
  expect(slackUnavailable(new HubRequestError('forbidden', 'owner only'))).toBe(false)
})

test('Slack navigation is owner-only; the isolated DM does not have a shared composer or a quick-send affordance', () => {
  const { client, calls } = fake()
  const seed = initialHubState()
  const view = (owner: boolean): string => {
    const state: HubState = { ...seed, connection: { state: 'live' }, autopilot: { ...seed.autopilot, canEdit: owner } }
    return renderToStaticMarkup(<SupervisorView state={state} slack={{ client, openOAuth: async () => false }} hubLabel="fake Hub" now={0}
      onSend={async () => false} onRespond={() => {}} onReopen={() => {}} onOpenSettings={() => {}}
      chooserOpen={false} onChooserChange={() => {}} onChooserShown={() => {}} onOptions={() => {}} onSaveModel={async () => false} />)
  }
  expect(view(true)).toContain('>Slack</button>')
  expect(view(false)).not.toContain('>Slack</button>')
  const html = renderToStaticMarkup(<SlackDirectMessage contact={contact} client={client} onBack={() => {}} />)
  expect(html).toContain('Private Slack direct message')
  expect(html).toContain(`<code>${dest.team_id}</code>`)
  expect(html).toContain(`<code>${dest.user_id}</code>`)
  expect(html).not.toContain('Message to XERXES')
  expect(html).not.toContain('Send to Slack')
  expect(calls).toEqual([])
})

const redirect = 'https://hub.example.test/api/hub/v1/slack/oauth/callback'

test('saving the Slack app sends only the trimmed ID and secret and needs sign-in to be ready afterwards', async () => {
  const ready = { configured: true, browser_sign_in: true, app: 'saved', redirect_url: redirect, connection: null, contacts: [] }
  const replies: Partial<Record<HubSlackMethod, unknown>> = { 'supervisor.slack.app.save': ready }
  const { client, calls } = fake(replies)
  for (const bad of [{ clientId: '', clientSecret: 's' }, { clientId: '1.2', clientSecret: 'has space' }]) {
    await expect(client.saveApp(bad)).rejects.toThrow('Client ID')
  }
  expect(calls).toEqual([])
  expect(await client.saveApp({ clientId: ' 123.456 ', clientSecret: ' abcdef0123 ' })).toEqual(ready)
  expect(calls).toEqual([['supervisor.slack.app.save', { client_id: '123.456', client_secret: 'abcdef0123' }]])
  replies['supervisor.slack.app.save'] = { ...ready, browser_sign_in: false }
  await expect(client.saveApp({ clientId: '123.456', clientSecret: 'abcdef0123' })).rejects.toThrow('Invalid Slack response')
})

test('status accepts only the Hub Slack callback as the manifest redirect', async () => {
  const replies: Partial<Record<HubSlackMethod, unknown>> = { 'supervisor.slack.status': { configured: true, app: 'none', redirect_url: redirect, connection: null, contacts: [] } }
  const { client } = fake(replies)
  expect((await client.status()).redirect_url).toBe(redirect)
  for (const bad of ['http://hub.example.test/api/hub/v1/slack/oauth/callback', 'https://hub.example.test/evil', `${redirect}?x=1`]) {
    replies['supervisor.slack.status'] = { configured: true, app: 'none', redirect_url: bad, connection: null, contacts: [] }
    await expect(client.status()).rejects.toThrow('Invalid Slack response')
  }
})

test('app setup is two in-app steps with a pre-filled user-scope manifest that points back at this Hub', () => {
  const html = renderToStaticMarkup(<SlackAppSetup redirectUrl={redirect} disabled={false} openApp={() => {}} onSave={() => {}} />)
  expect(html).toContain('Create XERXES app in Slack')
  expect(html).toContain('Copy manifest')
  expect(html).toContain('Client ID')
  expect(html).toContain('type="password"')
  expect(html).not.toContain('xoxp-')
  expect(html.match(/<li>/g)).toHaveLength(2)
  const url = new URL(slackCreateAppUrl(redirect))
  expect(url.origin + url.pathname).toBe('https://api.slack.com/apps')
  const manifest = JSON.parse(url.searchParams.get('manifest_json') ?? '{}')
  expect(manifest.oauth_config.redirect_urls).toEqual([redirect])
  expect(manifest.oauth_config.scopes.user).toContain('chat:write')
  expect(manifest.oauth_config.scopes.bot).toBeUndefined()
  expect(manifest.features).toBeUndefined()
  expect(manifest.settings.socket_mode_enabled).toBe(false)
})
