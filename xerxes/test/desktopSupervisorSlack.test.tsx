// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { renderToStaticMarkup } from 'react-dom/server'

import { HubRequestError } from '../src/client/hubConnection.js'
import { initialHubState, type HubState } from '../src/client/hubStore.js'
import { createSupervisorSlackClient, slackUnavailable, type SlackContact } from '../src/client/supervisorSlack.js'
import { SupervisorView } from '../src/desktop/renderer/SupervisorPage.js'
import { SlackDirectMessage } from '../src/desktop/renderer/SupervisorSlack.js'
import type { HubSlackMethod } from '../src/hub/protocol.js'

const contact = { team_id: 'T01TEST', user_id: 'U01PEER', display_name: 'Jordan', mode: 'draft' } satisfies SlackContact
const account = { team_id: contact.team_id, user_id: 'U01OWNER' }
const status = { configured: true, connection: account, contacts: [contact] }
const dest = { team_id: contact.team_id, user_id: contact.user_id }
const draft = { id: 'draft-1', text: 'Hello, Jordan.\n  Reviewed text.' }
const dm = { ...dest, messages: [], drafts: [draft] }

function fake(replies: Partial<Record<HubSlackMethod, unknown>> = {}) {
  const calls: [HubSlackMethod, Readonly<Record<string, unknown>>][] = []
  const client = createSupervisorSlackClient({ allowed: () => true, request: async (method, params) => {
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
  expect(await client.connect()).toBe(replies['supervisor.slack.connect'].authorize_url)
  for (const url of ['https://slack.com.evil.test/oauth/v2/authorize?state=fake', 'https://slack.com/redirect?state=fake',
    'https://slack.com/oauth/v2/authorize', 'https://fake@slack.com/oauth/v2/authorize?state=fake']) {
    replies['supervisor.slack.connect'].authorize_url = url
    await expect(client.connect()).rejects.toThrow('Invalid Slack response')
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
