// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { Database } from 'bun:sqlite'
import { afterEach, expect, test } from 'bun:test'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { SlackPolicy } from '../src/hub/slack/policy.js'
import { SlackTransportError, type SlackMessage, type SlackPeer, type SlackPeerOperation, type SlackTransport } from '../src/hub/slack/types.js'
import { SUPERVISOR_PRINCIPAL_ID } from '../src/hub/supervisor/audit.js'
import type { HubPrincipal } from '../src/security/principal.js'

const owner = (id: string, hubOwner = true): HubPrincipal => ({ kind: 'user', id, hubOwner, displayName: id, deviceId: `d-${id}` })
const agent = (onBehalfOf: string, id = SUPERVISOR_PRINCIPAL_ID): HubPrincipal => ({ kind: 'agent', id, onBehalfOf, displayName: 'Xerxes' })
const alice = owner('alice')
const bob = owner('bob')
const peer: SlackPeer = { teamId: 'T123', userId: 'UACCOUNT', peerId: 'UPEER' }
const other = { ...peer, peerId: 'UOTHER' }
const ts = '1729666850.100000'
const reviewed = { requestId: 'turn-1', text: 'Reviewed text.' }
const cleanup: (() => void)[] = []
afterEach(() => { for (const close of cleanup.splice(0).reverse()) close() })

function deferred() {
  let release = () => {}
  const wait = new Promise<void>(resolve => { release = resolve })
  return { wait, release }
}

function setup() {
  const dir = mkdtempSync(join(tmpdir(), 'hub-slack-policy-'))
  const path = join(dir, 'state.sqlite')
  cleanup.push(() => rmSync(dir, { recursive: true, force: true }))
  const owners = new Set([alice.id, bob.id])
  const calls: { ownerId: string; operation: string; peer?: SlackPeer }[] = []
  const sent: { ownerId: string; peer: SlackPeer; text: string }[] = []
  const fake = {
    owners,
    mismatchAuth: false,
    mismatchProfile: false,
    wrongChannel: false,
    wrongReceipt: false,
    beforeProfile: async () => {},
    afterRead: async () => {},
    beforeSendApproval: async () => {},
    afterSend: async () => {},
    messages: (_ownerId: string, _peer: SlackPeer): readonly SlackMessage[] => [],
  }
  function connect() {
    const db = new Database(path)
    db.exec('PRAGMA journal_mode = WAL; PRAGMA synchronous = FULL; PRAGMA busy_timeout = 5000;')
    return { db, policy: new SlackPolicy({ db, isHubOwner: id => owners.has(id), transportForOwner: (ownerId, approve) => {
      async function allowed(peer: SlackPeer, operation: SlackPeerOperation) {
        calls.push({ ownerId, operation, peer })
        if (!await approve(peer, operation)) throw new SlackTransportError('not_approved')
      }
      const port: SlackTransport = {
        authTest: async identity => {
          calls.push({ ownerId, operation: 'auth' })
          return fake.mismatchAuth ? { ...identity, userId: 'UWRONG' } : identity
        },
        lookupProfile: async (_identity, exact) => {
          calls.push({ ownerId, operation: 'profile' })
          await fake.beforeProfile()
          return { userId: 'userId' in exact && !fake.mismatchProfile ? exact.userId : 'UWRONG', displayName: 'Known peer' }
        },
        openPeer: async peer => {
          await allowed(peer, 'open')
          return { ...peer, channelId: 'C123' }
        },
        verifyDm: async dm => {
          await allowed(dm, 'verify')
          return fake.wrongChannel ? { ...dm, channelId: 'C456' } : dm
        },
        readDm: async dm => {
          await allowed(dm, 'read')
          await fake.afterRead()
          return fake.messages(ownerId, dm)
        },
        sendDm: async (dm, text) => {
          await fake.beforeSendApproval()
          await allowed(dm, 'send')
          sent.push({ ownerId, peer: dm, text })
          await fake.afterSend()
          return { channelId: fake.wrongReceipt ? 'C999' : dm.channelId, ts }
        },
      }
      return port
    } }) }
  }
  const { db, policy } = connect()
  cleanup.push(() => db.close())
  return { db, policy, connect, calls, sent, fake, context: (key = peer) => policy.supervisorContext(agent(alice.id), alice.id, key) }
}

test('defaults to deny; only a current human owner can confirm an exact Slack account and peer', async () => {
  const { policy, fake, calls, context } = setup()
  expect(policy.listContacts(alice)).toEqual([])
  expect(() => policy.readInbox(alice, peer)).toThrow(expect.objectContaining({ code: 'not_approved' }))
  expect(() => context()).toThrow(expect.objectContaining({ code: 'not_approved' }))
  await expect(policy.poll(alice.id, peer)).rejects.toMatchObject({ code: 'not_approved' })
  await expect(policy.confirmContact(agent(alice.id), peer)).rejects.toMatchObject({ code: 'forbidden' })
  await expect(policy.confirmContact(owner(alice.id, false), peer)).rejects.toMatchObject({ code: 'forbidden' })
  await expect(policy.confirmContact(owner('not-a-live-owner'), peer)).rejects.toMatchObject({ code: 'forbidden' })
  expect(calls).toEqual([])

  fake.mismatchAuth = true
  await expect(policy.confirmContact(alice, peer)).rejects.toMatchObject({ code: 'identity_mismatch' })
  expect(calls.map(call => call.operation)).toEqual(['auth'])
  fake.mismatchAuth = false
  fake.mismatchProfile = true
  await expect(policy.confirmContact(alice, peer)).rejects.toMatchObject({ code: 'identity_mismatch' })
  fake.mismatchProfile = false
  await policy.lookupContact(alice, peer)
  expect(policy.listContacts(alice)).toEqual([])
  expect(await policy.confirmContact(alice, peer)).toEqual({ ...peer, displayName: 'Known peer', mode: 'draft' })
  expect(() => policy.setContactMode(bob, peer, 'auto')).toThrow(expect.objectContaining({ code: 'not_approved' }))
  fake.owners.delete(alice.id)
  await expect(policy.poll(alice.id, peer)).rejects.toMatchObject({ code: 'forbidden' })
})

test('isolates cached DM messages by Hub owner, Slack workspace, account and peer; rejects forged turn contexts', async () => {
  const { policy, fake, context } = setup()
  const workspaces = { ...peer, teamId: 'T456' }
  const accounts = { ...peer, userId: 'UDIFFERENT' }
  for (const destination of [peer, other, workspaces, accounts]) await policy.confirmContact(alice, destination)
  await policy.confirmContact(bob, peer)
  fake.messages = (id, target) => [
    { ts: '1729666850.001000', userId: target.userId, text: 'earlier outbound' },
    { ts, userId: target.peerId, text: `${id}/${target.teamId}/${target.userId}/${target.peerId}` },
    { ts: '1729666850.110000', userId: 'USTRANGER', text: 'drop this' },
  ]
  for (const destination of [peer, other, workspaces, accounts]) await policy.poll(alice.id, destination)
  await policy.poll(bob.id, peer)
  expect(await policy.poll(alice.id, peer)).toEqual([])
  for (const target of [peer, other, workspaces, accounts]) {
    expect(policy.read(context(target))).toEqual([
      { id: '1729666850.001000', direction: 'outbound', text: 'earlier outbound', createdAt: '2024-10-23T07:00:50.001Z' },
      { id: ts, direction: 'inbound', text: `alice/${target.teamId}/${target.userId}/${target.peerId}`, createdAt: '2024-10-23T07:00:50.100Z' },
    ])
  }
  expect(policy.readInbox(bob, peer).messages.at(-1)?.text).toBe('bob/T123/UACCOUNT/UPEER')
  expect(() => policy.read({ ...context() })).toThrow(expect.objectContaining({ code: 'forbidden' }))
  expect(() => policy.supervisorContext(agent(alice.id, 'task-agent'), alice.id, peer)).toThrow(expect.objectContaining({ code: 'forbidden' }))
  expect(() => policy.supervisorContext(agent(bob.id), alice.id, peer)).toThrow(expect.objectContaining({ code: 'forbidden' }))
  expect(() => policy.readInbox(agent(alice.id), peer)).toThrow(expect.objectContaining({ code: 'forbidden' }))
})

test('persists contact, inbox, immutable draft and receipt across handles and restart', async () => {
  const { policy, connect, db, sent, context, fake } = setup()
  await policy.confirmContact(alice, peer)
  await policy.confirmContact(alice, other)
  await policy.confirmContact(bob, peer)
  fake.messages = () => [{ ts: '1729666850.000000', userId: peer.peerId, text: 'inbox' }]
  await policy.poll(alice.id, peer)
  const draft = await policy.reply(context(), reviewed)
  expect(draft.state).toBe('pending')
  expect(sent).toEqual([])
  db.close()
  const reopened = connect()
  cleanup.push(() => reopened.db.close())
  const p = reopened.policy
  expect(p.listContacts(alice)).toHaveLength(2)
  expect(p.readInbox(alice, peer).drafts).toEqual([{ id: draft.id, text: reviewed.text }])
  expect(p.readInbox(alice, peer).messages.map(row => row.text)).toEqual(['inbox'])
  const send = { draftId: draft.id, text: reviewed.text }
  await expect(p.sendDraft(alice, other, send)).rejects.toMatchObject({ code: 'not_found' })
  await expect(p.sendDraft(bob, peer, send)).rejects.toMatchObject({ code: 'not_found' })
  await expect(p.sendDraft(agent(alice.id), peer, send)).rejects.toMatchObject({ code: 'forbidden' })
  await expect(p.sendDraft(alice, peer, { ...send, text: `${send.text} ` })).rejects.toMatchObject({ code: 'conflict' })
  expect(await p.sendDraft(alice, peer, send)).toEqual({ id: draft.id, state: 'sent', ts })
  expect(p.readInbox(alice, peer).drafts).toEqual([])
  expect(await p.sendDraft(alice, peer, send)).toEqual({ id: draft.id, state: 'sent', ts })
  expect(sent).toHaveLength(1)
})

test('concurrent confirmations and reply replays cannot send through a second SQLite handle', async () => {
  const { policy, connect, fake, sent, context } = setup()
  const reopened = connect()
  cleanup.push(() => reopened.db.close())
  await policy.confirmContact(alice, peer)
  const draft = await policy.reply(context(), reviewed)
  const send = { draftId: draft.id, text: reviewed.text }
  const entered = deferred()
  const finish = deferred()
  fake.afterSend = async () => { entered.release(); await finish.wait }
  const first = policy.sendDraft(alice, peer, send)
  await entered.wait
  expect(await reopened.policy.sendDraft(alice, peer, send)).toEqual({ id: draft.id, state: 'dispatching' })
  const otherContext = reopened.policy.supervisorContext(agent(alice.id), alice.id, peer)
  expect(await reopened.policy.reply(otherContext, reviewed)).toEqual({ id: draft.id, state: 'dispatching' })
  await expect(reopened.policy.reply(otherContext, { ...reviewed, text: 'changed' })).rejects.toMatchObject({ code: 'conflict' })
  finish.release()
  expect((await first).state).toBe('sent')
  expect(sent).toHaveLength(1)
})

test('auto is opt-in, does not promote existing drafts, and a lost receipt is terminal across restart', async () => {
  const { policy, connect, fake, sent, context } = setup()
  await policy.confirmContact(alice, peer)
  const draft = await policy.reply(context(), reviewed)
  policy.setContactMode(alice, peer, 'auto')
  expect((await policy.reply(context(), reviewed)).state).toBe('pending')
  expect(sent).toHaveLength(0)
  fake.afterSend = async () => { throw new SlackTransportError('network_error') }
  const request = { ...reviewed, requestId: 'turn-2' }
  const uncertain = await policy.reply(context(), request)
  expect(uncertain.state).toBe('uncertain')
  const reopened = connect()
  cleanup.push(() => reopened.db.close())
  const ctx = reopened.policy.supervisorContext(agent(alice.id), alice.id, peer)
  expect(await reopened.policy.reply(ctx, request)).toEqual(uncertain)
  expect(await reopened.policy.sendDraft(alice, peer, { draftId: uncertain.id, text: request.text })).toEqual(uncertain)
  expect(reopened.policy.readInbox(alice, peer).drafts).toEqual([{ id: draft.id, text: reviewed.text }])
  expect(sent).toHaveLength(1)
})

test('removal invalidates in-flight poll, transport approval, draft and previously minted context across reapproval', async () => {
  const { policy, fake, sent, context } = setup()
  await policy.confirmContact(alice, peer)
  const ctx = context()
  const draft = await policy.reply(ctx, reviewed)
  const entered = deferred()
  const finish = deferred()
  fake.afterRead = async () => { entered.release(); await finish.wait }
  fake.messages = () => [{ ts, userId: peer.peerId, text: 'stale' }]
  const poll = policy.poll(alice.id, peer)
  await entered.wait
  policy.removeContact(alice, peer)
  await policy.confirmContact(alice, peer)
  finish.release()
  await expect(poll).rejects.toMatchObject({ code: 'not_approved' })
  expect(policy.readInbox(alice, peer)).toMatchObject({ messages: [], drafts: [] })
  expect(() => policy.read(ctx)).toThrow(expect.objectContaining({ code: 'not_approved' }))
  await expect(policy.reply(context(), reviewed)).rejects.toMatchObject({ code: 'conflict' })
  await expect(policy.sendDraft(alice, peer, { draftId: draft.id, text: reviewed.text })).rejects.toMatchObject({ code: 'not_found' })

  policy.setContactMode(alice, peer, 'auto')
  fake.beforeSendApproval = async () => { policy.removeContact(alice, peer) }
  await expect(policy.reply(context(), { ...reviewed, requestId: 'turn-2' })).rejects.toMatchObject({ code: 'not_approved' })
  expect(sent).toHaveLength(0)
})

test('does not read or claim delivery for a mismatched DM or an already aborted signal', async () => {
  const { policy, fake, calls, sent, context } = setup()
  await policy.confirmContact(alice, peer)
  fake.wrongChannel = true
  await expect(policy.poll(alice.id, peer)).rejects.toMatchObject({ code: 'not_dm' })
  const draft = await policy.reply(context(), reviewed)
  const send = { draftId: draft.id, text: reviewed.text }
  await expect(policy.sendDraft(alice, peer, send)).rejects.toMatchObject({ code: 'not_dm' })
  fake.wrongChannel = false
  const signal = AbortSignal.abort()
  await expect(policy.sendDraft(alice, peer, send, signal)).rejects.toMatchObject({ code: 'aborted' })
  expect(calls.map(call => call.operation)).not.toContain('read')
  expect(sent).toEqual([])
  fake.wrongReceipt = true
  expect((await policy.sendDraft(alice, peer, send)).state).toBe('uncertain')
  expect(policy.readInbox(alice, peer).messages).toEqual([])
})
