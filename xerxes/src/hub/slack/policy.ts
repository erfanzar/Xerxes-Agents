// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { randomUUID } from 'node:crypto'
import type { Database } from 'bun:sqlite'

import type { HubPrincipal } from '../../security/principal.js'
import { HubProtocolError } from '../protocol.js'
import { SUPERVISOR_PRINCIPAL_ID } from '../supervisor/audit.js'
import { SlackTransportError, type SlackDirectMessage, type SlackPeer, type SlackPeerApproval, type SlackTransport } from './types.js'

export type SlackContactMode = 'draft' | 'auto'
export interface SlackContact extends SlackPeer {
  readonly displayName: string
  readonly mode: SlackContactMode
}

export interface SlackInboxMessage {
  readonly id: string
  readonly direction: 'inbound' | 'outbound'
  /** External, untrusted content; fence it before placing it in a model prompt. */
  readonly text: string
  readonly createdAt: string
}

export interface SlackInbox {
  readonly contact: SlackContact
  readonly messages: readonly SlackInboxMessage[]
  readonly drafts: readonly { readonly id: string; readonly text: string }[]
}

export interface SlackDelivery {
  readonly id: string
  /** `dispatching` and `uncertain` may have reached Slack and are never retried. */
  readonly state: 'pending' | 'dispatching' | 'sent' | 'uncertain'
  readonly ts?: string
}

const slackTurn = Symbol('Slack supervisor turn')

/** Mint on the Hub from a resolved private owner turn and an exact peer, never from `onBehalfOf` alone. */
export interface SlackSupervisorContext {
  readonly [slackTurn]: true
}

export interface SlackPolicyOptions {
  /** Hub-private SQLite connection. Use durable commits (synchronous=FULL) for the dispatch ledger. */
  readonly db: Database
  readonly isHubOwner: (ownerId: string) => boolean
  /** Bind credentials to this owner and enforce this callback at the transport boundary, including just before POST.
   * The transport must not retry a send. Do not expose the resulting transport to the supervisor model. */
  readonly transportForOwner: (ownerId: string, approvePeer: SlackPeerApproval) => SlackTransport
}

type PeerKey = [ownerId: string, teamId: string, userId: string, peerId: string]
interface ContactRow {
  team_id: string
  user_id: string
  peer_id: string
  display_name: string
  mode: SlackContactMode | 'removed'
  approval_id: string
}
interface DraftRow {
  id: string
  text: string
  approval_id: string
  state: SlackDelivery['state']
  ts: string | null
}
interface MessageRow {
  ts: string
  direction: SlackInboxMessage['direction']
  text: string
}

const PAGE_SIZE = 50
const POLL_LIMIT = 15
const MAX_TEXT = 4000
const TEAM_ID = /^T[A-Z0-9]{1,63}$/
const USER_ID = /^[UW][A-Z0-9]{1,63}$/
// Slack IMs can use more than one ID prefix; verifyDm, not the prefix, proves membership.
const CHANNEL_ID = /^[CDG][A-Z0-9]{1,63}$/
const TIMESTAMP = /^(?:0|[1-9]\d{0,10})\.\d{6}$/

function matches(value: unknown, pattern: RegExp): value is string {
  // A full match, including the end, even for strings terminating in a newline.
  return typeof value === 'string' && pattern.exec(value)?.[0] === value
}

function keyFor(ownerId: string, peer: SlackPeer): PeerKey {
  if (!matches(peer?.teamId, TEAM_ID) || !matches(peer?.userId, USER_ID) || !matches(peer?.peerId, USER_ID) || peer.userId === peer.peerId) {
    throw new HubProtocolError('invalid_request', 'Expected an exact Slack workspace, account and distinct peer.')
  }
  return [ownerId, peer.teamId, peer.userId, peer.peerId]
}

function samePeer(peer: SlackPeer, [, teamId, userId, peerId]: PeerKey): boolean {
  return peer?.teamId === teamId && peer?.userId === userId && peer?.peerId === peerId
}

function contact(row: ContactRow & { mode: SlackContactMode }): SlackContact {
  return { teamId: row.team_id, userId: row.user_id, peerId: row.peer_id, displayName: row.display_name, mode: row.mode }
}

function message(row: MessageRow): SlackInboxMessage {
  return { id: row.ts, direction: row.direction, text: row.text, createdAt: new Date(Number(row.ts) * 1000).toISOString() }
}

function delivery(row: DraftRow): SlackDelivery {
  return { id: row.id, state: row.state, ...(row.ts === null ? {} : { ts: row.ts }) }
}

function checkText(text: string): void {
  if (typeof text !== 'string' || text.length > MAX_TEXT || text.includes('\0') || !text.trim()) {
    throw new HubProtocolError('invalid_request', 'Expected non-empty Slack text of at most 4000 characters.')
  }
}

function checkSignal(signal?: AbortSignal): void {
  if (signal?.aborted) throw new SlackTransportError('aborted')
}

/** Owner-only local inbox. There are no broadcasts, global inbox queries or model-facing transport credentials. */
export class SlackPolicy {
  private readonly db: Database
  private readonly turns = new WeakMap<SlackSupervisorContext, { key: PeerKey; approvalId: string }>()

  constructor(private readonly options: SlackPolicyOptions) {
    this.db = options.db
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS hub_slack_contacts (
        owner_id TEXT NOT NULL, team_id TEXT NOT NULL, user_id TEXT NOT NULL, peer_id TEXT NOT NULL,
        approval_id TEXT NOT NULL, display_name TEXT NOT NULL,
        mode TEXT NOT NULL CHECK (mode IN ('draft', 'auto', 'removed')),
        PRIMARY KEY (owner_id, team_id, user_id, peer_id)
      );
      CREATE TABLE IF NOT EXISTS hub_slack_outbox (
        owner_id TEXT NOT NULL, team_id TEXT NOT NULL, user_id TEXT NOT NULL, peer_id TEXT NOT NULL,
        approval_id TEXT NOT NULL, id TEXT NOT NULL, request_id TEXT NOT NULL, text TEXT NOT NULL,
        state TEXT NOT NULL CHECK (state IN ('pending', 'dispatching', 'sent', 'uncertain')), ts TEXT,
        PRIMARY KEY (owner_id, team_id, user_id, peer_id, id),
        UNIQUE (owner_id, team_id, user_id, peer_id, request_id)
      );
      CREATE INDEX IF NOT EXISTS hub_slack_pending ON hub_slack_outbox (owner_id, team_id, user_id, peer_id, approval_id, state);
      CREATE TABLE IF NOT EXISTS hub_slack_inbox (
        owner_id TEXT NOT NULL, team_id TEXT NOT NULL, user_id TEXT NOT NULL, peer_id TEXT NOT NULL,
        approval_id TEXT NOT NULL, ts TEXT NOT NULL, direction TEXT NOT NULL CHECK (direction IN ('inbound', 'outbound')), text TEXT NOT NULL,
        PRIMARY KEY (owner_id, team_id, user_id, peer_id, approval_id, ts)
      );
    `)
  }

  listContacts(principal: HubPrincipal): SlackContact[] {
    return this.activeContacts(this.humanOwner(principal))
  }

  /** Host-only enumeration for the polling scheduler, never a model tool. */
  contactsForPoll(ownerId: string): SlackContact[] {
    this.requireOwner(ownerId)
    return this.activeContacts(ownerId)
  }

  /** Exact lookup is owner-only and never creates an approval. */
  async lookupContact(principal: HubPrincipal, peer: SlackPeer, signal?: AbortSignal): Promise<{ userId: string; displayName: string }> {
    const key = keyFor(this.humanOwner(principal), peer)
    return this.lookup(key, signal)
  }

  /** Revalidates the authenticated Slack identity and exact peer. Re-confirmation never upgrades a mode. */
  async confirmContact(principal: HubPrincipal, peer: SlackPeer, signal?: AbortSignal): Promise<SlackContact> {
    const key = keyFor(this.humanOwner(principal), peer)
    const before = this.contactRow(key)?.approval_id
    const profile = await this.lookup(key, signal)
    return this.db.transaction(() => {
      const current = this.contactRow(key)
      if (current?.approval_id !== before) throw new HubProtocolError('conflict', 'Slack contact changed during confirmation.')
      this.requireOwner(key[0])
      checkSignal(signal)
      this.db.query(`INSERT INTO hub_slack_contacts VALUES (?, ?, ?, ?, ?, ?, 'draft')
        ON CONFLICT (owner_id, team_id, user_id, peer_id) DO UPDATE SET
        approval_id = excluded.approval_id, display_name = excluded.display_name, mode = 'draft'
        WHERE hub_slack_contacts.mode = 'removed'`).run(...key, randomUUID(), profile.displayName)
      return contact(this.requireContact(key))
    }).immediate()
  }

  setContactMode(principal: HubPrincipal, peer: SlackPeer, mode: SlackContactMode): SlackContact {
    const key = keyFor(this.humanOwner(principal), peer)
    if (mode !== 'draft' && mode !== 'auto') throw new HubProtocolError('invalid_request', 'Invalid Slack contact mode.')
    const row = this.db.query<ContactRow, [SlackContactMode, ...PeerKey]>(`UPDATE hub_slack_contacts SET mode = ?
      WHERE owner_id = ? AND team_id = ? AND user_id = ? AND peer_id = ? AND mode != 'removed' RETURNING *`).get(mode, ...key)
    if (row === null) throw new SlackTransportError('not_approved')
    return contact({ ...row, mode })
  }

  removeContact(principal: HubPrincipal, peer: SlackPeer): void {
    const key = keyFor(this.humanOwner(principal), peer)
    this.db.transaction(() => {
      // The tombstone also invalidates a confirmation in flight for a previously absent contact.
      this.db.query(`INSERT INTO hub_slack_contacts VALUES (?, ?, ?, ?, ?, '', 'removed')
        ON CONFLICT (owner_id, team_id, user_id, peer_id) DO UPDATE SET
        approval_id = excluded.approval_id, display_name = '', mode = 'removed'`).run(...key, randomUUID())
      this.db.query(`DELETE FROM hub_slack_inbox WHERE owner_id = ? AND team_id = ? AND user_id = ? AND peer_id = ?`).run(...key)
      // Keep replay keys/attempt outcomes, not removed contact content. Old drafts cannot become sendable after reapproval.
      this.db.query(`UPDATE hub_slack_outbox SET text = '' WHERE owner_id = ? AND team_id = ? AND user_id = ? AND peer_id = ?`).run(...key)
    }).immediate()
  }

  /** Poll only a confirmed peer. Returns newly cached messages, oldest first; only inbound messages should wake a turn. */
  async poll(ownerId: string, peer: SlackPeer, signal?: AbortSignal): Promise<SlackInboxMessage[]> {
    const key = keyFor(ownerId, peer)
    const approved = this.requireContact(key)
    const port = this.scopedTransport(key, approved.approval_id)
    const dm = await this.open(key, approved.approval_id, port, signal)
    const remote = await port.readDm(dm, POLL_LIMIT, signal)
    const batch: MessageRow[] = []
    if (!Array.isArray(remote) || remote.length > POLL_LIMIT) throw new SlackTransportError('invalid_response')
    for (const row of remote) {
      if (!matches(row?.ts, TIMESTAMP) || typeof row.text !== 'string') throw new SlackTransportError('invalid_response')
      if (row.userId !== key[2] && row.userId !== key[3]) continue
      batch.push({ ts: row.ts, text: row.text.slice(0, MAX_TEXT), direction: row.userId === key[3] ? 'inbound' : 'outbound' })
    }
    return this.db.transaction(() => {
      this.requireContact(key, approved.approval_id)
      checkSignal(signal)
      return batch.sort((a, b) => Number(a.ts) - Number(b.ts))
        .filter(row => this.insertMessage(key, approved.approval_id, row)).map(message)
    }).immediate()
  }

  readInbox(principal: HubPrincipal, peer: SlackPeer): SlackInbox {
    const key = keyFor(this.humanOwner(principal), peer)
    return this.db.transaction(() => {
      const approved = this.requireContact(key)
      const drafts = this.db.query<{ id: string; text: string }, [...PeerKey, string, number]>(`SELECT id, text FROM hub_slack_outbox
        WHERE owner_id = ? AND team_id = ? AND user_id = ? AND peer_id = ? AND approval_id = ? AND state = 'pending'
        ORDER BY rowid DESC LIMIT ?`).all(...key, approved.approval_id, PAGE_SIZE)
      return { contact: contact(approved), messages: this.messages(key, approved.approval_id), drafts }
    })()
  }

  /** Host only: call after resolving the private Slack turn, before starting model work. */
  supervisorContext(principal: HubPrincipal, ownerId: string, peer: SlackPeer): SlackSupervisorContext {
    if (principal?.kind !== 'agent' || principal.id !== SUPERVISOR_PRINCIPAL_ID || principal.onBehalfOf !== ownerId) {
      throw new HubProtocolError('forbidden', 'Slack replies and reads require a private main-supervisor DM turn.')
    }
    const key = keyFor(ownerId, peer)
    const { approval_id } = this.requireContact(key)
    const context: SlackSupervisorContext = { [slackTurn]: true }
    this.turns.set(context, { key, approvalId: approval_id })
    return context
  }

  /** Model-facing local read. The peer and sole audience come from the trusted turn context. */
  read(context: SlackSupervisorContext): SlackInboxMessage[] {
    const { key, approvalId } = this.requireTurn(context)
    return this.db.transaction(() => this.messages(key, this.requireContact(key, approvalId).approval_id))()
  }

  /** Use a stable host-generated request ID for one logical reply (never a model-supplied ID). */
  async reply(context: SlackSupervisorContext, request: { readonly requestId: string; readonly text: string }, signal?: AbortSignal): Promise<SlackDelivery> {
    const { key, approvalId } = this.requireTurn(context)
    const { requestId, text } = request
    if (!matches(requestId, /^[A-Za-z0-9._:-]{1,160}$/)) throw new HubProtocolError('invalid_request', 'Invalid Slack reply request ID.')
    checkText(text)
    checkSignal(signal)
    const { row, auto } = this.db.transaction(() => {
      const approved = this.requireContact(key, approvalId)
      const existing = this.db.query<DraftRow, [...PeerKey, string]>(`SELECT id, text, approval_id, state, ts FROM hub_slack_outbox
        WHERE owner_id = ? AND team_id = ? AND user_id = ? AND peer_id = ? AND request_id = ?`).get(...key, requestId)
      if (existing !== null) {
        if (existing.approval_id !== approved.approval_id || existing.text !== text) {
          throw new HubProtocolError('conflict', 'Slack reply request ID was already used.')
        }
        return { row: existing, auto: false }
      }
      const row: DraftRow = { id: randomUUID(), text, approval_id: approved.approval_id, state: 'pending', ts: null }
      this.db.query(`INSERT INTO hub_slack_outbox VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', NULL)`)
        .run(...key, row.approval_id, row.id, requestId, text)
      return { row, auto: approved.mode === 'auto' }
    }).immediate()
    return auto ? this.dispatch(key, row, true, signal) : delivery(row)
  }

  /** Owner confirmation matches the immutable, reviewed text and destination byte for byte. */
  async sendDraft(principal: HubPrincipal, peer: SlackPeer, request: { readonly draftId: string; readonly text: string }, signal?: AbortSignal): Promise<SlackDelivery> {
    const key = keyFor(this.humanOwner(principal), peer)
    checkText(request.text)
    const approved = this.requireContact(key)
    const row = this.draft(key, request.draftId, approved.approval_id)
    if (row.text !== request.text) throw new HubProtocolError('conflict', 'Slack draft text does not match the reviewed text.')
    return this.dispatch(key, row, false, signal)
  }

  private async dispatch(key: PeerKey, row: DraftRow, auto: boolean, signal?: AbortSignal): Promise<SlackDelivery> {
    if (row.state !== 'pending') return delivery(row)
    const port = this.scopedTransport(key, row.approval_id, row.id, auto)
    const dm = await this.open(key, row.approval_id, port, signal)
    checkSignal(signal)
    // Commit the unique claim before invoking send. A crash or a lost response must never cause a retry.
    const claimed = this.db.transaction(() => {
      const approved = this.requireContact(key, row.approval_id)
      if (auto && approved.mode !== 'auto') return false
      return this.db.query(`UPDATE hub_slack_outbox SET state = 'dispatching' WHERE owner_id = ? AND team_id = ? AND user_id = ?
        AND peer_id = ? AND id = ? AND approval_id = ? AND state = 'pending'`).run(...key, row.id, row.approval_id).changes === 1
    }).immediate()
    if (!claimed) return delivery(this.draft(key, row.id, row.approval_id))

    let ts: string | null = null
    try {
      const receipt = await port.sendDm(dm, row.text, signal)
      if (receipt?.channelId === dm.channelId && matches(receipt.ts, TIMESTAMP)) ts = receipt.ts
    } catch {
      // A throw (including timeout or cancellation) cannot prove the remote POST did not happen.
    }
    const state = ts === null ? 'uncertain' : 'sent'
    this.db.transaction(() => {
      this.db.query(`UPDATE hub_slack_outbox SET state = ?, ts = ? WHERE owner_id = ? AND team_id = ? AND user_id = ?
        AND peer_id = ? AND id = ? AND approval_id = ? AND state = 'dispatching'`).run(state, ts, ...key, row.id, row.approval_id)
      if (ts !== null && this.isCurrent(key, row.approval_id)) {
        this.insertMessage(key, row.approval_id, { ts, direction: 'outbound', text: row.text })
      }
    }).immediate()
    this.requireContact(key, row.approval_id)
    return delivery({ ...row, ts, state })
  }

  private async lookup(key: PeerKey, signal?: AbortSignal): Promise<{ userId: string; displayName: string }> {
    const [ownerId, teamId, userId, peerId] = key
    const port = this.options.transportForOwner(ownerId, () => false)
    const identity = { teamId, userId }
    checkSignal(signal)
    const who = await port.authTest(identity, signal)
    this.requireOwner(ownerId)
    checkSignal(signal)
    if (who?.teamId !== teamId || who?.userId !== userId) throw new SlackTransportError('identity_mismatch')
    const profile = await port.lookupProfile(identity, { userId: peerId }, signal)
    this.requireOwner(ownerId)
    checkSignal(signal)
    if (profile?.userId !== peerId) throw new SlackTransportError('identity_mismatch')
    if (typeof profile.displayName !== 'string') throw new SlackTransportError('invalid_response')
    return { userId: peerId, displayName: profile.displayName.slice(0, 256) }
  }

  private scopedTransport(key: PeerKey, approvalId: string, draftId?: string, auto = false): SlackTransport {
    return this.options.transportForOwner(key[0], (peer, operation) => {
      if (!samePeer(peer, key) || !this.isCurrent(key, approvalId)) return false
      if (operation !== 'send') return operation === 'open' || operation === 'verify' || (operation === 'read' && draftId === undefined)
      if (draftId === undefined || (auto && this.contactRow(key)?.mode !== 'auto')) return false
      return this.draft(key, draftId, approvalId).state === 'dispatching'
    })
  }

  private async open(key: PeerKey, approvalId: string, port: SlackTransport, signal?: AbortSignal): Promise<SlackDirectMessage> {
    const [, teamId, userId, peerId] = key
    this.requireContact(key, approvalId)
    checkSignal(signal)
    const dm = await port.openPeer({ teamId, userId, peerId }, signal)
    this.requireContact(key, approvalId)
    checkSignal(signal)
    if (!samePeer(dm, key) || !matches(dm.channelId, CHANNEL_ID)) throw new SlackTransportError('not_dm')
    const verified = await port.verifyDm({ teamId, userId, peerId, channelId: dm.channelId }, signal)
    this.requireContact(key, approvalId)
    checkSignal(signal)
    if (!samePeer(verified, key) || verified.channelId !== dm.channelId) throw new SlackTransportError('not_dm')
    return { teamId, userId, peerId, channelId: dm.channelId }
  }

  private messages(key: PeerKey, approvalId: string): SlackInboxMessage[] {
    return this.db.query<MessageRow, [...PeerKey, string, number]>(`SELECT ts, text, direction FROM hub_slack_inbox
      WHERE owner_id = ? AND team_id = ? AND user_id = ? AND peer_id = ? AND approval_id = ?
      ORDER BY length(ts) DESC, ts DESC LIMIT ?`).all(...key, approvalId, PAGE_SIZE).reverse().map(message)
  }

  private insertMessage(key: PeerKey, approvalId: string, row: MessageRow): boolean {
    return this.db.query(`INSERT INTO hub_slack_inbox VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING`)
      .run(...key, approvalId, row.ts, row.direction, row.text).changes === 1
  }

  private draft(key: PeerKey, id: string, approvalId: string): DraftRow {
    if (!matches(id, /^[a-f0-9-]{36}$/)) throw new HubProtocolError('not_found', 'No such Slack draft.')
    const row = this.db.query<DraftRow, [...PeerKey, string, string]>(`SELECT id, text, approval_id, state, ts FROM hub_slack_outbox
      WHERE owner_id = ? AND team_id = ? AND user_id = ? AND peer_id = ? AND approval_id = ? AND id = ?`).get(...key, approvalId, id)
    if (row === null) throw new HubProtocolError('not_found', 'No such Slack draft.')
    return row
  }

  private activeContacts(ownerId: string): SlackContact[] {
    return this.db.query<ContactRow & { mode: SlackContactMode }, [string]>(`SELECT * FROM hub_slack_contacts
      WHERE owner_id = ? AND mode != 'removed' ORDER BY team_id, user_id, peer_id`).all(ownerId).map(contact)
  }

  private contactRow(key: PeerKey): ContactRow | null {
    return this.db.query<ContactRow, PeerKey>(`SELECT * FROM hub_slack_contacts
      WHERE owner_id = ? AND team_id = ? AND user_id = ? AND peer_id = ?`).get(...key)
  }

  private requireContact(key: PeerKey, approvalId?: string): ContactRow & { mode: SlackContactMode } {
    this.requireOwner(key[0])
    const row = this.contactRow(key)
    if (row === null || row.mode === 'removed' || (approvalId !== undefined && row.approval_id !== approvalId)) {
      throw new SlackTransportError('not_approved')
    }
    return { ...row, mode: row.mode }
  }

  private isCurrent(key: PeerKey, approvalId: string): boolean {
    const row = this.contactRow(key)
    return this.ownerNow(key[0]) && row !== null && row.mode !== 'removed' && row.approval_id === approvalId
  }

  private requireTurn(context: SlackSupervisorContext): { key: PeerKey; approvalId: string } {
    const turn = this.turns.get(context)
    if (turn === undefined) throw new HubProtocolError('forbidden', 'Slack replies and reads require a private main-supervisor DM turn.')
    return turn
  }

  private humanOwner(principal: HubPrincipal): string {
    if (principal?.kind !== 'user' || principal.hubOwner !== true) throw new HubProtocolError('forbidden', 'Hub owner required.')
    this.requireOwner(principal.id)
    return principal.id
  }

  private ownerNow(ownerId: string): boolean {
    try {
      return typeof ownerId === 'string' && ownerId.length > 0 && this.options.isHubOwner(ownerId) === true
    } catch {
      return false
    }
  }

  private requireOwner(ownerId: string): void {
    if (!this.ownerNow(ownerId)) throw new HubProtocolError('forbidden', 'Hub owner required.')
  }
}
