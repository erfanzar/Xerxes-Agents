// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { isRecord, type HubSlackMethod } from '../hub/protocol.js'
import { HubRequestError } from './hubConnection.js'

/**
 * Desktop ↔ Hub Slack contract (owner-only Hub WS; no Slack secrets go to desktop).
 * All methods use the `supervisor.slack.` prefix. Desktop never replays a mutation.
 *
 * status {} → { configured, connection: { team_id, team_name?, user_id } | null, contacts: SlackContact[] }
 * connect {} → { authorize_url } (Slack HTTPS OAuth v2 URL; callback owned by Hub; poll status)
 * disconnect { team_id, user_id } → status shape
 * contact.lookup { team_id, user_id } → { identity: SlackIdentity } (exact Slack IDs, no name matching)
 * contact.confirm { team_id, user_id } → status shape (Hub revalidates exact IDs; defaults to draft)
 * contact.remove { team_id, user_id } → status shape
 * contact.mode { team_id, user_id, mode: 'auto'|'draft' } → status shape
 * dm.open { team_id, user_id } → { team_id, user_id, messages: SlackMessage[], drafts: SlackDraft[] }
 * draft.send { team_id, user_id, draft_id, text } → DM shape (Hub must match the reviewed text and
 *   pending draft to the exact destination and prevent repeat delivery; no free-form send method)
 *
 * Older Hubs reject an unknown method with `invalid_request`. DM data is never included in shared supervisor events.
 */

export interface SlackIdentity {
  readonly team_id: string
  readonly user_id: string
  readonly display_name?: string
}

export type SlackMode = 'auto' | 'draft'
export interface SlackContact extends SlackIdentity { readonly mode: SlackMode }
export interface SlackStatus {
  readonly configured: boolean
  readonly connection: { readonly team_id: string; readonly team_name?: string; readonly user_id: string } | null
  readonly contacts: readonly SlackContact[]
}
export interface SlackMessage {
  readonly id: string
  readonly direction: 'inbound' | 'outbound'
  readonly text: string
  readonly created_at: string
}
export interface SlackDraft { readonly id: string; readonly text: string }
export interface SlackDm extends SlackIdentity {
  readonly messages: readonly SlackMessage[]
  readonly drafts: readonly SlackDraft[]
}

type SlackIds = Pick<SlackIdentity, 'team_id' | 'user_id'>


export interface SupervisorSlackClient {
  status(): Promise<SlackStatus>
  connect(): Promise<string>
  disconnect(account: SlackIds): Promise<SlackStatus>
  lookup(ids: SlackIds): Promise<SlackIdentity>
  confirm(contact: SlackIdentity): Promise<SlackStatus>
  remove(contact: SlackContact): Promise<SlackStatus>
  mode(contact: SlackContact, mode: SlackMode): Promise<SlackStatus>
  dm(contact: SlackContact): Promise<SlackDm>
  send(contact: SlackContact, draft: SlackDraft): Promise<SlackDm>
}

export function isSlackUserId(value: string): boolean { return /^[UW][A-Z0-9]{2,31}$/u.exec(value)?.[0] === value }
export function sameSlackContact(a: SlackIds, b: SlackIds): boolean { return a.team_id === b.team_id && a.user_id === b.user_id }
export function slackUnavailable(error: unknown): boolean {
  return error instanceof HubRequestError && (error.code === 'unknown_method' || error.code === 'unsupported'
    || (error.code === 'invalid_request' && error.message.startsWith('unknown hub method')))
}

function invalid(): never { throw new Error('Invalid Slack response from the Hub.') }
function record(value: unknown): Record<string, unknown> { return isRecord(value) ? value : invalid() }
function text(value: unknown): string { return typeof value === 'string' ? value : invalid() }
function id(value: unknown): string { const s = text(value); return s.trim() !== '' ? s : invalid() }
function list<T>(value: unknown, parse: (item: unknown) => T): T[] { return Array.isArray(value) ? value.map(parse) : invalid() }
function unique<T>(items: T[], key: (item: T) => string): T[] {
  return new Set(items.map(key)).size === items.length ? items : invalid()
}
function ids(value: unknown): SlackIds {
  const raw = record(value)
  const team_id = text(raw.team_id); const user_id = text(raw.user_id)
  if (!/^T[A-Z0-9]{2,31}$/.test(team_id) || !isSlackUserId(user_id)) invalid()
  return { team_id, user_id }
}
function identity(value: unknown): SlackIdentity {
  const raw = record(value)
  return { ...ids(raw), ...(typeof raw.display_name === 'string' ? { display_name: raw.display_name } : {}) }
}
function parseStatus(value: unknown): SlackStatus {
  const raw = record(value)
  if (typeof raw.configured !== 'boolean') return invalid()
  const account = raw.connection === null ? null : record(raw.connection)
  const connection = account === null ? null : { ...ids(account), ...(typeof account.team_name === 'string' ? { team_name: account.team_name } : {}) }
  const contacts = unique(list(raw.contacts, item => {
    const row = record(item)
    if (connection === null || row.team_id !== connection.team_id || (row.mode !== 'auto' && row.mode !== 'draft')) return invalid()
    return { ...identity(row), mode: row.mode } satisfies SlackContact
  }), row => row.user_id)
  return { configured: raw.configured, connection, contacts }
}
function parseDm(value: unknown, expected: SlackIds): SlackDm {
  const raw = record(value)
  const actual = identity(raw)
  if (!sameSlackContact(actual, expected)) invalid()
  return { ...actual, messages: unique(list(raw.messages, value => {
    const row = record(value)
    if (row.direction !== 'inbound' && row.direction !== 'outbound') return invalid()
    return { id: id(row.id), text: text(row.text), created_at: text(row.created_at), direction: row.direction } satisfies SlackMessage
  }), row => row.id), drafts: unique(list(raw.drafts, value => {
    const row = record(value)
    return { id: id(row.id), text: id(row.text) }
  }), row => row.id) }
}

function authorizationUrl(value: unknown): string {
  const url = new URL(text(record(value).authorize_url))
  if (url.origin !== 'https://slack.com' || url.pathname !== '/oauth/v2/authorize' || url.username || url.password || url.hash || !url.searchParams.get('state')) invalid()
  return url.href
}

export function createSupervisorSlackClient(options: {
  readonly allowed: () => boolean
  readonly request: (method: HubSlackMethod, params: Readonly<Record<string, unknown>>) => Promise<unknown>
}): SupervisorSlackClient {
  const call = async (method: HubSlackMethod, params: Readonly<Record<string, unknown>> = {}): Promise<unknown> => {
    if (!options.allowed()) throw new Error('Slack requires a live connection as the Hub owner.')
    const value = await options.request(method, params)
    if (!options.allowed()) throw new Error('Owner access or connection changed.')
    return value
  }
  return {
    status: async () => parseStatus(await call('supervisor.slack.status')),
    connect: async () => authorizationUrl(await call('supervisor.slack.connect')),
    disconnect: async account => parseStatus(await call('supervisor.slack.disconnect', ids(account))),
    lookup: async contact => {
      const result = identity(record(await call('supervisor.slack.contact.lookup', ids(contact))).identity)
      return sameSlackContact(result, contact) ? result : invalid()
    },
    confirm: async contact => parseStatus(await call('supervisor.slack.contact.confirm', ids(contact))),
    remove: async contact => parseStatus(await call('supervisor.slack.contact.remove', ids(contact))),
    mode: async (contact, mode) => parseStatus(await call('supervisor.slack.contact.mode', { ...ids(contact), mode })),
    dm: async contact => parseDm(await call('supervisor.slack.dm.open', ids(contact)), contact),
    send: async (contact, draft) => {
      const dm = parseDm(await call('supervisor.slack.draft.send', { ...ids(contact), draft_id: draft.id, text: draft.text }), contact)
      return dm.drafts.some(item => item.id === draft.id) ? invalid() : dm
    },
  }
}
