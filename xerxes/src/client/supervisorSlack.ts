// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { isRecord, type HubSlackMethod } from '../hub/protocol.js'
import { HubRequestError } from './hubConnection.js'

/**
 * Desktop ↔ Hub Slack contract (owner-only Hub WS; no Slack secrets go to desktop).
 * All methods use the `supervisor.slack.` prefix. Desktop never replays a mutation.
 *
 * status {} → { configured, connection, pending_identity?, contacts, … }
 * connect {} → { authorize_url, expires_at?, pairing_code? } (trusted browser endpoint; poll status)
 * connect_token { token } → status shape with pending_identity (Hub verifies the owner's own app user token;
 *   nothing is stored until confirm_connection; the token is never echoed back)
 * confirm_connection { team_id, user_id } → status shape (exact identity after browser authorization)
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
  /** The Hub also offers browser sign-in (operator OAuth app or managed broker). Absent on older Hubs. */
  readonly browser_sign_in?: boolean
  readonly connection: { readonly team_id: string; readonly team_name?: string; readonly user_id: string } | null
  readonly pending_identity?: SlackIds
  readonly oauth_status?: 'pending' | 'denied' | 'expired'
  readonly contacts: readonly SlackContact[]
}
export interface SlackAuthorization {
  readonly url: string
  readonly expiresAt?: number
  readonly pairingCode?: string
}

export class SlackBrokerUnavailable extends Error {
  constructor() { super('Managed Slack sign-in is unavailable on this Desktop.') }
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
  connect(): Promise<SlackAuthorization>
  connectToken(token: string): Promise<SlackStatus>
  confirmConnection(account: SlackIds): Promise<SlackStatus>
  disconnect(account: SlackIds): Promise<SlackStatus>
  lookup(ids: SlackIds): Promise<SlackIdentity>
  confirm(contact: SlackIdentity): Promise<SlackStatus>
  remove(contact: SlackContact): Promise<SlackStatus>
  mode(contact: SlackContact, mode: SlackMode): Promise<SlackStatus>
  dm(contact: SlackContact): Promise<SlackDm>
  send(contact: SlackContact, draft: SlackDraft): Promise<SlackDm>
}

/** User scopes the Hub requires (mirrors hub/slack/oauth.ts USER_SCOPES). */
export const SLACK_USER_SCOPES = ['users:read', 'users:read.email', 'im:write', 'im:read', 'im:history', 'chat:write'] as const

/** The Slack app manifest for the owner's own XERXES app: user scopes only, no bot, no token rotation. */
export function slackAppManifest(): string {
  return JSON.stringify({
    display_information: { name: 'XERXES', description: 'Lets XERXES read and reply to direct messages you approve, as you.' },
    oauth_config: { scopes: { user: [...SLACK_USER_SCOPES] } },
    settings: { org_deploy_enabled: false, socket_mode_enabled: false, token_rotation_enabled: false },
  })
}

/** Opens Slack's "create app" page with the manifest pre-filled. */
export function slackCreateAppUrl(): string {
  return `https://api.slack.com/apps?new_app=1&manifest_json=${encodeURIComponent(slackAppManifest())}`
}

/** A Slack user token as Slack issues it (xoxp-…). Checked locally before it is sent to the Hub. */
export function isSlackUserToken(value: string): boolean { return /^xoxp-[A-Za-z0-9_-]{1,1019}$/u.test(value) }

export function isSlackUserId(value: string): boolean { return /^[UW][A-Z0-9]{2,31}$/u.exec(value)?.[0] === value }
export function sameSlackContact(a: SlackIds, b: SlackIds): boolean { return a.team_id === b.team_id && a.user_id === b.user_id }
export function slackUnavailable(error: unknown): boolean {
  return error instanceof SlackBrokerUnavailable || (error instanceof HubRequestError && (error.code === 'unknown_method'
    || error.code === 'unsupported' || error.code === 'unconfigured' || error.code === 'unavailable'
    || (error.code === 'invalid_request' && error.message.startsWith('unknown hub method'))))
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
  const pending_identity = raw.pending_identity == null ? undefined : ids(raw.pending_identity)
  if (pending_identity !== undefined && connection !== null) return invalid()
  const oauth_status = raw.oauth_status
  return { configured: raw.configured, connection, contacts,
    ...(typeof raw.browser_sign_in === 'boolean' ? { browser_sign_in: raw.browser_sign_in } : {}),
    ...(pending_identity === undefined ? {} : { pending_identity }),
    ...(oauth_status === 'pending' || oauth_status === 'denied' || oauth_status === 'expired' ? { oauth_status } : {}),
  }
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

function authorization(value: unknown, trustedBrokerOrigin: string | undefined): SlackAuthorization {
  const raw = record(value)
  const href = text(raw.authorize_url)
  if (href.length > 4096 || !href.startsWith('https://') || /[\s\\#]/u.test(href)) invalid()
  let url: URL
  try { url = new URL(href) } catch { return invalid() }
  if (url.username || url.password) invalid()
  let pairingCode: string | undefined
  if (url.origin === 'https://slack.com') {
    if (url.pathname !== '/oauth/v2/authorize' || !url.searchParams.get('state') || url.searchParams.getAll('state').length !== 1) invalid()
  } else {
    // The origin is a local build setting, never a value announced by the Hub.
    if (url.origin !== trustedBrokerOrigin) throw new SlackBrokerUnavailable()
    const flow = url.searchParams.get('id')
    if (url.pathname !== '/v1/slack/authorize' || flow === null || url.search !== `?id=${flow}` || !/^[A-Za-z0-9_-]{1,256}$/u.test(flow)) invalid()
    pairingCode = text(raw.pairing_code)
    if (pairingCode.length !== 22 || !/^[A-Za-z0-9_-]+$/u.test(pairingCode) || url.href.includes(pairingCode)) invalid()
  }
  const expiry = raw.expires_at
  const expiresAt = expiry == null ? undefined : typeof expiry === 'number' ? expiry : Date.parse(text(expiry))
  if (expiresAt !== undefined && (!Number.isFinite(expiresAt) || expiresAt <= 0)) invalid()
  return { url: url.href, ...(expiresAt === undefined ? {} : { expiresAt }), ...(pairingCode === undefined ? {} : { pairingCode }) }
}

export function createSupervisorSlackClient(options: {
  readonly allowed: () => boolean
  readonly trustedBrokerOrigin?: string
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
    connect: async () => authorization(await call('supervisor.slack.connect'), options.trustedBrokerOrigin),
    connectToken: async token => {
      const trimmed = token.trim()
      if (!isSlackUserToken(trimmed)) throw new Error('Paste the User OAuth Token from your Slack app. It starts with xoxp-.')
      const result = parseStatus(await call('supervisor.slack.connect_token', { token: trimmed }))
      return result.pending_identity !== undefined && result.connection === null ? result : invalid()
    },
    confirmConnection: async account => {
      const result = parseStatus(await call('supervisor.slack.confirm_connection', ids(account)))
      return result.connection !== null && sameSlackContact(result.connection, account) ? result : invalid()
    },
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
