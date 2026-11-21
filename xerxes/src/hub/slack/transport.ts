// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { SlackCredentialStore } from './credentials.js'
import { checkAbort, SlackHttp, type SlackFetch } from './http.js'
import { parseUserGrant, SlackOAuthState, type SlackOAuthCallback, type SlackOAuthConfig, type SlackOAuthStart, type SlackUserGrant } from './oauth.js'
import {
  SlackTransportError, type SlackDirectMessage, type SlackIdentity, type SlackMessage, type SlackPeer, type SlackPeerApproval,
  type SlackPeerOperation, type SlackProfile, type SlackTransport,
} from './types.js'
import { channelId, identity, record, timestamp, userId } from './wire.js'

export * from './types.js'
export type { SlackFetch, SlackOAuthConfig, SlackOAuthStart, SlackOAuthCallback }

export interface HubSlackTransportOptions {
  readonly stateDir: string
  readonly oauth?: SlackOAuthConfig
  readonly approvePeer?: SlackPeerApproval
  readonly fetch?: SlackFetch
  readonly timeoutMs?: number
  /** Clock for deterministic tests of OAuth state expiry. */
  readonly now?: () => number
}

const MAX_HISTORY = 15
const MAX_TEXT = 4000

function normalizePeer(input: SlackPeer): SlackPeer {
  const account = identity(input)
  if (!userId(input.peerId) || input.peerId === account.userId) throw new SlackTransportError('invalid_request')
  return { ...account, peerId: input.peerId }
}

function normalizeDm(input: SlackDirectMessage): SlackDirectMessage {
  const peer = normalizePeer(input)
  if (!channelId(input.channelId)) throw new SlackTransportError('invalid_request')
  return { ...peer, channelId: input.channelId }
}

const NONLOCAL_OR_NON_DM = [
  'is_mpim', 'is_channel', 'is_group', 'is_archived', 'is_shared', 'is_org_shared',
  'is_ext_shared', 'is_ext_ws_shared', 'is_pending_ext_shared', 'is_frozen',
] as const

function verifiedChannel(data: unknown, peer: SlackPeer, expected?: string): string {
  const channel = record(data)
  if (!channelId(channel?.id) || (expected !== undefined && channel.id !== expected) || channel.user !== peer.peerId
    || channel.is_im !== true || channel.is_member === false
    || NONLOCAL_OR_NON_DM.some(flag => channel[flag] !== undefined && channel[flag] !== false)
    || (channel.context_team_id !== undefined && channel.context_team_id !== peer.teamId)) throw new SlackTransportError('not_dm')
  return channel.id
}

function sameAccount(a: SlackIdentity, b: SlackIdentity): boolean {
  return a.teamId === b.teamId && a.userId === b.userId
}

/** Hub-only transport. It exposes neither individual credentials nor arbitrary Slack calls. */
export class HubSlackTransport implements SlackTransport {
  readonly #credentials: SlackCredentialStore
  readonly #oauth: SlackOAuthState
  readonly #http: SlackHttp
  readonly #approval: SlackPeerApproval | undefined

  constructor(options: HubSlackTransportOptions) {
    this.#credentials = new SlackCredentialStore(options.stateDir)
    this.#oauth = new SlackOAuthState(options.oauth, options.now ?? Date.now)
    this.#http = new SlackHttp(options.fetch ?? ((url, init) => fetch(url, init)), options.timeoutMs ?? 10_000)
    this.#approval = options.approvePeer
  }

  beginOAuth(binding: string, expectedIdentity?: SlackIdentity): SlackOAuthStart {
    return this.#oauth.begin(binding, expectedIdentity)
  }

  async completeOAuth(binding: string, callback: SlackOAuthCallback, signal?: AbortSignal): Promise<SlackIdentity> {
    const grant = await this.verifyOAuth(binding, callback, signal)
    await this.saveGrant(grant, signal)
    return grant.account
  }

  /** Exchange and verify, without writing any credential before the Hub owner's review. */
  async verifyOAuth(binding: string, callback: SlackOAuthCallback, signal?: AbortSignal): Promise<SlackUserGrant> {
    checkAbort(signal)
    const { body, authorization, expected } = this.#oauth.consume(binding, callback)
    const data = await this.#http.request('oauth.v2.access', authorization, body, signal)
    const grant = parseUserGrant(data)
    if (expected && !sameAccount(expected, grant.account)) throw new SlackTransportError('identity_mismatch')
    await this.verifyGrant(grant, signal)
    return grant
  }

  async verifyGrant(grant: SlackUserGrant, signal?: AbortSignal): Promise<void> {
    await this.#checkIdentity(grant.account, `Bearer ${grant.token}`, signal)
  }

  async saveGrant(grant: SlackUserGrant, signal?: AbortSignal): Promise<void> {
    checkAbort(signal)
    await this.#credentials.save(grant.account, grant.token, signal)
  }

  /** Removes the local credential only. Does not revoke the Slack grant remotely. */
  async disconnect(account: SlackIdentity): Promise<void> {
    await this.#credentials.remove(account)
  }

  async #bearer(account: SlackIdentity, signal?: AbortSignal): Promise<string> {
    checkAbort(signal)
    const token = await this.#credentials.load(account)
    checkAbort(signal)
    if (!token) throw new SlackTransportError('not_connected')
    return `Bearer ${token}`
  }

  async #checkIdentity(account: SlackIdentity, bearer: string, signal?: AbortSignal): Promise<SlackIdentity> {
    const data = await this.#http.request('auth.test', bearer, new URLSearchParams(), signal)
    if (data.team_id !== account.teamId || data.user_id !== account.userId || data.bot_id !== undefined) {
      throw new SlackTransportError('identity_mismatch')
    }
    return account
  }

  async authTest(input: SlackIdentity, signal?: AbortSignal): Promise<SlackIdentity> {
    const account = identity(input)
    return this.#checkIdentity(account, await this.#bearer(account, signal), signal)
  }

  async lookupProfile(input: SlackIdentity, exact: { readonly userId: string } | { readonly email: string }, signal?: AbortSignal): Promise<SlackProfile> {
    const account = identity(input)
    const query = record(exact)
    if (!query || Object.keys(query).length !== 1) throw new SlackTransportError('invalid_request')
    const id = query.userId
    const email = typeof query.email === 'string' ? query.email.trim().toLowerCase() : undefined
    if (!userId(id) && (!email || email.length > 254 || !/^[^\s<>@\x00-\x1f\x7f]+@[^\s<>@\x00-\x1f\x7f]+\.[^\s<>@\x00-\x1f\x7f]+$/u.test(email))) {
      throw new SlackTransportError('invalid_request')
    }
    const bearer = await this.#bearer(account, signal)
    const data = userId(id)
      ? await this.#http.request('users.info', bearer, new URLSearchParams({ user: id }), signal)
      : await this.#http.request('users.lookupByEmail', bearer, new URLSearchParams({ email: email ?? '' }), signal)
    const user = record(data.user)
    const profile = record(user?.profile)
    if (!userId(user?.id) || (userId(id) && id !== user.id) || (user.team_id !== undefined && user.team_id !== account.teamId)
      || user.deleted === true || user.is_bot === true || user.is_app_user === true
      || (email !== undefined && (typeof profile?.email !== 'string' || profile.email.toLowerCase() !== email))) throw new SlackTransportError('not_found')
    const name = [profile?.display_name, profile?.real_name, user.real_name].find(value => typeof value === 'string' && value.trim())
    return { userId: user.id, displayName: typeof name === 'string' ? name.trim().slice(0, 160) : user.id }
  }

  async #approve(peer: SlackPeer, operation: SlackPeerOperation, signal?: AbortSignal): Promise<void> {
    checkAbort(signal)
    const { teamId, userId: connectedUserId, peerId } = peer
    let allowed: boolean | undefined
    try { allowed = await this.#approval?.({ teamId, userId: connectedUserId, peerId }, operation) }
    catch { checkAbort(signal); throw new SlackTransportError('not_approved') }
    checkAbort(signal)
    if (allowed !== true) throw new SlackTransportError('not_approved')
  }

  async openPeer(input: SlackPeer, signal?: AbortSignal): Promise<SlackDirectMessage> {
    const peer = normalizePeer(input)
    await this.#approve(peer, 'open', signal)
    const bearer = await this.#bearer(peer, signal)
    const data = await this.#http.request('conversations.open', bearer, { users: peer.peerId, return_im: true }, signal)
    return { ...peer, channelId: verifiedChannel(data.channel, peer) }
  }

  async #verify(dm: SlackDirectMessage, operation: SlackPeerOperation, signal?: AbortSignal): Promise<string> {
    await this.#approve(dm, operation, signal)
    const bearer = await this.#bearer(dm, signal)
    const data = await this.#http.request('conversations.info', bearer, new URLSearchParams({ channel: dm.channelId }), signal)
    verifiedChannel(data.channel, dm, dm.channelId)
    return bearer
  }

  async verifyDm(input: SlackDirectMessage, signal?: AbortSignal): Promise<SlackDirectMessage> {
    const dm = normalizeDm(input)
    await this.#verify(dm, 'verify', signal)
    return dm
  }

  async #beforeContent(dm: SlackDirectMessage, operation: 'read' | 'send', signal?: AbortSignal): Promise<string> {
    const bearer = await this.#verify(dm, operation, signal)
    // Disconnect/reconnect or approval revocation during conversations.info must stop this request.
    if (bearer !== await this.#bearer(dm, signal)) throw new SlackTransportError('identity_mismatch')
    await this.#approve(dm, operation, signal)
    return bearer
  }

  async readDm(input: SlackDirectMessage, limit = MAX_HISTORY, signal?: AbortSignal): Promise<readonly SlackMessage[]> {
    const dm = normalizeDm(input)
    if (!Number.isSafeInteger(limit) || limit < 1 || limit > MAX_HISTORY) throw new SlackTransportError('invalid_request')
    const bearer = await this.#beforeContent(dm, 'read', signal)
    const data = await this.#http.request('conversations.history', bearer, new URLSearchParams({ channel: dm.channelId, limit: String(limit) }), signal)
    if (!Array.isArray(data.messages)) throw new SlackTransportError('invalid_response')
    const messages: SlackMessage[] = []
    for (const value of data.messages.slice(0, limit)) {
      const msg = record(value)
      if (msg?.type === 'message' && msg.subtype === undefined && (msg.user === dm.userId || msg.user === dm.peerId)
        && msg.bot_id === undefined && timestamp(msg.ts) && typeof msg.text === 'string') {
        messages.push({ ts: msg.ts, userId: msg.user, text: msg.text.slice(0, MAX_TEXT) })
      }
    }
    return messages
  }

  async sendDm(input: SlackDirectMessage, text: string, signal?: AbortSignal): Promise<{ channelId: string; ts: string }> {
    const dm = normalizeDm(input)
    if (typeof text !== 'string' || !text.trim() || text.length > MAX_TEXT || text.includes('\0')) throw new SlackTransportError('invalid_request')
    const bearer = await this.#beforeContent(dm, 'send', signal)
    const plain = text.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;')
    const data = await this.#http.request('chat.postMessage', bearer, {
      channel: dm.channelId, text: plain, mrkdwn: false, parse: 'none', unfurl_links: false, unfurl_media: false,
    }, signal)
    if (data.channel !== dm.channelId || !timestamp(data.ts)) throw new SlackTransportError('invalid_response')
    return { channelId: dm.channelId, ts: data.ts }
  }
}
