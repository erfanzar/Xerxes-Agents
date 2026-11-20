// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash, randomUUID } from 'node:crypto'
import { join } from 'node:path'
import { Database } from 'bun:sqlite'
import type { AuthenticatedContext, HubIdentityPort } from '../identityPort.js'
import { HUB_LIMITS, HubProtocolError, readId, utf8ByteLength, type HubSlackMethod } from '../protocol.js'
import type { HubMethodHandler } from '../server.js'
import { SUPERVISOR_DISPLAY_NAME, SUPERVISOR_PRINCIPAL_ID } from '../supervisor/audit.js'
import type { SupervisorSlackPort } from '../supervisor/slackTools.js'
import type { UserPrincipal } from '../supervisor/threads.js'
import { SlackPolicy, type SlackInbox } from './policy.js'
import { HubSlackTransport, SlackTransportError, type SlackFetch, type SlackIdentity, type SlackOAuthConfig, type SlackPeer } from './transport.js'

export const SLACK_CALLBACK_PATH = '/api/hub/v1/slack/oauth/callback'
const CLIENT_ID = 'XERXES_HUB_SLACK_CLIENT_ID'
const CLIENT_SECRET = 'XERXES_HUB_SLACK_CLIENT_SECRET'
const REDIRECT_URI = 'XERXES_HUB_SLACK_REDIRECT_URI'

export function slackOAuthFromEnvironment(env: Readonly<Record<string, string | undefined>>): SlackOAuthConfig | undefined {
  const clientId = env[CLIENT_ID]?.trim()
  const clientSecret = env[CLIENT_SECRET]?.trim()
  const redirectUri = env[REDIRECT_URI]?.trim()
  if (!clientId && !clientSecret && !redirectUri) return undefined
  if (!clientId || !clientSecret || !redirectUri) throw new Error(`Slack requires ${CLIENT_ID}, ${CLIENT_SECRET}, and ${REDIRECT_URI}.`)
  return { clientId, clientSecret, redirectUri }
}

export interface HubSlackIntegration {
  readonly handlers: Readonly<Record<HubSlackMethod, HubMethodHandler>>
  readonly tools?: SupervisorSlackPort
  callback(request: Request): Promise<Response>
  close(): Promise<void>
}

export interface HubSlackOptions {
  readonly databasePath: string
  readonly stateDir: string
  readonly identity: Pick<HubIdentityPort, 'audit' | 'onRevoked'>
  readonly isHubOwner: (id: string) => boolean
  readonly publicOrigins: readonly string[]
  readonly oauth?: SlackOAuthConfig
  readonly fetch?: SlackFetch
  readonly now?: () => number
}

const digest = (text: string): string => createHash('sha256').update(text).digest('hex')
const team = (id: unknown): id is string => typeof id === 'string' && /^T[A-Z0-9]{2,31}$/u.exec(id)?.[0] === id
const user = (id: unknown): id is string => typeof id === 'string' && /^[UW][A-Z0-9]{2,31}$/u.exec(id)?.[0] === id
const sameAccount = (a: SlackIdentity | undefined, b: SlackIdentity): boolean => a?.teamId === b.teamId && a.userId === b.userId

function requireKeys(params: Readonly<Record<string, unknown>>, ...keys: string[]): void {
  if (Object.keys(params).length !== keys.length || keys.some(key => !Object.hasOwn(params, key))) {
    throw new HubProtocolError('invalid_request', 'Invalid parameters for this Slack method.')
  }
}

function boundedRows<T>(newestFirst: readonly T[]): T[] {
  let remaining = HUB_LIMITS.normalFrameBytes / 2 - 1024
  const rows: T[] = []
  for (const row of newestFirst) {
    remaining -= utf8ByteLength(JSON.stringify(row)) + 1
    if (remaining < 0) break
    rows.push(row)
  }
  return rows
}

function dmToWire(inbox: SlackInbox): Record<string, unknown> {
  const messages = inbox.messages.map(row => ({ id: row.id, direction: row.direction, text: row.text, created_at: row.createdAt }))
  // Preserve exact reviewed draft text; drop whole older rows rather than clip individual messages.
  return { team_id: inbox.contact.teamId, user_id: inbox.contact.peerId, display_name: inbox.contact.displayName,
    messages: boundedRows(messages.reverse()).reverse(), drafts: boundedRows(inbox.drafts.map(row => ({ id: row.id, text: row.text }))) }
}

function rpcError(error: unknown): HubProtocolError {
  if (error instanceof HubProtocolError) return error
  if (!(error instanceof SlackTransportError)) return new HubProtocolError('unavailable', 'Slack is temporarily unavailable.')
  switch (error.code) {
    case 'not_approved': return new HubProtocolError('forbidden', error.message)
    case 'not_found': case 'not_connected': return new HubProtocolError('not_found', error.message)
    case 'invalid_request': return new HubProtocolError('invalid_request', error.message)
    case 'rate_limited': return new HubProtocolError('rate_limited', error.message)
    case 'oauth_unsupported': return new HubProtocolError('unconfigured', error.message)
    default: return new HubProtocolError('unavailable', error.message)
  }
}

export function createHubSlackIntegration(options: HubSlackOptions): HubSlackIntegration {
  let redirect: URL | undefined
  if (options.oauth !== undefined) {
    try { redirect = new URL(options.oauth.redirectUri) } catch { /* checked below */ }
    if (!redirect || redirect.protocol !== 'https:' || redirect.pathname !== SLACK_CALLBACK_PATH || redirect.username || redirect.password
      || redirect.search || redirect.hash || !options.publicOrigins.includes(redirect.origin)) {
      throw new Error(`Slack redirect must use an explicitly configured Hub HTTPS origin and ${SLACK_CALLBACK_PATH}.`)
    }
  }
  const transportOptions = (ownerId: string) => ({ stateDir: join(options.stateDir, `slack-owner-${digest(ownerId)}`),
    ...(options.fetch === undefined ? {} : { fetch: options.fetch }), ...(options.now === undefined ? {} : { now: options.now }) })
  const oauthOptions = options.oauth === undefined ? {} : { oauth: options.oauth }
  // Validate the operator's settings at startup; per-owner instances keep credentials and OAuth state disjoint.
  const oauthForOwner = (ownerId: string) => new HubSlackTransport({ ...transportOptions(ownerId), ...oauthOptions })
  if (options.oauth !== undefined) oauthForOwner('validation-only')
  const db = new Database(options.databasePath, { create: true, strict: true })
  const active = new Set<Promise<unknown>>()
  const shutdown = new AbortController()
  type Flow = { owner: string; deviceId: string; sessionId: string; stateHash: string; expiresAt: number; binding: string; abort: AbortController; oauth: HubSlackTransport }
  let pending: Flow | undefined
  let exchanging: Flow | undefined
  const disconnecting = new Set<string>()
  const epochs = new Map<string, AbortController>()
  const now = options.now ?? Date.now
  const isOwner = (id: string): boolean => !shutdown.signal.aborted && options.isHubOwner(id) === true
  const owner = (principal: AuthenticatedContext['principal']): UserPrincipal => {
    if (principal.kind !== 'user' || !principal.hubOwner || !isOwner(principal.id)) throw new HubProtocolError('forbidden', 'Only the current Hub owner can access Slack.')
    return principal
  }
  let policy: SlackPolicy
  const accountOf = (id: string): SlackIdentity | undefined => db.query<SlackIdentity, [string]>(
    'SELECT team_id AS teamId, user_id AS userId FROM hub_slack_connection WHERE owner_id = ?').get(id) ?? undefined
  // Epoch aborts in-flight data-plane operations when the local account is disconnected.
  const signalFor = (ownerId: string, caller?: AbortSignal): AbortSignal => {
    let epoch = epochs.get(ownerId)
    if (epoch === undefined) { epoch = new AbortController(); epochs.set(ownerId, epoch) }
    return AbortSignal.any([shutdown.signal, epoch.signal, ...(caller === undefined ? [] : [caller])])
  }
  try {
    db.exec(`PRAGMA journal_mode = WAL; PRAGMA synchronous = FULL; PRAGMA busy_timeout = 5000;
      CREATE TABLE IF NOT EXISTS hub_slack_connection (owner_id TEXT PRIMARY KEY, team_id TEXT NOT NULL, user_id TEXT NOT NULL);`)
    policy = new SlackPolicy({ db, isHubOwner: isOwner, transportForOwner: (id, approve) => new HubSlackTransport({
      ...transportOptions(id),
      approvePeer: (peer, operation) => isOwner(id) && sameAccount(accountOf(id), peer) && approve(peer, operation),
    }) })
  } catch (error) { db.close(); throw error }

  const offRevoked = options.identity.onRevoked(event => {
    for (const flow of [pending, exchanging]) {
      if (flow?.owner !== event.principalId || (event.deviceId !== undefined && flow.deviceId !== event.deviceId)
        || (event.browserSessionId !== undefined && flow.sessionId !== event.browserSessionId)) continue
      flow.abort.abort()
      if (pending === flow) pending = undefined
    }
    if (event.deviceId === undefined && event.browserSessionId === undefined) {
      epochs.get(event.principalId)?.abort()
      epochs.delete(event.principalId)
    }
  })
  const track = <T>(promise: Promise<T>): Promise<T> => {
    active.add(promise)
    return promise.finally(() => { active.delete(promise) })
  }
  const status = (principal: UserPrincipal): Record<string, unknown> => {
    owner(principal)
    const account = accountOf(principal.id)
    const contacts = account === undefined ? [] : policy.listContacts(principal).filter(contact => sameAccount(account, contact))
    return { configured: redirect !== undefined, connection: account === undefined ? null : { team_id: account.teamId, user_id: account.userId },
      contacts: contacts.map(row => ({ team_id: row.teamId, user_id: row.peerId, display_name: row.displayName, mode: row.mode })) }
  }
  const peerFor = (id: string, teamId: unknown, peerId: unknown): SlackPeer => {
    if (!team(teamId) || !user(peerId)) throw new HubProtocolError('invalid_request', 'Exact Slack team and user IDs are required.')
    const account = accountOf(id)
    if (account?.teamId !== teamId) throw new HubProtocolError('not_found', 'No connected Slack account for this team.')
    if (account.userId === peerId) throw new HubProtocolError('invalid_request', 'Select a Slack user other than the connected account.')
    return { ...account, peerId }
  }
  const audit = (context: AuthenticatedContext, method: HubSlackMethod, ids: readonly string[] = []): Promise<void> => options.identity.audit({
    actor: context.actor, action: method, requestHash: digest(JSON.stringify([method, ...ids])), decision: 'allowed', outcome: 'n/a',
  })
  const run = (method: HubSlackMethod): HubMethodHandler => ({ context, signal: socketSignal }, params) => track((async () => {
    const principal = owner(context.principal)
    socketSignal?.throwIfAborted()
    const id = principal.id
    if (method === 'supervisor.slack.status') { requireKeys(params); return status(principal) }
    if (method === 'supervisor.slack.connect') {
      requireKeys(params)
      if (!redirect) throw new HubProtocolError('unconfigured', 'Slack OAuth is not configured on this Hub.')
      await audit(context, method)
      owner(principal); socketSignal?.throwIfAborted()
      if (accountOf(id) || exchanging || disconnecting.has(id)) throw new HubProtocolError('conflict', 'Disconnect the current Slack account or finish the connection first.')
      if (pending && pending.expiresAt > now() && isOwner(pending.owner)) throw new HubProtocolError('conflict', 'Slack authorization is already pending. Finish it or wait for it to expire.')
      const binding = randomUUID()
      const oauth = oauthForOwner(id)
      const { authorizeUrl, expiresAt } = oauth.beginOAuth(binding)
      const state = new URL(authorizeUrl).searchParams.get('state')
      if (!state) throw new HubProtocolError('unavailable', 'Could not start Slack authorization.')
      pending = { owner: id, deviceId: principal.deviceId, sessionId: context.browserSessionId, stateHash: digest(state), expiresAt, binding, abort: new AbortController(), oauth }
      return { authorize_url: authorizeUrl }
    }
    if (method === 'supervisor.slack.disconnect') {
      requireKeys(params, 'team_id', 'user_id')
      if (!team(params.team_id) || !user(params.user_id)) throw new HubProtocolError('invalid_request', 'Exact Slack account IDs are required.')
      const account = { teamId: params.team_id, userId: params.user_id }
      await audit(context, method, [account.teamId, account.userId]); owner(principal); socketSignal?.throwIfAborted()
      if (!sameAccount(accountOf(id), account) || disconnecting.has(id)) throw new HubProtocolError('not_found', 'No matching connected Slack account.')
      disconnecting.add(id)
      try {
        db.transaction(() => {
          for (const contact of policy.listContacts(principal)) if (sameAccount(account, contact)) policy.removeContact(principal, contact)
          db.query('DELETE FROM hub_slack_connection WHERE owner_id = ?').run(id)
        }).immediate()
        epochs.get(id)?.abort(); epochs.delete(id)
        await oauthForOwner(id).disconnect(account)
      } finally { disconnecting.delete(id) }
      return status(principal)
    }
    if (!redirect) throw new HubProtocolError('unconfigured', 'Slack OAuth is not configured on this Hub.')
    const extra = method === 'supervisor.slack.contact.mode' ? ['mode'] : method === 'supervisor.slack.draft.send' ? ['draft_id', 'text'] : []
    requireKeys(params, 'team_id', 'user_id', ...extra)
    const peer = peerFor(id, params.team_id, params.user_id)
    const signal = signalFor(id, socketSignal)
    if (method === 'supervisor.slack.contact.lookup') {
      const profile = await policy.lookupContact(principal, peer, signal)
      owner(principal); signal.throwIfAborted()
      return { identity: { team_id: peer.teamId, user_id: profile.userId, display_name: profile.displayName } }
    }
    if (method === 'supervisor.slack.dm.open') {
      await policy.poll(id, peer, signal)
      owner(principal); signal.throwIfAborted()
      return dmToWire(policy.readInbox(principal, peer))
    }
    const mode = params.mode
    const text = params.text
    const draftId = method === 'supervisor.slack.draft.send' ? readId(params.draft_id, 'draft_id') : undefined
    if (method === 'supervisor.slack.contact.mode' && mode !== 'draft' && mode !== 'auto') throw new HubProtocolError('invalid_request', 'Invalid Slack contact mode.')
    await audit(context, method, [peer.teamId, peer.userId, peer.peerId, ...(draftId === undefined ? [] : [draftId]), ...(typeof mode === 'string' ? [mode] : [])]); owner(principal); signal.throwIfAborted()
    switch (method) {
      case 'supervisor.slack.contact.confirm': await policy.confirmContact(principal, peer, signal); break
      case 'supervisor.slack.contact.remove': policy.removeContact(principal, peer); break
      case 'supervisor.slack.contact.mode': if (mode === 'auto' || mode === 'draft') policy.setContactMode(principal, peer, mode); break
      case 'supervisor.slack.draft.send': {
        if (draftId === undefined || typeof text !== 'string') throw new HubProtocolError('invalid_request', 'Exact reviewed Slack text is required.')
        const result = await policy.sendDraft(principal, peer, { draftId, text }, signal)
        owner(principal); signal.throwIfAborted()
        if (result.state !== 'sent') throw new HubProtocolError('conflict', 'Slack delivery could not be confirmed. Check the DM before taking further action.')
        return dmToWire(policy.readInbox(principal, peer))
      }
    }
    return status(principal)
  })().catch((error: unknown) => { throw rpcError(error) }))

  const callback = async (request: Request): Promise<Response> => {
    const response = (status: number, text: string, allow?: string): Response => new Response(text, { status, headers: {
      'content-type': 'text/plain; charset=utf-8', 'cache-control': 'no-store', 'referrer-policy': 'no-referrer',
      ...(allow === undefined ? {} : { allow }),
    } })
    if (!redirect || shutdown.signal.aborted || request.headers.get('host') !== redirect.host) return response(404, 'Not found.')
    if (request.method !== 'GET') return response(405, 'Method not allowed.', 'GET')
    const flow = pending
    const url = new URL(request.url)
    if (url.pathname !== SLACK_CALLBACK_PATH) return response(404, 'Not found.')
    const states = url.searchParams.getAll('state')
    const state = states[0]
    if (!flow || states.length !== 1 || !state || !/^[A-Za-z0-9_-]{43}$/u.test(state) || digest(state) !== flow.stateHash
      || flow.expiresAt <= now() || !isOwner(flow.owner)) return response(400, 'Invalid or expired Slack authorization. Return to Xerxes.')
    // The external browser need not share desktop cookies. The unguessable state binds it to the audited owner initiation.
    pending = undefined
    exchanging = flow
    try {
      const codes = url.searchParams.getAll('code')
      const errors = url.searchParams.getAll('error')
      const account = await flow.oauth.completeOAuth(flow.binding, { state,
        ...(codes.length === 1 && errors.length === 0 && codes[0] !== undefined ? { code: codes[0] } : { error: 'denied' }) },
      AbortSignal.any([shutdown.signal, flow.abort.signal]))
      if (!team(account.teamId) || !user(account.userId) || flow.abort.signal.aborted || !isOwner(flow.owner)) {
        await flow.oauth.disconnect(account)
        return response(400, 'Slack authorization could not be saved. Return to Xerxes.')
      }
      try { db.query('INSERT INTO hub_slack_connection VALUES (?, ?, ?)').run(flow.owner, account.teamId, account.userId) }
      catch (error) { await flow.oauth.disconnect(account); throw error }
      return response(200, 'Slack is connected. Return to Xerxes.')
    } catch { return response(400, 'Slack authorization did not complete. Return to Xerxes.') }
    finally { exchanging = undefined }
  }

  const tools: SupervisorSlackPort = {
    isOwner: id => isOwner(id) && accountOf(id) !== undefined,
    contacts: id => {
      if (!isOwner(id)) throw new Error('Hub owner required.')
      const account = accountOf(id)
      return account === undefined ? [] : policy.contactsForPoll(id).filter(row => sameAccount(account, row))
    },
    prepare: (id, target, text, signal, requestId) => track((async () => {
      if (!isOwner(id)) throw new Error('Hub owner required.')
      const peer = peerFor(id, target.teamId, target.peerId)
      const turn = policy.supervisorContext({ kind: 'agent', id: SUPERVISOR_PRINCIPAL_ID,
        displayName: SUPERVISOR_DISPLAY_NAME, onBehalfOf: id }, id, peer)
      const result = await policy.reply(turn, { requestId, text }, signalFor(id, signal))
      if (result.state === 'pending') return 'drafted' as const
      if (result.state === 'sent') return 'sent' as const
      throw new Error('Slack delivery is uncertain.')
    })()),
  }
  let closing: Promise<void> | undefined
  return {
    handlers: {
      'supervisor.slack.status': run('supervisor.slack.status'), 'supervisor.slack.connect': run('supervisor.slack.connect'),
      'supervisor.slack.disconnect': run('supervisor.slack.disconnect'), 'supervisor.slack.contact.lookup': run('supervisor.slack.contact.lookup'),
      'supervisor.slack.contact.confirm': run('supervisor.slack.contact.confirm'), 'supervisor.slack.contact.remove': run('supervisor.slack.contact.remove'),
      'supervisor.slack.contact.mode': run('supervisor.slack.contact.mode'), 'supervisor.slack.dm.open': run('supervisor.slack.dm.open'),
      'supervisor.slack.draft.send': run('supervisor.slack.draft.send'),
    },
    ...(redirect === undefined ? {} : { tools }),
    callback: request => track(callback(request)),
    close: () => closing ??= (async () => {
      shutdown.abort(); offRevoked(); pending = undefined
      await Promise.allSettled(active)
      db.close()
    })(),
  }
}
