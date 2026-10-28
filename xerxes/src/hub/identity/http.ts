// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { HubHttpRequestInfo } from '../identityPort.js'
import { auditAction } from './audit.js'
import { clearHostCookie, parseCookieHeader, REFRESH_COOKIE, SESSION_COOKIE } from './cookies.js'
import { HubIdentityError, errorResponse, jsonResponse, sha256Hex, type IdentityContext, type IssuedTokens } from './core.js'
import { listDevices, revokeDevice } from './devices.js'
import { updateGrants } from './grants.js'
import { acceptInvite, createInvite } from './invites.js'
import { subscribePush, unsubscribePush } from './notifications.js'
import { confirmPairing, consumePairing, createPairing } from './pairing.js'
import type { DeviceRow, UserRow } from './store.js'
import {
  authenticateRequest,
  cookiesFor,
  csrfFor,
  deviceForRefreshToken,
  originAllowed,
  refreshTokens,
  type Authed,
} from './tokens.js'
import {
  assertionOptions,
  passkeysFor,
  registrationOptions,
  removePasskey,
  requireRecentStepUp,
  verifyAssertion,
  verifyRegistration,
} from './webauthn.js'
import { isHubRole, type HubRole } from '../../security/principal.js'

/**
 * Identity HTTP routes under `/api/hub/v1/`. Unauthenticated routes
 * (`pair/consume`, `invites/accept`, `session/refresh`, `session/logout`) take
 * secrets only in the JSON body or the `__Host-` cookies. Every other route
 * needs a live browser session; mutating ones also need `x-xerxes-csrf`.
 * Sensitive administration (invites, grants, pairing for other devices,
 * revoking another user's device, extra passkeys) additionally needs the owner
 * and a recent passkey step-up; without a passkey it is refused and must be done
 * from the local admin API.
 */

const PREFIX = '/api/hub/v1/'
const MAX_BODY_BYTES = 64 * 1024

interface Call {
  readonly ctx: IdentityContext
  readonly request: Request
  readonly info: HubHttpRequestInfo
}
type Route = (call: Call) => Promise<Response> | Response
type Method = 'GET' | 'POST'

function mergeInfo(request: Request, info: HubHttpRequestInfo): HubHttpRequestInfo {
  const headers = request.headers
  const cookie = info.cookie ?? headers.get('cookie') ?? undefined
  const origin = info.origin ?? headers.get('origin') ?? undefined
  const host = info.host ?? headers.get('host') ?? undefined
  const csrfToken = info.csrfToken ?? headers.get('x-xerxes-csrf') ?? undefined
  return {
    method: request.method,
    url: request.url,
    ...(cookie === undefined ? {} : { cookie }),
    ...(origin === undefined ? {} : { origin }),
    ...(host === undefined ? {} : { host }),
    ...(csrfToken === undefined ? {} : { csrfToken }),
    ...(info.remoteAddress === undefined ? {} : { remoteAddress: info.remoteAddress }),
  }
}

async function readJson(request: Request): Promise<Record<string, unknown>> {
  const type = request.headers.get('content-type') ?? ''
  if (!type.toLowerCase().startsWith('application/json')) throw new HubIdentityError('invalid_request', 'content-type must be application/json')
  if (Number(request.headers.get('content-length') ?? '0') > MAX_BODY_BYTES) throw new HubIdentityError('invalid_request', 'request body is too large')
  const text = await request.text()
  if (Buffer.byteLength(text, 'utf8') > MAX_BODY_BYTES) throw new HubIdentityError('invalid_request', 'request body is too large')
  let value: unknown
  try {
    value = JSON.parse(text) as unknown
  } catch {
    throw new HubIdentityError('invalid_request', 'request body is not valid JSON')
  }
  if (typeof value !== 'object' || value === null || Array.isArray(value)) throw new HubIdentityError('invalid_request', 'request body must be a JSON object')
  return value as Record<string, unknown>
}

function text(body: Record<string, unknown>, field: string, max = 500): string {
  const value = body[field]
  if (typeof value !== 'string' || value.length === 0 || value.length > max) {
    throw new HubIdentityError('invalid_request', `${field} must be a non-empty string of at most ${max} characters`)
  }
  return value
}

function optionalText(body: Record<string, unknown>, field: string, max = 500): string | undefined {
  return body[field] === undefined ? undefined : text(body, field, max)
}

/** Unauthenticated POSTs: an `Origin`, when a browser sends one, must be ours. */
function gateOrigin(call: Call, options: { readonly required?: boolean } = {}): void {
  const origin = call.info.origin
  if (origin === undefined) {
    if (options.required) throw new HubIdentityError('forbidden', 'an allowed Origin header is required')
    return
  }
  if (!originAllowed(call.ctx, origin)) throw new HubIdentityError('forbidden', 'origin is not allowed')
}

function requireUser(call: Call): Authed {
  const result = authenticateRequest(call.ctx, call.info)
  if (result.ok) return result.value
  if (result.reason === 'origin') throw new HubIdentityError('forbidden', 'origin is not allowed')
  if (result.reason === 'csrf') throw new HubIdentityError('forbidden', 'csrf token missing or invalid')
  throw new HubIdentityError('unauthorized', 'a valid session is required')
}

function requireOwnerStepUp(call: Call, authed: Authed): void {
  if (authed.user.is_owner !== 1) throw new HubIdentityError('forbidden', 'only the hub owner can do this')
  requireRecentStepUp(call.ctx, authed.session.id)
}

function iso(ms: number): string {
  return new Date(ms).toISOString()
}

function sessionResponse(
  ctx: IdentityContext,
  status: 'active' | 'pending_confirmation',
  tokens: IssuedTokens,
  user: Pick<UserRow, 'id' | 'display_name'>,
  device: Pick<DeviceRow, 'id'>,
  extra: Record<string, unknown> = {},
): Response {
  return jsonResponse(
    200,
    {
      status,
      user: { id: user.id, display_name: user.display_name },
      device_id: device.id,
      csrf_token: tokens.csrfToken,
      session_expires_at: iso(tokens.sessionExpiresAt),
      ...extra,
    },
    cookiesFor(ctx, tokens),
  )
}

/**
 * `reason` tells the client why (for example `no_session`: this browser holds no sign-in at all,
 * which on iPhone usually means Safari was paired but the Home Screen app was not). It names no secret.
 */
function endedResponse(reason?: string): Response {
  return jsonResponse(401, { error: { code: 'unauthorized', message: 'the session ended; pair this device again', ...(reason === undefined ? {} : { reason }) } }, [clearHostCookie(REFRESH_COOKIE), clearHostCookie(SESSION_COOKIE)])
}

/** Record why a refresh was refused, so a phone that "signs out" leaves evidence. Never stores the token. */
function auditRefusal(call: Call, reason: string, token?: string): void {
  try {
    let device = ''
    if (token !== undefined) {
      try { device = deviceForRefreshToken(call.ctx, token)?.device.id ?? '' } catch { device = '' }
    }
    auditAction(call.ctx, {
      actor: { principal_kind: 'user', principal_id: 'anonymous', display_name: 'Unauthenticated caller' },
      action: `identity.refresh.refused:${reason}${device ? `:${device}` : ''}`,
      decision: 'denied',
      outcome: 'rejected',
    })
  } catch { /* Diagnostics must never change the answer. */ }
}

const consume: Route = async call => {
  gateOrigin(call)
  const body = await readJson(call.request)
  const consumed = consumePairing(call.ctx, {
    pairingId: text(body, 'pairing_id', 100),
    code: text(body, 'code', 200),
    ...(optionalText(body, 'device_label', 80) === undefined ? {} : { deviceLabel: optionalText(body, 'device_label', 80) as string }),
    ...(call.info.remoteAddress === undefined ? {} : { remoteAddress: call.info.remoteAddress }),
  })
  return sessionResponse(call.ctx, consumed.status, consumed.tokens, consumed.user, consumed.device, {
    comparison_code: consumed.comparisonCode,
    hub_fingerprint: call.ctx.signerFingerprint,
  })
}

const accept: Route = async call => {
  gateOrigin(call)
  const body = await readJson(call.request)
  const displayName = optionalText(body, 'display_name', 60)
  const deviceLabel = optionalText(body, 'device_label', 80)
  const accepted = acceptInvite(call.ctx, {
    inviteId: text(body, 'invite_id', 100),
    code: text(body, 'code', 200),
    ...(displayName === undefined ? {} : { displayName }),
    ...(deviceLabel === undefined ? {} : { deviceLabel }),
    ...(call.info.remoteAddress === undefined ? {} : { remoteAddress: call.info.remoteAddress }),
  })
  return sessionResponse(call.ctx, accepted.status, accepted.tokens, accepted.user, accepted.device, {
    comparison_code: accepted.comparisonCode,
    hub_fingerprint: call.ctx.signerFingerprint,
  })
}

const refresh: Route = call => {
  gateOrigin(call, { required: true })
  const now = call.ctx.clock.now()
  if (!call.ctx.rate.allow(`refresh:${call.info.remoteAddress ?? 'unknown'}`, 60, 60_000, now)) {
    throw new HubIdentityError('rate_limited', 'too many refresh attempts')
  }
  const token = parseCookieHeader(call.info.cookie).get(REFRESH_COOKIE)
  if (token === undefined) {
    auditRefusal(call, 'no_session')
    return endedResponse('no_session')
  }
  const outcome = refreshTokens(call.ctx, token)
  if (outcome.kind === 'ok') return sessionResponse(call.ctx, 'active', outcome.tokens, outcome.user, outcome.device)
  if (outcome.kind === 'pending') return jsonResponse(202, { status: 'pending_confirmation' })
  const reason = outcome.kind === 'invalid' ? outcome.reason ?? 'invalid' : outcome.kind
  auditRefusal(call, reason, token)
  return endedResponse(reason)
}

const logout: Route = call => {
  gateOrigin(call, { required: true })
  const cookies = parseCookieHeader(call.info.cookie)
  const refreshToken = cookies.get(REFRESH_COOKIE)
  let target = refreshToken === undefined ? undefined : deviceForRefreshToken(call.ctx, refreshToken)
  if (!target) {
    const authed = authenticateRequest(call.ctx, call.info, { requireCsrf: false })
    if (authed.ok) target = { device: authed.value.device, user: authed.value.user }
  }
  if (target) revokeDevice(call.ctx, target.device.id, 'logout', { principal_kind: 'user', principal_id: target.user.id, display_name: target.user.display_name, user_id: target.user.id, device_id: target.device.id })
  return jsonResponse(200, { ok: true }, [clearHostCookie(REFRESH_COOKIE), clearHostCookie(SESSION_COOKIE)])
}

const whoami: Route = call => {
  const authed = requireUser(call)
  return jsonResponse(200, {
    user: { id: authed.user.id, display_name: authed.user.display_name, owner: authed.user.is_owner === 1 },
    device_id: authed.device.id,
    csrf_token: csrfFor(call.ctx, authed.sessionToken),
    session_expires_at: iso(authed.session.expires_at),
    passkeys: passkeysFor(call.ctx, authed.user.id).length,
  })
}

const devicesList: Route = call => {
  const authed = requireUser(call)
  const all = new URL(call.request.url).searchParams.get('all') === '1' && authed.user.is_owner === 1
  const devices = listDevices(call.ctx, all ? {} : { userId: authed.user.id }).map(device => ({
    id: device.id,
    user_id: device.userId,
    display_name: device.displayName,
    label: device.label,
    status: device.status,
    created_at: iso(device.createdAt),
    last_seen_at: device.lastSeenAt === null ? null : iso(device.lastSeenAt),
    expires_at: iso(device.expiresAt),
    current: device.id === authed.device.id,
  }))
  return jsonResponse(200, { devices })
}

const devicesRevoke: Route = async call => {
  const authed = requireUser(call)
  const body = await readJson(call.request)
  const deviceId = text(body, 'device_id', 100)
  const target = call.ctx.store.one<DeviceRow>('SELECT * FROM hub_devices WHERE id = ?', deviceId)
  if (!target) throw new HubIdentityError('not_found', 'device does not exist')
  if (target.user_id !== authed.user.id) requireOwnerStepUp(call, authed)
  revokeDevice(call.ctx, deviceId, 'revoked_by_user', authed.auth.actor)
  return jsonResponse(200, { ok: true })
}

const pairingsCreate: Route = async call => {
  const authed = requireUser(call)
  requireOwnerStepUp(call, authed)
  const body = await readJson(call.request)
  const role = optionalText(body, 'role', 20)
  if (role !== undefined && !isHubRole(role)) throw new HubIdentityError('invalid_request', 'unknown role')
  const label = optionalText(body, 'label', 80)
  const created = createPairing(
    call.ctx,
    { userId: optionalText(body, 'user_id', 100) ?? authed.user.id, ...(label === undefined ? {} : { label }), ...(role === undefined ? {} : { role: role as HubRole }) },
    authed.auth.actor,
  )
  return jsonResponse(201, {
    pairing_id: created.pairingId,
    url: created.url,
    comparison_code: created.comparisonCode,
    expires_at: iso(created.expiresAt),
    qr_svg: created.qrSvg,
  })
}

const pairingsConfirm: Route = async call => {
  const authed = requireUser(call)
  requireOwnerStepUp(call, authed)
  const body = await readJson(call.request)
  const confirmed = confirmPairing(call.ctx, text(body, 'pairing_id', 100), text(body, 'comparison_code', 20), authed.auth.actor)
  return jsonResponse(200, { ok: true, device_id: confirmed.deviceId })
}

const invitesCreate: Route = async call => {
  const authed = requireUser(call)
  requireOwnerStepUp(call, authed)
  const body = await readJson(call.request)
  const role = text(body, 'role', 20)
  if (!isHubRole(role)) throw new HubIdentityError('invalid_request', 'unknown role')
  const sessionId = optionalText(body, 'session_id', 200)
  const label = optionalText(body, 'label', 80)
  const expires = body.expires_in_ms
  if (expires !== undefined && typeof expires !== 'number') throw new HubIdentityError('invalid_request', 'expires_in_ms must be a number')
  const created = createInvite(
    call.ctx,
    {
      createdBy: authed.user.id,
      workspaceId: text(body, 'workspace_id', 200),
      ...(sessionId === undefined ? {} : { sessionId }),
      role,
      ...(typeof expires === 'number' ? { expiresInMs: expires } : {}),
      ...(label === undefined ? {} : { label }),
    },
    authed.auth.actor,
  )
  return jsonResponse(201, {
    invite_id: created.inviteId,
    url: created.url,
    comparison_code: created.comparisonCode,
    expires_at: iso(created.expiresAt),
    resources: created.resources,
    role: created.role,
    qr_svg: created.qrSvg,
  })
}

const grantsUpdate: Route = async call => {
  const authed = requireUser(call)
  requireOwnerStepUp(call, authed)
  const body = await readJson(call.request)
  if (!Array.isArray(body.grants)) throw new HubIdentityError('invalid_request', 'grants must be an array')
  const mode = body.mode
  if (mode !== undefined && mode !== 'replace' && mode !== 'add') throw new HubIdentityError('invalid_request', 'mode must be replace or add')
  const grants = updateGrants(call.ctx, { principalId: text(body, 'principal_id', 100), grants: body.grants as unknown[], ...(mode === undefined ? {} : { mode }) }, authed.auth.actor)
  return jsonResponse(200, { grants })
}

const passkeysList: Route = call => {
  const authed = requireUser(call)
  return jsonResponse(200, {
    passkeys: passkeysFor(call.ctx, authed.user.id).map(key => ({ id: key.id, label: key.label, alg: key.alg, created_at: iso(key.created_at), last_used_at: key.last_used_at === null ? null : iso(key.last_used_at) })),
  })
}

const registerOptions: Route = call => {
  const authed = requireUser(call)
  // The first passkey bootstraps step-up; further ones need an existing one.
  if (passkeysFor(call.ctx, authed.user.id).length > 0) requireRecentStepUp(call.ctx, authed.session.id)
  return jsonResponse(200, registrationOptions(call.ctx, authed.user, authed.session))
}

const registerVerify: Route = async call => {
  const authed = requireUser(call)
  if (passkeysFor(call.ctx, authed.user.id).length > 0) requireRecentStepUp(call.ctx, authed.session.id)
  const body = await readJson(call.request)
  const created = verifyRegistration(call.ctx, authed.user, authed.session, { clientDataJSON: body.clientDataJSON, attestationObject: body.attestationObject, label: body.label })
  auditAction(call.ctx, { actor: authed.auth.actor, action: `identity.passkey.register:${sha256Hex(created.id).slice(0, 16)}` })
  return jsonResponse(201, { id: created.id, alg: created.alg, label: created.label })
}

const assertOptions: Route = call => {
  const authed = requireUser(call)
  return jsonResponse(200, assertionOptions(call.ctx, authed.user, authed.session))
}

const assertVerify: Route = async call => {
  const authed = requireUser(call)
  const body = await readJson(call.request)
  const result = verifyAssertion(call.ctx, authed.user, authed.session, { id: body.id, clientDataJSON: body.clientDataJSON, authenticatorData: body.authenticatorData, signature: body.signature })
  auditAction(call.ctx, { actor: authed.auth.actor, action: 'identity.passkey.step_up' })
  return jsonResponse(200, { ok: true, step_up_at: iso(result.stepUpAt) })
}

const passkeysRemove: Route = async call => {
  const authed = requireUser(call)
  requireRecentStepUp(call.ctx, authed.session.id)
  const id = text(await readJson(call.request), 'id', 1400)
  removePasskey(call.ctx, authed.user.id, id)
  auditAction(call.ctx, { actor: authed.auth.actor, action: `identity.passkey.remove:${sha256Hex(id).slice(0, 16)}` })
  return jsonResponse(200, { ok: true })
}

const vapidKey: Route = call => {
  requireUser(call)
  return jsonResponse(200, { public_key: call.ctx.vapid.publicKey })
}

const pushSubscribe: Route = async call => {
  const authed = requireUser(call)
  const body = await readJson(call.request)
  const row = subscribePush(call.ctx, { userId: authed.user.id, deviceId: authed.device.id }, { endpoint: body.endpoint, keys: body.keys })
  auditAction(call.ctx, { actor: authed.auth.actor, action: `identity.push.subscribe:${row.id}` })
  return jsonResponse(201, { subscription_id: row.id })
}

const pushUnsubscribe: Route = async call => {
  const authed = requireUser(call)
  const body = await readJson(call.request)
  const removed = unsubscribePush(call.ctx, { deviceId: authed.device.id }, body.endpoint)
  return jsonResponse(200, { removed })
}

const ROUTES: ReadonlyMap<string, Partial<Record<Method, Route>>> = new Map([
  ['pair/consume', { POST: consume }],
  ['invites/accept', { POST: accept }],
  ['session/refresh', { POST: refresh }],
  ['session/logout', { POST: logout }],
  ['session', { GET: whoami }],
  ['devices', { GET: devicesList }],
  ['devices/revoke', { POST: devicesRevoke }],
  ['pairings', { POST: pairingsCreate }],
  ['pairings/confirm', { POST: pairingsConfirm }],
  ['invites', { POST: invitesCreate }],
  ['grants', { POST: grantsUpdate }],
  ['passkeys', { GET: passkeysList }],
  ['passkeys/register/options', { POST: registerOptions }],
  ['passkeys/register/verify', { POST: registerVerify }],
  ['passkeys/assert/options', { POST: assertOptions }],
  ['passkeys/assert/verify', { POST: assertVerify }],
  ['passkeys/remove', { POST: passkeysRemove }],
  ['push/vapid-key', { GET: vapidKey }],
  ['push/subscribe', { POST: pushSubscribe }],
  ['push/unsubscribe', { POST: pushUnsubscribe }],
])

/** Returns undefined when the path is not an identity route. */
export async function handleIdentityHttp(ctx: IdentityContext, request: Request, info: HubHttpRequestInfo): Promise<Response | undefined> {
  const path = new URL(request.url).pathname
  if (!path.startsWith(PREFIX)) return undefined
  const methods = ROUTES.get(path.slice(PREFIX.length))
  if (!methods) return undefined
  const method = request.method.toUpperCase()
  const route = method === 'GET' || method === 'POST' ? methods[method] : undefined
  if (!route) return new Response(null, { status: 405, headers: { allow: Object.keys(methods).join(', ') } })
  try {
    return await route({ ctx, request, info: mergeInfo(request, info) })
  } catch (error) {
    if (error instanceof HubIdentityError) return errorResponse(error)
    // Never leak storage or crypto details to the caller; the failure is fail-closed.
    console.error('[hub identity] request failed:', error instanceof Error ? error.message : String(error))
    return errorResponse(new HubIdentityError('internal', 'identity service failed'))
  }
}
