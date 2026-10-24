// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Principals, scopes and grants for the Xerxes Hub.
 *
 * Authority is always the intersection of what the user was granted, what the
 * device is allowed (its ceiling), and what the resource ACL permits. Explicit
 * denials win. Agent principals (the XERXES supervisor) can never hold a
 * human-only scope, whatever a grant record says: that rule is enforced here,
 * structurally, so no configuration mistake can hand an agent approval power.
 *
 * This module is pure and browser-safe.
 */

export const HUB_SCOPES = [
  'session.read',
  'usage.read',
  'presence.read',
  'session.steer',
  'session.submit',
  'session.cancel',
  'session.configure',
  'goal.write',
  'permission.respond',
  'question.respond',
  'grants.manage',
  'approvals.manage',
  'devices.manage',
  /** Run one model completion on a daemon's local provider profile (the Hub's XERXES supervisor). */
  'supervisor.model',
  /**
   * Answer a task's pending QUESTION on a Hub owner's behalf (XERXES Autopilot).
   * Not human-only: it exists so an agent can hold it. It never covers a
   * permission request; `permission.respond` stays human-only everywhere.
   */
  'question.delegated',
] as const
export type HubScope = (typeof HUB_SCOPES)[number]

/** Scopes only an authenticated human may ever exercise. */
export const HUMAN_ONLY_SCOPES: ReadonlySet<HubScope> = new Set<HubScope>([
  'permission.respond',
  'question.respond',
  'grants.manage',
  'approvals.manage',
  'devices.manage',
])

export const HUB_ROLES = ['viewer', 'steer', 'operator', 'owner'] as const
export type HubRole = (typeof HUB_ROLES)[number]

const VIEWER: readonly HubScope[] = ['session.read', 'usage.read', 'presence.read']
const STEER: readonly HubScope[] = [...VIEWER, 'session.steer']
const OPERATOR: readonly HubScope[] = [
  ...STEER,
  'session.submit',
  'session.cancel',
  'session.configure',
  'goal.write',
  'permission.respond',
  'question.respond',
]
const OWNER: readonly HubScope[] = [...OPERATOR, 'grants.manage', 'approvals.manage', 'devices.manage', 'supervisor.model', 'question.delegated']

export const ROLE_SCOPES: Readonly<Record<HubRole, ReadonlySet<HubScope>>> = Object.freeze({
  viewer: new Set(VIEWER),
  steer: new Set(STEER),
  operator: new Set(OPERATOR),
  owner: new Set(OWNER),
})

/** The most an agent principal can ever be granted. */
export const AGENT_SCOPE_CEILING: ReadonlySet<HubScope> = new Set<HubScope>([
  'session.read',
  'usage.read',
  'session.steer',
  'session.submit',
  'session.cancel',
  'session.configure',
  'goal.write',
  'supervisor.model',
  'question.delegated',
])

export type HubPrincipal =
  | {
      readonly kind: 'user'
      readonly id: string
      readonly displayName: string
      readonly deviceId: string
      /** Optional per-device ceiling; undefined means the device adds no restriction. */
      readonly deviceScopeCeiling?: ReadonlySet<HubScope>
      readonly hubOwner: boolean
    }
  | {
      readonly kind: 'agent'
      readonly id: string
      readonly displayName: string
      /** The human on whose request the agent acts. Attribution only, never authority. */
      readonly onBehalfOf?: string
    }

/** A resource a grant can name. `session` grants are narrower than `workspace` grants. */
export type GrantResource =
  | { readonly type: 'hub' }
  | { readonly type: 'workspace'; readonly workspaceId: string }
  | { readonly type: 'session'; readonly workspaceId: string; readonly sessionId: string }

export interface HubGrant {
  readonly principalId: string
  readonly resource: GrantResource
  readonly effect: 'allow' | 'deny'
  /** Either a role preset or explicit scopes; both may be present and are unioned. */
  readonly role?: HubRole
  readonly scopes?: readonly HubScope[]
  readonly expiresAt?: string
}

export interface AccessTarget {
  readonly workspaceId: string
  readonly sessionId?: string
}

export function isHubScope(value: unknown): value is HubScope {
  return typeof value === 'string' && (HUB_SCOPES as readonly string[]).includes(value)
}

export function isHubRole(value: unknown): value is HubRole {
  return typeof value === 'string' && (HUB_ROLES as readonly string[]).includes(value)
}

function grantScopes(grant: HubGrant): Set<HubScope> {
  const scopes = new Set<HubScope>(grant.role ? ROLE_SCOPES[grant.role] : [])
  for (const scope of grant.scopes ?? []) scopes.add(scope)
  return scopes
}

function grantApplies(grant: HubGrant, target: AccessTarget, now: number): boolean {
  if (grant.expiresAt !== undefined) {
    const expires = Date.parse(grant.expiresAt)
    if (!Number.isFinite(expires) || expires <= now) return false
  }
  const resource = grant.resource
  if (resource.type === 'hub') return true
  if (resource.type === 'workspace') return resource.workspaceId === target.workspaceId
  return resource.workspaceId === target.workspaceId && target.sessionId !== undefined && resource.sessionId === target.sessionId
}

/**
 * Effective scopes for one principal on one target.
 *
 * allow ∪ (matching allow grants) − (matching deny grants), then ∩ device
 * ceiling, then ∩ agent ceiling for agents. A session-scoped grant applies only
 * when a session is named. A hub owner holds every scope on every target, still
 * subject to explicit denials and the device ceiling.
 */
export function effectiveScopes(
  principal: HubPrincipal,
  grants: readonly HubGrant[],
  target: AccessTarget,
  now: number = Date.now(),
): ReadonlySet<HubScope> {
  const allowed = new Set<HubScope>()
  const denied = new Set<HubScope>()
  if (principal.kind === 'user' && principal.hubOwner) for (const scope of HUB_SCOPES) allowed.add(scope)
  for (const grant of grants) {
    if (grant.principalId !== principal.id || !grantApplies(grant, target, now)) continue
    const scopes = grantScopes(grant)
    const into = grant.effect === 'deny' ? denied : allowed
    for (const scope of scopes) into.add(scope)
  }
  for (const scope of denied) allowed.delete(scope)
  if (principal.kind === 'user' && principal.deviceScopeCeiling) {
    for (const scope of [...allowed]) if (!principal.deviceScopeCeiling.has(scope)) allowed.delete(scope)
  }
  if (principal.kind === 'agent') {
    for (const scope of [...allowed]) if (!AGENT_SCOPE_CEILING.has(scope)) allowed.delete(scope)
  }
  return allowed
}

export function hasScope(
  principal: HubPrincipal,
  grants: readonly HubGrant[],
  target: AccessTarget,
  scope: HubScope,
  now: number = Date.now(),
): boolean {
  if (principal.kind === 'agent' && HUMAN_ONLY_SCOPES.has(scope)) return false
  return effectiveScopes(principal, grants, target, now).has(scope)
}

/**
 * Whether the principal may see a workspace at all (it appears in lists).
 * A session-only grant reveals only that session, never the workspace inventory.
 */
export function canSeeWorkspace(principal: HubPrincipal, grants: readonly HubGrant[], workspaceId: string, now: number = Date.now()): boolean {
  if (principal.kind === 'user' && principal.hubOwner) return true
  return effectiveScopes(principal, grants, { workspaceId }, now).has('session.read')
}

/** Sessions in a workspace the principal was granted individually (session-only invites). */
export function grantedSessionIds(principal: HubPrincipal, grants: readonly HubGrant[], workspaceId: string, now: number = Date.now()): string[] {
  const ids = new Set<string>()
  for (const grant of grants) {
    if (grant.principalId !== principal.id || grant.effect !== 'allow' || grant.resource.type !== 'session') continue
    if (grant.resource.workspaceId !== workspaceId) continue
    if (!grantApplies(grant, { workspaceId, sessionId: grant.resource.sessionId }, now)) continue
    if (hasScope(principal, grants, { workspaceId, sessionId: grant.resource.sessionId }, 'session.read', now)) ids.add(grant.resource.sessionId)
  }
  return [...ids].sort()
}

/** Validate an untrusted grant record. Throws a descriptive Error. */
export function parseGrant(value: unknown): HubGrant {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) throw new Error('grant must be an object')
  const record = value as Record<string, unknown>
  const principalId = record.principalId
  if (typeof principalId !== 'string' || !principalId) throw new Error('grant.principalId is required')
  const effect = record.effect
  if (effect !== 'allow' && effect !== 'deny') throw new Error('grant.effect must be allow or deny')
  const resource = parseResource(record.resource)
  const role = record.role
  if (role !== undefined && !isHubRole(role)) throw new Error(`grant.role ${JSON.stringify(role)} is not a known role`)
  let scopes: HubScope[] | undefined
  if (record.scopes !== undefined) {
    if (!Array.isArray(record.scopes)) throw new Error('grant.scopes must be an array')
    scopes = record.scopes.map(scope => {
      if (!isHubScope(scope)) throw new Error(`grant scope ${JSON.stringify(scope)} is not known`)
      return scope
    })
  }
  if (role === undefined && (scopes === undefined || scopes.length === 0)) throw new Error('grant needs a role or at least one scope')
  const expiresAt = record.expiresAt
  if (expiresAt !== undefined && (typeof expiresAt !== 'string' || !Number.isFinite(Date.parse(expiresAt)))) {
    throw new Error('grant.expiresAt must be an ISO timestamp')
  }
  return {
    principalId,
    effect,
    resource,
    ...(role === undefined ? {} : { role }),
    ...(scopes === undefined ? {} : { scopes }),
    ...(expiresAt === undefined ? {} : { expiresAt: expiresAt as string }),
  }
}

function parseResource(value: unknown): GrantResource {
  if (typeof value !== 'object' || value === null) throw new Error('grant.resource is required')
  const record = value as Record<string, unknown>
  if (record.type === 'hub') return { type: 'hub' }
  if (typeof record.workspaceId !== 'string' || !record.workspaceId) throw new Error('grant.resource.workspaceId is required')
  if (record.type === 'workspace') return { type: 'workspace', workspaceId: record.workspaceId }
  if (record.type === 'session') {
    if (typeof record.sessionId !== 'string' || !record.sessionId) throw new Error('grant.resource.sessionId is required')
    return { type: 'session', workspaceId: record.workspaceId, sessionId: record.sessionId }
  }
  throw new Error('grant.resource.type must be hub, workspace or session')
}
