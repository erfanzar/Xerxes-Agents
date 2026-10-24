// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { HUMAN_ONLY_SCOPES, ROLE_SCOPES, parseGrant, type HubGrant, type HubRole, type HubScope } from '../../security/principal.js'
import type { HubActor } from '../protocol.js'
import { LOCAL_ADMIN_ACTOR, auditAction } from './audit.js'
import { HubIdentityError, type IdentityContext } from './core.js'
import type { GrantRow, HubIdentityStore, UserRow } from './store.js'

export const MAX_GRANTS_PER_PRINCIPAL = 100

export function grantFromRow(row: GrantRow): HubGrant {
  const resource: HubGrant['resource'] =
    row.resource_type === 'hub'
      ? { type: 'hub' }
      : row.resource_type === 'workspace'
        ? { type: 'workspace', workspaceId: row.workspace_id ?? '' }
        : { type: 'session', workspaceId: row.workspace_id ?? '', sessionId: row.session_id ?? '' }
  return {
    principalId: row.principal_id,
    resource,
    effect: row.effect,
    ...(row.role === null ? {} : { role: row.role as HubRole }),
    ...(row.scopes === null ? {} : { scopes: JSON.parse(row.scopes) as HubScope[] }),
    ...(row.expires_at === null ? {} : { expiresAt: row.expires_at }),
  }
}

export function listGrants(store: HubIdentityStore, principalId: string): HubGrant[] {
  return store.all<GrantRow>('SELECT * FROM hub_grants WHERE principal_id = ? ORDER BY id ASC', principalId).map(grantFromRow)
}

export function grantScopeSet(grant: HubGrant): Set<HubScope> {
  const scopes = new Set<HubScope>(grant.role ? ROLE_SCOPES[grant.role] : [])
  for (const scope of grant.scopes ?? []) scopes.add(scope)
  return scopes
}

/** Agent principals can never be *granted* a human-only scope, whatever the record says. */
export function assertGrantAllowed(kind: UserRow['kind'], grant: HubGrant): void {
  if (grant.effect === 'deny') return
  // question.delegated exists only inside one XERXES Autopilot dispatch (the supervisor's grant
  // overlay). It can never be stored as an explicit scope, for anyone; a Hub owner holds it through ownership.
  if ((grant.scopes ?? []).includes('question.delegated') || (kind === 'agent' && grantScopeSet(grant).has('question.delegated'))) {
    throw new HubIdentityError('forbidden', 'question.delegated is only delegated by XERXES Autopilot and cannot be stored in a grant')
  }
  if (kind !== 'agent') return
  for (const scope of grantScopeSet(grant)) {
    if (HUMAN_ONLY_SCOPES.has(scope)) {
      throw new HubIdentityError('forbidden', `agent principals cannot be granted the human-only scope ${scope}`)
    }
  }
}

export function insertGrant(ctx: Pick<IdentityContext, 'store' | 'clock'>, grant: HubGrant, createdBy: string): void {
  const resource = grant.resource
  ctx.store.run(
    `INSERT INTO hub_grants (principal_id, resource_type, workspace_id, session_id, effect, role, scopes, expires_at, created_at, created_by)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    grant.principalId,
    resource.type,
    resource.type === 'hub' ? null : resource.workspaceId,
    resource.type === 'session' ? resource.sessionId : null,
    grant.effect,
    grant.role ?? null,
    grant.scopes === undefined ? null : JSON.stringify(grant.scopes),
    grant.expiresAt ?? null,
    ctx.clock.now(),
    createdBy,
  )
}

export function parseGrants(principalId: string, raw: readonly unknown[]): HubGrant[] {
  if (raw.length > MAX_GRANTS_PER_PRINCIPAL) {
    throw new HubIdentityError('invalid_request', `at most ${MAX_GRANTS_PER_PRINCIPAL} grants per principal`)
  }
  return raw.map(item => {
    let grant: HubGrant
    try {
      grant = parseGrant(item)
    } catch (error) {
      throw new HubIdentityError('invalid_request', error instanceof Error ? error.message : 'invalid grant', { cause: error })
    }
    if (grant.principalId !== principalId) throw new HubIdentityError('invalid_request', 'every grant must name the principal being updated')
    return grant
  })
}

export interface UpdateGrantsInput {
  readonly principalId: string
  readonly grants: readonly unknown[]
  /** `replace` (default) swaps the principal's whole grant set; `add` appends. */
  readonly mode?: 'replace' | 'add'
}

/** Validate and write grants atomically with an audit record. */
export function updateGrants(ctx: IdentityContext, input: UpdateGrantsInput, actor: HubActor = LOCAL_ADMIN_ACTOR): HubGrant[] {
  const grants = parseGrants(input.principalId, input.grants)
  return ctx.store.transaction(() => {
    const user = ctx.store.one<UserRow>('SELECT * FROM hub_users WHERE id = ?', input.principalId)
    if (!user) throw new HubIdentityError('not_found', 'principal does not exist')
    if (user.revoked_at !== null) throw new HubIdentityError('conflict', 'principal is revoked')
    for (const grant of grants) assertGrantAllowed(user.kind, grant)
    const mode = input.mode ?? 'replace'
    if (mode === 'replace') ctx.store.run('DELETE FROM hub_grants WHERE principal_id = ?', user.id)
    const existing = ctx.store.one<{ n: number }>('SELECT COUNT(*) AS n FROM hub_grants WHERE principal_id = ?', user.id)?.n ?? 0
    if (existing + grants.length > MAX_GRANTS_PER_PRINCIPAL) throw new HubIdentityError('invalid_request', `at most ${MAX_GRANTS_PER_PRINCIPAL} grants per principal`)
    for (const grant of grants) insertGrant(ctx, grant, actor.principal_id)
    auditAction(ctx, { actor, action: `identity.grants.${mode}:${user.id}`, params: { grants } })
    return listGrants(ctx.store, user.id)
  })
}
