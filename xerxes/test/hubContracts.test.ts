// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, test } from 'bun:test'

import { EventCursorTracker, newCommandId } from '../src/client/bridge.js'
import { cursorScopeKey, decodeCursor, encodeCursor, resolveClientCursor } from '../src/hub/cursor.js'
import {
  HUB_LIMITS,
  HubProtocolError,
  encodeHubFrame,
  hubError,
  hubEvent,
  hubResponse,
  parseHubRequestFrame,
  parseHubServerFrame,
  readIdList,
  utf8ByteLength,
  type HubEvent,
} from '../src/hub/protocol.js'
import {
  SUPERVISOR_TOOLS,
  assertSupervisorToolCeiling,
  validateSupervisorArguments,
} from '../src/hub/supervisor/schemas.js'
import {
  AGENT_SCOPE_CEILING,
  HUMAN_ONLY_SCOPES,
  canSeeWorkspace,
  effectiveScopes,
  grantedSessionIds,
  hasScope,
  parseGrant,
  type HubGrant,
  type HubPrincipal,
} from '../src/security/principal.js'

function caught(run: () => unknown): HubProtocolError {
  try {
    run()
  } catch (error) {
    if (error instanceof HubProtocolError) return error
    throw error
  }
  throw new Error('expected a HubProtocolError')
}

describe('hub protocol frames', () => {
  test('parses a valid request and defaults params', () => {
    const frame = parseHubRequestFrame(JSON.stringify({ v: 1, kind: 'request', id: 'r1', method: 'hub.workspaces.list' }))
    expect(frame).toEqual({ v: 1, kind: 'request', id: 'r1', method: 'hub.workspaces.list', params: {} })
  })

  test('rejects unknown methods, wrong versions and bad ids, keeping the request id when known', () => {
    expect(caught(() => parseHubRequestFrame('{nope')).code).toBe('invalid_request')
    const unknown = caught(() => parseHubRequestFrame(JSON.stringify({ v: 1, kind: 'request', id: 'a', method: 'turn.submit' })))
    expect(unknown.code).toBe('invalid_request')
    expect(unknown.requestId).toBe('a')
    expect(caught(() => parseHubRequestFrame(JSON.stringify({ v: 2, kind: 'request', id: 'b', method: 'hub.hello' }))).code).toBe('unsupported')
    expect(caught(() => parseHubRequestFrame(JSON.stringify({ v: 1, kind: 'request', id: '', method: 'hub.hello' }))).code).toBe('invalid_request')
    expect(caught(() => parseHubRequestFrame(JSON.stringify({ v: 1, kind: 'request', id: 'c', method: 'hub.hello', params: [] }))).requestId).toBe('c')
  })

  test('refuses frames over the byte cap in both directions', () => {
    const big = 'x'.repeat(100)
    expect(caught(() => parseHubRequestFrame(JSON.stringify({ v: 1, kind: 'request', id: 'r', method: 'hub.hello', params: { big } }), 50)).code).toBe('frame_too_large')
    expect(caught(() => encodeHubFrame(hubResponse('r', { big }), 50)).code).toBe('frame_too_large')
  })

  test('utf8ByteLength counts multi-byte and surrogate pairs', () => {
    expect(utf8ByteLength('a')).toBe(1)
    expect(utf8ByteLength('é')).toBe(2)
    expect(utf8ByteLength('€')).toBe(3)
    expect(utf8ByteLength('😀')).toBe(4)
    expect(utf8ByteLength('a😀é')).toBe(new TextEncoder().encode('a😀é').length)
  })

  test('server frames round-trip; malformed ones are refused', () => {
    expect(parseHubServerFrame(encodeHubFrame(hubResponse('r', { ok: true })))).toMatchObject({ kind: 'response', ok: true, id: 'r' })
    expect(parseHubServerFrame(encodeHubFrame(hubError(null, 'forbidden', 'no')))).toMatchObject({ ok: false, error: { code: 'forbidden' } })
    expect(parseHubServerFrame(encodeHubFrame(hubEvent('hub.presence', { users: [] })))).toMatchObject({ kind: 'event', type: 'hub.presence' })
    expect(caught(() => parseHubServerFrame(JSON.stringify({ v: 1, kind: 'event', type: 'made.up', payload: {} }))).code).toBe('invalid_request')
  })

  test('readIdList bounds, dedupes and validates', () => {
    expect(readIdList(['a', 'a', 'b'], 'ids')).toEqual(['a', 'b'])
    expect(readIdList(undefined, 'ids')).toEqual([])
    expect(caught(() => readIdList(undefined, 'ids', { required: true })).code).toBe('invalid_request')
    expect(caught(() => readIdList(Array.from({ length: HUB_LIMITS.maxIdsPerRequest + 1 }, (_, i) => `w${i}`), 'ids')).code).toBe('invalid_request')
    expect(caught(() => readIdList([''], 'ids')).code).toBe('invalid_request')
  })
})

describe('hub cursors', () => {
  const scopeKey = cursorScopeKey({ workspaceIds: ['b', 'a'] })

  test('round-trip and scope key is order-independent', () => {
    expect(cursorScopeKey({ workspaceIds: ['a', 'b'] })).toBe(scopeKey)
    const text = encodeCursor({ streamId: 's1', sequence: 42, scopeKey })
    expect(decodeCursor(text)).toEqual({ streamId: 's1', sequence: 42, scopeKey })
  })

  test('resolve: start, after, replaced stream, foreign scope, malformed', () => {
    const expected = { streamId: 's1', scopeKey }
    expect(resolveClientCursor(undefined, expected)).toEqual({ kind: 'start' })
    expect(resolveClientCursor(encodeCursor({ streamId: 's1', sequence: 7, scopeKey }), expected)).toEqual({ kind: 'after', sequence: 7 })
    expect(resolveClientCursor(encodeCursor({ streamId: 's0', sequence: 7, scopeKey }), expected)).toEqual({ kind: 'stream_replaced' })
    const foreign = encodeCursor({ streamId: 's1', sequence: 7, scopeKey: cursorScopeKey({ workspaceIds: ['c'] }) })
    expect(caught(() => resolveClientCursor(foreign, expected)).code).toBe('invalid_request')
    expect(caught(() => decodeCursor('hc1.%%%')).code).toBe('invalid_request')
    expect(() => encodeCursor({ streamId: 's', sequence: -1, scopeKey })).toThrow(RangeError)
  })
})

describe('principals and grants', () => {
  const alice: HubPrincipal = { kind: 'user', id: 'alice', displayName: 'Alice', deviceId: 'd1', hubOwner: false }
  const owner: HubPrincipal = { kind: 'user', id: 'own', displayName: 'Owner', deviceId: 'd0', hubOwner: true }
  const xerxes: HubPrincipal = { kind: 'agent', id: 'sup', displayName: 'XERXES', onBehalfOf: 'own' }

  test('roles expand to scopes and denials win', () => {
    const grants: HubGrant[] = [
      { principalId: 'alice', resource: { type: 'workspace', workspaceId: 'w1' }, effect: 'allow', role: 'operator' },
      { principalId: 'alice', resource: { type: 'workspace', workspaceId: 'w1' }, effect: 'deny', scopes: ['session.cancel'] },
    ]
    const scopes = effectiveScopes(alice, grants, { workspaceId: 'w1' })
    expect(scopes.has('session.submit')).toBe(true)
    expect(scopes.has('permission.respond')).toBe(true)
    expect(scopes.has('session.cancel')).toBe(false)
    expect(scopes.has('grants.manage')).toBe(false)
    expect(effectiveScopes(alice, grants, { workspaceId: 'w2' }).size).toBe(0)
  })

  test('hub owner holds everything except explicit denials and the device ceiling', () => {
    expect(hasScope(owner, [], { workspaceId: 'any' }, 'grants.manage')).toBe(true)
    const ceilinged: HubPrincipal = { ...owner, deviceScopeCeiling: new Set(['session.read']) } as HubPrincipal
    expect(hasScope(ceilinged, [], { workspaceId: 'any' }, 'session.submit')).toBe(false)
    expect(hasScope(ceilinged, [], { workspaceId: 'any' }, 'session.read')).toBe(true)
  })

  test('an agent principal can never hold a human-only scope, even with an owner grant', () => {
    const grants: HubGrant[] = [{ principalId: 'sup', resource: { type: 'hub' }, effect: 'allow', role: 'owner' }]
    for (const scope of HUMAN_ONLY_SCOPES) expect(hasScope(xerxes, grants, { workspaceId: 'w1' }, scope)).toBe(false)
    const scopes = effectiveScopes(xerxes, grants, { workspaceId: 'w1' })
    for (const scope of scopes) expect(AGENT_SCOPE_CEILING.has(scope)).toBe(true)
    expect(scopes.has('session.steer')).toBe(true)
  })

  test('session-only grants reveal the session, not the workspace', () => {
    const grants: HubGrant[] = [
      { principalId: 'alice', resource: { type: 'session', workspaceId: 'w1', sessionId: 's9' }, effect: 'allow', role: 'steer' },
    ]
    expect(canSeeWorkspace(alice, grants, 'w1')).toBe(false)
    expect(grantedSessionIds(alice, grants, 'w1')).toEqual(['s9'])
    expect(hasScope(alice, grants, { workspaceId: 'w1', sessionId: 's9' }, 'session.steer')).toBe(true)
    expect(hasScope(alice, grants, { workspaceId: 'w1', sessionId: 's8' }, 'session.read')).toBe(false)
  })

  test('expired grants do not apply', () => {
    const grants: HubGrant[] = [
      { principalId: 'alice', resource: { type: 'hub' }, effect: 'allow', role: 'viewer', expiresAt: '2020-01-01T00:00:00Z' },
    ]
    expect(hasScope(alice, grants, { workspaceId: 'w1' }, 'session.read')).toBe(false)
  })

  test('parseGrant validates untrusted records', () => {
    expect(parseGrant({ principalId: 'a', effect: 'allow', resource: { type: 'hub' }, role: 'viewer' })).toMatchObject({ role: 'viewer' })
    expect(() => parseGrant({ principalId: 'a', effect: 'allow', resource: { type: 'hub' } })).toThrow(/role or at least one scope/)
    expect(() => parseGrant({ principalId: 'a', effect: 'allow', resource: { type: 'hub' }, scopes: ['root'] })).toThrow(/not known/)
    expect(() => parseGrant({ principalId: 'a', effect: 'maybe', resource: { type: 'hub' }, role: 'viewer' })).toThrow(/effect/)
    expect(() => parseGrant({ principalId: 'a', effect: 'allow', resource: { type: 'session', workspaceId: 'w' }, role: 'viewer' })).toThrow(/sessionId/)
  })
})

describe('supervisor tool schemas', () => {
  test('every tool has a provider-safe name and an object schema', () => {
    expect(SUPERVISOR_TOOLS.size).toBe(16)
    expect([...SUPERVISOR_TOOLS.values()].filter(spec => spec.gate === 'autopilot').map(spec => spec.providerName)).toEqual(['answer_question', 'answer_permission'])
    expect(SUPERVISOR_TOOLS.get('task.pending')).toMatchObject({ providerName: 'pending_interactions', gate: 'read' })
    for (const spec of SUPERVISOR_TOOLS.values()) {
      expect(spec.providerName).toMatch(/^[a-z_]+$/)
      expect(spec.parameters.type).toBe('object')
      expect(spec.parameters.additionalProperties).toBe(false)
    }
    expect([...SUPERVISOR_TOOLS.values()].filter(spec => spec.gate === 'confirm').map(spec => spec.name).sort()).toEqual(
      ['agent.message', 'goal.set', 'model.set', 'task.start', 'task.steer', 'task.stop'],
    )
  })

  test('validates required fields, unknown fields and bounds', () => {
    expect(validateSupervisorArguments('task_steer', { workspace_id: 'w', task_id: 't', text: 'go' })).toMatchObject({ ok: true })
    expect(validateSupervisorArguments('task.steer', { workspace_id: 'w', task_id: 't' })).toMatchObject({ ok: false })
    expect(validateSupervisorArguments('task_steer', { workspace_id: 'w', task_id: 't', text: 'x', extra: 1 })).toMatchObject({ ok: false })
    const long = validateSupervisorArguments('task_steer', { workspace_id: 'w', task_id: 't', text: 'x'.repeat(2001) })
    expect(long.ok).toBe(false)
    expect(validateSupervisorArguments('tasks_list', { limit: 101 })).toMatchObject({ ok: false })
    expect(validateSupervisorArguments('tasks_list', { limit: '5' })).toMatchObject({ ok: true, args: { limit: 5 } })
    expect(validateSupervisorArguments('notify_user', { recipients: ['a', 'a'], text: 'hi', severity: 'info' })).toMatchObject({ ok: false })
    expect(validateSupervisorArguments('notify_user', { recipients: ['a'], text: 'hi', severity: 'loud' })).toMatchObject({ ok: false })
    const badCriterion = validateSupervisorArguments('goal_set', {
      workspace_id: 'w', task_id: 't', expected_revision: 0, objective: 'o', criteria: [{ id: 'a', description: 'd', extra: true }],
    })
    expect(badCriterion.ok).toBe(false)
    expect(validateSupervisorArguments('permission_response', {})).toMatchObject({ ok: false })
  })

  test('the tool ceiling refuses human-only and unknown tools', () => {
    expect(() => assertSupervisorToolCeiling(['task_digest', 'tasks.list'])).not.toThrow()
    expect(() => assertSupervisorToolCeiling(['permission_response'])).toThrow(/human-only/)
    expect(() => assertSupervisorToolCeiling(['learning.review'])).toThrow(/human-only/)
    expect(() => assertSupervisorToolCeiling(['ReadFile'])).toThrow(/outside/)
  })
})

describe('client event cursor tracker', () => {
  const event = (sequence: number, id = `e${sequence}`): HubEvent => ({
    event_id: id, sequence, cursor: `c${sequence}`, workspace_id: 'w', type: 't', payload: {}, occurred_at: '2026-10-21T00:00:00Z',
  })

  test('applies in order, dedupes, and reports gaps without moving', () => {
    const tracker = new EventCursorTracker()
    expect(tracker.classify(event(1))).toEqual({ kind: 'applied' })
    tracker.commit(event(1))
    expect(tracker.classify(event(1))).toEqual({ kind: 'duplicate' })
    expect(tracker.classify(event(2, 'e1'))).toEqual({ kind: 'duplicate' })
    expect(tracker.classify(event(4))).toEqual({ kind: 'gap', expected: 2, received: 4 })
    expect(tracker.cursor).toBe('c1')
    tracker.reset({ sequence: 3, cursor: 'c3' })
    expect(tracker.classify(event(4))).toEqual({ kind: 'applied' })
  })

  test('command ids are unique', () => {
    expect(newCommandId()).not.toBe(newCommandId())
  })
})
