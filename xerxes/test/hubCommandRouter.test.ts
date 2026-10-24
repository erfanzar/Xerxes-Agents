// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { Database } from 'bun:sqlite'
import { createHash } from 'node:crypto'
import { afterEach, describe, expect, test } from 'bun:test'
import { join } from 'node:path'

import { COMMAND_SPECS, HUB_AUTH_LIFETIME_MS, HubCommandRouter, commandDaemonMethod, commandScope, type HubCommandRouterOptions } from '../src/hub/commandRouter.js'
import { HUB_COMMAND_ACTIONS, type HubCommandAction } from '../src/hub/protocol.js'
import { HUB_DAEMON_METHODS } from '../src/hub/runtimePool.js'
import { verifyHubAuth } from '../src/security/hubAuthEnvelope.js'
import { SupervisorGrantOverlay } from '../src/hub/supervisor/actions.js'
import { SUPERVISOR_PRINCIPAL_ID } from '../src/hub/supervisor/audit.js'
import { HUMAN_ONLY_SCOPES, type HubGrant, type HubScope } from '../src/security/principal.js'
import {
  FakeBackend, FakeIdentity, agentPrincipal, disposeAll, registerCleanup, standardUsers, tempDir, testSigner, workspaceGrant,
} from './hubHarness.js'

afterEach(disposeAll)

async function setup(extra: Partial<HubCommandRouterOptions> = {}) {
  const dir = await tempDir('xh-cmd-')
  const identity = standardUsers(new FakeIdentity())
  const backend = new FakeBackend()
  const signer = testSigner()
  const time = { now: Date.now() }
  const errors: string[] = []
  const databasePath = join(dir, 'hub.sqlite')
  const options: HubCommandRouterOptions = { databasePath, backend, identity, signer, clock: () => time.now, onError: (_error, context) => { errors.push(context) }, ...extra }
  const router = new HubCommandRouter(options)
  registerCleanup(() => router.close())
  const mutations = () => backend.calls.filter(call => call.method !== 'session.active_list')
  return { dir, identity, backend, signer, time, errors, databasePath, router, options, mutations }
}

/** A router wired like the Hub's: the supervisor's grants come only from the per-dispatch overlay. */
function autopilotRouter(t: Awaited<ReturnType<typeof setup>>) {
  const overlay = new SupervisorGrantOverlay()
  if (!t.identity.users.has(SUPERVISOR_PRINCIPAL_ID)) t.identity.add(agentPrincipal(SUPERVISOR_PRINCIPAL_ID), [])
  const router = new HubCommandRouter({
    ...t.options, databasePath: join(t.dir, 'hub-overlay.sqlite'),
    identity: overlay.wrapIdentity(t.identity), overlayGrants: () => overlay.current(),
  })
  registerCleanup(() => router.close())
  return { router, overlay }
}
const delegated = (scopes: HubScope[] = ['question.delegated', 'session.read']): HubGrant[] =>
  [{ principalId: SUPERVISOR_PRINCIPAL_ID, resource: { type: 'session', workspaceId: 'w1', sessionId: 's1' }, effect: 'allow', scopes }]

const submit = (id = 'c1', text = 'hello') => ({ workspace_id: 'w1', session_id: 's1', command_id: id, action: 'turn.submit', args: { text } })

const EXPECTED: Record<HubCommandAction, readonly [HubScope, string]> = {
  'session.open': ['session.submit', 'session.open'],
  'turn.submit': ['session.submit', 'turn.submit'],
  'turn.steer': ['session.steer', 'turn.steer'],
  'turn.cancel': ['session.cancel', 'turn.cancel'],
  'permission.respond': ['permission.respond', 'permission_response'],
  'question.respond': ['question.respond', 'question_response'],
  'question.answer': ['question.delegated', 'question_response'],
  'goal.set': ['goal.write', 'session.goal'],
  'model.set': ['session.configure', 'set_model'],
  'subagent.message': ['session.steer', 'subagent.steer'],
}
const VALID_ARGS: Record<HubCommandAction, Record<string, unknown>> = {
  'session.open': {},
  'turn.submit': { text: 'go' },
  'turn.steer': { text: 'steer' },
  'turn.cancel': {},
  'permission.respond': { request_id: 'p1', response: 'approve' },
  'question.respond': { request_id: 'q1', answers: { a: 'b' } },
  'question.answer': { request_id: 'q1', answers: { a: 'b' } },
  'goal.set': { input: 'ship it' },
  'model.set': { model: 'm1' },
  'subagent.message': { task: 't1', message: 'hi' },
}
const command = (action: HubCommandAction, id: string) => ({
  workspace_id: 'w1', ...(action === 'session.open' ? {} : { session_id: 's1' }), command_id: id, action, args: VALID_ARGS[action],
})

describe('Hub command router', () => {
  test('maps every action to one scope and exactly one allowlisted daemon method', async () => {
    expect(Object.keys(COMMAND_SPECS).sort()).toEqual([...HUB_COMMAND_ACTIONS].sort())
    const t = await setup()
    const { router, overlay } = autopilotRouter(t)
    for (const action of HUB_COMMAND_ACTIONS) {
      const [scope, method] = EXPECTED[action]
      expect(commandScope(action)).toBe(scope)
      expect(commandDaemonMethod(action)).toBe(method)
      expect((HUB_DAEMON_METHODS as readonly string[]).includes(method)).toBe(true)
      t.backend.calls.length = 0
      // question.answer is the Autopilot route: the supervisor principal, scopes only from the overlay.
      const result = action === 'question.answer'
        ? await overlay.run(delegated(), () => router.execute(t.identity.context(SUPERVISOR_PRINCIPAL_ID), command(action, `map-${action}`)))
        : await router.execute(t.identity.context('alice'), command(action, `map-${action}`))
      expect(result.state).toBe('accepted')
      expect(t.mutations().map(call => call.method)).toEqual([method])
    }
  })

  test('refuses unknown actions, parameters and arguments before touching a daemon', async () => {
    const t = await setup()
    const alice = t.identity.context('alice')
    const bad: Record<string, unknown>[] = [
      { ...submit(), action: 'shell.exec' },
      { ...submit(), actor: { principal_id: 'owner' } },
      { ...submit(), args: { text: 'x', cmd: 'rm -rf' } },
      { ...submit(), args: {} },
      { ...submit(), args: { text: 'x'.repeat(40_000) } },
      { ...submit(), session_id: undefined },
      { ...submit(), command_id: '' },
      { ...command('permission.respond', 'p'), args: { request_id: 'p1', response: 'always' } },
      { ...command('permission.respond', 'p'), args: { request_id: 'p1', response: 'approve', extra: 1 } },
      { ...command('question.respond', 'q'), args: { request_id: 'q1', answers: { a: 5 } } },
    ]
    for (const params of bad) await expect(t.router.execute(alice, params)).rejects.toMatchObject({ code: 'invalid_request' })
    expect(t.backend.calls).toEqual([])
  })

  test('denies actions that lack the required scope and audits the denial', async () => {
    const t = await setup()
    for (const who of ['viewer', 'guest']) {
      for (const action of ['turn.submit', 'turn.steer', 'turn.cancel', 'goal.set', 'model.set'] as const) {
        await expect(t.router.execute(t.identity.context(who), command(action, `deny-${who}-${action}`))).rejects.toMatchObject({ code: 'forbidden' })
      }
    }
    // alice has no grant in w2 at all.
    await expect(t.router.execute(t.identity.context('alice'), { ...submit('x'), workspace_id: 'w2' })).rejects.toMatchObject({ code: 'forbidden' })
    expect(t.backend.calls).toEqual([])
    expect(t.identity.audits.length).toBeGreaterThan(0)
    expect(t.identity.audits.every(entry => entry.decision === 'denied' && entry.outcome === 'rejected')).toBe(true)
  })

  test('an agent principal can never answer permissions or questions, whatever its grants say', async () => {
    const t = await setup()
    t.identity.add(agentPrincipal('rogue'), [{ principalId: 'rogue', resource: { type: 'hub' }, effect: 'allow', scopes: ['permission.respond', 'question.respond', 'session.read', 'session.submit'] }])
    for (const who of ['robot', 'rogue']) {
      for (const action of ['permission.respond', 'question.respond'] as const) {
        await expect(t.router.execute(t.identity.context(who), command(action, `agent-${who}-${action}`))).rejects.toMatchObject({ code: 'forbidden' })
      }
    }
    expect(t.mutations()).toEqual([])
    // Structural: both interaction actions require a human-only scope.
    expect(HUMAN_ONLY_SCOPES.has(commandScope('permission.respond'))).toBe(true)
    expect(HUMAN_ONLY_SCOPES.has(commandScope('question.respond'))).toBe(true)
    // The agent can still do what its ceiling permits, attributed as an agent.
    expect((await t.router.execute(t.identity.context('robot'), submit('ok'))).state).toBe('accepted')
    // A human operator can answer.
    expect((await t.router.execute(t.identity.context('alice'), command('permission.respond', 'human'))).state).toBe('accepted')
  })

  test('Autopilot: question.answer only for the supervisor with overlay grants, never a permission, whatever it is granted', async () => {
    const t = await setup()
    const { router, overlay } = autopilotRouter(t)
    const supervisor = t.identity.context(SUPERVISOR_PRINCIPAL_ID)
    // Another agent with a stored question.delegated grant is refused: the scope must come from the overlay.
    t.identity.add(agentPrincipal('auto'), [{ principalId: 'auto', resource: { type: 'hub' }, effect: 'allow', scopes: ['question.delegated', 'session.read'] }])
    await expect(router.execute(t.identity.context('auto'), command('question.answer', 'auto-qa'))).rejects.toMatchObject({ code: 'forbidden' })
    await expect(overlay.run(delegated(), () => router.execute(t.identity.context('auto'), command('question.answer', 'auto-qa2')))).rejects.toMatchObject({ code: 'forbidden' })
    await expect(router.execute(t.identity.context('robot'), command('question.answer', 'robot-qa'))).rejects.toMatchObject({ code: 'forbidden' })
    // The supervisor outside an overlay dispatch has nothing.
    await expect(router.execute(supervisor, command('question.answer', 'sup-bare'))).rejects.toMatchObject({ code: 'forbidden' })
    // A router without overlayGrants refuses the route outright, even inside a dispatch.
    await expect(overlay.run(delegated(), () => t.router.execute(supervisor, command('question.answer', 'no-overlay')))).rejects.toMatchObject({ code: 'forbidden' })
    // Even with every scope in the overlay, the supervisor can never answer a permission or use question.respond.
    const everything = delegated(['question.delegated', 'permission.respond', 'question.respond', 'session.read'])
    for (const action of ['permission.respond', 'question.respond'] as const) {
      await expect(overlay.run(everything, () => router.execute(supervisor, command(action, `sup-${action}`)))).rejects.toMatchObject({ code: 'forbidden' })
    }
    // A person (even the owner, who holds question.delegated) answers through question.respond instead.
    await expect(router.execute(t.identity.context('owner'), command('question.answer', 'owner-qa'))).rejects.toMatchObject({ code: 'forbidden' })
    expect(t.mutations()).toEqual([])
    expect(HUMAN_ONLY_SCOPES.has('question.delegated')).toBe(false)
    expect((await overlay.run(delegated(), () => router.execute(supervisor, command('question.answer', 'sup-qa')))).state).toBe('accepted')
    expect(t.mutations().map(call => call.method)).toEqual(['question_response'])
    expect(t.mutations()[0]!.params).toMatchObject({ request_id: 'q1', answers: { a: 'b' } })
  })

  test('replays the recorded outcome for the same command id and content, durably', async () => {
    const t = await setup()
    const alice = t.identity.context('alice')
    const first = await t.router.execute(alice, submit('c1'))
    const second = await t.router.execute(alice, { ...submit('c1'), args: { text: 'hello' } })
    expect(second).toEqual(first)
    expect(t.backend.callsTo('turn.submit')).toHaveLength(1)
    expect(t.router.status(alice, { workspace_id: 'w1', session_id: 's1', command_id: 'c1' })).toEqual(first)

    // A new router over the same database (a Hub restart) still replays instead of resending.
    const again = new HubCommandRouter(t.options)
    registerCleanup(() => again.close())
    expect(await again.execute(alice, submit('c1'))).toEqual(first)
    expect(t.backend.callsTo('turn.submit')).toHaveLength(1)

    // Ids are namespaced by principal, workspace and session.
    await t.router.execute(t.identity.context('owner'), submit('c1'))
    await t.router.execute(alice, { ...submit('c1'), session_id: 's2' })
    expect(t.backend.callsTo('turn.submit')).toHaveLength(3)
  })

  test('reusing a command id with different content is a submission_conflict', async () => {
    const t = await setup()
    const alice = t.identity.context('alice')
    await t.router.execute(alice, submit('c1', 'first'))
    await expect(t.router.execute(alice, submit('c1', 'second'))).rejects.toMatchObject({ code: 'submission_conflict' })
    expect(t.backend.callsTo('turn.submit')).toHaveLength(1)
  })

  test('concurrent duplicates share one daemon call; concurrent conflicts are refused', async () => {
    const t = await setup()
    t.backend.handler = async call => { if (call.method === 'turn.submit') await Bun.sleep(30); return undefined }
    const alice = t.identity.context('alice')
    const [a, b] = await Promise.all([t.router.execute(alice, submit('c1')), t.router.execute(alice, submit('c1'))])
    expect(b).toEqual(a)
    expect(t.backend.callsTo('turn.submit')).toHaveLength(1)
    const winner = t.router.execute(alice, submit('c2', 'a'))
    const loser = t.router.execute(alice, submit('c2', 'b'))
    await expect(loser).rejects.toMatchObject({ code: 'submission_conflict' })
    expect((await winner).state).toBe('accepted')
  })

  test('refuses a mutation when the audit write fails, and does not poison the id', async () => {
    const t = await setup()
    const alice = t.identity.context('alice')
    t.identity.failAudit = true
    await expect(t.router.execute(alice, submit('c1'))).rejects.toMatchObject({ code: 'unavailable' })
    expect(t.backend.calls).toEqual([])
    // A denial is still a denial when the audit log is down.
    await expect(t.router.execute(t.identity.context('viewer'), submit('c2'))).rejects.toMatchObject({ code: 'forbidden' })
    t.identity.failAudit = false
    expect((await t.router.execute(alice, submit('c1'))).state).toBe('accepted')
    expect(t.backend.callsTo('turn.submit')).toHaveLength(1)
  })

  test('a failing post-call audit never changes an outcome that already happened', async () => {
    const t = await setup()
    t.backend.handler = call => { if (call.method === 'turn.submit') t.identity.failAudit = true; return undefined }
    const result = await t.router.execute(t.identity.context('alice'), submit('c1'))
    expect(result.state).toBe('accepted')
    expect(t.errors).toContain('audit outcome')
  })

  test('audit records carry a request hash, never raw arguments', async () => {
    const t = await setup()
    await t.router.execute(t.identity.context('alice'), submit('c1', 'super secret prompt text'))
    expect(t.identity.audits.length).toBeGreaterThanOrEqual(2)
    for (const entry of t.identity.audits) {
      expect(entry.requestHash).toMatch(/^[0-9a-f]{64}$/)
      expect(JSON.stringify(entry)).not.toContain('super secret prompt text')
      expect(entry.action).toBe('command.turn.submit')
      expect(entry.actor.principal_id).toBe('alice')
      expect(entry.submissionId).toBe('c1')
    }
  })

  test('forwards a signed hub_auth envelope bound to the exact params and the authenticated actor', async () => {
    const t = await setup()
    await t.router.execute(t.identity.context('alice'), submit('c1'))
    await t.router.execute(t.identity.context('robot'), submit('c2'))
    await t.router.execute(t.identity.context('alice'), command('permission.respond', 'c3'))
    const [human, agent, interaction] = t.backend.calls.filter(call => call.method !== 'session.active_list')
    for (const call of [human!, agent!, interaction!]) {
      const { hub_auth: envelope, ...params } = call.params
      const verified = verifyHubAuth(envelope, { publicKey: t.signer.publicKeyPem, method: call.method, params, now: t.time.now })
      expect(verified.ok).toBe(true)
      if (!verified.ok) continue
      expect(verified.claims.hub_id).toBe(t.signer.hubId)
      expect(verified.claims.workspace_id).toBe('w1')
      expect(verified.claims.session_id).toBe('s1')
      expect(verified.claims.expires_at - verified.claims.issued_at).toBe(HUB_AUTH_LIFETIME_MS)
      expect(params.session_key).toBe('k1')
      expect(params.project_dir).toBe('/proj/w1')
    }
    const claims = (call: typeof human) => (call!.params.hub_auth as { claims: Record<string, unknown> }).claims
    expect(claims(human)).toMatchObject({ principal_id: 'alice', principal_kind: 'user', user_id: 'alice', device_id: 'dev-alice', scopes: ['session.submit'] })
    expect(claims(human).human_action_id).toBeUndefined()
    expect(claims(agent)).toMatchObject({ principal_id: 'robot', principal_kind: 'agent', requested_by: 'alice' })
    expect(claims(agent).user_id).toBeUndefined()
    expect(claims(interaction)).toMatchObject({ principal_id: 'alice', method: 'permission_response', scopes: ['permission.respond'], human_action_id: 'c3' })
    expect(claims(human).nonce).not.toBe(claims(agent).nonce)

    const { hub_auth: envelope, ...params } = human!.params
    const tampered = verifyHubAuth(envelope, { publicKey: t.signer.publicKeyPem, method: 'turn.submit', params: { ...params, text: 'different' }, now: t.time.now })
    expect(tampered).toEqual({ ok: false, reason: 'params_mismatch' })
  })

  test('an unknown outcome is recorded and never replayed, even after a restart', async () => {
    const t = await setup()
    const alice = t.identity.context('alice')
    t.backend.handler = call => call.method === 'turn.submit'
      ? { state: 'unknown', error: { code: 'outcome_unknown', message: 'written but unanswered' } }
      : undefined
    const first = await t.router.execute(alice, submit('c1'))
    expect(first).toMatchObject({ ok: false, state: 'unknown' })
    t.backend.handler = undefined
    expect(await t.router.execute(alice, submit('c1'))).toEqual(first)
    expect(t.router.status(alice, { workspace_id: 'w1', session_id: 's1', command_id: 'c1' })).toMatchObject({ state: 'unknown' })
    const restarted = new HubCommandRouter(t.options)
    registerCleanup(() => restarted.close())
    expect(await restarted.execute(alice, submit('c1'))).toMatchObject({ state: 'unknown' })
    expect(t.backend.callsTo('turn.submit')).toHaveLength(1)
  })

  test('a command still in flight when the Hub stopped becomes unknown at the next open', async () => {
    const t = await setup()
    const alice = t.identity.context('alice')
    const key = createHash('sha256').update(JSON.stringify(['alice', 'w1', 's1', 'c9'])).digest('hex')
    const db = new Database(t.databasePath, { strict: true })
    db.query(`INSERT INTO hub_commands (key, principal_id, workspace_id, session_id, command_id, action, request_hash, state, created_at, updated_at)
      VALUES (?, 'alice', 'w1', 's1', 'c9', 'turn.submit', 'h', 'in_flight', 1, 1)`).run(key)
    db.close()
    const restarted = new HubCommandRouter(t.options)
    registerCleanup(() => restarted.close())
    expect(restarted.status(alice, { workspace_id: 'w1', session_id: 's1', command_id: 'c9' })).toMatchObject({ ok: false, state: 'unknown' })
  })

  test('daemon rejections are recorded and mapped; pre-write unavailability is retryable under the same id', async () => {
    const t = await setup()
    const alice = t.identity.context('alice')
    t.backend.handler = call => call.method === 'turn.submit' ? { state: 'rejected', error: { code: 'turn-active', message: 'busy' } }
      : call.method === 'permission_response' ? { state: 'rejected', error: { code: 'daemon_rejected', message: 'gone' } } : undefined
    expect(await t.router.execute(alice, submit('c1'))).toMatchObject({ state: 'rejected', error: { code: 'conflict' } })
    expect(await t.router.execute(alice, submit('c1'))).toMatchObject({ state: 'rejected' })
    expect(t.backend.callsTo('turn.submit')).toHaveLength(1)
    expect(await t.router.execute(alice, command('permission.respond', 'p1'))).toMatchObject({ state: 'rejected', error: { code: 'interaction_already_resolved' } })

    let online = false
    t.backend.handler = call => call.method === 'steer' || call.method === 'turn.steer'
      ? (online ? undefined : { state: 'rejected', error: { code: 'unavailable', message: 'offline' } }) : undefined
    await expect(t.router.execute(alice, command('turn.steer', 's1'))).rejects.toMatchObject({ code: 'unavailable' })
    online = true
    expect((await t.router.execute(alice, command('turn.steer', 's1'))).state).toBe('accepted')
    expect(t.backend.callsTo('turn.steer')).toHaveLength(2)
  })

  test('refuses unknown workspaces and sessions that are not active', async () => {
    const t = await setup()
    await expect(t.router.execute(t.identity.context('owner'), { ...submit('c1'), workspace_id: 'nope' })).rejects.toMatchObject({ code: 'not_found' })
    await expect(t.router.execute(t.identity.context('alice'), { ...submit('c2'), session_id: 'ghost' })).rejects.toMatchObject({ code: 'not_found' })
    expect(t.backend.callsTo('turn.submit')).toEqual([])
  })

  test('expected_revision is checked when supported and refused rather than ignored when not', async () => {
    const plain = await setup()
    await expect(plain.router.execute(plain.identity.context('alice'), { ...submit('c1'), expected_revision: 3 })).rejects.toMatchObject({ code: 'unsupported' })
    const t = await setup({ revision: () => 7 })
    const alice = t.identity.context('alice')
    await expect(t.router.execute(alice, { ...submit('c1'), expected_revision: 3 })).rejects.toMatchObject({ code: 'conflict' })
    expect((await t.router.execute(alice, { ...submit('c1'), expected_revision: 7 })).state).toBe('accepted')
  })

  test('status only answers for the caller\'s own commands', async () => {
    const t = await setup()
    await t.router.execute(t.identity.context('alice'), submit('c1'))
    const ref = { workspace_id: 'w1', session_id: 's1', command_id: 'c1' }
    expect(t.router.status(t.identity.context('alice'), ref)).toMatchObject({ state: 'accepted' })
    expect(() => t.router.status(t.identity.context('owner'), ref)).toThrow(/no such command/)
    expect(() => t.router.status(t.identity.context('alice'), { ...ref, command_id: 'zzz' })).toThrow(/no such command/)
    expect(() => t.router.status(t.identity.context('alice'), { ...ref, extra: 1 })).toThrow(/unknown parameter/)
  })

  test('workspace grants for other principals do not leak into command scope', async () => {
    const t = await setup()
    t.identity.add(agentPrincipal('worker'), [workspaceGrant('worker', 'w1', 'viewer')])
    await expect(t.router.execute(t.identity.context('worker'), submit('c1'))).rejects.toMatchObject({ code: 'forbidden' })
  })
})
