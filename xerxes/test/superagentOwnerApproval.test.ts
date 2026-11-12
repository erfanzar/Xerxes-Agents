// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, expect, test } from 'bun:test'
import { existsSync, readFileSync, unlinkSync } from 'node:fs'
import { basename, join } from 'node:path'

import { emptyProjection } from '../src/hub/projections.js'
import { actionToWire, SupervisorGrantOverlay } from '../src/hub/supervisor/actions.js'
import { createHubSupervisor } from '../src/hub/supervisor/index.js'
import { authenticate, cleanupAll, makeHarness, must, pairDevice } from './hubIdentityHarness.js'

afterEach(cleanupAll)

test('real isolated Hub owner approval executes once and audits; rejecting a second card executes nothing', async () => {
  // Real SQLite identity, HTTP pairing and cookie authentication, not FakeIdentity.
  const h = await makeHarness({ clock: { now: Date.now, advance() { throw new Error('This proof uses wall-clock time') } } })
  const browser = await pairDevice(h)
  const context = must(await authenticate(h, browser))
  if (context.principal.kind !== 'user') throw new Error('Expected the paired human owner')
  const owner = context.principal
  expect(owner.hubOwner).toBe(true)
  const ctx = { context, grants: h.identity.grantsFor(owner.id) }
  const approvedPath = join(h.dir, 'owned-approved-marker.txt')
  const rejectedPath = join(h.dir, 'owned-rejected-marker.txt')
  const contents = 'Disposable owner-confirmation proof marker.\n'
  await Bun.write(approvedPath, contents)
  await Bun.write(rejectedPath, contents)
  const executions: string[] = []
  const workspaceId = 'owned-confirmation-proof'
  const projection = emptyProjection(workspaceId)
  const supervisor = createHubSupervisor({
    databasePath: h.paths.databasePath,
    identity: h.identity,
    overlay: new SupervisorGrantOverlay(),
    clock: h.clock.now,
    isHubOwner: id => h.admin.ownerUserId() === id,
    workspaces: {
      list: () => [{ id: workspaceId, label: 'Owned temporary confirmation proof' }],
      projection: id => id === workspaceId ? projection : undefined,
      health: id => id === workspaceId ? { state: 'online', degraded: false, instanceId: 'owned-proof' } : undefined,
    },
    backend: { async call() { throw new Error('The marker action must not call a daemon') } },
    execute: async () => { throw new Error('The marker action must use its existing typed operation port') },
    model: { id: 'not-used-for-owner-confirmation', async complete() { throw new Error('Confirmations must not invoke a model') } },
    autopilot: { checkInMs: 0 },
    risk: {
      revision: (workspace, target) => workspace === workspaceId && [approvedPath, rejectedPath].includes(target)
        && existsSync(target) && readFileSync(target, 'utf8') === contents ? 'owned-marker-v1' : undefined,
      async execute(action, confirmer) {
        const target = action.args.target
        if (confirmer.id !== owner.id || action.args.operation !== 'delete' || typeof target !== 'string'
          || ![approvedPath, rejectedPath].includes(target)) throw new Error('Unexpected proof operation')
        unlinkSync(target)
        executions.push(target)
        return { deleted: basename(target) }
      },
    },
  })
  try {
    const thread = supervisor.threads.privateFor(owner)
    const reader = supervisor.threads.scopeReader(thread, owner)
    const propose = (target: string) => supervisor.actions.createIntent({
      threadId: thread.id, tool: 'risk.execute', requester: owner, reader,
      args: { workspace_id: workspaceId, operation: 'delete', target, expected_revision: 'owned-marker-v1',
        details: `Delete only the disposable ${basename(target)} created for this proof.` },
    })
    const respond = must(supervisor.handlers['supervisor.confirmation.respond'])
    const approved = await propose(approvedPath)
    expect(approved.state).toBe('pending')
    expect(approved.auto).toBe(false)
    expect(executions).toEqual([])
    expect(existsSync(approvedPath)).toBe(true)
    console.info(JSON.stringify({ proof: 'owner-confirmation-card', card: actionToWire(approved), executions: 0 }))

    const response = await respond(ctx, { confirmation_id: approved.id, decision: 'confirm', response_id: 'owner-approve-proof' })
    expect(response).toMatchObject({ confirmation: { state: 'confirmed', confirmed_by: owner.displayName } })
    expect(existsSync(approvedPath)).toBe(false)
    expect(executions).toEqual([approvedPath])
    await respond(ctx, { confirmation_id: approved.id, decision: 'confirm', response_id: 'duplicate-owner-approve-proof' })
    expect(executions).toEqual([approvedPath])

    const rejected = await propose(rejectedPath)
    expect(rejected.state).toBe('pending')
    const rejection = await respond(ctx, { confirmation_id: rejected.id, decision: 'cancel', response_id: 'owner-reject-proof' })
    expect(rejection).toMatchObject({ confirmation: { state: 'cancelled' } })
    // An approval arriving after the rejection must not resurrect the operation.
    await respond(ctx, { confirmation_id: rejected.id, decision: 'confirm', response_id: 'late-approve-proof' })
    expect(executions).toEqual([approvedPath])
    expect(existsSync(rejectedPath)).toBe(true)

    const audit = h.admin.listAudit({ limit: 1000 })
    const approvedAudit = audit.filter(event => event.request_hash === approved.argsHash)
    const rejectedAudit = audit.filter(event => event.request_hash === rejected.argsHash)
    expect(approvedAudit.map(event => event.action)).toEqual([
      'supervisor.intent:risk.execute', 'supervisor.confirm:risk.execute',
      'supervisor.dispatch:risk.execute', 'supervisor.outcome:risk.execute',
    ])
    expect(approvedAudit.at(-1)?.outcome).toBe('accepted')
    expect(rejectedAudit.map(event => event.action)).toEqual(['supervisor.intent:risk.execute', 'supervisor.cancel:risk.execute'])
    expect(h.admin.verifyAuditChain()).toMatchObject({ ok: true })
    console.info(JSON.stringify({ proof: 'real-owner-approve-once-reject-zero', approved: approved.id,
      rejected: rejected.id, approvedExecutions: executions.filter(path => path === approvedPath).length,
      rejectedExecutions: executions.filter(path => path === rejectedPath).length, rejectedMarkerPresent: existsSync(rejectedPath),
      approvedAudit: approvedAudit.map(event => ({ sequence: event.sequence, action: event.action, outcome: event.outcome })),
      rejectedAudit: rejectedAudit.map(event => ({ sequence: event.sequence, action: event.action, outcome: event.outcome })),
      auditChain: h.admin.verifyAuditChain() }))
  } finally {
    supervisor.close()
  }
})
