// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, expect, test } from 'bun:test'
import { SuperagentAbilities } from '../src/hub/supervisor/abilities/service.js'
import { createSupervisorToolRegistry, TURN_METADATA_KEY } from '../src/hub/supervisor/tools.js'
import { disposeAll } from './hubHarness.js'
import { makeWorld, toolCall } from './hubSupervisorHarness.js'

afterEach(disposeAll)

test('production ability composition registers scoped schemas and audits notebook persistence', async () => {
  const world = makeWorld(); const audit: string[] = []
  const abilities = new SuperagentAbilities({ db: world.db, clock: world.clock, backend: world.backend,
    authorized: owner => owner === 'owner', deliver: async () => {}, audit: async action => { audit.push(action) } })
  try {
    const registry = createSupervisorToolRegistry({ ...world, abilities })
    const turn = { threadId: 'shared', requester: world.user('owner'), reader: world.reader('owner'), denial: undefined }
    expect(registry.definitions().some(definition => definition.function.name === 'notebook_put')).toBe(true)
    await registry.execute(toolCall('notebook_put', { workspace_id: 'w1', key: 'promise', kind: 'promise', text: 'remember me' }), { metadata: { [TURN_METADATA_KEY]: turn } })
    const result: unknown = JSON.parse(await registry.execute(toolCall('notebook_list', { workspace_id: 'w1' }), { metadata: { [TURN_METADATA_KEY]: turn } }))
    expect(result).toMatchObject({ entries: [{ text: 'remember me' }] })
    expect(audit).toContain('intent:notebook.put'); expect(audit).toContain('outcome:notebook.put')
  } finally { abilities.dispose() }
})

test('workspace-only reads reject session guests before calling daemon and fail closed without audit', async () => {
  const world = makeWorld()
  const abilities = new SuperagentAbilities({ db: world.db, clock: world.clock, backend: world.backend,
    authorized: () => true, deliver: async () => {}, audit: async () => { throw new Error('audit offline') } })
  try {
    const turn = { threadId: 'shared', requester: world.user('guest'), reader: world.reader('guest'), denial: undefined }
    await expect(abilities.execute('workspace.read', { workspace_id: 'w1', path: 'a.txt' }, turn)).rejects.toThrow('cannot read')
    abilities.notebook.put({ owner: 'owner', thread: 'shared', workspace: 'w1' }, 'private', 'plan', 'workspace-private note')
    expect(abilities.context('owner', 'shared', ['w1'], world.reader('owner'))).toBe('')
    const privateThread = world.threads.privateFor(world.user('owner'))
    const owner = { ...turn, threadId: privateThread.id, requester: world.user('owner'), reader: world.reader('owner', privateThread.id) }
    await expect(abilities.execute('workspace.read', { workspace_id: 'w1', path: 'a.txt' }, owner)).rejects.toThrow('audit offline')
    expect(world.backendCalls).toHaveLength(0)
  } finally { abilities.dispose() }
})

test('opt-in disk polling only reads subscribed authorized workspaces and stays silent while unchanged', async () => {
  const world = makeWorld(); let authorized = true; let reads = 0; let delivered = 0
  const abilities = new SuperagentAbilities({ db: world.db, clock: world.clock,
    backend: { call: async (_workspace, method, params) => {
      expect(method).toBe('superagent.observe')
      expect(params).toEqual({ ability: 'machine.health', view: 'disk' })
      reads++
      return { state: 'accepted', result: { ok: true, observation: { disk: { available_bytes: 5, total_bytes: 100 } } } }
    } }, authorized: () => authorized, deliver: async () => { delivered++ }, audit: async () => {},
    diskPolling: { intervalMs: 5 } })
  try {
    await Bun.sleep(15); expect(reads).toBe(0)
    abilities.wakes.subscribe({ owner: 'owner', thread: 'shared', workspace: 'w1' }, { kind: 'disk_threshold', threshold: 90 })
    const deadline = Date.now() + 1000
    while (delivered === 0 && Date.now() < deadline) await Bun.sleep(5)
    expect(reads).toBeGreaterThan(0); expect(delivered).toBe(1)
    await Bun.sleep(20); expect(delivered).toBe(1)
    authorized = false
    await Bun.sleep(10); const stoppedReads = reads
    await Bun.sleep(20); expect(reads).toBe(stoppedReads)
  } finally { abilities.dispose() }
  const closedReads = reads
  await Bun.sleep(15); expect(reads).toBe(closedReads)
})
