// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, describe, expect, test } from 'bun:test'
import { createHash } from 'node:crypto'
import { basename, join } from 'node:path'

import { ensureHubDirectories, hubPaths } from '../src/hub/paths.js'
import { derivedWorkspaceId } from '../src/hub/runtimePool.js'
import { createHubService, type HubServiceOptions } from '../src/hub/service.js'
import { SupervisorModelStore } from '../src/hub/supervisor/modelControl.js'
import { WorkspaceRegistry } from '../src/hub/workspaceRegistry.js'
import { verifyHubAuth } from '../src/security/hubAuthEnvelope.js'
import { FakeIdentity, TestClient, disposeAll, registerCleanup, standardUsers, tempDir, testSigner, until } from './hubHarness.js'

afterEach(disposeAll)

type Frame = { id: number; method: string; params: Record<string, unknown> }
const MACHINE = 'mach1234'

function row(cwd: string, id = `s:${cwd}`): Record<string, unknown> {
  return { id, key: `k:${id}`, cwd, title: `Task ${basename(cwd)}`, status: 'idle', model: 'm' }
}

/** A v35 daemon that discovers projects from its chats and answers the three hub.model* methods. */
async function machineDaemon(dir: string, initial: { active?: Record<string, unknown>[]; saved?: Record<string, unknown>[]; models?: boolean }) {
  const socketPath = join(dir, 'm.sock')
  const requests: Frame[] = []
  const state = { active: initial.active ?? [], saved: initial.saved ?? [], models: initial.models ?? true, text: 'pong from the chosen model' }
  const sockets = new Set<Bun.Socket<{ buffer: string }>>()
  const server = Bun.listen<{ buffer: string }>({
    unix: socketPath,
    socket: {
      open(socket) { sockets.add(socket); socket.data = { buffer: '' } },
      close(socket) { sockets.delete(socket) },
      data(socket, bytes) {
        socket.data.buffer += bytes.toString()
        for (;;) {
          const newline = socket.data.buffer.indexOf('\n')
          if (newline < 0) return
          const line = socket.data.buffer.slice(0, newline)
          socket.data.buffer = socket.data.buffer.slice(newline + 1)
          const frame = JSON.parse(line) as Frame
          requests.push(frame)
          let result: Record<string, unknown> = { ok: true }
          switch (frame.method) {
            case 'initialize': result = { ok: true, daemon_protocol: 35, session: { id: 'metadata', key: frame.params.session_key }, connection_lease_supported: true }; break
            case 'connection.lease': result = { ok: true, token: 'lease' }; break
            case 'runtime.status':
              result = { ok: true, daemon_protocol: 35, runtime_id: 'rt-m', instance_id: 'i1', event_stream_supported: true, principal_auth_v1_supported: true, ...(state.models ? { hub_models_v1_supported: true } : {}) }
              break
            case 'session.active_list': result = { ok: true, sessions: state.active }; break
            case 'session.list': result = { ok: true, sessions: state.saved }; break
            case 'events.subscribe': result = { ok: true, subscription_id: 'sub', stream_id: 'stream', watermark: 'c0', oldest_cursor: 'c0', reset_required: false }; break
            case 'events.read': result = { ok: true, events: [], through_cursor: frame.params.after_cursor ?? 'c0' }; break
            case 'hub.models.catalog':
              result = { ok: true, hostname: 'mac', profiles: [
                { name: 'p1', label: 'Profile one', provider: 'openai', default_model: 'm-a', signs_in: false },
                { name: 'p2', label: 'Profile two', provider: 'anthropic', default_model: 'c-1', signs_in: true },
              ] }
              break
            case 'hub.models.options':
              result = frame.params.profile === 'p1' || frame.params.profile === 'p2'
                ? { ok: true, profile: frame.params.profile, provider: 'openai', models: [{ id: 'm-a', label: 'Model A' }, { id: 'm-b' }], model: frame.params.model ?? 'm-a',
                    efforts: [{ effort: 'low', description: 'fast' }, { effort: 'high' }], default_effort: 'low' }
                : { ok: false, code: 'unknown_profile', error: 'No provider profile' }
              break
            case 'hub.model.complete': result = { ok: true, text: state.text, tool_calls: [], usage: { input_tokens: 5, output_tokens: 2 } }; break
          }
          socket.write(JSON.stringify({ jsonrpc: '2.0', id: frame.id, result }) + '\n')
        }
      },
    },
  })
  registerCleanup(() => { for (const socket of sockets) socket.terminate(); server.stop(true) })
  return { socketPath, requests, state, callsTo: (method: string): Frame[] => requests.filter(request => request.method === method) }
}

async function machineFixture(options: {
  active?: Record<string, unknown>[]; saved?: Record<string, unknown>[]; models?: boolean
  supervisor?: HubServiceOptions['supervisor']; registryPollMs?: number; legacy?: boolean
} = {}) {
  const home = await tempDir('xh-mach-')
  const paths = hubPaths({ XERXES_HOME: home })
  await ensureHubDirectories(paths)
  const daemon = await machineDaemon(home, options)
  const ids = [MACHINE, 'w1']
  const registry = new WorkspaceRegistry(paths.workspacesPath, { generateId: () => ids.shift() ?? crypto.randomUUID() })
  await registry.upsertMachine({ label: 'Mac', socketPath: daemon.socketPath })
  if (options.legacy) {
    // A per-project entry on the same daemon keeps working beside the machine entry.
    await registry.add({
      label: 'Legacy', projectDir: join(home, 'legacy'), endpoint: { kind: 'local', socketPath: daemon.socketPath },
      connection: { autoConnect: true }, consent: { bootstrap: false, update: false },
    }, (await registry.load()).revision)
  }
  const identity = standardUsers(new FakeIdentity())
  const signer = testSigner()
  const errors: string[] = []
  const service = createHubService({
    paths, signer, registry, exposure: { port: 0 }, identity, registryPollMs: options.registryPollMs ?? 0,
    pool: { clientOptions: { retryBaseMs: 10, retryMaxMs: 25, random: () => 0.5, requestTimeoutMs: 2000 }, pollIntervalMs: 40 },
    ...(options.supervisor === undefined ? {} : { supervisor: options.supervisor }),
    onError: (error, context) => { errors.push(`${context}: ${error instanceof Error ? error.message : String(error)}`) },
  })
  registerCleanup(() => service.stop())
  const info = await service.start()
  await service.workspacesSettled
  await until(() => service.pool?.health(MACHINE)?.state === 'online' || undefined)
  const connect = async (who: string): Promise<TestClient> => {
    const client = await TestClient.connect(info.port, who)
    if (!client) throw new Error(`websocket for ${who} was refused`)
    return client
  }
  return { home, paths, daemon, identity, signer, service, info, errors, connect }
}

const derived = (dir: string): string => derivedWorkspaceId(MACHINE, dir)

describe('machine entries discover their projects', () => {
  test('one derived workspace per distinct cwd with a deterministic id; found live; the machine entry is never listed; routing works; removal takes them along', async () => {
    const t = await machineFixture({
      active: [row('/work/alpha'), row('/work/alpha', 's-alpha-2'), row('/work/beta')],
      saved: [row('/work/gamma'), { ...row('/work/sub'), parent_session_id: 'parent' }, row('relative')],
      registryPollMs: 30,
    })
    const dirs = ['/work/alpha', '/work/beta', '/work/gamma']
    await until(() => t.service.workspaceHealth().filter(item => item.machine_id === MACHINE).length === 3 || undefined)
    await until(() => dirs.every(dir => t.service.pool?.health(derived(dir))?.state === 'online') || undefined)

    // id = m-<first 8 chars of the entry id>-<sha256(cwd) hex 12>; subagent and relative cwds are not projects.
    expect(derived('/work/alpha')).toBe(`m-mach1234-${createHash('sha256').update('/work/alpha').digest('hex').slice(0, 12)}`)
    const ids = t.service.workspaceHealth().map(item => item.id).sort()
    expect(ids).toEqual(dirs.map(derived).sort())
    expect(ids).not.toContain(MACHINE)
    expect(t.service.status().workspaces.map(item => item.id).sort()).toEqual(ids)

    const owner = await t.connect('owner')
    const listed = (await owner.ok('hub.workspaces.list')).workspaces as Array<{ id: string; label: string; health: string }>
    expect(listed.map(item => item.id).sort()).toEqual(ids)
    expect(listed.find(item => item.id === derived('/work/alpha'))).toMatchObject({ label: 'Mac · alpha', health: 'online' })
    const snapshot = (await owner.ok('hub.snapshot')).workspaces as Array<{ id: string; sessions: unknown[] }>
    expect(snapshot.map(item => item.id)).not.toContain(MACHINE)
    expect(snapshot.find(item => item.id === derived('/work/alpha'))?.sessions).toHaveLength(2)
    expect(snapshot.find(item => item.id === derived('/work/beta'))?.sessions).toHaveLength(1)

    // A project that appears later joins on the next poll.
    t.daemon.state.active.push(row('/work/delta'))
    await until(() => t.service.pool?.derivedWorkspaces().some(item => item.projectDir === '/work/delta') || undefined)
    expect(t.service.pool?.derivedProjectDir(derived('/work/delta'))).toBe('/work/delta')
    expect(t.service.pool?.derivedProjectDir(MACHINE)).toBeUndefined()

    // The router resolves the derived id to its folder and signs the forwarded command.
    const opened = await owner.ok('hub.command', { workspace_id: derived('/work/delta'), command_id: 'open-1', action: 'session.open', args: {} })
    expect(opened).toMatchObject({ ok: true, state: 'accepted' })
    const sent = t.daemon.requests.find(request => request.method === 'session.open' && request.params.project_dir === '/work/delta' && request.params.hub_auth !== undefined)
    expect(sent).toBeDefined()
    // The machine entry itself is not a routable workspace.
    expect(await owner.fails('hub.command', { workspace_id: MACHINE, command_id: 'open-2', action: 'session.open', args: {} })).toBe('not_found')

    // Removing the machine removes everything derived from it.
    const registry = new WorkspaceRegistry(t.paths.workspacesPath)
    await registry.remove(MACHINE, (await registry.load()).revision)
    await until(() => t.service.pool?.derivedWorkspaces().length === 0 || undefined)
    expect(((await owner.ok('hub.workspaces.list')).workspaces as unknown[])).toEqual([])
    expect(t.errors).toEqual([])
  })

  test('hub.machines.list: the owner sees every machine, others only machines holding a workspace they can read', async () => {
    const t = await machineFixture({ active: [row('/work/alpha')], legacy: true })
    await until(() => t.service.pool?.derivedWorkspaces().length === 1 || undefined)
    const owner = await t.connect('owner')
    const alice = await t.connect('alice')
    const all = (await owner.ok('hub.machines.list')).machines as Array<Record<string, unknown>>
    expect(all).toHaveLength(1)
    expect(all[0]).toMatchObject({ id: MACHINE, label: 'Mac', health: 'online', read_only: false, models_supported: true })
    expect((all[0]!.workspace_ids as string[]).sort()).toEqual(['w1', derived('/work/alpha')].sort())
    const mine = (await alice.ok('hub.machines.list')).machines as Array<Record<string, unknown>>
    expect(mine).toHaveLength(1)
    expect(mine[0]!.workspace_ids).toEqual(['w1'])
    const viewer = await t.connect('viewer')
    expect(((await viewer.ok('hub.machines.list')).machines as Array<Record<string, unknown>>)[0]!.workspace_ids).toEqual(['w1'])
    const guest = await t.connect('guest')
    expect((await guest.ok('hub.machines.list')).machines).toEqual([])
  })
})

describe('XERXES model chosen at runtime', () => {
  test('unconfigured: thread opens, submit is refused with code unconfigured, settings are owner only', async () => {
    const t = await machineFixture({ legacy: true })
    const alice = await t.connect('alice')
    const owner = await t.connect('owner')
    expect(await alice.ok('supervisor.thread.open')).toMatchObject({ config_status: 'unconfigured' })
    expect(await alice.fails('supervisor.message.submit', { message_id: 'm1', text: 'hello', source: 'text' })).toBe('unconfigured')
    expect(await alice.ok('supervisor.config.get')).toMatchObject({ status: 'unconfigured', can_edit: false, message: "Choose XERXES's model in XERXES settings" })
    expect(await owner.ok('supervisor.config.get')).toMatchObject({ status: 'unconfigured', can_edit: true })
    expect(await alice.fails('supervisor.config.options', { machine_id: MACHINE })).toBe('forbidden')
    expect(await alice.fails('supervisor.config.set', { machine_id: MACHINE, profile: 'p1', model: 'm-a', effort: null })).toBe('forbidden')
    expect(t.identity.audits.some(record => record.action === 'supervisor.config.set' && record.decision === 'denied')).toBe(true)
    expect(new SupervisorModelStore(t.paths.databasePath).get()).toBeUndefined()
    expect(t.daemon.callsTo('hub.model.complete')).toHaveLength(0)
  })

  test('options are proxied from the daemon; set validates, persists, audits, broadcasts and hot-swaps; calls go signed to the chosen machine', async () => {
    const t = await machineFixture({ legacy: true })
    const alice = await t.connect('alice')
    const owner = await t.connect('owner')

    const catalog = await owner.ok('supervisor.config.options', { machine_id: MACHINE })
    expect(catalog).toMatchObject({ machine_id: MACHINE, models: [], efforts: [], default_effort: null })
    expect(catalog.profiles).toEqual([
      { name: 'p1', label: 'Profile one', provider: 'openai', default_model: 'm-a' },
      { name: 'p2', label: 'Profile two', provider: 'anthropic', default_model: 'c-1' },
    ])
    const detail = await owner.ok('supervisor.config.options', { machine_id: MACHINE, profile: 'p1' })
    expect(detail).toMatchObject({ profile: 'p1', model: 'm-a', default_effort: 'low', models: [{ id: 'm-a', label: 'Model A' }, { id: 'm-b' }], efforts: [{ effort: 'low', description: 'fast' }, { effort: 'high' }] })
    expect(await owner.fails('supervisor.config.options', { machine_id: 'nope' })).toBe('not_found')
    expect(await owner.fails('supervisor.config.options', { machine_id: MACHINE, profile: 'ghost' })).toBe('not_found')
    // Machine-side requests were signed by the supervisor agent with only the model scope, on behalf of the owner.
    const catalogCall = t.daemon.callsTo('hub.models.catalog')[0]!
    const { hub_auth: catalogAuth, ...catalogParams } = catalogCall.params
    const verifiedCatalog = verifyHubAuth(catalogAuth, { publicKey: t.signer.publicKeyPem, method: 'hub.models.catalog', params: catalogParams })
    expect(verifiedCatalog.ok).toBe(true)
    if (verifiedCatalog.ok) expect(verifiedCatalog.claims).toMatchObject({ principal_kind: 'agent', principal_id: 'xerxes-supervisor', requested_by: 'owner', scopes: ['supervisor.model'] })

    expect(await owner.fails('supervisor.config.set', { machine_id: MACHINE, profile: 'p1', model: 'not-offered', effort: null })).toBe('invalid_request')
    expect(await owner.fails('supervisor.config.set', { machine_id: MACHINE, profile: 'p1', model: 'm-a', effort: 'extreme' })).toBe('invalid_request')
    expect(await owner.fails('supervisor.config.set', { machine_id: 'nope', profile: 'p1', model: 'm-a', effort: null })).toBe('not_found')
    expect(await owner.fails('supervisor.config.set', { machine_id: MACHINE, profile: 'p1', model: 'm-a', effort: 'low', extra: 1 })).toBe('invalid_request')
    expect(new SupervisorModelStore(t.paths.databasePath).get()).toBeUndefined()

    const set = await owner.ok('supervisor.config.set', { machine_id: MACHINE, profile: 'p1', model: 'm-a', effort: 'low' })
    expect(set).toEqual({ status: 'configured', config: { machine_id: MACHINE, machine_label: 'Mac', profile: 'p1', model: 'm-a', effort: 'low' }, can_edit: true })
    await alice.event('supervisor.config_changed', payload => payload.status === 'configured' && payload.can_edit === false && JSON.stringify(payload).includes('"model":"m-a"'))
    await owner.event('supervisor.config_changed', payload => payload.can_edit === true)
    const store = new SupervisorModelStore(t.paths.databasePath)
    expect(store.get()).toMatchObject({ machineId: MACHINE, profile: 'p1', model: 'm-a', effort: 'low', updatedBy: 'owner' })
    store.close()
    expect(t.identity.audits.some(record => record.action === 'supervisor.config.set' && record.decision === 'allowed' && record.outcome === 'accepted')).toBe(true)
    expect(await alice.ok('supervisor.config.get')).toMatchObject({ status: 'configured', can_edit: false, config: { model: 'm-a', effort: 'low' } })

    // The next message runs on the chosen machine through a signed hub.model.complete.
    expect(await alice.ok('supervisor.thread.open')).toMatchObject({ config_status: 'configured', model: 'm-a' })
    await alice.ok('supervisor.message.submit', { message_id: 'm1', text: 'what is running?', source: 'text' })
    await alice.event('supervisor.thread_event', payload => JSON.stringify(payload).includes('pong from the chosen model'))
    const first = t.daemon.callsTo('hub.model.complete')[0]!
    expect(first.params).toMatchObject({ profile: 'p1', model: 'm-a', effort: 'low' })
    expect(typeof first.params.system).toBe('string')
    expect(Array.isArray(first.params.messages)).toBe(true)
    expect((first.params.tools as unknown[]).length).toBeGreaterThan(0)
    expect(typeof first.params.max_tokens).toBe('number')
    const { hub_auth: auth, ...completeParams } = first.params
    const verified = verifyHubAuth(auth, { publicKey: t.signer.publicKeyPem, method: 'hub.model.complete', params: completeParams })
    expect(verified.ok).toBe(true)
    if (verified.ok) expect(verified.claims).toMatchObject({ hub_id: t.signer.hubId, principal_kind: 'agent', principal_id: 'xerxes-supervisor', scopes: ['supervisor.model'], method: 'hub.model.complete' })

    // Swapping the choice needs no restart: the very next call uses the new model and no effort.
    t.daemon.state.text = 'second answer'
    expect(await owner.ok('supervisor.config.set', { machine_id: MACHINE, profile: 'p1', model: 'm-b', effort: null })).toMatchObject({ status: 'configured', config: { model: 'm-b', effort: null } })
    await alice.ok('supervisor.message.submit', { message_id: 'm2', text: 'and now?', source: 'text' })
    await alice.event('supervisor.thread_event', payload => JSON.stringify(payload).includes('second answer'))
    const second = t.daemon.callsTo('hub.model.complete')[1]!
    expect(second.params.model).toBe('m-b')
    expect(second.params).not.toHaveProperty('effort')
    expect(t.errors).toEqual([])
  })

  test('the XERXES_HUB_SUPERVISOR_* model is only a fallback while nothing is stored', async () => {
    const fallback = { id: 'env-model', reportsCost: false, complete: async () => ({ text: 'from the env model', toolCalls: [] }) }
    const t = await machineFixture({ legacy: true, supervisor: { kind: 'configured', model: fallback } })
    const owner = await t.connect('owner')
    expect(await owner.ok('supervisor.config.get')).toMatchObject({ status: 'configured', can_edit: true })
    expect(await owner.ok('supervisor.thread.open')).toMatchObject({ model: 'env-model', config_status: 'configured' })
    await owner.ok('supervisor.message.submit', { message_id: 'm1', text: 'hi', source: 'text' })
    await owner.event('supervisor.thread_event', payload => JSON.stringify(payload).includes('from the env model'))
    expect(t.daemon.callsTo('hub.model.complete')).toHaveLength(0)
  })

  test('a machine that cannot run model calls is reported and refused', async () => {
    const t = await machineFixture({ legacy: true, models: false })
    const owner = await t.connect('owner')
    expect(((await owner.ok('hub.machines.list')).machines as Array<Record<string, unknown>>)[0]).toMatchObject({ models_supported: false })
    expect(await owner.fails('supervisor.config.options', { machine_id: MACHINE })).toBe('unsupported')
    expect(await owner.fails('supervisor.config.set', { machine_id: MACHINE, profile: 'p1', model: 'm-a', effort: null })).toBe('unsupported')
    expect(t.daemon.callsTo('hub.models.catalog')).toHaveLength(0)
  })
})
