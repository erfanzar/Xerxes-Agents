// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdir, mkdtemp, realpath, rm } from 'node:fs/promises'
import { generateKeyPairSync, randomUUID } from 'node:crypto'
import { enrollHub } from '../src/security/hubIdentityVerifier.js'
import { paramsHash, signHubAuth, type HubAuthClaims } from '../src/security/hubAuthEnvelope.js'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { InMemoryDaemonRuntime } from '../src/daemon/runtime.js'
import { DaemonServer } from '../src/daemon/server.js'
import { createGitHubObservationHost } from '../src/daemon/superagentGitHub.js'
import { HubDaemonClient } from '../src/hub/daemonClient.js'
import type { ObservationHosts } from '../src/hub/supervisor/abilities/observations.js'

test('daemon composition selects a host only from the verified workspace, never model repository/credentials', async () => {
  const root = await realpath(await mkdtemp(join(tmpdir(), 'xerxes-observation-host-')))
  const selected: string[] = []; const requests: string[] = []
  const github = createGitHubObservationHost({ repository: 'acme/project', dnsLookup: async () => ['93.184.216.34'],
    fetcher: async url => { requests.push(url); return Response.json([]) } })
  const runtime = new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: root, sessionDirectory: join(root, 'sessions') })
  const server = new DaemonServer({ runtime, socketPath: join(root, 'daemon.sock'), projectDirectory: root,
    eventJournal: { directory: join(root, 'events') }, autoTitle: false, autoSnapshotTurns: false,
    observationHosts: (workspace: string): ObservationHosts => { selected.push(workspace); return { github } },
  })
  const client = new HubDaemonClient({ socketPath: join(root, 'daemon.sock'), projectDir: root, requestTimeoutMs: 1000 })
  try {
    await server.start(); await client.connect()
    expect(await client.call('superagent.observe', { ability: 'github.read', view: 'issues' })).toMatchObject({ ok: true, observation: { repository: 'acme/project', items: [], untrusted: true } })
    expect(selected).toEqual([root]); expect(requests).toHaveLength(1)
    await expect(client.call('superagent.observe', { ability: 'github.read', view: 'issues', repository: 'other/repo', token: 'untrusted' })).rejects.toThrow()
    expect(requests).toHaveLength(1)
  } finally { client.close(); await server.stop(); await rm(root, { recursive: true, force: true }) }
})

test('signed root-B reads cannot select the daemon-default root-A host or credential', async () => {
  const scratch = await realpath(await mkdtemp(join(tmpdir(), 'xerxes-observe-roots-')))
  const rootA = join(scratch, 'a'); const rootB = join(scratch, 'b')
  await mkdir(rootA); await mkdir(rootB)
  const pair = generateKeyPairSync('ed25519'); const enrollment = join(scratch, 'enrollment.json')
  enrollHub(enrollment, { hub_id: 'fixture-observer', public_key_pem: pair.publicKey.export({ type: 'spki', format: 'pem' }).toString(), project_dirs: [rootB] })
  const selected: string[] = []
  const runtime = new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: rootA, sessionDirectory: join(scratch, 'sessions') })
  const server = new DaemonServer({ runtime, socketPath: join(scratch, 'daemon.sock'), projectDirectory: rootA, hubEnrollmentPath: enrollment,
    eventJournal: { directory: join(scratch, 'events') }, autoTitle: false, autoSnapshotTurns: false,
    observationHosts: (root: string): ObservationHosts => { selected.push(root); return { github: async () => ({ repository: root === rootB ? 'B' : 'A-private' }) } },
  })
  const client = new HubDaemonClient({ socketPath: join(scratch, 'daemon.sock'), projectDir: rootB })
  const signed = (params: Readonly<Record<string, unknown>>) => {
    const claims: HubAuthClaims = { hub_id: 'fixture-observer', principal_id: 'observer', principal_kind: 'agent', display_name: 'Observer', workspace_id: 'B',
      method: 'superagent.observe', params_hash: paramsHash(params), request_id: randomUUID(), nonce: randomUUID(), issued_at: Date.now(), expires_at: Date.now() + 10_000, scopes: ['session.read'] }
    return { ...params, hub_auth: signHubAuth(claims, pair.privateKey) }
  }
  try {
    await server.start(); await client.connect()
    expect(await client.call('superagent.observe', signed({ ability: 'github.read', view: 'issues' }))).toMatchObject({ observation: { repository: 'B' } })
    expect(selected).toEqual([rootB])
    await expect(client.call('superagent.observe', signed({ ability: 'github.read', view: 'issues', project_dir: rootA }))).rejects.toThrow()
    expect(selected).toEqual([rootB])
  } finally { client.close(); await server.stop(); await rm(scratch, { recursive: true, force: true }) }
})

test('closing the test client aborts its observation host and suppresses a late result', async () => {
  const root = await realpath(await mkdtemp(join(tmpdir(), 'xerxes-observation-cancel-')))
  let signal: AbortSignal | undefined; let ready = (): void => {}; let finish = (): void => {}
  const started = new Promise<void>(resolve => { ready = resolve })
  const pending = new Promise<void>(resolve => { finish = resolve })
  const runtime = new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: root, sessionDirectory: join(root, 'sessions') })
  const server = new DaemonServer({ runtime, socketPath: join(root, 'daemon.sock'), projectDirectory: root,
    eventJournal: { directory: join(root, 'events') }, autoTitle: false, autoSnapshotTurns: false,
    observationHosts: (): ObservationHosts => ({ github: async (_view, received) => { signal = received; ready(); await pending; return { late: true } } }),
  })
  const client = new HubDaemonClient({ socketPath: join(root, 'daemon.sock'), projectDir: root, requestTimeoutMs: 1000 })
  try {
    await server.start(); await client.connect()
    const work = client.call('superagent.observe', { ability: 'github.read', view: 'issues' }).then(() => 'returned', () => 'closed')
    expect(await Promise.race([started.then(() => true), Bun.sleep(300).then(() => false)])).toBe(true)
    client.close()
    for (let i = 0; i < 100 && !signal?.aborted; i += 1) await Bun.sleep(2)
    expect(signal?.aborted).toBe(true)
    finish(); expect(await work).toBe('closed')
  } finally { finish(); client.close(); await server.stop(); await rm(root, { recursive: true, force: true }) }
})
