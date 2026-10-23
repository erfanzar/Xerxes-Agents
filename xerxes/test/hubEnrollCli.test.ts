// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdtemp, readFile, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { afterEach, expect, test } from 'bun:test'

import { InMemoryDaemonRuntime } from '../src/daemon/runtime.js'
import { DaemonServer } from '../src/daemon/server.js'
import { runHubCli } from '../src/hub/cli.js'
import { hubPaths } from '../src/hub/paths.js'
import { loadOrCreateHubSigner } from '../src/hub/signer.js'
import { WorkspaceRegistry } from '../src/hub/workspaceRegistry.js'

const cleanup: Array<() => Promise<void> | void> = []
afterEach(async () => { for (const close of cleanup.splice(0).reverse()) await close() })

test('workspaces enroll pins the Hub key on a local daemon through its local hub.enroll RPC', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xh-enroll-'))
  cleanup.push(() => rm(directory, { recursive: true, force: true }))
  const socketPath = join(directory, 'd.sock')
  const enrollmentPath = join(directory, 'hub-enrollments.json')
  const server = new DaemonServer({
    socketPath, projectDirectory: directory, hubEnrollmentPath: enrollmentPath,
    runtime: new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: directory, sessionDirectory: join(directory, 'sessions') }),
  })
  await server.start()
  cleanup.push(() => server.stop())

  const paths = hubPaths({ XERXES_HOME: join(directory, 'home') })
  const registry = new WorkspaceRegistry(paths.workspacesPath)
  const saved = await registry.add({
    label: 'local', projectDir: directory, endpoint: { kind: 'local', socketPath },
    connection: { autoConnect: false }, consent: { bootstrap: false, update: false },
  }, (await registry.load()).revision)
  const id = saved.workspaces.at(-1)!.id

  const lines: string[] = []
  const errors: string[] = []
  const code = await runHubCli(['workspaces', 'enroll', id], { paths, stdout: line => lines.push(line), stderr: line => errors.push(line) })
  expect(errors).toEqual([])
  expect(code).toBe(0)
  const signer = await loadOrCreateHubSigner({ paths })
  expect(lines.join('\n')).toContain(signer.fingerprint)
  const pinned = JSON.parse(await readFile(enrollmentPath, 'utf8')) as { hubs: { hub_id: string; fingerprint: string }[] }
  expect(pinned.hubs).toEqual([expect.objectContaining({ hub_id: signer.hubId, fingerprint: signer.fingerprint })])

  expect(await runHubCli(['workspaces', 'enroll', 'missing'], { paths, stdout: () => {}, stderr: () => {} })).toBe(1)
})
