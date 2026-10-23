// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, describe, expect, test } from 'bun:test'
import { mkdtemp, readFile, readdir, rm, stat, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { ensureHubDirectories, hubPaths } from '../src/hub/paths.js'
import { HubRegistryConflictError, HubRegistryError, WorkspaceRegistry } from '../src/hub/workspaceRegistry.js'

const temporary: string[] = []
afterEach(async () => { await Promise.all(temporary.splice(0).map(path => rm(path, { recursive: true, force: true }))) })
async function fixture() {
  const home = await mkdtemp(join(tmpdir(), 'xh-reg-'))
  temporary.push(home)
  const paths = hubPaths({ XERXES_HOME: home })
  return { home, paths, registry: new WorkspaceRegistry(paths.workspacesPath) }
}
const local = { label: 'Local', projectDir: '/projects/local', endpoint: { kind: 'local' }, connection: { autoConnect: false }, consent: { bootstrap: false, update: false } }

describe('Hub workspace registry', () => {
  test('home paths honor XERXES_HOME and directories/files are private', async () => {
    const { home, paths, registry } = await fixture()
    expect(paths.stateDir).toBe(join(home, 'hub'))
    expect(paths.databasePath).toBe(join(home, 'hub', 'hub.sqlite'))
    await ensureHubDirectories(paths)
    expect((await stat(paths.stateDir)).mode & 0o777).toBe(0o700)
    expect((await stat(paths.logsDir)).mode & 0o777).toBe(0o700)
    expect(await registry.load()).toEqual({ version: 1, revision: 0, workspaces: [] })
    const saved = await registry.add(local, 0)
    expect(saved.revision).toBe(1)
    expect((await stat(paths.workspacesPath)).mode & 0o777).toBe(0o600)
  })

  test('stable ids, updates and removal are revision checked', async () => {
    const { registry } = await fixture()
    const first = await registry.add(local, 0)
    const id = first.workspaces[0]!.id
    const changed = await registry.update(id, { label: 'Renamed', connection: { autoConnect: true } }, 1)
    expect(changed.workspaces[0]!.id).toBe(id)
    expect(changed.workspaces[0]!.label).toBe('Renamed')
    expect(changed.workspaces[0]!.createdAt).toBe(first.workspaces[0]!.createdAt)
    await expect(registry.remove(id, 1)).rejects.toBeInstanceOf(HubRegistryConflictError)
    expect((await registry.remove(id, 2)).workspaces).toEqual([])
  })

  test('cross-process lock prevents concurrent lost updates and leaves atomic JSON without temp files', async () => {
    const { paths, registry } = await fixture()
    const other = new WorkspaceRegistry(paths.workspacesPath)
    const outcomes = await Promise.allSettled([registry.add(local, 0), other.add({ ...local, label: 'Other' }, 0)])
    expect(outcomes.filter(result => result.status === 'fulfilled')).toHaveLength(1)
    const failure = outcomes.find(result => result.status === 'rejected')
    expect(failure?.status === 'rejected' && failure.reason).toBeInstanceOf(HubRegistryConflictError)
    expect(JSON.parse(await readFile(paths.workspacesPath, 'utf8')).revision).toBe(1)
    const before = await readFile(paths.workspacesPath, 'utf8')
    await expect(registry.update((await registry.load()).workspaces[0]!.id, { projectDir: 'relative' }, 1)).rejects.toBeInstanceOf(HubRegistryError)
    expect(await readFile(paths.workspacesPath, 'utf8')).toBe(before)
    expect((await readdir(paths.stateDir)).filter(name => name.endsWith('.tmp') || name.endsWith('.lock'))).toEqual([])
  })

  test('separate Bun processes cannot both commit the same expected revision', async () => {
    const { paths, registry } = await fixture()
    const module = new URL('../src/hub/workspaceRegistry.js', import.meta.url).href
    const source = `import { WorkspaceRegistry } from ${JSON.stringify(module)};
      try { await new WorkspaceRegistry(${JSON.stringify(paths.workspacesPath)}).add(${JSON.stringify(local)}, 0) }
      catch (error) { if (error.code === 'revision_conflict') process.exit(2); throw error }`
    const children = [0, 1].map(() => Bun.spawn([process.execPath, '--eval', source], { stdin: 'ignore', stdout: 'pipe', stderr: 'pipe' }))
    try {
      const results = await Promise.all(children.map(async child => ({ code: await child.exited, error: await new Response(child.stderr).text() })))
      expect(results.map(result => result.code).sort()).toEqual([0, 2])
      expect(results.map(result => result.error)).toEqual(['', ''])
      expect((await registry.load()).revision).toBe(1)
      expect((await registry.load()).workspaces).toHaveLength(1)
    } finally { for (const child of children) if (child.exitCode === null) child.kill() }
  })

  test('corrupt registries are preserved and never replaced by mutations', async () => {
    const { paths, registry } = await fixture()
    await ensureHubDirectories(paths)
    for (const content of ['{broken', '{"version":2,"revision":0,"workspaces":[]}', '{"version":1,"revision":0,"workspaces":[{"password":"do-not-echo"}]}']) {
      await writeFile(paths.workspacesPath, content)
      await expect(registry.load()).rejects.toMatchObject({ code: 'corrupt_registry' })
      await expect(registry.add(local, 0)).rejects.toMatchObject({ code: 'corrupt_registry' })
      expect(await readFile(paths.workspacesPath, 'utf8')).toBe(content)
    }
  })

  test('untrusted SSH entries reject key material, passwords, unsafe aliases and unexpected fields', async () => {
    const { registry } = await fixture()
    const ssh = { ...local, endpoint: { kind: 'ssh' as const, sshAlias: 'server', remoteProjectDir: '/srv/work' }, projectDir: '/srv/work' }
    for (const input of [
      { ...ssh, label: '-----BEGIN OPENSSH PRIVATE KEY----- secret' },
      { ...ssh, endpoint: { ...ssh.endpoint, sshAlias: 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIsecret' } },
      { ...ssh, password: 'do-not-store' },
      { ...ssh, endpoint: { ...ssh.endpoint, privateKey: 'do-not-store' } },
      { ...ssh, endpoint: { ...ssh.endpoint, sshAlias: '-oProxyCommand=bad' } },
      { ...ssh, endpoint: { ...ssh.endpoint, remoteProjectDir: 'relative' } },
    ]) await expect(registry.add(input, 0)).rejects.toBeInstanceOf(HubRegistryError)
    expect((await registry.add(ssh, 0)).workspaces[0]!.endpoint).toEqual(ssh.endpoint)
  })

  test('explicit machines import is idempotent, requires a revision and never rewrites the source', async () => {
    const { paths, registry } = await fixture()
    const source = JSON.stringify({ machines: [{ alias: 'Build', target: 'user@build:2222', workspacePath: '/srv/build' }] }, null, 2)
    await writeFile(paths.machinesPath, source)
    const imported = await registry.importMachines(0)
    expect(imported.workspaces).toHaveLength(1)
    expect(imported.workspaces[0]).toMatchObject({ label: 'Build', projectDir: '/srv/build', endpoint: { kind: 'ssh', sshAlias: 'user@build:2222', remoteProjectDir: '/srv/build' }, connection: { autoConnect: false }, consent: { bootstrap: false, update: false } })
    expect((await registry.importMachines(1, paths.machinesPath)).workspaces).toHaveLength(1)
    expect(await readFile(paths.machinesPath, 'utf8')).toBe(source)
    await expect(registry.importMachines(0, paths.machinesPath)).rejects.toBeInstanceOf(HubRegistryConflictError)
  })

  test('machine routes: idempotent by endpoint, sorted hub before mac, only on machine entries, validated', async () => {
    const { registry } = await fixture()
    const { id } = await registry.upsertMachine({ label: 'TPU', socketPath: '/home/xerxes/.xerxes/tunnels/tpu-daemon.sock' })
    const mac = { via: 'mac', endpoint: { kind: 'local', socketPath: '/home/xerxes/.xerxes/tunnels/mac-relay-tpu.sock' } }
    const hub = { via: 'hub', endpoint: { kind: 'ssh', sshAlias: 'tpu', remoteSocketPath: '/home/erfan/.xerxes/daemon/global.sock' } }
    expect((await registry.addRoute(id, mac)).added).toBe(true)
    expect((await registry.addRoute(id, hub)).added).toBe(true)
    // Same endpoint again, or the machine's own endpoint: nothing changes.
    const again = await registry.addRoute(id, mac)
    expect(again.added).toBe(false)
    expect((await registry.addRoute(id, { via: 'mac', endpoint: { kind: 'local', socketPath: '/home/xerxes/.xerxes/tunnels/tpu-daemon.sock' } })).added).toBe(false)
    const machine = (await registry.load()).workspaces.find(row => row.id === id)!
    expect(machine.routes?.map(route => route.via)).toEqual(['hub', 'mac'])
    // Survives a reload exactly.
    expect((await new WorkspaceRegistry(registry.path).load()).workspaces.find(row => row.id === id)?.routes).toEqual(machine.routes)
    for (const bad of [
      { via: 'mac', endpoint: { kind: 'local', socketPath: 'relative.sock' } },
      { via: 'mac', endpoint: { kind: 'local', socketPath: '/tmp/a:b.sock' } },
      { via: 'hub', endpoint: { kind: 'ssh', sshAlias: '-oProxyCommand=x', remoteSocketPath: '/r.sock' } },
      { via: 'hub', endpoint: { kind: 'ssh', sshAlias: 'user@host', remoteSocketPath: '/r.sock' } },
      { via: 'phone', endpoint: { kind: 'local', socketPath: '/r.sock' } },
      { ...mac, extra: true },
    ]) await expect(registry.addRoute(id, bad)).rejects.toBeInstanceOf(HubRegistryError)
    // A per-project entry never carries routes.
    const project = await registry.add(local, (await registry.load()).revision)
    const projectId = project.workspaces.find(row => row.label === 'Local')!.id
    await expect(registry.addRoute(projectId, mac)).rejects.toBeInstanceOf(HubRegistryError)
    await expect(registry.add({ ...local, routes: [mac] }, (await registry.load()).revision)).rejects.toBeInstanceOf(HubRegistryError)
    // Removal drops just that route; removing the last one leaves no routes field.
    await registry.removeRoute(id, hub)
    expect((await registry.load()).workspaces.find(row => row.id === id)?.routes?.map(route => route.via)).toEqual(['mac'])
    await registry.removeRoute(id, mac)
    expect((await registry.load()).workspaces.find(row => row.id === id)?.routes).toBeUndefined()
  })
})
