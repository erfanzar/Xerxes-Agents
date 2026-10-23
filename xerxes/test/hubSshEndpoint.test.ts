// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { afterEach, expect, test } from 'bun:test'

import { runHubCli } from '../src/hub/cli.js'
import { hubPaths } from '../src/hub/paths.js'
import { toPoolWorkspace } from '../src/hub/service.js'
import { WorkspaceRegistry } from '../src/hub/workspaceRegistry.js'

const cleanup: Array<() => Promise<void>> = []
afterEach(async () => { for (const close of cleanup.splice(0)) await close() })

async function home(): Promise<ReturnType<typeof hubPaths>> {
  const directory = await mkdtemp(join(tmpdir(), 'xh-ssh-endpoint-'))
  cleanup.push(() => rm(directory, { recursive: true, force: true }))
  return hubPaths({ XERXES_HOME: directory })
}

const ssh = (remoteSocketPath?: string) => ({
  label: 'box', projectDir: '/srv/project',
  endpoint: { kind: 'ssh', sshAlias: 'box', remoteProjectDir: '/srv/project', ...(remoteSocketPath === undefined ? {} : { remoteSocketPath }) },
  connection: { autoConnect: false }, consent: { bootstrap: false, update: false },
})

test('an SSH workspace keeps its remote socket path through the registry and into the pool', async () => {
  const paths = await home()
  const registry = new WorkspaceRegistry(paths.workspacesPath)
  const saved = await registry.add(ssh('/home/xerxes/.xerxes/daemon/global-abc.sock'), 0)
  const reloaded = await new WorkspaceRegistry(paths.workspacesPath).load()
  const entry = reloaded.workspaces.find(row => row.id === saved.workspaces.at(-1)!.id)!
  expect(entry.endpoint).toMatchObject({ kind: 'ssh', remoteSocketPath: '/home/xerxes/.xerxes/daemon/global-abc.sock' })
  expect(toPoolWorkspace(entry).endpoint).toEqual({ kind: 'ssh', sshAlias: 'box', remoteSocketPath: '/home/xerxes/.xerxes/daemon/global-abc.sock' })
})

test('relative or colon-bearing remote socket paths are refused', async () => {
  const paths = await home()
  const registry = new WorkspaceRegistry(paths.workspacesPath)
  await expect(registry.add(ssh('relative.sock'), 0)).rejects.toMatchObject({ code: 'invalid_input' })
  await expect(registry.add(ssh('/tmp/a:b.sock'), 0)).rejects.toMatchObject({ code: 'invalid_input' })
})

test('add-ssh --socket stores the path; without it the CLI warns that the workspace cannot connect', async () => {
  const paths = await home()
  const errors: string[] = []
  const quiet = { paths, stdout: () => {}, stderr: (line: string) => errors.push(line) }
  expect(await runHubCli(['workspaces', 'add-ssh', 'box', '/srv/a', '--socket', '/run/x/d.sock', '--label', 'A'], quiet)).toBe(0)
  expect(errors).toEqual([])
  expect(await runHubCli(['workspaces', 'add-ssh', 'box', '/srv/b'], quiet)).toBe(0)
  expect(errors).toEqual([expect.stringContaining('--socket')])
  const rows = (await new WorkspaceRegistry(paths.workspacesPath).load()).workspaces
  expect(rows.find(row => row.projectDir === '/srv/a')).toMatchObject({ label: 'A', endpoint: { remoteSocketPath: '/run/x/d.sock' } })
  expect(rows.find(row => row.projectDir === '/srv/b')?.endpoint).not.toHaveProperty('remoteSocketPath')
})
