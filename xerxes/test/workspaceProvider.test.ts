// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { WorkspaceProviderRegistry, createLocalWorkspaceProvider } from '../src/runtime/workspaceProvider.js'

test('workspace registry returns registered providers and rejects unknown kinds', () => {
  const registry = new WorkspaceProviderRegistry()
  const local = createLocalWorkspaceProvider({
    spawn: async () => ({ exitCode: 0, stdout: '', stderr: '' }),
    readFile: async path => `read ${path}`,
    writeFile: async () => {},
    mkdir: async () => {},
  })
  registry.register(local)
  expect(registry.has('local')).toBeTrue()
  expect(registry.get('local')).toBe(local)
  expect(() => registry.get('docker')).toThrow('workspace provider docker is not registered')
})

test('local workspace provider prepares, runs, and files through the host port', async () => {
  const spawns: Array<{ command: readonly string[]; cwd: string }> = []
  const files: Record<string, string> = {}
  const provider = createLocalWorkspaceProvider({
    spawn: async (command, options) => {
      spawns.push({ command, cwd: options.cwd })
      return { exitCode: 0, stdout: 'ok', stderr: '' }
    },
    readFile: async path => files[path] ?? '',
    writeFile: async (path, content) => { files[path] = content },
    mkdir: async () => {},
  })

  const connection = await provider.prepare({ id: 'ws-1', kind: 'local' })
  expect(connection.workingDir).toContain('ws-1')

  const result = await provider.exec(connection, ['echo', 'hello'])
  expect(result.exitCode).toBe(0)
  expect(result.stdout).toBe('ok')
  expect(spawns).toEqual([{ command: ['echo', 'hello'], cwd: connection.workingDir }])

  await provider.writeFile(connection, 'notes.md', '# hello')
  const content = await provider.readFile(connection, 'notes.md')
  expect(content).toBe('# hello')
})

test('workspace paths cannot escape the workspace', async () => {
  const reads: string[] = []
  const writes: string[] = []
  const made: string[] = []
  const provider = createLocalWorkspaceProvider({
    mkdir: async path => { made.push(path) },
    readFile: async path => { reads.push(path); return 'content' },
    writeFile: async path => { writes.push(path) },
    spawn: async () => ({ exitCode: 0, stdout: '', stderr: '' }),
  })
  const connection = await provider.prepare({ id: 'ws', kind: 'local', workingDir: '/tmp/ws-root' })

  // The abstraction whose stated purpose is isolation joined the path raw, so
  // these read and wrote straight through it.
  await expect(provider.readFile(connection, '../../../../etc/passwd')).rejects.toThrow('escapes workspace')
  await expect(provider.writeFile(connection, '../../.ssh/authorized_keys', 'k')).rejects.toThrow('escapes workspace')
  // Only revealed as an escape after normalization.
  await expect(provider.readFile(connection, 'a/../../outside.txt')).rejects.toThrow('escapes workspace')
  expect(reads).toEqual([])
  expect(writes).toEqual([])

  // Ordinary paths still work, and a nested write creates its parents rather
  // than failing with ENOENT.
  await provider.readFile(connection, 'notes.md')
  await provider.writeFile(connection, 'deep/nested/out.txt', 'x')
  expect(reads).toEqual(['/tmp/ws-root/notes.md'])
  expect(writes).toEqual(['/tmp/ws-root/deep/nested/out.txt'])
  expect(made).toContain('/tmp/ws-root/deep/nested')
})
