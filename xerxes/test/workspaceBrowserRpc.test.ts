// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { InMemoryDaemonRuntime } from '../src/daemon/runtime.js'
import { DaemonServer } from '../src/daemon/server.js'
import { DaemonRpc } from '../src/desktop/main/daemon.js'

test('workspace RPC pages folder siblings and previews untracked content without staging it', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xr-browser-rpc-'))
  const runtime = new InMemoryDaemonRuntime({ async *run() {} }, { currentProjectDirectory: root, sessionDirectory: join(root, 'sessions') })
  const server = new DaemonServer({ socketPath: join(root, 'test.sock'), runtime, projectDirectory: root })
  const client = new DaemonRpc({ projectDir: root, socketPath: join(root, 'test.sock') })
  try {
    await Bun.spawn(['git', 'init', '-q', root]).exited
    for (let i = 0; i < 65; i++) await Bun.write(join(root, `f${String(i).padStart(2, '0')}.ts`), `export const value = ${i}\n`)
    await runtime.openSession('browser', undefined, { cwd: root })
    await server.start()
    const call = (method: string, params = {}) => client.call<Record<string, unknown>>(method, { session_key: 'browser', ...params })
    const first = await call('complete', { path_prefix: './f' })
    const next = await call('complete', { path_prefix: './f', path_offset: 50 })
    expect((first.completions as unknown[]).length).toBe(50)
    expect((next.completions as unknown[]).length).toBe(15)
    await expect(call('complete', { path_prefix: './', path_offset: -1 })).rejects.toThrow('path_offset')
    const overview = await call('workspace.diff')
    expect((overview.diff as {untracked: string[]}).untracked).toHaveLength(50)
    const expanded = await call('workspace.diff', {untracked_limit: 150})
    expect((expanded.diff as {untracked: string[]}).untracked).toHaveLength(65)
    await expect(call('workspace.diff', {untracked_limit: 10001})).rejects.toThrow('untracked_limit')
    const preview = await call('workspace.diff', { path: 'f64.ts' })
    expect(preview.kind).toBe('ok')
    expect(JSON.stringify(preview)).toContain('+export const value = 64')
    const index = Bun.spawn(['git', '-C', root, 'ls-files'], { stdout: 'pipe' })
    expect(await new Response(index.stdout).text()).toBe('')
    expect(await index.exited).toBe(0)
    await expect(call('workspace.diff', { path: 42 })).rejects.toThrow('path')
  } finally { client.dispose(); await server.stop(); await rm(root, { recursive: true, force: true }) }
})
