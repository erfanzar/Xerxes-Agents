// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { mkdtemp, mkdir, rm } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { spawn } from 'node:child_process'
import { browseSshFolders, readSshHosts } from '../src/daemon/machineDiscovery.js'
import { runMachineCommand } from '../src/daemon/machineCommand.js'

test('SSH discovery follows includes, ignores patterns/comments and never executes Match', async () => {
  const home = await mkdtemp(join(tmpdir(), 'ssh-discovery-'))
  try {
    await mkdir(join(home, '.ssh', 'hosts'), { recursive: true })
    await Bun.write(join(home, '.ssh/config'), 'Host gpu dev # not-a-host\nHost * !excluded host?\nInclude "hosts/*.conf"\nMatch exec "touch /tmp/do-not-execute"\nHost=other\n')
    await Bun.write(join(home, '.ssh/hosts/a.conf'), 'Host "included" gpu\nInclude config\n')
    expect(await readSshHosts(home)).toEqual(['dev', 'gpu', 'included', 'other'])
    expect(await readSshHosts(join(home, 'missing'))).toEqual([])
  } finally { await rm(home, { recursive: true, force: true }) }
})

test('remote browse command lists actual directories safely and preserves unusual path characters', async () => {
  const home = await mkdtemp(join(tmpdir(), 'ssh-folders-'))
  try {
    const root = join(home, "repo's $(echo nope)")
    await mkdir(join(root, 'space name'), { recursive: true })
    await mkdir(join(root, '.hidden'))
    await Bun.write(join(root, 'file'), 'not a directory')
    const local = ((_file: string, args: readonly string[]) => {
      expect(args.slice(2, -1)).toEqual(['-T', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'ForwardAgent=no', '-o', 'ForwardX11=no', '-o', 'ConnectTimeout=10', '--', 'my-host'])
      return spawn('sh', ['-c', args.at(-1)!], { stdio: ['ignore', 'pipe', 'pipe'] })
    })
    expect(await browseSshFolders('my-host', root, { spawnProcess: local })).toEqual({ path: root, directories: ['.hidden', 'space name'], truncated: false })
    await expect(browseSshFolders('my-host', '/nonexistent-xerxes-folder', { spawnProcess: local })).rejects.toThrow('SSH browse failed')
    await expect(browseSshFolders('-oProxyCommand=bad', root)).rejects.toThrow('Choose an SSH alias')
  } finally { await rm(home, { recursive: true, force: true }) }
})

test('remote browse terminates on cancellation and timeout and reports process errors', async () => {
  const slow = (() => spawn('sh', ['-c', 'exec sleep 10'], { stdio: ['ignore', 'pipe', 'pipe'] }))
  const controller = new AbortController()
  const pending = browseSshFolders('host', '', { spawnProcess: slow, signal: controller.signal })
  controller.abort()
  await expect(pending).rejects.toThrow('cancelled')
  await expect(browseSshFolders('host', '', { spawnProcess: slow, timeoutMs: 10 })).rejects.toThrow('timed out')
  const missing = (() => spawn('/no-such-ssh-binary', [], { stdio: ['ignore', 'pipe', 'pipe'] }))
  await expect(browseSshFolders('host', '', { spawnProcess: missing })).rejects.toThrow()
})

test('remote browse classifies failures without exposing remote diagnostics', async () => {
  const denied = (() => spawn('sh', ['-c', "printf 'Permission denied private-sentinel-credential' >&2; exit 255"], { stdio: ['ignore', 'pipe', 'pipe'] }))
  const error = await browseSshFolders('host', '', { spawnProcess: denied }).catch(error => error as Error)
  expect(error).toBeInstanceOf(Error)
  expect(String(error)).toContain('SSH authentication failed')
  expect(String(error)).not.toContain('private-sentinel-credential')
})

test('machine discovery command contract returns typed results and actionable errors', async () => {
  const discovery = { hosts: async () => ['gpu'], browse: async (target: string, path: string) => {
    expect(target).toBe('gpu'); expect(path).toBe('/my folder'); return { path, directories: ['src'], truncated: false }
  } }
  expect(await runMachineCommand('/unused', 'hosts', discovery)).toEqual({ ok: true, hosts: ['gpu'] })
  expect(await runMachineCommand('/unused', `browse gpu ${Buffer.from('/my folder').toString('base64url')}`, discovery)).toEqual({ ok: true, path: '/my folder', directories: ['src'], truncated: false })
  expect(await runMachineCommand('/unused', 'hosts', { ...discovery, hosts: async () => { throw new Error('Cannot read SSH config') } })).toEqual({ ok: false, error: 'Cannot read SSH config' })
})
