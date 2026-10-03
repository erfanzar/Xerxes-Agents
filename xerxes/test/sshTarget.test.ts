// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { spawn } from 'node:child_process'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { isSshTarget, sshDestination } from '../src/daemon/sshTarget.js'
import { browseSshFolders } from '../src/daemon/machineDiscovery.js'
import { runMachineCommand } from '../src/daemon/machineCommand.js'
import { remoteTarget } from '../src/desktop/main/remote.js'
import { sshDestination as tuiSshDestination } from '../src/ui/lib/machineHandoff.js'

test('an SSH target may carry a port, and nothing else after the host', () => {
  for (const target of ['gpu', 'me@host', 'root@office.example.ir:7860', 'host:22', 'a.b-c_d:65535']) expect(isSshTarget(target)).toBe(true)
  for (const target of ['', 'host:', 'host:0', 'host:65536', 'host:123456', '-oProxyCommand=x', 'me@host:22:33', 'me@host:/srv', 'host name', 'host;id', 'a'.repeat(256)])
    expect(isSshTarget(target)).toBe(false)
})

test('a ported target reaches ssh as an ssh:// URI; plain targets and aliases are unchanged', () => {
  for (const convert of [sshDestination, tuiSshDestination]) {
    expect(convert('root@office.example.ir:7860')).toBe('ssh://root@office.example.ir:7860')
    expect(convert('host:2222')).toBe('ssh://host:2222')
    expect(convert('me@host')).toBe('me@host')
    expect(convert('gpu')).toBe('gpu')
  }
})

test('desktop and terminal accept a saved workspace whose target has a port', async () => {
  expect(remoteTarget({ alias: 'asr', target: 'root@office.example.ir:7860', workspacePath: '/root/asr' }).target).toBe('root@office.example.ir:7860')
  expect(() => remoteTarget({ alias: 'asr', target: 'root@host:99999', workspacePath: '/root/asr' })).toThrow('Invalid SSH')
  const home = await mkdtemp(join(tmpdir(), 'ssh-target-'))
  try {
    const file = join(home, 'machines.json')
    expect(await runMachineCommand(file, 'add asr root@office.example.ir:7860 /root/asr')).toMatchObject({ ok: true })
    expect(await runMachineCommand(file, 'connect asr')).toEqual({ ok: true, machine: { alias: 'asr', target: 'root@office.example.ir:7860', workspacePath: '/root/asr' } })
    expect(await runMachineCommand(file, 'add bad root@host:0 /root/asr')).toMatchObject({ ok: false })
  } finally { await rm(home, { recursive: true, force: true }) }
})

test('folder browsing passes the port to ssh as part of the destination', async () => {
  const seen: string[] = []
  const local = ((_file: string, args: readonly string[]) => {
    seen.push(args.at(-2)!)
    return spawn('sh', ['-c', args.at(-1)!], { stdio: ['ignore', 'pipe', 'pipe'] })
  })
  const root = await mkdtemp(join(tmpdir(), 'ssh-target-browse-'))
  try {
    expect((await browseSshFolders('root@office.example.ir:7860', root, { spawnProcess: local })).path).toBe(root)
    expect(seen).toEqual(['ssh://root@office.example.ir:7860'])
    await expect(browseSshFolders('root@host:70000', root, { spawnProcess: local })).rejects.toThrow('Choose an SSH alias')
  } finally { await rm(root, { recursive: true, force: true }) }
})
