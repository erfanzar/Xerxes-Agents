// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { spawn } from 'node:child_process'
import { createHash } from 'node:crypto'
import { mkdir, mkdtemp, realpath, rm } from 'node:fs/promises'
import { createServer, type Socket } from 'node:net'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { openRemote, runCaptured, type RemoteOpenPorts } from '../src/desktop/main/remote.js'
import { remoteAttachAddress, remoteAttachProgram } from '../src/desktop/main/remoteAttach.js'

const machine = { alias: 'worker', target: 'worker', workspacePath: '/srv/project' }
const address = { socketPath: '/remote/global.sock', projectDir: machine.workspacePath, version: '0.8.1', buildId: 'a'.repeat(16) }
const attached = 'XERXES_REMOTE_ATTACH ' + JSON.stringify(address)
const ready = 'XERXES_REMOTE_READY ' + JSON.stringify({ ...address, expectedBuildId: 'b'.repeat(16) })
const signal = () => new AbortController().signal
const localTunnel: NonNullable<RemoteOpenPorts['spawnTunnel']> = args => {
  const socketPath = args[args.indexOf('-L') + 1]!.split(':')[0]!
  return spawn(process.execPath, ['-e', `const {createServer}=await import('node:net');createServer(()=>{}).listen(${JSON.stringify(socketPath)});`], { stdio: ['ignore', 'ignore', 'pipe'] })
}

async function fixture(exercise: (root: string, path: string, requests: string[], sockets: Set<Socket>) => Promise<void>, reply: boolean | string = true): Promise<void> {
  const root = await realpath(await mkdtemp(join(tmpdir(), 'xd-attach-')))
  await mkdir(join(root, 'daemon'))
  const path = join(root, 'daemon', `global-${createHash('sha256').update(root).digest('hex').slice(0, 16)}.sock`)
  const sockets = new Set<Socket>(), requests: string[] = []
  const server = createServer(socket => {
    sockets.add(socket)
    socket.on('error', () => {})
    socket.on('close', () => sockets.delete(socket))
    let buffer = ''
    socket.on('data', chunk => {
      buffer += chunk.toString()
      for (;;) {
        const newline = buffer.indexOf('\n')
        if (newline < 0) break
        const frame = JSON.parse(buffer.slice(0, newline)); buffer = buffer.slice(newline + 1)
        requests.push(frame.method)
        expect(frame.params.project_dir).toBe(root)
        if (frame.method === 'session.list') expect(frame.params).toEqual({ project_dir: root, kind: 'main', scope: 'global', limit: 60 })
        if (frame.method === 'session.active_list') expect(frame.params).toEqual({ project_dir: root, history_limit: 0 })
        if (!reply) continue
        if (typeof reply === 'string') { socket.end(reply); continue }
        const result = frame.method === 'runtime.status' ? { ok: true, daemon_version: '0.8.1', daemon_build_id: 'a'.repeat(16) }
          : { ok: true, sessions: [{ id: 'chat-1', cwd: '/other', title: frame.id === 2 ? 'Saved title' : '', status: frame.id === 3 ? 'working' : 'idle' }] }
        socket.write(JSON.stringify({ jsonrpc: '2.0', id: frame.id, result }) + '\n')
      }
    })
  })
  await new Promise<void>(resolve => server.listen(path, resolve))
  try { await exercise(root, path, requests, sockets) }
  finally {
    for (const socket of sockets) socket.destroy()
    await new Promise<void>(resolve => server.close(() => resolve()))
    await rm(root, { recursive: true, force: true })
  }
}
function program(home: string, workspace: string, sessions = false): string {
  return `process.env.XERXES_HOME=${JSON.stringify(home)};delete process.env.XERXES_DAEMON_SOCKET;` + remoteAttachProgram(workspace, sessions)
}

test.skipIf(process.platform === 'win32')('attach probe resolves XERXES_HOME global socket and reads version without setup', async () => {
  await fixture(async (root, socketPath, requests) => {
    const output = await runCaptured(process.execPath, ['-e', program(root, root)], signal(), 10000)
    expect(remoteAttachAddress(output)).toEqual({ socketPath, projectDir: root, version: '0.8.1', buildId: 'a'.repeat(16) })
    expect(requests).toEqual(['runtime.status'])
  })
})
test.skipIf(process.platform === 'win32')('attach lister only requests bounded main/global and history-free live sessions', async () => {
  await fixture(async (root, _socketPath, requests) => {
    const output = await runCaptured(process.execPath, ['-e', program(root, root, true)], signal(), 10000)
    expect(remoteAttachAddress(output)?.projectDir).toBe(root)
    expect(output).toContain('XERXES_REMOTE_SESSIONS ')
    expect(requests).toEqual(['runtime.status', 'session.list', 'session.active_list'])
  })
})
test.skipIf(process.platform === 'win32')('attach probe reports MISSING only for an absent daemon', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xd-absent-'))
  try {
    const output = await runCaptured(process.execPath, ['-e', program(root, root)], signal(), 10000)
    expect(output.trim()).toBe('XERXES_REMOTE_MISSING')
    expect(remoteAttachAddress(output)).toBeNull()
  } finally { await rm(root, { recursive: true, force: true }) }
})
test.skipIf(process.platform === 'win32')('attach probe canonicalizes a git workspace to its toplevel', async () => {
  await fixture(async (root) => {
    await runCaptured('git', ['init', '--quiet', root], signal(), 5000)
    const nested = join(root, 'nested'); await mkdir(nested)
    const output = await runCaptured(process.execPath, ['-e', program(root, nested)], signal(), 10000)
    expect(remoteAttachAddress(output)?.projectDir).toBe(root)
  })
})
test.skipIf(process.platform === 'win32')('attach protocol failures never report a missing runtime or echo private RPC errors', async () => {
  for (const reply of ['not json\n', 'null\n', '{"id":1,"result":{"ok":false}}\n', '{"id":1,"error":{"message":"private-provider-token"}}\n']) {
    await fixture(async root => {
      await expect(runCaptured(process.execPath, ['-e', program(root, root)], signal(), 10000)).rejects.toThrow(/Invalid remote read-only response|rejected its read-only probe/)
    }, reply)
  }
})
test('attach address parser rejects control characters, malformed metadata, and ambiguous output', () => {
  for (const output of ['garbage', attached + '\nXERXES_REMOTE_MISSING', attached + '\n' + attached,
    ...[{ socketPath: '/a:b' }, { projectDir: '/bad\npath' }, { version: 'bad\nversion' }, { buildId: 'bad\nbuild' }].map(extra => 'XERXES_REMOTE_ATTACH ' + JSON.stringify({ ...address, ...extra }))]) {
    expect(() => remoteAttachAddress(output)).toThrow()
  }
  expect(remoteAttachAddress('XERXES_REMOTE_ATTACH ' + JSON.stringify({ ...address, buildId: 'bun-runtime-v0.8.1' }))?.buildId).toBe('bun-runtime-v0.8.1')
})
test.skipIf(process.platform === 'win32')('openRemote attaches and reconnects without bootstrap or background updates; only explicit update installs', async () => {
  const commands: string[][] = []
  const connection = await openRemote(machine, signal(), () => {}, undefined, {
    run: async (_binary, args) => { commands.push(args); return commands.length === 1 ? attached : commands.length === 2 ? 'XERXES_REMOTE_ALIVE' : ready },
    spawnTunnel: localTunnel,
  })
  let replacement: Awaited<ReturnType<typeof openRemote>> | undefined
  try {
    expect(connection.projectDir).toBe(address.projectDir)
    expect(connection.expectedBuildId).toBeUndefined()
    await new Promise(resolve => setTimeout(resolve, 25))
    expect(commands).toHaveLength(1)
    expect(commands[0]!.join(' ')).toContain('XERXES_REMOTE_ATTACH')
    expect(commands[0]!.join(' ')).not.toContain('XERXES_REMOTE_READY')
    replacement = await connection.reconnect(signal(), () => {})
    await new Promise(resolve => setTimeout(resolve, 25))
    expect(commands).toHaveLength(2)
    expect(await replacement.update()).toEqual({ ok: true })
    expect(commands).toHaveLength(3)
    expect(replacement.expectedBuildId).toBe('b'.repeat(16))
  } finally { await replacement?.close(); await connection.close() }
})
test.skipIf(process.platform === 'win32')('openRemote bootstraps only after a definitive MISSING response', async () => {
  const commands: string[][] = []
  const connection = await openRemote(machine, signal(), () => {}, undefined, {
    run: async (_binary, args) => { commands.push(args); return commands.length === 1 ? 'XERXES_REMOTE_MISSING' : ready }, spawnTunnel: localTunnel,
  })
  try {
    expect(commands).toHaveLength(2)
    expect(commands[1]!.join(' ')).toContain('XERXES_REMOTE_READY')
    expect(connection.expectedBuildId).toBe('b'.repeat(16))
  } finally { await connection.close() }
  let calls = 0
  await expect(openRemote(machine, signal(), () => {}, undefined, {
    run: async () => { calls++; return 'unrecognized probe reply' }, spawnTunnel: () => { throw new Error('must not forward') },
  })).rejects.toThrow('valid status')
  expect(calls).toBe(1)
})
test.skipIf(process.platform === 'win32')('reconnect discovers a running global daemon when a previous address disappears, without setup', async () => {
  let calls = 0
  const connection = await openRemote(machine, signal(), () => {}, { socketPath: '/old/daemon.sock', projectDir: machine.workspacePath }, {
    run: async () => ++calls === 1 ? 'XERXES_REMOTE_MISSING' : attached,
    spawnTunnel: localTunnel,
  })
  try { expect(calls).toBe(2); expect(connection.expectedBuildId).toBeUndefined() }
  finally { await connection.close() }
})
test.skipIf(process.platform === 'win32')('attach cancellation closes an unresponsive fake daemon without bootstrapping', async () => {
  await fixture(async (root, _path, requests) => {
    const controller = new AbortController()
    const pending = runCaptured(process.execPath, ['-e', program(root, root)], controller.signal, 10000)
    while (!requests.length) await new Promise(resolve => setTimeout(resolve, 5))
    controller.abort()
    await expect(pending).rejects.toThrow('cancelled')
  }, false)
})
