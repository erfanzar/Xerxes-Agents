// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { writeFile, readFile } from 'node:fs/promises'
import { HubOperatorError } from '../src/hub/ports.js'
import type { SshPort, SshProcess } from '../src/hub/ports.js'
import { openSshTunnel } from '../src/hub/sshTunnel.js'

function fakeSsh(failure?: string) {
  const calls: Array<{ command: string; args: readonly string[]; killed: boolean }> = []
  let emit: ((text: string) => void) | undefined
  const port: SshPort = { spawn(command, args): SshProcess {
    const call = { command, args, killed: false }; calls.push(call)
    let resolveExit: (code: number) => void = () => {}
    let controller: ReadableStreamDefaultController<Uint8Array>
    let closed = false
    const stderr = new ReadableStream<Uint8Array>({ start(value) { controller = value } })
    const exited = new Promise<number>(resolve => { resolveExit = resolve })
    const finish = (code: number) => { if (!closed) { closed = true; controller.close(); resolveExit(code) } }
    emit = text => { if (!closed) controller.enqueue(new TextEncoder().encode(text)) }
    if (calls.length === 1) {
      if (failure) queueMicrotask(() => { emit?.(failure); finish(255) })
      else {
        const control = args[args.indexOf('-S') + 1]!
        void writeFile(control, '').catch(() => finish(1))
      }
    } else {
      const forward = args[args.indexOf('-L') + 1]!
      void writeFile(forward.slice(0, forward.indexOf(':')), '').then(() => finish(0), () => finish(1))
    }
    return { stderr, exited, kill() { call.killed = true; finish(0) } }
  } }
  return { port, calls, emit(text: string) { emit?.(text) } }
}

test('SSH master uses a private config, strict host keys, no agent/X11/env forwarding, and one explicit socket forward', async () => {
  const fake = fakeSsh()
  const tunnel = await openSshTunnel({ sshAlias: 'prod', remoteSocketPath: '/tmp/xerxes.sock', port: fake.port, startupTimeoutMs: 500 })
  try {
    expect(fake.calls).toHaveLength(2)
    const master = fake.calls[0]!, forward = fake.calls[1]!
    for (const value of ['BatchMode=yes', 'StrictHostKeyChecking=yes', 'ForwardAgent=no', 'ForwardX11=no', 'ClearAllForwardings=yes', 'ServerAliveInterval=15', 'ServerAliveCountMax=3', 'ControlPersist=no']) expect(master.args).toContain(value)
    expect(master.args).not.toContain('-A')
    expect(master.args).not.toContain('ForwardAgent=yes')
    const config = await readFile(master.args[master.args.indexOf('-F') + 1]!, 'utf8')
    expect(config).toContain('SendEnv -*')
    expect(config).toContain('PermitLocalCommand no')
    expect(config).toContain('SetEnv XERXES_SSH=1')
    expect(forward.args).toContain('/dev/null')
    expect(forward.args).toContain('forward')
    expect(forward.args).toContain(`${tunnel.socketPath}:/tmp/xerxes.sock`)
    expect(tunnel).not.toHaveProperty('healthy')
  } finally { await tunnel.close() }
  expect(fake.calls.every(call => call.killed)).toBe(true)
})

test('host-key and authentication failures require an operator, with no internal retry storm', async () => {
  for (const failure of ['Host key verification failed.', 'Permission denied (publickey).']) {
    const fake = fakeSsh(failure)
    await expect(openSshTunnel({ sshAlias: 'prod', remoteSocketPath: '/tmp/d.sock', port: fake.port, startupTimeoutMs: 500 })).rejects.toBeInstanceOf(HubOperatorError)
    expect(fake.calls).toHaveLength(1)
    expect(fake.calls[0]?.killed).toBe(true)
  }
})

test('missing socket without explicit bootstrap consent is actionable and never bootstraps', async () => {
  const fake = fakeSsh()
  await expect(openSshTunnel({ sshAlias: 'prod', port: fake.port })).rejects.toThrow('consent')
  expect(fake.calls).toHaveLength(0)
  const missing = fakeSsh('channel 2: open failed: connect failed: No such file or directory')
  await expect(openSshTunnel({ sshAlias: 'prod', remoteSocketPath: '/tmp/d.sock', port: missing.port })).rejects.toBeInstanceOf(HubOperatorError)
  expect(missing.calls).toHaveLength(1)
})

test('abort cleans up the owned tunnel and unsafe alias/socket values are refused', async () => {
  const fake = fakeSsh(), abort = new AbortController()
  const tunnel = await openSshTunnel({ sshAlias: 'prod', remoteSocketPath: '/tmp/d.sock', port: fake.port, signal: abort.signal })
  abort.abort()
  await tunnel.close()
  expect(fake.calls.every(call => call.killed)).toBe(true)
  await expect(openSshTunnel({ sshAlias: '-evil', remoteSocketPath: '/tmp/d.sock', port: fake.port })).rejects.toBeInstanceOf(HubOperatorError)
  await expect(openSshTunnel({ sshAlias: 'prod', remoteSocketPath: '/tmp/bad:socket', port: fake.port })).rejects.toBeInstanceOf(HubOperatorError)
})

test('an unresponsive SSH child has a bounded startup and is killed on timeout', async () => {
  let killed = false, resolveExit: (code: number) => void = () => {}
  const exited = new Promise<number>(resolve => { resolveExit = resolve })
  const port: SshPort = { spawn() { return { stderr: new ReadableStream<Uint8Array>(), exited, kill() { killed = true; resolveExit(0) } } } }
  await expect(openSshTunnel({ sshAlias: 'prod', remoteSocketPath: '/tmp/d.sock', port, startupTimeoutMs: 20 })).rejects.toThrow('timed out')
  expect(killed).toBe(true)
})
