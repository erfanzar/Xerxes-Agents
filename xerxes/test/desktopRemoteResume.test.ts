// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { createServer, type Socket } from 'node:net'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { remoteResumeProgram } from '../src/desktop/main/remoteResume.js'
import { runCaptured } from '../src/desktop/main/remote.js'

async function probeFixture(reply: string | null, exercise: (path: string, connected: Promise<void>) => Promise<void>): Promise<void> {
  const root = await mkdtemp(join(tmpdir(), 'remote-resume-'))
  const path = join(root, "daemon's socket.sock")
  const sockets = new Set<Socket>()
  let accepted!: () => void
  const connected = new Promise<void>(resolve => { accepted = resolve })
  const server = createServer(socket => {
    sockets.add(socket)
    socket.on('error', () => {})
    socket.on('close', () => sockets.delete(socket))
    socket.once('data', data => {
      expect(JSON.parse(data.toString())).toMatchObject({ method: 'runtime.status', params: {} })
      accepted()
      if (reply !== null) socket.end(reply)
    })
  })
  await new Promise<void>(resolve => server.listen(path, resolve))
  try { await exercise(path, connected) }
  finally {
    for (const socket of sockets) socket.destroy()
    await new Promise<void>(resolve => server.close(() => resolve()))
    await rm(root, { recursive: true, force: true })
  }
}

test.skipIf(process.platform === 'win32')('SSH reconnect probe recognizes an existing daemon without setup', async () => {
  await probeFixture('{"jsonrpc":"2.0","id":1,"result":{"ok":true}}\n', async path => {
    const result = await runCaptured(process.execPath, ['-e', remoteResumeProgram(path)], new AbortController().signal, 7000)
    expect(result.trim()).toBe('XERXES_REMOTE_ALIVE')
  })
})

test.skipIf(process.platform === 'win32')('SSH reconnect probe distinguishes a missing daemon from protocol failures', async () => {
  const root = await mkdtemp(join(tmpdir(), 'remote-missing-'))
  try {
    expect((await runCaptured(process.execPath, ['-e', remoteResumeProgram(join(root, 'absent.sock'))], new AbortController().signal, 7000)).trim()).toBe('XERXES_REMOTE_MISSING')
  } finally { await rm(root, { recursive: true, force: true }) }
  for (const reply of ['not json\n', 'null\n', '{"id":1,"error":{"message":"authentication required"}}\n', '{"id":1,"result":{"ok":false}}\n']) {
    await probeFixture(reply, async path => {
      await expect(runCaptured(process.execPath, ['-e', remoteResumeProgram(path)], new AbortController().signal, 7000)).rejects.toThrow()
    })
  }
})

test.skipIf(process.platform === 'win32')('SSH reconnect probe bounds an unresponsive daemon without treating it as missing', async () => {
  await probeFixture(null, async path => {
    await expect(runCaptured(process.execPath, ['-e', remoteResumeProgram(path)], new AbortController().signal, 7000)).rejects.toThrow('did not answer')
  })
}, 10000)

test.skipIf(process.platform === 'win32')('SSH reconnect probe cancellation closes its socket', async () => {
  await probeFixture(null, async (path, connected) => {
    const controller = new AbortController()
    const operation = runCaptured(process.execPath, ['-e', remoteResumeProgram(path)], controller.signal, 7000)
    await connected
    controller.abort()
    await expect(operation).rejects.toThrow('cancelled')
  })
})
