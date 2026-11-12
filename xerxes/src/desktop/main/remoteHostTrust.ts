// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { createServer, type Socket } from 'node:net'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import type { RemoteRunner } from './remote.js'

export type ConfirmRemoteHostKey = (target: string, prompt: string, signal: AbortSignal) => Promise<boolean>

/** Force a fresh handshake: a cached master can hide a missing known_hosts entry. */
export const freshSshOptions = ['-S', 'none', '-o', 'ControlMaster=no', '-o', 'ForkAfterAuthentication=no']

/** OpenSSH owns key lookup, changed-key rejection and persistence. We only answer its first-trust prompt. */
export async function verifyRemoteHost(
  destination: string,
  signal: AbortSignal,
  confirm: ConfirmRemoteHostKey,
  run: RemoteRunner,
  strictOptions: readonly string[],
): Promise<void> {
  signal.throwIfAborted()
  // The packaged desktop currently uses Unix stream-local forwarding. Do not
  // weaken verification on a host without the executable askpass bridge.
  if (process.platform === 'win32') {
    await run('ssh', [...strictOptions, '-T', '--', destination, 'true'], signal, 20000)
    return
  }
  const promptController = new AbortController()
  const promptSignal = AbortSignal.any([signal, promptController.signal])
  const directory = await mkdtemp(join(tmpdir(), 'xt-'))
  const socketPath = join(directory, 'trust.sock')
  const helper = join(directory, 'askpass.cjs')
  const executable = join(directory, 'askpass')
  const sockets = new Set<Socket>()
  let asked = false
  let promptFailure: unknown
  const server = createServer(socket => {
    sockets.add(socket)
    socket.on('close', () => sockets.delete(socket))
    socket.on('error', () => {})
    let input = ''
    socket.on('data', chunk => {
      input += chunk.toString()
      if (input.length > 8192) { socket.destroy(); return }
      const newline = input.indexOf('\n')
      if (newline < 0) return
      socket.removeAllListeners('data')
      let prompt: unknown
      try { prompt = JSON.parse(input.slice(0, newline)) } catch { socket.end('no\n'); return }
      // Never turn this into a password/passphrase prompt or a way to replace a
      // changed host key. OpenSSH rejects changed keys under checking=ask.
      if (asked || signal.aborted || typeof prompt !== 'string' || prompt.length > 8192 ||
        /[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]/.test(prompt) ||
        !prompt.includes('Are you sure you want to continue connecting') || !/SHA256:[A-Za-z0-9+/=]+/.test(prompt)) {
        socket.end('no\n'); return
      }
      asked = true
      void Promise.resolve().then(() => confirm(destination, prompt, promptSignal)).then(approved => {
        if (!socket.destroyed) socket.end(approved && !signal.aborted ? 'yes\n' : 'no\n')
      }, error => {
        promptFailure = error
        if (!socket.destroyed) socket.end('no\n')
      })
    })
  })
  const abort = () => { for (const socket of sockets) socket.destroy() }
  try {
    await writeFile(helper, `const {connect}=require('node:net');
const socket=connect(${JSON.stringify(socketPath)});
let answer='';
socket.setTimeout(120000,()=>socket.destroy());
socket.on('connect',()=>socket.write(JSON.stringify(process.argv[2]||'')+'\\n'));
socket.on('data',chunk=>{answer+=chunk;if(answer.length>16)socket.destroy();});
socket.on('end',()=>{process.stdout.write(answer==='yes\\n'?'yes\\n':'no\\n');});
socket.on('error',()=>{process.exitCode=1;});
`, { mode: 0o600 })
    const quote = (value: string) => "'" + value.replaceAll("'", "'\\''") + "'"
    await writeFile(executable, `#!/bin/sh\nELECTRON_RUN_AS_NODE=1 exec ${quote(process.execPath)} ${quote(helper)} "$@"\n`, { mode: 0o700 })
    await new Promise<void>((resolve, reject) => {
      server.once('error', reject)
      server.listen(socketPath, () => { server.removeListener('error', reject); resolve() })
    })
    signal.addEventListener('abort', abort, { once: true })
    signal.throwIfAborted()
    try {
      await run('ssh', [
        '-o', 'BatchMode=no', '-o', 'StrictHostKeyChecking=ask',
        '-o', 'UpdateHostKeys=no', '-o', 'PasswordAuthentication=no', '-o', 'KbdInteractiveAuthentication=no',
        ...strictOptions, '-T', '--', destination, 'true',
      ], signal, 120000, {
        ...process.env, LC_ALL: 'C', SSH_ASKPASS: executable, SSH_ASKPASS_REQUIRE: 'force', DISPLAY: 'xerxes:0',
      })
    } catch (error) {
      if (promptFailure) throw promptFailure
      throw error
    }
  } finally {
    promptController.abort()
    signal.removeEventListener('abort', abort)
    for (const socket of sockets) socket.destroy()
    if (server.listening) await new Promise<void>(resolve => server.close(() => resolve()))
    await rm(directory, { recursive: true, force: true })
  }
}
