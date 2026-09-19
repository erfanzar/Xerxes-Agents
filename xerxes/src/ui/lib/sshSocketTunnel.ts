// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { spawn } from 'node:child_process'
import { existsSync, rmSync } from 'node:fs'
import { setTimeout as delay } from 'node:timers/promises'
import { sshFailure } from '../../security/sshDiagnostics.js'

/** Own a fresh master with no inherited forwards, then add exactly one socket
 * through its private control socket. The control request reads no user config.
 * ClearAllForwardings cannot be used with an ordinary -L: it clears that too. */
export function startSshSocketTunnel(options: {
  ssh: readonly string[]
  target: string
  controlPath: string
  localSocket: string
  remoteSocket: string
  onFailure: (error: Error) => void
  spawnProcess?: typeof spawn
  startupTimeoutMs?: number
}): { close(): void } {
  const launch = options.spawnProcess ?? spawn
  rmSync(options.controlPath, { force: true })
  let closed = false
  let failed = false
  let forwarding: ReturnType<typeof spawn> | undefined
  let stderr = ''
  const master = launch('ssh', [...options.ssh, '-M', '-S', options.controlPath,
    '-o', 'ControlPersist=no', '-o', 'ForkAfterAuthentication=no',
    '-o', 'ClearAllForwardings=yes', '-N', '-T', '--', options.target],
  { stdio: ['ignore', 'ignore', 'pipe'] })
  const close = () => {
    if (closed) return
    closed = true
    forwarding?.kill('SIGTERM')
    master.kill('SIGTERM')
  }
  const fail = (error: Error) => {
    if (closed || failed) return
    failed = true
    close()
    options.onFailure(error)
  }
  master.stderr?.on('data', chunk => { stderr = (stderr + String(chunk)).slice(-8192) })
  master.once('error', () => fail(new Error('Could not start SSH. Check that the SSH client is installed and executable.')))
  master.once('close', () => fail(sshFailure('tunnel', stderr)))
  const deadline = Date.now() + (options.startupTimeoutMs ?? 15000)
  void (async () => {
    while (!existsSync(options.controlPath)) {
      if (closed) return
      if (Date.now() >= deadline) { fail(new Error('SSH tunnel startup timed out.')); return }
      await delay(25)
    }
    if (closed) return
    // An explicit control operation fails if the master is unavailable; it
    // cannot fall back to a new unaudited connection or user config.
    forwarding = launch('ssh', ['-F', '/dev/null', '-S', options.controlPath,
      '-o', 'ControlMaster=no', '-O', 'forward', '-L', `${options.localSocket}:${options.remoteSocket}`,
      '--', options.target], { stdio: ['ignore', 'ignore', 'pipe'] })
    let forwardError = ''
    forwarding.stderr?.on('data', chunk => { forwardError = (forwardError + String(chunk)).slice(-8192) })
    forwarding.once('error', () => fail(new Error('Could not start SSH socket forwarding. Check that the SSH client is installed and executable.')))
    forwarding.once('close', code => {
      if (code !== 0) fail(sshFailure('tunnel', forwardError))
    })
    while (!closed && !existsSync(options.localSocket)) {
      if (Date.now() >= deadline) { fail(new Error('SSH tunnel startup timed out.')); return }
      await delay(25)
    }
  })().catch(() => fail(new Error('Could not open the private SSH socket. Reconnect with /machine.')))
  return { close }
}
