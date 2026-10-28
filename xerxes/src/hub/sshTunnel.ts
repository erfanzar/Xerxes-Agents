// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { existsSync } from 'node:fs'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { writeSshConnectionConfig } from '../security/sshConnectionConfig.js'
import { HubOperatorError } from './ports.js'
import type { HubTunnel, SshPort, SshProcess } from './ports.js'

/** Local auth is allowed; SSH forwards no agent, X11, environment or existing sockets. */
export const bunSshPort: SshPort = {
  spawn(command, args) {
    const env: Record<string, string> = {}
    for (const key of ['PATH', 'HOME', 'USER', 'LOGNAME', 'SSH_AUTH_SOCK']) if (process.env[key] !== undefined) env[key] = process.env[key]!
    const child = Bun.spawn([command, ...args], { stdin: 'ignore', stdout: 'ignore', stderr: 'pipe', env, windowsHide: true })
    return { stderr: child.stderr, exited: child.exited, kill: signal => { child.kill(signal ?? 'SIGTERM') } }
  },
}
export interface SshTunnelOptions {
  readonly sshAlias: string
  readonly remoteSocketPath?: string
  readonly bootstrapConsent?: boolean
  readonly signal?: AbortSignal
  readonly port?: SshPort
  readonly startupTimeoutMs?: number
  /** Fixed diagnostics only; raw SSH stderr never becomes a Hub/phone event. */
  readonly onFailure?: (error: Error) => void
}
function failure(stderr: string, consent: boolean): Error {
  if (/host key|host identification|REMOTE HOST IDENTIFICATION/i.test(stderr)) return new HubOperatorError('SSH host-key verification failed. Verify the host in a local terminal before reconnecting this workspace.')
  if (/permission denied|authentication failed|no supported authentication/i.test(stderr)) return new HubOperatorError('SSH authentication failed. Restore access to this SSH alias in a local terminal, then reconnect.')
  if (/no such file|connect failed: connection refused|cannot connect to.*socket/i.test(stderr)) return new HubOperatorError(consent
    ? 'The remote daemon socket is missing. Start the consented runtime through an explicit operator bootstrap port, then reconnect; this tunnel never installs or updates a runtime.'
    : 'The remote daemon socket is missing. Start Xerxes on the host or explicitly grant remote-bootstrap consent in the workspace registry, then reconnect. Nothing was installed or updated.')
  return new Error('SSH tunnel disconnected. Reconnecting requires a fresh daemon handshake; remote work was not cancelled.')
}

/**
 * Own a new private master, clear inherited forwards, then add exactly one
 * Unix forward using a config-free control request. A listening socket proves
 * only transport readiness: the runtime pool MUST initialize and runtime.status
 * before marking a workspace online. No implicit remote command/bootstrap.
 */
export async function openSshTunnel(options: SshTunnelOptions): Promise<HubTunnel> {
  if (!/^[a-zA-Z0-9][a-zA-Z0-9_.-]*$/.test(options.sshAlias)) throw new HubOperatorError('Use a saved SSH alias, not a destination, option or shell command.')
  if (!options.remoteSocketPath) throw new HubOperatorError(options.bootstrapConsent
    ? 'No remote daemon socket is registered. Run the explicitly consented bootstrap on the host and register its socket path.'
    : 'No remote daemon socket is registered and remote-bootstrap consent is absent. Start Xerxes on the host and register its socket path, or grant explicit bootstrap consent.')
  if (!options.remoteSocketPath.startsWith('/') || /[:\0\r\n]/.test(options.remoteSocketPath)) throw new HubOperatorError('Remote daemon socket must be an absolute Unix path without colons or control characters.')
  options.signal?.throwIfAborted()
  const directory = await mkdtemp(join(tmpdir(), 'xh-ssh-'))
  const socketPath = join(directory, 'rpc.sock'), controlPath = join(directory, 'control.sock')
  const processes: SshProcess[] = []
  const readers = new Set<ReadableStreamDefaultReader<Uint8Array>>()
  let closed = false, startup = true, failureError: Error | undefined
  let closePromise: Promise<void> | undefined
  const close = (): Promise<void> => {
    if (closePromise) return closePromise
    closed = true
    options.signal?.removeEventListener('abort', abort)
    closePromise = (async () => {
      await Promise.all(processes.map(async child => {
        child.kill('SIGTERM')
        let timer: ReturnType<typeof setTimeout> | undefined
        await Promise.race([child.exited, new Promise<void>(resolve => { timer = setTimeout(() => { child.kill('SIGKILL'); resolve() }, 1000) })])
        clearTimeout(timer)
      }))
      await Promise.all([...readers].map(reader => reader.cancel()))
      await rm(directory, { recursive: true, force: true })
    })()
    return closePromise
  }
  const fail = (error: Error): void => {
    if (closed || failureError) return
    failureError = error
    void close()
    if (!startup) options.onFailure?.(error)
  }
  const abort = (): void => fail(new Error('SSH connection cancelled'))
  options.signal?.addEventListener('abort', abort, { once: true })
  const launch = options.port ?? bunSshPort
  const spawn = (args: readonly string[], forward: boolean): void => {
    const child = launch.spawn('ssh', args); processes.push(child)
    let stderr = ''
    const pump = (async () => {
      const reader = child.stderr.getReader(), decoder = new TextDecoder()
      readers.add(reader)
      try {
        for (;;) {
          const next = await reader.read(); if (next.done) break
          stderr = (stderr + decoder.decode(next.value, { stream: true })).slice(-8192)
          // A missing forwarded socket can leave ssh alive. It is an operator
          // condition, not evidence that a listening local socket is healthy.
          if (/host key|host identification|permission denied|authentication failed|no such file|connect failed: connection refused/i.test(stderr)) fail(failure(stderr, options.bootstrapConsent === true))
        }
      } finally { readers.delete(reader); reader.releaseLock() }
    })().catch(() => fail(new Error('Could not read SSH diagnostics. Inspect the local SSH client before reconnecting.')))
    void child.exited.then(async code => {
      await pump
      if (!closed && (!forward || code !== 0)) fail(failure(stderr, options.bootstrapConsent === true))
    }).catch(() => fail(new Error('Could not monitor the SSH process. Check that the local SSH client is available.')))
  }
  const waitFor = async (path: string, deadline: number): Promise<void> => {
    while (!existsSync(path)) {
      if (failureError) throw failureError
      if (closed || options.signal?.aborted) throw new Error('SSH connection cancelled')
      if (Date.now() >= deadline) throw new Error('SSH tunnel startup timed out. Check connectivity and the remote daemon socket.')
      await Bun.sleep(10)
    }
    if (failureError) throw failureError
  }
  try {
    const config = await writeSshConnectionConfig(directory)
    options.signal?.throwIfAborted()
    const hardened = ['-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'ForwardAgent=no', '-o', 'ForwardX11=no',
      '-o', 'ForwardX11Trusted=no', '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3', '-o', 'PermitLocalCommand=no']
    spawn(['-F', config, ...hardened, '-M', '-S', controlPath, '-o', 'ControlPersist=no', '-o', 'ForkAfterAuthentication=no',
      '-o', 'ClearAllForwardings=yes', '-N', '-T', '--', options.sshAlias], false)
    const deadline = Date.now() + (options.startupTimeoutMs ?? 15_000)
    await waitFor(controlPath, deadline)
    spawn(['-F', '/dev/null', ...hardened, '-S', controlPath, '-o', 'ControlMaster=no', '-O', 'forward',
      '-L', `${socketPath}:${options.remoteSocketPath}`, '--', options.sshAlias], true)
    await waitFor(socketPath, deadline)
    if (closed || options.signal?.aborted) throw failureError ?? new Error('SSH connection cancelled')
    startup = false
    return { socketPath, close }
  } catch (error) {
    await close()
    throw failureError ?? error
  }
}
