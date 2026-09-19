// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { spawn } from 'node:child_process'
import { existsSync, rmSync, writeFileSync } from 'node:fs'
import { mkdtemp, readFile, rm } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { setTimeout as delay } from 'node:timers/promises'

import { withTerminalSuspended } from './terminalRuntime.opentui.js'
import { remoteBootstrapScript } from './remoteBootstrap.js'
import { sshFailure } from '../../security/sshDiagnostics.js'
import { writeSshConnectionConfig } from '../../security/sshConnectionConfig.js'
import { startSshSocketTunnel } from './sshSocketTunnel.js'

/** Runs while the parent renderer is still available for review/consent. */
export interface RemoteWorkspacePreparation {
  readonly machine: RemoteMachine
  readonly socketPath: string
  readonly projectDir: string
  readonly resumeSessionId?: string
  readonly signal: AbortSignal
}
export interface PreparedRemoteWorkspace {
  readonly sessionId: string
  readonly sessionKey?: string
  close(): Promise<void>
}

export interface RemoteMachine {
  alias: string
  target: string
  workspacePath: string
}

export function parseRemoteMachine(value: unknown): RemoteMachine {
  if (!value || typeof value !== 'object') throw new Error('Invalid remote machine response')
  const row = value as Record<string, unknown>
  if (typeof row.alias !== 'string' || !/^[a-zA-Z0-9][a-zA-Z0-9_.-]*$/.test(row.alias) ||
      typeof row.target !== 'string' || !/^[a-zA-Z0-9][a-zA-Z0-9_.@:-]*$/.test(row.target) ||
      typeof row.workspacePath !== 'string' || !row.workspacePath.startsWith('/') || /[\0\r\n]/.test(row.workspacePath)) {
    throw new Error('Invalid remote machine response')
  }
  return { alias: row.alias, target: row.target, workspacePath: row.workspacePath }
}

const quoteShell = (value: string): string => `'${value.replaceAll("'", "'\\''")}'`

/** A renderer already owned this workspace. Never replace it in an outer retry. */
class RemoteSessionEndedError extends Error {}

function remoteFailureHint(error: Error): string {
  if (/host key|host identification/i.test(error.message)) return 'SSH host verification failed. Verify the destination in a terminal before reconnecting.'
  if (/permission denied|authentication/i.test(error.message)) return 'SSH authentication failed. Restore access to this destination before reconnecting.'
  return 'SSH reconnection failed. Your draft is still here; copy it before exiting and reconnecting with /machine.'
}

export function remoteMachineCommand(machine: RemoteMachine): string {
  const validated = parseRemoteMachine(machine)
  return `exec sh -c ${quoteShell(remoteBootstrapScript(validated.workspacePath, 'daemon'))}`
}

/** Start the remote daemon, forward its private socket, and run a LOCAL TUI. */
export async function connectRemoteMachine(
  machine: RemoteMachine,
  options: {
    signal?: AbortSignal
    spawnProcess?: typeof spawn
    suspend?: (run: () => Promise<void>) => Promise<void>
    tuiEntry?: string
    resumeSessionId?: string
    prepare?: (remote: RemoteWorkspacePreparation) => Promise<PreparedRemoteWorkspace>
    onSessionId?: (id: string) => void
    onProgress?: (message: string) => void
  } = {}
): Promise<void> {
  const command = remoteMachineCommand(machine)
  if (options.signal?.aborted) throw new Error('Remote connection cancelled')
  const launch = options.spawnProcess ?? spawn
  // Authentication stays local. A saved SSH alias may enable agent/X11
  // forwarding or relax host-key checks; opening a workspace grants none of
  // those capabilities. Apply these to setup as well as the RPC tunnel.
  const ssh = ['-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
    '-o', 'ForwardAgent=no', '-o', 'ForwardX11=no',
    '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3']
  const directory = await mkdtemp(join(tmpdir(), 'xr-'))
  const socket = join(directory, 'rpc.sock')
  const sessionFile = join(directory, "active-session")
  const statusFile = join(directory, 'transport-status')
  try {
    ssh.unshift('-F', await writeSshConnectionConfig(directory))
    options.onProgress?.("Checking remote runtime…")
    const output = await new Promise<string>((resolve, reject) => {
      const child = launch('ssh', [...ssh, '-T', '--', machine.target, command], { stdio: ['ignore', 'pipe', 'pipe'] })
      let stdout = '', stderr = ''
      let killTimer: ReturnType<typeof setTimeout> | undefined
      const stop = (error: Error) => {
        cleanup()
        child.kill('SIGTERM')
        killTimer = setTimeout(() => child.kill('SIGKILL'), 1000)
        killTimer.unref?.()
        reject(error)
      }
      const abort = () => stop(new Error('Remote connection cancelled'))
      const timer = setTimeout(() => stop(new Error('Remote setup timed out; inspect ~/.xerxes/remote-runtime/setup.log.')), 300_000)
      const cleanup = () => { clearTimeout(timer); options.signal?.removeEventListener('abort', abort) }
      options.signal?.addEventListener('abort', abort, { once: true })
      if (options.signal?.aborted) abort()
      child.stdout?.on('data', chunk => { stdout = (stdout + String(chunk)).slice(-65536) })
      child.stderr?.on('data', chunk => { stderr = (stderr + String(chunk)).slice(-8192) })
      child.once('error', () => { cleanup(); clearTimeout(killTimer); reject(new Error('Could not start SSH. Check that the SSH client is installed and executable.')) })
      child.once('close', code => {
        cleanup()
        clearTimeout(killTimer)
        if (options.signal?.aborted) reject(new Error('Remote connection cancelled'))
        else if (code !== 0) reject(sshFailure('setup', stderr || stdout))
        else resolve(stdout)
      })
    })
    const ready = output.split('\n').find(line => line.startsWith('XERXES_REMOTE_READY '))
    if (!ready) throw new Error('Remote setup did not return a daemon address.')
    let remote: unknown
    try { remote = JSON.parse(ready.slice('XERXES_REMOTE_READY '.length)) }
    catch { throw new Error('Remote setup returned an invalid daemon address. Check the installed remote runtime.') }
    if (!remote || typeof remote !== 'object' || !('socketPath' in remote) || !('projectDir' in remote) ||
      typeof remote.socketPath !== 'string' || !remote.socketPath.startsWith('/') || /[:\r\n\0]/u.test(remote.socketPath) ||
      typeof remote.projectDir !== 'string' || !remote.projectDir.startsWith('/') || /[\r\n\0]/u.test(remote.projectDir)) throw new Error('Invalid remote daemon address.')
    // Own this connection: multiplexed -N can exit successfully after handing
    // the forwarding to an unrelated persistent master, defeating cleanup.
    options.onProgress?.("Opening SSH tunnel…")
    let tunnel: ReturnType<typeof startSshSocketTunnel> | undefined
    let failure: Error | undefined
    let local: ReturnType<typeof spawn> | undefined
    let closing = false
    let prepared: PreparedRemoteWorkspace | undefined
    const preparationAbort = new AbortController()
    let retryTimer: ReturnType<typeof setTimeout> | undefined
    let retries = 0
    const status = (state: 'reconnecting' | 'failed', message: string) => {
      // Only fixed user-facing diagnostics, never raw SSH/remote output.
      writeFileSync(statusFile, JSON.stringify({ state, message }), { mode: 0o600 })
    }
    const startTunnel = () => {
      if (closing || options.signal?.aborted) return
      failure = undefined
      let settled = false
      const failed = (error: Error) => {
        if (settled || closing) return
        settled = true
        failure = error
        if (!local) preparationAbort.abort()
        // The pathname belongs only to this connection's private directory.
        // A stale socket must not be mistaken for a ready replacement.
        rmSync(socket, { force: true })
        if (!local || options.signal?.aborted) return
        if (retries < 3 && retryableRemoteFailure(error)) {
          const wait = [250, 1000, 2000][retries++]!
          status('reconnecting', 'SSH disconnected; reconnecting to the same workspace…')
          retryTimer = setTimeout(startTunnel, wait)
          retryTimer.unref?.()
        } else {
          // Keep the renderer alive: the user can still inspect/copy the draft
          // and transcript. Gateway recovery reads this nonsecret local status.
          status('failed', remoteFailureHint(error))
        }
      }
      tunnel = startSshSocketTunnel({ ssh, target: machine.target, controlPath: join(directory, 'control.sock'),
        localSocket: socket, remoteSocket: remote.socketPath as string, onFailure: failed, spawnProcess: launch })
    }
    const abort = () => { failure = new Error('Remote connection cancelled'); clearTimeout(retryTimer); tunnel?.close(); local?.kill('SIGTERM') }
    startTunnel()
    options.signal?.addEventListener('abort', abort, { once: true })
    try {
      const deadline = Date.now() + 15000
      while (!existsSync(socket)) {
        if (options.signal?.aborted) abort()
        if (failure) throw failure
        if (Date.now() >= deadline) throw new Error('SSH tunnel startup timed out.')
        await delay(25)
      }
      if (failure) throw failure
      if (options.prepare) {
        options.onProgress?.('Review remote task setup…')
        prepared = await options.prepare({ machine, socketPath: socket, projectDir: remote.projectDir as string,
          ...(options.resumeSessionId ? { resumeSessionId: options.resumeSessionId } : {}),
          signal: options.signal ? AbortSignal.any([options.signal, preparationAbort.signal]) : preparationAbort.signal })
        if (failure) throw failure
        if (options.signal?.aborted || preparationAbort.signal.aborted) throw new Error('Remote connection cancelled')
        options.onSessionId?.(prepared.sessionId)
      }
      options.onProgress?.("Opening remote workspace…")
      await (options.suspend ?? withTerminalSuspended)(() => new Promise<void>((resolve, reject) => {
        // Suspending the parent can yield; the tunnel may die in that gap.
        if (failure) { reject(failure); return }
        if (options.signal?.aborted) { reject(new Error('Remote connection cancelled')); return }
        local = launch(process.execPath, [options.tuiEntry ?? process.argv[1]!], {
          stdio: 'inherit',
          env: { ...process.env, XERXES_REMOTE_SOCKET: socket, XERXES_PROJECT_DIR: remote.projectDir as string,
            XERXES_REMOTE_LABEL: machine.alias,
            XERXES_REMOTE_STATUS_FILE: statusFile,
            XERXES_CWD: remote.projectDir as string, XERXES_TUI_RESUME: prepared?.sessionId ?? options.resumeSessionId ?? '',
            XERXES_TUI_PREPARED_SESSION_KEY: prepared?.sessionKey ?? '', XERXES_TUI_QUERY: '', XERXES_TUI_ACTIVE_SESSION_FILE: sessionFile }
        })
        local.once('error', reject)
        local.once('close', code => failure ? reject(new RemoteSessionEndedError(remoteFailureHint(failure))) : code === 0 ? resolve() : reject(new RemoteSessionEndedError(`Local remote-workspace TUI exited (${code}).`)))
        if (options.signal?.aborted) abort()
      }))
    } finally {
      closing = true
      clearTimeout(retryTimer)
      options.signal?.removeEventListener('abort', abort)
      preparationAbort.abort()
      try { await prepared?.close() } finally { tunnel?.close() }
    }
  } finally {
    try {
      const saved: unknown = JSON.parse(await readFile(sessionFile, 'utf8'))
      if (saved && typeof saved === 'object' && 'session_id' in saved && typeof saved.session_id === 'string' && /^[a-zA-Z0-9_-]{1,128}$/.test(saved.session_id)) options.onSessionId?.(saved.session_id)
    } catch (error) {
      if (!(error && typeof error === 'object' && 'code' in error && error.code === 'ENOENT')) options.onProgress?.('Could not recover the session ID; use the remote session picker.')
    }
    await rm(directory, { recursive: true, force: true })
  }
}

/** Retry transport interruptions, never authentication or host-key failures. */
export function retryableRemoteFailure(error: unknown): boolean {
  if (error instanceof RemoteSessionEndedError) return false
  const message = error instanceof Error ? error.message : String(error)
  if (/permission denied|authentication|host key|host identification|cancelled|canceled/i.test(message)) return false
  return /SSH tunnel closed|connection (?:reset|refused|closed)|timed out|network is unreachable|no route to host|broken pipe/i.test(message)
}

export async function reconnectRemoteMachine(
  machine: RemoteMachine,
  options: NonNullable<Parameters<typeof connectRemoteMachine>[1]> = {},
  connect = connectRemoteMachine,
  wait: (ms: number, signal?: AbortSignal) => Promise<void> = async (ms, signal) => { await delay(ms, undefined, { signal }) },
): Promise<void> {
  let resumeSessionId = options.resumeSessionId
  for (let attempt = 0; ; attempt++) {
    options.signal?.throwIfAborted()
    try {
      await connect(machine, { ...options, resumeSessionId, onSessionId: id => { resumeSessionId = id; options.onSessionId?.(id) } })
      return
    } catch (error) {
      if (options.signal?.aborted || attempt >= 3 || !retryableRemoteFailure(error)) throw error
      const seconds = [2, 5, 10][attempt]!
      options.onProgress?.(`Connection interrupted. Retry ${attempt + 1}/3 in ${seconds}s · Esc cancel`)
      await wait(seconds * 1000, options.signal)
    }
  }
}
