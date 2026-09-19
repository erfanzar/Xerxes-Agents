// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { spawn, type ChildProcess } from 'node:child_process'
import { mkdtemp, rm, stat } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { setTimeout as delay } from 'node:timers/promises'
import { remoteBootstrapScript } from '../../ui/lib/remoteBootstrap.js'
import { remoteResumeProgram } from './remoteResume.js'

export interface RemoteTarget {
  alias: string
  target: string
  workspacePath: string
}
export function remoteTarget(value: unknown): RemoteTarget {
  if (!value || typeof value !== 'object') throw new Error('Invalid remote workspace')
  const row = value as Record<string, unknown>
  if (
    typeof row.alias !== 'string' ||
    !/^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$/.test(row.alias) ||
    typeof row.target !== 'string' ||
    row.target.length > 255 ||
    !/^(?:[a-zA-Z0-9_][a-zA-Z0-9_.-]*@)?[a-zA-Z0-9_][a-zA-Z0-9_.-]*$/.test(row.target) ||
    typeof row.workspacePath !== 'string' ||
    !row.workspacePath.startsWith('/') ||
    row.workspacePath.length > 4096 ||
    /[\x00-\x1f\x7f]/.test(row.workspacePath)
  )
    throw new Error('Invalid SSH alias, target, or absolute project folder')
  return { alias: row.alias, target: row.target, workspacePath: row.workspacePath }
}
export function remoteAddress(output: string): { socketPath: string; projectDir: string; expectedBuildId?: string; busy?: boolean } {
  const line = output.split('\n').find((value) => value.startsWith('XERXES_REMOTE_READY '))
  if (!line) throw new Error('Remote setup did not return a daemon address')
  const value: unknown = JSON.parse(line.slice('XERXES_REMOTE_READY '.length))
  if (!value || typeof value !== 'object') throw new Error('Invalid remote daemon address')
  const row = value as Record<string, unknown>
  if (
    typeof row.socketPath !== 'string' ||
    !row.socketPath.startsWith('/') ||
    /[:\x00-\x1f\x7f]/.test(row.socketPath) ||
    typeof row.projectDir !== 'string' ||
    !row.projectDir.startsWith('/') ||
    /[\x00-\x1f\x7f]/.test(row.projectDir)
  )
    throw new Error('Invalid remote daemon address')
  if (row.expectedBuildId !== undefined && (typeof row.expectedBuildId !== 'string' || !/^[a-f0-9]{16,64}$/.test(row.expectedBuildId))) throw new Error('Invalid remote build identity')
  return { socketPath: row.socketPath, projectDir: row.projectDir,
    ...(typeof row.expectedBuildId === 'string' ? { expectedBuildId: row.expectedBuildId } : {}),
    ...(row.busy === true ? { busy: true } : {}),
  }
}
export function runCaptured(
  binary: string,
  args: string[],
  signal: AbortSignal,
  timeout: number,
): Promise<string> {
  return new Promise((accept, reject) => {
    signal.throwIfAborted()
    const child = spawn(binary, args, { stdio: ['ignore', 'pipe', 'pipe'] })
    let output = '',
      errors = '',
      settled = false
    const finish = (error?: Error) => {
      if (settled) return
      settled = true
      clearTimeout(timer)
      signal.removeEventListener('abort', abort)
      if (error) {
        child.kill('SIGKILL')
        reject(error)
      } else accept(output)
    }
    const abort = () => finish(new Error('Remote connection cancelled'))
    const timer = setTimeout(
      () =>
        finish(
          new Error('Remote setup timed out. Check ~/.xerxes/remote-runtime/setup.log and retry.'),
        ),
      timeout,
    )
    signal.addEventListener('abort', abort, { once: true })
    child.stdout?.on('data', (chunk) => {
      output = (output + String(chunk)).slice(-1024 * 1024)
    })
    child.stderr?.on('data', (chunk) => {
      errors = (errors + String(chunk)).slice(-8192)
    })
    child.once('error', finish)
    child.once('close', (code) =>
      finish(
        code === 0
          ? undefined
          : new Error(`Remote operation failed (${code}): ${errors || output}`),
      ),
    )
    if (signal.aborted) abort()
  })
}
export interface RemoteConnection {
  socketPath: string
  projectDir: string
  expectedBuildId?: string | undefined
  update(): Promise<Record<string, unknown>>
  reconnect(signal: AbortSignal, onFailure: (error: Error) => void): Promise<RemoteConnection>
  close(): Promise<void>
}
/** Own only the forwarding process; the remote daemon and its durable sessions survive disconnect. */
export async function openRemote(
  value: unknown,
  signal: AbortSignal,
  onFailure: (error: Error) => void,
  previousAddress?: ReturnType<typeof remoteAddress>,
): Promise<RemoteConnection> {
  const machine = remoteTarget(value)
  const ssh = [
    '-o',
    'BatchMode=yes',
    '-o',
    'StrictHostKeyChecking=yes',
    '-o',
    'ConnectTimeout=15',
    '-o',
    'ServerAliveInterval=15',
    '-o',
    'ServerAliveCountMax=3',
  ]
  const quote = (value: string) => "'" + value.replaceAll("'", "'\\''") + "'"
  let address: ReturnType<typeof remoteAddress> | undefined
  if (previousAddress) {
    const probe = await runCaptured('ssh', [...ssh, '-T', '--', machine.target,
      'exec sh -c ' + quote('PATH="$HOME/.bun/bin:$HOME/.local/bin:$PATH"; export PATH; exec bun -e ' + quote(remoteResumeProgram(previousAddress.socketPath)))], signal, 20000)
    if (probe.split('\n').includes('XERXES_REMOTE_ALIVE')) address = previousAddress
    else if (!probe.split('\n').includes('XERXES_REMOTE_MISSING')) throw new Error('Remote runtime probe did not return a valid status.')
  }
  if (!address) {
    const output = await runCaptured(
      'ssh',
      [
        ...ssh,
        '-T',
        '--',
        machine.target,
        'exec sh -c ' + quote(remoteBootstrapScript(machine.workspacePath, 'daemon')),
      ],
      signal,
      300000,
    )
    address = remoteAddress(output)
  }
  signal.throwIfAborted()
  const directory = await mkdtemp(join(tmpdir(), 'xd-')),
    socketPath = join(directory, 'rpc.sock')
  let tunnel: ChildProcess | undefined,
    closing = false,
    failure: Error | undefined,
    errors = ''
  const updateController = new AbortController()
  const close = async () => {
    if (closing) return
    closing = true
    updateController.abort()
    signal.removeEventListener('abort', abort)
    tunnel?.kill('SIGKILL')
    await rm(directory, { recursive: true, force: true })
  }
  const abort = () => {
    failure = new Error('Remote connection cancelled')
    void close()
  }
  try {
    signal.addEventListener('abort', abort, { once: true })
    signal.throwIfAborted()
    tunnel = spawn(
      'ssh',
      [
        ...ssh,
        '-S',
        'none',
        '-o',
        'ControlMaster=no',
        '-o',
        'ForkAfterAuthentication=no',
        '-N',
        '-T',
        '-o',
        'ExitOnForwardFailure=yes',
        '-L',
        `${socketPath}:${address.socketPath}`,
        '--',
        machine.target,
      ],
      { stdio: ['ignore', 'ignore', 'pipe'] },
    )
    const failed = (error: Error) => {
      if (!closing) {
        failure = error
        onFailure(error)
      }
    }
    tunnel.stderr?.on('data', (chunk) => {
      errors = (errors + String(chunk)).slice(-8192)
    })
    tunnel.once('error', failed)
    tunnel.once('close', (code) => failed(new Error(`SSH tunnel closed (${code}). ${errors}`)))
    const deadline = Date.now() + 15000
    while (true) {
      signal.throwIfAborted()
      if (failure) throw failure
      const socket = await stat(socketPath).catch((error) => {
        if (error.code === 'ENOENT') return null
        throw error
      })
      if (socket?.isSocket()) break
      if (Date.now() > deadline) throw new Error('SSH tunnel startup timed out')
      await delay(25, undefined, { signal })
    }
    let reconnectAddress = address
    const connection: RemoteConnection = { socketPath, projectDir: address.projectDir, expectedBuildId: address.expectedBuildId, close,
      reconnect: (retrySignal, retryFailure) => openRemote(machine, retrySignal, retryFailure, reconnectAddress),
      async update() {
        const updated = remoteAddress(await runCaptured('ssh', [...ssh, '-T', '--', machine.target,
          'exec sh -c ' + quote(remoteBootstrapScript(machine.workspacePath, 'daemon'))], updateController.signal, 300000))
        if (updated.socketPath !== address.socketPath || updated.projectDir !== address.projectDir) throw new Error('Remote daemon address changed; reconnect this workspace.')
        reconnectAddress = updated
        connection.expectedBuildId = updated.expectedBuildId
        return updated.busy ? { ok: false, busy: true } : { ok: true }
      },
    }
    return connection
  } catch (error) {
    await close()
    throw error
  }
}
