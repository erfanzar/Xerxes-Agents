// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * The network of one SSH workspace, reached from this Mac:
 * - a SOCKS listener (`ssh -D`) used only by the in-app browser tab's own
 *   session, so localhost, 10.x hosts and internal names resolve as they do
 *   on the remote machine;
 * - per-port forwards (`ssh -L`) for the Mac's own browser.
 * Every listener binds 127.0.0.1 on this Mac. Nothing listens on the remote,
 * nothing is installed there, and no system proxy is changed. All of it
 * closes with the workspace.
 */

import { spawn, type ChildProcess } from 'node:child_process'
import { createConnection, createServer } from 'node:net'
import { setTimeout as delay } from 'node:timers/promises'

import { sshDestination } from '../../daemon/sshTarget.js'
import { sshOptions, type RemoteTarget } from './remote.js'

export const MAX_PORT_FORWARDS = 16
export interface PortForward { readonly remote: number; readonly local: number; readonly state: 'starting' | 'ready' | 'failed'; readonly error?: string }
export interface RemoteNetworkPorts {
  readonly spawn?: (args: readonly string[]) => ChildProcess
  readonly freePort?: (preferred?: number) => Promise<number>
  readonly waitReady?: (port: number, signal: AbortSignal) => Promise<void>
}

export function validRemotePort(value: unknown): number {
  const port = Number(value)
  if (!Number.isSafeInteger(port) || port < 1 || port > 65535) throw new Error('Port must be an integer from 1 to 65535')
  return port
}

/** A free loopback port; the preferred number first when it is free. */
export async function freeLoopbackPort(preferred?: number): Promise<number> {
  const tryPort = (port: number) => new Promise<number | undefined>(resolve => {
    const server = createServer()
    server.once('error', () => resolve(undefined))
    server.listen(port, '127.0.0.1', () => {
      const address = server.address()
      const chosen = address && typeof address === 'object' ? address.port : undefined
      server.close(() => resolve(chosen))
    })
  })
  if (preferred && preferred >= 1024) { const port = await tryPort(preferred); if (port) return port }
  const port = await tryPort(0)
  if (!port) throw new Error('No free local port is available')
  return port
}

/** Resolves once 127.0.0.1:port accepts a TCP connection. */
export async function waitForLoopback(port: number, signal: AbortSignal, timeoutMs = 15_000): Promise<void> {
  const deadline = Date.now() + timeoutMs
  for (;;) {
    signal.throwIfAborted()
    const open = await new Promise<boolean>(resolve => {
      const socket = createConnection({ host: '127.0.0.1', port })
      socket.once('connect', () => { socket.destroy(); resolve(true) })
      socket.once('error', () => { socket.destroy(); resolve(false) })
    })
    if (open) return
    if (Date.now() > deadline) throw new Error('SSH forward did not start in time')
    await delay(50, undefined, { signal })
  }
}

/** argv for one forwarding ssh process: loopback-only listener, no remote command, fail fast. */
export function forwardArgs(machine: RemoteTarget, forward: { kind: 'socks'; local: number } | { kind: 'port'; local: number; remote: number }): string[] {
  const bind = forward.kind === 'socks'
    ? ['-D', `127.0.0.1:${forward.local}`]
    : ['-L', `127.0.0.1:${forward.local}:127.0.0.1:${forward.remote}`]
  return [...sshOptions, '-N', '-T', '-o', 'ExitOnForwardFailure=yes', '-o', 'GatewayPorts=no', ...bind, '--', sshDestination(machine.target)]
}

interface Running { child?: ChildProcess; stopped?: boolean; local: number; state: 'starting' | 'ready' | 'failed'; error?: string; ready: Promise<number> }

export class RemoteNetwork {
  private socks: Running | undefined
  private readonly forwards = new Map<number, Running>()
  private readonly controller = new AbortController()
  private closed = false
  constructor(readonly machine: RemoteTarget, private readonly ports: RemoteNetworkPorts = {}) {}

  get label(): string { return this.machine.alias }

  /** The SOCKS port for the browser tab's session; started on first use. */
  socksPort(): Promise<number> {
    if (this.closed) return Promise.reject(new Error('The SSH workspace is closed'))
    if (this.socks && this.socks.state !== 'failed') return this.socks.ready
    this.socks = this.start(local => ({ kind: 'socks', local }))
    return this.socks.ready
  }

  /** Forward a remote loopback port to a local one (same number when free). */
  forward(remotePort: unknown): Promise<number> {
    if (this.closed) return Promise.reject(new Error('The SSH workspace is closed'))
    const remote = validRemotePort(remotePort)
    const existing = this.forwards.get(remote)
    if (existing && existing.state !== 'failed') return existing.ready
    if (!existing && this.forwards.size >= MAX_PORT_FORWARDS) return Promise.reject(new Error(`At most ${MAX_PORT_FORWARDS} ports can be forwarded per workspace`))
    const running = this.start(local => ({ kind: 'port', local, remote }), remote)
    this.forwards.set(remote, running)
    return running.ready
  }

  list(): readonly PortForward[] {
    return [...this.forwards.entries()].map(([remote, row]) => ({ remote, local: row.local, state: row.state, ...(row.error ? { error: row.error } : {}) }))
  }

  stop(remotePort: unknown): void {
    const remote = validRemotePort(remotePort)
    const row = this.forwards.get(remote)
    if (!row) return
    this.forwards.delete(remote)
    row.stopped = true
    row.child?.kill('SIGTERM')
  }

  close(): void {
    if (this.closed) return
    this.closed = true
    this.controller.abort()
    this.socks?.child?.kill('SIGTERM')
    for (const row of this.forwards.values()) row.child?.kill('SIGTERM')
    this.forwards.clear()
    this.socks = undefined
  }

  private start(spec: (local: number) => Parameters<typeof forwardArgs>[1], preferred?: number): Running {
    const row = { local: 0, state: 'starting' } as Running
    row.ready = (async () => {
      const local = await (this.ports.freePort ?? freeLoopbackPort)(preferred)
      if (this.closed) throw new Error('The SSH workspace is closed')
      if (row.stopped) throw new Error('The SSH forward was stopped')
      row.local = local
      const child = (this.ports.spawn ?? (args => spawn('ssh', [...args], { stdio: ['ignore', 'ignore', 'pipe'] })))(forwardArgs(this.machine, spec(local)))
      row.child = child
      let errors = ''
      child.stderr?.on('data', chunk => { errors = (errors + String(chunk)).slice(-2048) })
      const exited = new Promise<never>((_, reject) => {
        const fail = (message: string) => {
          row.state = 'failed'
          row.error = message
          reject(new Error(message))
        }
        child.once('error', () => fail('Could not start ssh'))
        child.once('exit', code => fail(`SSH forward closed (${code ?? 'signal'})${errors.trim() ? ': ' + errors.trim().split('\n').at(-1) : ''}`))
      })
      exited.catch(() => {})
      if (this.closed) { child.kill('SIGTERM'); throw new Error('The SSH workspace is closed') }
      await Promise.race([(this.ports.waitReady ?? waitForLoopback)(local, this.controller.signal), exited])
      row.state = 'ready'
      return local
    })()
    row.ready.catch(error => { row.state = 'failed'; row.error ??= error instanceof Error ? error.message : String(error); row.child?.kill('SIGTERM') })
    return row
  }
}

/**
 * What the tab loads for what the person typed. A bare `localhost:8080` or
 * `8080` becomes http; anything without a scheme gets http unless it looks
 * like a public name. Only http(s) is ever loaded.
 */
/**
 * What the Browser tab says when a page fails to load. Over an SSH workspace
 * Chromium reports every refused or unreachable destination as the same
 * opaque ERR_SOCKS_CONNECTION_FAILED (-120), which read as "the tunnel is
 * broken" when the tunnel was fine and nothing listened at the address.
 * The raw code stays at the end for diagnosis.
 */
export function browserLoadError(code: number, description: string, url: string, via: string | null): string {
  let host = ''
  try { host = new URL(url).host } catch { /* keep the generic wording */ }
  const target = host || 'this address'
  const raw = `${description} (${code})`
  // The local SOCKS listener (or its SSH process) itself did not answer.
  if (via && code === -130) return `The connection to ${via} is down. Reload to reconnect. ${raw}`
  // ssh -D refuses the CONNECT when the destination refuses, times out or does not resolve on the remote.
  if (via && [-120, -121, -102, -104, -105, -109, -118].includes(code)) {
    return `${via} could not reach ${target}. Nothing accepted the connection there: the service may be stopped, on another port, or listening only on localhost. ${raw}`
  }
  if ([-102, -105, -109, -118].includes(code)) return `Could not reach ${target}. ${raw}`
  return raw
}

export function browserUrl(input: string): string {
  const text = input.trim()
  if (!text) throw new Error('Type an address')
  if (/^\d{1,5}$/.test(text)) return `http://localhost:${validRemotePort(text)}/`
  const withScheme = /^[a-z][a-z0-9+.-]*:\/\//i.test(text) ? text : `http://${text}`
  const url = new URL(withScheme)
  if (url.protocol !== 'http:' && url.protocol !== 'https:') throw new Error('Only http and https addresses open here')
  return url.href
}
