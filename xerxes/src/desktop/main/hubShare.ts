// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Shares an SSH workspace this app is connected to with the Hub, as that
 * machine's last-resort route ("via mac").
 *
 * The app already holds a local Unix socket forwarded to the remote daemon.
 * This pushes that socket onto the Hub host with `ssh -R`, then asks the Hub
 * (`hub machines attach`) to treat it as a route of the machine whose daemon
 * has the same runtime identity, or as a machine of its own. Finally the Hub's
 * key is pinned on the remote daemon (`hub trust`) so the Hub can sign commands
 * there. Nothing here carries a credential: SSH uses the user's own config and
 * keys, and only socket paths and labels cross to the Hub.
 *
 * Sharing is best effort. Every failure is logged and leaves the workspace
 * connection itself untouched.
 */

import { execFile, spawn } from 'node:child_process'
import { existsSync, readFileSync } from 'node:fs'
import { join } from 'node:path'

import type { CliRunner } from './hubCli.js'
import { parseCliJson } from './hubCli.js'
import { HUB_SETTINGS_FILE, parseStoredHub } from './hubStorage.js'

export interface ShareProcess {
  readonly exited: Promise<number | null>
  kill(): void
}

export interface HubShareDeps {
  /** The saved Hub SSH target (`user@host` or alias), or undefined when no Hub is set up. */
  readonly target: () => string | undefined
  /** Run `ssh <args>` to completion. */
  readonly ssh: (args: readonly string[], timeoutMs: number) => Promise<{ readonly code: number | null; readonly stdout: string; readonly stderr: string }>
  /** Start a long-running `ssh <args>`. */
  readonly spawnSsh: (args: readonly string[]) => ShareProcess
  readonly cli: CliRunner
  readonly log: (message: string) => void
  readonly sleep?: (ms: number) => Promise<void>
}

export interface HubShareHandle {
  /** Stop forwarding. The Hub keeps the route; it is skipped while its socket is gone. */
  close(): Promise<void>
  /** Settles when the first attempt finished (shared, or failed and logged). */
  readonly ready: Promise<HubShareOutcome>
}
export type HubShareOutcome =
  | { readonly state: 'shared'; readonly machineId: string; readonly attached: 'route' | 'machine'; readonly trusted: boolean }
  | { readonly state: 'skipped'; readonly reason: string }
  | { readonly state: 'failed'; readonly reason: string }

const BASE = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ControlMaster=no', '-o', 'ControlPath=none',
  '-o', 'ForwardAgent=no', '-o', 'ForwardX11=no']

export function shareSlug(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 40) || 'remote'
}
const quote = (value: string): string => `'${value.replaceAll("'", `'\\''`)}'`
const clean = (text: string): string => text.split('\n').map(line => line.trim()).find(Boolean)?.slice(0, 200) ?? 'no output'

function checkSocket(path: string): boolean {
  return path.startsWith('/') && !/[\s:\0]/.test(path)
}

export function shareTunnelArgs(target: string, remoteSocket: string, localSocket: string): string[] {
  return ['-N', ...BASE, '-o', 'ExitOnForwardFailure=yes', '-o', 'ServerAliveInterval=20', '-o', 'ServerAliveCountMax=3',
    '-R', `${remoteSocket}:${localSocket}`, '--', target]
}

/**
 * Real processes for the Electron main process (Node, not Bun). The Hub target is read from the
 * desktop's saved Hub address each time, so linking or forgetting a Hub takes effect on the next
 * connection. `XERXES_HUB_SHARE=off` turns sharing off.
 */
export function createNodeShareDeps(options: { readonly home: string; readonly cli: CliRunner; readonly env?: Readonly<Record<string, string | undefined>>; readonly log?: (message: string) => void }): HubShareDeps {
  const env = options.env ?? process.env
  const ssh = env.XERXES_SSH?.trim() || '/usr/bin/ssh'
  return {
    target: () => {
      if (env.XERXES_HUB_SHARE?.trim().toLowerCase() === 'off') return undefined
      const path = join(options.home, HUB_SETTINGS_FILE)
      if (!existsSync(path)) return undefined
      try { return parseStoredHub(readFileSync(path, 'utf8'))?.target } catch { return undefined }
    },
    ssh: (args, timeoutMs) => new Promise(resolve => {
      execFile(ssh, [...args], { timeout: timeoutMs, maxBuffer: 1024 * 1024, encoding: 'utf8', windowsHide: true }, (error, stdout, stderr) => {
        const code = error === null ? 0 : typeof (error as { code?: unknown }).code === 'number' ? (error as { code: number }).code : null
        resolve({ code, stdout, stderr })
      })
    }),
    spawnSsh: args => {
      const child = spawn(ssh, [...args], { stdio: ['ignore', 'ignore', 'ignore'], windowsHide: true })
      return {
        exited: new Promise(resolve => { child.once('exit', code => resolve(code)); child.once('error', () => resolve(null)) }),
        kill: () => { child.kill('SIGTERM') },
      }
    },
    cli: options.cli,
    log: options.log ?? (message => console.warn(message)),
  }
}

export function shareRemoteWorkspace(deps: HubShareDeps, input: { readonly label: string; readonly localSocket: string }): HubShareHandle {
  const sleep = deps.sleep ?? (ms => new Promise<void>(resolve => setTimeout(resolve, ms)))
  let closed = false
  let child: ShareProcess | undefined
  let remoteSocket: string | undefined
  const target = deps.target()?.trim()

  const run = async (): Promise<HubShareOutcome> => {
    if (!target) return { state: 'skipped', reason: 'no Hub is set up in this app' }
    if (!/^[A-Za-z0-9_.@][A-Za-z0-9_.@:[\]-]{0,252}$/.test(target)) return { state: 'skipped', reason: 'the saved Hub address is not a plain SSH target' }
    if (!checkSocket(input.localSocket)) return { state: 'failed', reason: 'the local forward socket path is not usable' }
    const prepared = await deps.ssh([...BASE, '--', target, 'mkdir -p "$HOME/.xerxes/tunnels" && chmod 700 "$HOME/.xerxes/tunnels" && printf "%s\\n" "$HOME"'], 30_000)
    if (prepared.code !== 0) return { state: 'failed', reason: `could not reach the Hub host: ${clean(prepared.stderr || prepared.stdout)}` }
    const home = prepared.stdout.trim().split('\n').at(-1)?.trim() ?? ''
    if (!home.startsWith('/')) return { state: 'failed', reason: 'the Hub host reported no home directory' }
    remoteSocket = `${home}/.xerxes/tunnels/mac-relay-${shareSlug(input.label)}.sock`
    if (!checkSocket(remoteSocket)) return { state: 'failed', reason: 'the Hub home directory makes an unusable socket path' }
    const socket = remoteSocket
    // Keep the forward up for as long as this connection lives; sshd refuses to bind over a stale file.
    void (async () => {
      let delay = 1000
      while (!closed) {
        await deps.ssh([...BASE, '--', target, `rm -f ${quote(socket)}`], 20_000).catch(() => undefined)
        if (closed) break
        const started = Date.now()
        child = deps.spawnSsh(shareTunnelArgs(target, socket, input.localSocket))
        await child.exited.catch(() => null)
        child = undefined
        if (closed) break
        delay = Date.now() - started > 60_000 ? 1000 : Math.min(delay * 2, 60_000)
        await sleep(delay)
      }
    })()
    let up = false
    for (let attempt = 0; attempt < 40 && !closed; attempt++) {
      const probe = await deps.ssh([...BASE, '--', target, `test -S ${quote(socket)}`], 15_000)
      if (probe.code === 0) { up = true; break }
      await sleep(500)
    }
    if (!up) return { state: 'failed', reason: 'the relayed socket never appeared on the Hub host' }
    const attached = await deps.cli.run(['remote', target, 'machines', 'attach', '--socket', socket, '--label', input.label, '--json'], { timeoutMs: 60_000 })
    const json = attached.exitCode === 0 ? parseCliJson(attached.stdout) : undefined
    const machineId = typeof json?.machine_id === 'string' ? json.machine_id : undefined
    if (machineId === undefined || (json?.attached !== 'route' && json?.attached !== 'machine')) {
      return { state: 'failed', reason: `the Hub did not attach it: ${clean(attached.stderr || attached.stdout || attached.spawnError || '')}` }
    }
    // The Hub signs its commands; the remote daemon must trust its key. Older Hubs lack this command.
    const trust = await deps.cli.run(['trust', target, '--socket', input.localSocket, '--json'], { timeoutMs: 60_000 })
    const trusted = trust.exitCode === 0 && parseCliJson(trust.stdout)?.enrolled === true
    if (!trusted) deps.log(`Hub share: ${input.label} is shared but its daemon does not trust the Hub key yet (${clean(trust.stderr || trust.stdout || trust.spawnError || '')}); the Hub can only read it.`)
    return { state: 'shared', machineId, attached: json.attached, trusted }
  }

  const ready = run().catch((error: unknown): HubShareOutcome => ({ state: 'failed', reason: error instanceof Error ? error.message : String(error) }))
    .then(outcome => {
      if (outcome.state === 'failed') deps.log(`Hub share for ${input.label} failed: ${outcome.reason}`)
      return outcome
    })
  return {
    ready,
    async close() {
      if (closed) return
      closed = true
      child?.kill()
      if (target && remoteSocket) await deps.ssh([...BASE, '--', target, `rm -f ${quote(remoteSocket)}`], 20_000).catch(() => undefined)
    },
  }
}
