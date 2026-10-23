// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { readFile } from 'node:fs/promises'
import { isAbsolute } from 'node:path'

import { firstLine, shellQuote, type CommandResult, type CommandRunner } from './commandRunner.js'
import { isRecord } from './protocol.js'
import { installUserService, userServicePath, type UserServiceContext, type UserServiceSpec } from './serviceUnits.js'

/**
 * `xerxes hub link <ssh-target>`, run on a workspace machine (for example a Mac
 * behind a home router). The machine dials OUT to the Hub over SSH and forwards
 * its daemon socket there, so the Hub never needs to reach in. A per-user
 * background service keeps the forward alive and reconnects.
 *
 * The Hub gets ONE machine entry for that socket (its projects are discovered by
 * asking the daemon), and the Hub's signing key is pinned on this machine's own
 * daemon for every project, so nothing has to be listed or enrolled by hand.
 */

export class HubLinkError extends Error {
  constructor(message: string) { super(message); this.name = 'HubLinkError' }
}

export interface HubLinkDeps extends UserServiceContext {
  readonly ssh: string
  readonly hostname: string
  readonly socketExists: (path: string) => boolean
  /** Progress lines (stderr in `--json` mode). */
  readonly out: (line: string) => void
  /** Pin the Hub key on THIS machine's daemon for every project (plain local connection, `hub.enroll {all_projects:true}`). */
  readonly enrollLocal: (input: { readonly socketPath: string; readonly hubId: string; readonly publicKeyPem: string }) => Promise<void>
}

export interface HubLinkOptions {
  readonly target: string
  /** This machine's daemon socket. */
  readonly localSocket: string
  /** Shell text that runs Xerxes on the Hub host; used verbatim. Default: `$HOME/.xerxes/bin/xerxes` if executable, else `xerxes`. */
  readonly remoteXerxes?: string
  readonly machine?: string
  readonly service?: boolean
  /** Absolute Bun executable and CLI entry for the background service. */
  readonly executable: string
  readonly scriptPath: string
  readonly environment: Record<string, string | undefined>
  readonly logPath: string
  readonly socketWaitMs?: number
}

export interface HubLinkResult {
  readonly machineId: string
  readonly remoteSocket: string
  readonly servicePath?: string
  readonly enrolled: true
}

/** Non-interactive, and never through a shared master: a multiplexed `-R` would exit at once and loop the service. */
const SSH_BASE = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ControlMaster=no', '-o', 'ControlPath=none']

export function validateSshTarget(target: string): string {
  const trimmed = target.trim()
  // No leading dash (option injection), no spaces or shell syntax.
  if (!/^[A-Za-z0-9_.@][A-Za-z0-9_.@:[\]-]{0,252}$/.test(trimmed)) throw new HubLinkError(`Invalid SSH target ${JSON.stringify(target)}. Use user@host or a Host alias from ~/.ssh/config.`)
  return trimmed
}

export function slug(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 40) || 'machine'
}

export function linkServiceSpec(options: Pick<HubLinkOptions, 'environment' | 'executable' | 'scriptPath' | 'logPath' | 'localSocket'>, remoteSocket: string, target: string): UserServiceSpec {
  const environment: Record<string, string> = {}
  const path = options.environment.PATH?.trim()
  if (path) environment.PATH = path
  return {
    name: `xerxes-hub-link-${slug(target)}`.slice(0, 64).replace(/-+$/, ''),
    label: `dev.xerxes.hub-link.${slug(target)}`,
    description: `Xerxes Hub link to ${target}`,
    argv: [options.executable, options.scriptPath, 'hub', 'link-tunnel', target, '--remote-socket', remoteSocket, '--local-socket', options.localSocket],
    environment,
    logPath: options.logPath,
  }
}

/**
 * The long-running forward: this machine's daemon socket appears on the Hub host.
 * The tunnel must be its own connection so its exit means "down".
 */
export function tunnelArgv(ssh: string, target: string, remoteSocket: string, localSocket: string): string[] {
  return [ssh, '-N', ...SSH_BASE,
    '-o', 'ExitOnForwardFailure=yes', '-o', 'ServerAliveInterval=20', '-o', 'ServerAliveCountMax=3',
    '-R', `${remoteSocket}:${localSocket}`, '--', target]
}

/** The `--remote-socket` an existing link service for this target forwards to, when it is one of ours. */
async function previousRemoteSocket(spec: Pick<UserServiceSpec, 'name' | 'label'>, context: UserServiceContext, remoteHome: string): Promise<string | undefined> {
  const path = userServicePath(spec, context.platform, context.home)
  if (path === undefined) return undefined
  let text: string
  try { text = await readFile(path, 'utf8') } catch { return undefined }
  const match = /--remote-socket(?:<\/string>\s*<string>|"\s+")([^<"\s]+)/.exec(text)
  const socket = match?.[1]
  const tunnels = `${remoteHome}/.xerxes/tunnels/`
  if (socket === undefined || !socket.startsWith(tunnels) || !socket.endsWith('-daemon.sock') || /[\s:\0]/.test(socket)) return undefined
  if (socket.slice(tunnels.length).includes('/')) return undefined
  return socket
}

function checkSocketPath(path: string, what: string): void {
  if (!isAbsolute(path) || /[\s:\0]/.test(path)) throw new HubLinkError(`${what} must be an absolute path without spaces or colons: ${path}`)
}

/**
 * Shell text that runs `xerxes <args>` on the Hub host. Without an override it is
 * `$HOME/.xerxes/bin/xerxes` (written by `hub setup`) when executable, else `xerxes` on PATH.
 */
export function remoteXerxesCommand(args: readonly string[], override?: string): string {
  const quoted = args.map(shellQuote).join(' ')
  const explicit = override?.trim()
  if (explicit) return `${explicit} ${quoted}`
  return `if [ -x "$HOME/.xerxes/bin/xerxes" ]; then exec "$HOME/.xerxes/bin/xerxes" ${quoted}; else exec xerxes ${quoted}; fi`
}

async function remote(ssh: string, run: CommandRunner, target: string, command: string, timeoutMs = 30_000): Promise<CommandResult> {
  return run([ssh, ...SSH_BASE, '--', target, command], { timeoutMs })
}

/** One JSON object from a command's stdout: the whole text, else the last line that parses (shell banners are tolerated). */
export function parseJsonObject(stdout: string): Record<string, unknown> | undefined {
  const candidates = [stdout.trim(), ...stdout.split('\n').map(line => line.trim()).filter(line => line.startsWith('{')).reverse()]
  for (const text of candidates) {
    if (!text.startsWith('{')) continue
    try {
      const value: unknown = JSON.parse(text)
      if (isRecord(value)) return value
    } catch { /* try the next candidate */ }
  }
  return undefined
}

export interface HubRemoteInput {
  readonly ssh: string
  readonly target: string
  /** Arguments of `xerxes hub ...` on the Hub host (without the leading `hub`). */
  readonly args: readonly string[]
  readonly remoteXerxes?: string
  readonly timeoutMs?: number
}

/** `hub remote <target> <hub-args...>`: run `xerxes hub <args>` on the Hub host. The caller relays the result. */
export async function runHubRemote(input: HubRemoteInput, run: CommandRunner): Promise<CommandResult> {
  const target = validateSshTarget(input.target)
  if (input.args.length === 0) throw new HubLinkError('usage: xerxes hub remote <user@hub-host> <hub-args...>')
  return remote(input.ssh, run, target, remoteXerxesCommand(['hub', ...input.args], input.remoteXerxes), input.timeoutMs ?? 120_000)
}

async function remoteJson(deps: HubLinkDeps, target: string, args: readonly string[], remoteXerxes: string | undefined, what: string): Promise<Record<string, unknown>> {
  const result = await remote(deps.ssh, deps.run, target, remoteXerxesCommand(['hub', ...args], remoteXerxes))
  if (result.code === 127) throw new HubLinkError(`Xerxes is not installed on ${target}. Run \`xerxes hub setup\` there (it writes ~/.xerxes/bin/xerxes), or pass --remote-xerxes with the command that runs it, e.g. --remote-xerxes '~/.bun/bin/bun ~/xerxes/dist/cli.js'.`)
  if (result.code !== 0 || result.timedOut) throw new HubLinkError(`${what} on ${target} failed: ${firstLine(result.stderr || result.stdout)}`)
  const parsed = parseJsonObject(result.stdout)
  if (parsed === undefined) throw new HubLinkError(`${what} on ${target} did not return JSON. Update Xerxes on the Hub host.`)
  return parsed
}

export async function runHubLink(options: HubLinkOptions, deps: HubLinkDeps): Promise<HubLinkResult> {
  const out = deps.out
  const target = validateSshTarget(options.target)
  checkSocketPath(options.localSocket, 'The local daemon socket')
  if (!deps.socketExists(options.localSocket)) {
    throw new HubLinkError(`No Xerxes daemon is listening at ${options.localSocket}. Open the Xerxes app (or run \`xerxes daemon\`) on this machine first.`)
  }
  const machine = options.machine?.trim() || deps.hostname.split('.')[0] || 'machine'

  out(`1/4 Reaching ${target}`)
  const prepared = await remote(deps.ssh, deps.run, target, 'mkdir -p "$HOME/.xerxes/tunnels" && chmod 700 "$HOME/.xerxes/tunnels" && printf "%s\\n" "$HOME"')
  if (prepared.code !== 0 || prepared.timedOut) {
    throw new HubLinkError(`Could not log in to ${target} over SSH without a password: ${firstLine(prepared.stderr || prepared.stdout)}. Check \`ssh ${target}\` works with your key first.`)
  }
  const remoteHome = prepared.stdout.trim().split('\n').at(-1)?.trim() ?? ''
  if (!isAbsolute(remoteHome)) throw new HubLinkError(`${target} did not report a home directory.`)
  // One link service per Hub target, so re-linking (from the app, with a different machine
  // name) replaces the tunnel. Keep its socket: the Hub's machine entry is keyed by it, and a
  // new socket would strand that entry (and its projects) on a socket nothing serves.
  const previous = options.service === false ? undefined : await previousRemoteSocket(linkServiceSpec(options, '/', target), deps, remoteHome)
  const remoteSocket = previous ?? `${remoteHome}/.xerxes/tunnels/${slug(machine)}-daemon.sock`
  checkSocketPath(remoteSocket, 'The remote tunnel socket')

  out('2/4 Background tunnel')
  let servicePath: string | undefined
  if (options.service === false) {
    out(`    Skipped. Keep this running yourself: ${tunnelArgv(deps.ssh, target, remoteSocket, options.localSocket).join(' ')}`)
  } else {
    const installed = await installUserService(linkServiceSpec(options, remoteSocket, target), deps)
    servicePath = installed.path
    for (const note of installed.notes) out(`    Note: ${note}`)
    const deadline = Date.now() + (options.socketWaitMs ?? 20_000)
    let up = false
    for (;;) {
      const probe = await remote(deps.ssh, deps.run, target, `test -S ${shellQuote(remoteSocket)}`, 15_000)
      if (probe.code === 0 && !probe.timedOut) { up = true; break }
      if (Date.now() >= deadline) break
      await deps.sleep(500)
    }
    if (!up) throw new HubLinkError(`The tunnel service is installed (${installed.path}) but the socket never appeared on ${target}. See ${options.logPath}.`)
    out(`    This machine's daemon is reachable on ${target} at ${remoteSocket}. The service reconnects on its own.`)
  }

  out('3/4 This machine on the Hub')
  const added = await remoteJson(deps, target, ['machines', 'add-local', machine, '--socket', remoteSocket, '--json'], options.remoteXerxes, 'Registering this machine')
  const machineId = added.machine_id
  if (typeof machineId !== 'string' || machineId === '') throw new HubLinkError(`The Hub on ${target} did not return a machine id. Update Xerxes on the Hub host.`)
  const replaced = Array.isArray(added.removed_workspace_ids) ? added.removed_workspace_ids.length : 0
  out(`    ${machine} is one machine on the Hub; its projects appear on their own${replaced > 0 ? ` (replaced ${replaced} older per-project entr${replaced === 1 ? 'y' : 'ies'})` : ''}.`)

  out('4/4 Hub key on this machine')
  const info = await remoteJson(deps, target, ['info', '--json'], options.remoteXerxes, 'Reading the Hub key')
  const hubId = info.hub_id, publicKeyPem = info.public_key_pem
  if (typeof hubId !== 'string' || !/^[A-Za-z0-9_-]{1,100}$/.test(hubId) || typeof publicKeyPem !== 'string' || !publicKeyPem.includes('-----BEGIN PUBLIC KEY-----')) {
    throw new HubLinkError(`The Hub on ${target} returned no usable signing key. Update Xerxes on the Hub host.`)
  }
  try {
    await deps.enrollLocal({ socketPath: options.localSocket, hubId, publicKeyPem })
  } catch (error) {
    throw new HubLinkError(`This machine's daemon refused the Hub key: ${error instanceof Error ? error.message : String(error)}. Update Xerxes here if it is older than the Hub features.`)
  }
  out(`    Hub ${hubId} is trusted for every project on this machine.`)
  out('    The running Hub picks this up within a few seconds; no restart is needed.')
  return { machineId, remoteSocket, enrolled: true, ...(servicePath === undefined ? {} : { servicePath }) }
}

export interface LinkTunnelDeps {
  readonly run: CommandRunner
  /** Run a process attached to this one; resolves with its exit code. */
  readonly runAttached: (argv: readonly string[]) => Promise<number>
}

/**
 * One tunnel attempt, run by the background service (which restarts it). A
 * dropped connection leaves the forwarded socket file behind and sshd refuses
 * to bind over it, so the stale file is removed before every attempt.
 */
export async function runLinkTunnel(input: { readonly ssh: string; readonly target: string; readonly remoteSocket: string; readonly localSocket: string }, deps: LinkTunnelDeps): Promise<number> {
  const target = validateSshTarget(input.target)
  checkSocketPath(input.remoteSocket, 'The remote tunnel socket')
  checkSocketPath(input.localSocket, 'The local daemon socket')
  await deps.run([input.ssh, ...SSH_BASE, '--', target, `rm -f ${shellQuote(input.remoteSocket)}`], { timeoutMs: 20_000 })
  return deps.runAttached(tunnelArgv(input.ssh, target, input.remoteSocket, input.localSocket))
}
