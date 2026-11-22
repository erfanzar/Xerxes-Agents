// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdir, readFile, rename, writeFile } from 'node:fs/promises'
import { dirname, isAbsolute } from 'node:path'

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
export function tunnelArgv(ssh: string, target: string, remoteSocket: string, localSocket: string, sshOptions: readonly string[] = SSH_BASE): string[] {
  return [ssh, '-N', ...sshOptions,
    '-o', 'ExitOnForwardFailure=yes', '-o', 'ServerAliveInterval=20', '-o', 'ServerAliveCountMax=3',
    '-R', `${remoteSocket}:${localSocket}`, '--', target]
}

/** What `ssh -G` resolves for a target: the name its host keys are stored under and the user's known_hosts files. */
export interface SshHostKeyLookup {
  readonly name: string
  readonly files: readonly string[]
}

export function parseSshConfigDump(stdout: string): SshHostKeyLookup | undefined {
  const values = new Map<string, string>()
  for (const line of stdout.split('\n')) {
    const match = /^(\S+)\s+(.*)$/.exec(line.trim())
    if (match === null) continue
    const key = (match[1] ?? '').toLowerCase()
    if (!values.has(key)) values.set(key, (match[2] ?? '').trim())
  }
  const host = values.get('hostname')
  if (!host) return undefined
  const port = values.get('port') ?? '22'
  const alias = values.get('hostkeyalias')
  const name = alias && alias !== 'none' ? alias : port === '22' ? host : `[${host}]:${port}`
  const files = (values.get('userknownhostsfile') ?? '').split(/\s+/).filter(file => isAbsolute(file))
  return { name, files }
}

function checkKnownHostsPath(path: string): void {
  if (!isAbsolute(path) || /["\n\0]/.test(path)) throw new HubLinkError(`The link's known_hosts must be an absolute path without quotes: ${path}`)
}

/** `UserKnownHostsFile` reading the link's pinned keys first, then the files the user's SSH config already uses. */
export function knownHostsOption(pinned: string, files: readonly string[]): string {
  const all = [pinned, ...files.filter(file => file !== pinned)]
  return `UserKnownHostsFile=${all.map(file => (/\s/.test(file) ? `"${file}"` : file)).join(' ')}`
}

function isMissingFile(error: unknown): boolean {
  return error instanceof Error && 'code' in error && error.code === 'ENOENT'
}

/**
 * Copy the target's host keys from the user's own known_hosts files into the link's pinned file.
 * Only keys the user's SSH already trusts are copied, and pinned keys are never removed, so cleaning
 * up ~/.ssh/known_hosts later does not silently cut the machine off its Hub.
 */
export async function pinHostKeys(input: { readonly ssh: string; readonly sshKeygen: string; readonly target: string; readonly knownHostsPath: string }, run: CommandRunner): Promise<SshHostKeyLookup | undefined> {
  checkKnownHostsPath(input.knownHostsPath)
  const dump = await run([input.ssh, '-G', '--', input.target], { timeoutMs: 10_000 })
  const lookup = dump.code === 0 && !dump.timedOut ? parseSshConfigDump(dump.stdout) : undefined
  if (lookup === undefined) return undefined
  const found: string[] = []
  for (const file of lookup.files) {
    if (file === input.knownHostsPath) continue
    const result = await run([input.sshKeygen, '-F', lookup.name, '-f', file], { timeoutMs: 10_000 })
    if (result.code !== 0 || result.timedOut) continue
    for (const line of result.stdout.split('\n')) {
      const trimmed = line.trim()
      if (trimmed !== '' && !trimmed.startsWith('#')) found.push(trimmed)
    }
  }
  let existing = ''
  try { existing = await readFile(input.knownHostsPath, 'utf8') } catch (error) {
    if (!isMissingFile(error)) throw error
  }
  const lines = existing.split('\n').map(line => line.trim()).filter(line => line !== '')
  const added = [...new Set(found)].filter(line => !lines.includes(line))
  if (added.length > 0) {
    await mkdir(dirname(input.knownHostsPath), { recursive: true, mode: 0o700 })
    const temp = `${input.knownHostsPath}.${process.pid}.tmp`
    await writeFile(temp, `${[...lines, ...added].join('\n')}\n`, { mode: 0o600 })
    await rename(temp, input.knownHostsPath)
  }
  return lookup
}

/** One actionable line for a tunnel attempt that SSH itself refused (exit 255). */
export function linkFailureMessage(stderr: string, target: string, knownHostsPath?: string): string {
  if (/remote host identification has changed/i.test(stderr)) {
    return `The SSH host key of ${target} has CHANGED. The link stays down until you confirm the new key with the server's administrator and replace the old entry (ssh-keygen -R <host>).`
  }
  if (/host key verification failed/i.test(stderr)) {
    return `${target} has no trusted SSH host key on this machine${knownHostsPath ? ` (checked ${knownHostsPath} and your known_hosts)` : ''}. Compare its fingerprint with a trusted source, then run \`ssh ${target}\` once and accept it; the link reconnects on its own.`
  }
  const detail = firstLine(stderr)
  if (/permission denied/i.test(stderr)) return `${target} refused this machine's SSH key: ${detail}.`
  return `Could not reach ${target}: ${detail || 'ssh exited without a message'}.`
}

/** Wait before the service restarts a failed attempt: 5 s, doubling, at most 5 minutes. */
export function linkRetryDelayMs(failures: number): number {
  return Math.min(300_000, 5_000 * 2 ** Math.min(Math.max(failures, 1) - 1, 6))
}

export interface LinkTunnelState {
  readonly failures: number
  readonly last_error?: string
  readonly last_failure_at?: string
  readonly connected_at?: string
}

async function readLinkState(path: string | undefined): Promise<LinkTunnelState> {
  if (path === undefined) return { failures: 0 }
  try {
    const value: unknown = JSON.parse(await readFile(path, 'utf8'))
    if (isRecord(value) && typeof value.failures === 'number' && Number.isInteger(value.failures) && value.failures >= 0) return { failures: value.failures }
  } catch { /* a missing or unreadable status file means no recorded failures */ }
  return { failures: 0 }
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

/**
 * `hub trust <ssh-target> --socket <path>`: pin the Hub's signing key on the daemon behind a local
 * socket (this machine's own, or a forward to a remote daemon a desktop app holds). Only step 4 of
 * `hub link`: no tunnel and no machine entry.
 */
export async function runHubTrust(input: { readonly target: string; readonly socketPath: string; readonly remoteXerxes?: string }, deps: Pick<HubLinkDeps, 'ssh' | 'run' | 'socketExists' | 'enrollLocal'>): Promise<{ readonly hubId: string; readonly enrolled: true }> {
  const target = validateSshTarget(input.target)
  checkSocketPath(input.socketPath, 'The daemon socket')
  if (!deps.socketExists(input.socketPath)) throw new HubLinkError(`No Xerxes daemon is listening at ${input.socketPath}.`)
  const info = await remoteJson(deps as HubLinkDeps, target, ['info', '--json'], input.remoteXerxes, 'Reading the Hub key')
  const hubId = info.hub_id, publicKeyPem = info.public_key_pem
  if (typeof hubId !== 'string' || !/^[A-Za-z0-9_-]{1,100}$/.test(hubId) || typeof publicKeyPem !== 'string' || !publicKeyPem.includes('-----BEGIN PUBLIC KEY-----')) {
    throw new HubLinkError(`The Hub on ${target} returned no usable signing key. Update Xerxes on the Hub host.`)
  }
  try {
    await deps.enrollLocal({ socketPath: input.socketPath, hubId, publicKeyPem })
  } catch (error) {
    throw new HubLinkError(`The daemon refused the Hub key: ${error instanceof Error ? error.message : String(error)}. Update Xerxes there if it is older than the Hub features.`)
  }
  return { hubId, enrolled: true }
}

export interface LinkTunnelDeps {
  readonly run: CommandRunner
  /** Run a process attached to this one; resolves with its exit code. */
  readonly runAttached: (argv: readonly string[]) => Promise<number>
  readonly sleep?: (ms: number) => Promise<void>
  /** Diagnostic lines (the service's log). */
  readonly log?: (line: string) => void
  readonly now?: () => Date
}

export interface LinkTunnelInput {
  readonly ssh: string
  readonly target: string
  readonly remoteSocket: string
  readonly localSocket: string
  readonly sshKeygen?: string
  /** The link's own pinned host keys; checked before the user's known_hosts. */
  readonly knownHostsPath?: string
  /** Small JSON status file: consecutive failures and the last error. */
  readonly statePath?: string
}

/**
 * One tunnel attempt, run by the background service (which restarts it). A
 * dropped connection leaves the forwarded socket file behind and sshd refuses
 * to bind over it, so the stale file is removed before every attempt.
 * That first connection also tells why SSH refuses: the reason is logged once
 * per attempt and repeated failures back off instead of retrying every 5 s.
 */
export async function runLinkTunnel(input: LinkTunnelInput, deps: LinkTunnelDeps): Promise<number> {
  const target = validateSshTarget(input.target)
  checkSocketPath(input.remoteSocket, 'The remote tunnel socket')
  checkSocketPath(input.localSocket, 'The local daemon socket')
  const log = deps.log ?? ((line: string) => { process.stderr.write(`${line}\n`) })
  const now = deps.now ?? (() => new Date())
  let sshOptions: readonly string[] = SSH_BASE
  if (input.knownHostsPath !== undefined) {
    let lookup: SshHostKeyLookup | undefined
    try {
      lookup = await pinHostKeys({ ssh: input.ssh, sshKeygen: input.sshKeygen ?? 'ssh-keygen', target, knownHostsPath: input.knownHostsPath }, deps.run)
    } catch (error) {
      log(`${now().toISOString()} Could not update ${input.knownHostsPath}: ${error instanceof Error ? error.message : String(error)}`)
    }
    if (lookup !== undefined) sshOptions = [...SSH_BASE, '-o', knownHostsOption(input.knownHostsPath, lookup.files)]
  }
  const writeState = async (state: LinkTunnelState): Promise<void> => {
    if (input.statePath === undefined) return
    try {
      await writeFile(input.statePath, `${JSON.stringify(state)}\n`, { mode: 0o600 })
    } catch (error) {
      log(`${now().toISOString()} Could not write ${input.statePath}: ${error instanceof Error ? error.message : String(error)}`)
    }
  }
  const cleanup = await deps.run([input.ssh, ...sshOptions, '--', target, `rm -f ${shellQuote(input.remoteSocket)}`], { timeoutMs: 20_000 })
  if (cleanup.code === 255 || cleanup.timedOut) {
    const failures = (await readLinkState(input.statePath)).failures + 1
    const delayMs = linkRetryDelayMs(failures)
    const message = cleanup.timedOut ? `Could not reach ${target}: SSH timed out.` : linkFailureMessage(cleanup.stderr, target, input.knownHostsPath)
    const at = now().toISOString()
    log(`${at} ${message} Retrying in ${Math.round(delayMs / 1000)} s (failure ${failures}).`)
    await writeState({ failures, last_error: message, last_failure_at: at })
    await (deps.sleep ?? Bun.sleep)(delayMs)
    return 255
  }
  await writeState({ failures: 0, connected_at: now().toISOString() })
  return deps.runAttached(tunnelArgv(input.ssh, target, input.remoteSocket, input.localSocket, sshOptions))
}
