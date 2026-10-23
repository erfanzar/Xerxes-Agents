// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { basename, resolve } from 'node:path'
import { ProfileStore } from '../bridge/profiles.js'
import { createLlmClient } from '../llms/client.js'
import { exposureConfigFromEnvironment } from './exposure.js'
import { createHubIdentity } from './identity/index.js'
import { runHubIdentityCli } from './identity/cli.js'
import { createHubService, type HubService } from './service.js'
import { loadOrCreateHubSigner } from './signer.js'
import { SUPERVISOR_ENV, SUPERVISOR_SETUP_HINT, createLlmModelPort } from './supervisor/index.js'
import { createVoiceRuntimeConfig } from './voice/providerFactory.js'
import type { VoiceRuntimeConfig } from './voice/config.js'
import type { HubServiceSupervisorConfig } from './service.js'
import { getHubStatus, restartHub, runHub, startHub, stopHub, type HubLifecycleOptions, type HubStatus } from './lifecycle.js'
import { hubPaths, type HubPaths } from './paths.js'
import { WorkspaceRegistry } from './workspaceRegistry.js'
import { HubDaemonClient } from './daemonClient.js'
import { daemonPaths } from '../daemon/paths.js'
import { existsSync } from 'node:fs'
import { homedir, hostname, userInfo } from 'node:os'
import { join } from 'node:path'
import { createInterface } from 'node:readline/promises'
import { realSleep, spawnCommand } from './commandRunner.js'
import { resolveTailscaleCli, runHubSetup } from './hubSetup.js'
import { runHubLink, runHubRemote, runLinkTunnel, slug } from './hubLink.js'
import { version as packageVersion } from '../../package.json' with { type: 'json' }
import { loadHubSettings, settingsEnvironment } from './settings.js'

export const HUB_HELP = `Usage: xerxes hub [command]

  setup [--name <owner>] [--port <n>] [--no-service] [--no-pair]
                                   On the Hub machine: Tailscale HTTPS, owner, background
                                   service and a phone pairing link. Safe to run again.
  link <user@hub-host> [--machine <name>] [--remote-xerxes <cmd>] [--socket <daemon-socket>]
       [--no-service] [--json]
                                   On a workspace machine: keep an SSH tunnel to the Hub as a
                                   background service, register this machine ONCE on the Hub (its
                                   projects appear on their own) and trust the Hub key on this
                                   machine's daemon for every project. Safe to run again.
  remote <user@hub-host> [--remote-xerxes <cmd>] <hub-args...>
                                   Run "xerxes hub <hub-args...>" on the Hub host over ssh
  info [--json]                    Hub id, key fingerprint, public key and public origin
  machines list [--json]           Machines and the projects the Hub found on them
  machines add-local <label> --socket <path> [--json]
                                   Register one machine (its projects are discovered)
  machines remove <id> [--json]
  run                              Run in the foreground (default)
  start                            Start detached; wait for control readiness
  stop                             Stop only the Hub, never workspace daemons
  restart                          Stop and start the Hub
  status [--json]                  Show process, listeners and workspace health
  workspaces list [--json]          List registered workspaces
  workspaces add-local <dir> [--socket <daemon-socket>] [--label <label>]
  workspaces add-ssh <alias> <remote-dir> --socket <remote-socket> [--label <label>]
                                   The remote daemon socket, e.g. ~/.xerxes/daemon/global-<id>.sock (absolute path)
  workspaces remove <id>
  workspaces import-machines       Explicitly import machines.json (read-only)
  workspaces enroll <id> [--replace]
                                   Pin this Hub's signing key on a local workspace's daemon

  owner init --name <name>         Create the first Hub owner (local only)
  pair [--label <text>] [--no-confirm]
                                   Print a one-time pairing link, QR and comparison code
  pair confirm <pairing-id> <code> Confirm a phone that consumed a pairing link
  devices list [--all] | devices revoke <device-id>
  users list | users revoke <user-id>
  invite create --workspace <id> [--session <id>] --role <viewer|steer|operator> [--expires 24h]
  audit verify                     Check the audit log hash chain

Environment: XERXES_HUB_HOST, XERXES_HUB_PORT, XERXES_HUB_PUBLIC_ORIGINS (the phone's https origin;
required for pairing), XERXES_HUB_VAPID_SUBJECT (mailto: or https: contact; required for push).
XERXES_HUB_SUPERVISOR_PROFILE + XERXES_HUB_SUPERVISOR_MODEL enable the XERXES supervisor (a provider
profile name and a cheap, tool-capable model id). Unset, the supervisor answers "unsupported".
XERXES_HUB_VOICE_ENABLED=true turns on voice (also needs the supervisor); providers and keys come from
XERXES_HUB_VOICE_STT_*, _TTS_*, _REALTIME_* and XERXES_VOICE_<STT|TTS|REALTIME>_API_KEY. Audio is never stored.

State: <XERXES_HOME>/hub (default: ~/.xerxes/hub).
SSH aliases use the host's SSH configuration. Never pass passwords or keys.
Adding/importing a workspace does not connect, bootstrap or update a daemon.
`

const IDENTITY_COMMANDS: ReadonlySet<string> = new Set(['owner', 'pair', 'devices', 'users', 'invite', 'audit'])

/**
 * The supervisor model comes only from an explicitly named provider profile and
 * model id. Missing or unusable configuration yields an `unconfigured` state
 * (every supervisor method answers `unsupported`), never a fallback model.
 */
export function supervisorConfigFromEnvironment(
  environment: Record<string, string | undefined>,
  profiles: Pick<ProfileStore, 'get'> = new ProfileStore(),
): HubServiceSupervisorConfig {
  const profileName = environment[SUPERVISOR_ENV.profile]?.trim()
  const model = environment[SUPERVISOR_ENV.model]?.trim()
  if (!profileName && !model) return { kind: 'unconfigured' }
  if (!profileName || !model) {
    return { kind: 'unconfigured', reason: `${SUPERVISOR_SETUP_HINT} Both ${SUPERVISOR_ENV.profile} and ${SUPERVISOR_ENV.model} are required.` }
  }
  const profile = profiles.get(profileName)
  if (!profile) {
    return { kind: 'unconfigured', reason: `The supervisor provider profile ${JSON.stringify(profileName)} does not exist on this Hub machine. ${SUPERVISOR_SETUP_HINT}` }
  }
  try {
    const client = createLlmClient(model, { provider: profile.provider, api_key: profile.api_key, base_url: profile.base_url })
    return { kind: 'configured', model: createLlmModelPort(client, model) }
  } catch (error) {
    return { kind: 'unconfigured', reason: `The supervisor model could not be created: ${error instanceof Error ? error.message : String(error)}` }
  }
}

export interface HubCliOptions {
  readonly paths?: HubPaths
  readonly lifecycle?: HubLifecycleOptions
  readonly stdout?: (line: string) => void
  readonly stderr?: (line: string) => void
}

export function formatHubStatus(status: HubStatus): string {
  return [
    `Xerxes Hub ${status.version} · running (pid ${status.pid})`,
    `Started: ${status.started_at} · ${status.hostname}`,
    `Listeners: control ready · HTTP ${status.listeners.http ? 'ready' : 'not listening'}`,
    `Workspaces: ${status.workspaces.length}`,
    ...status.workspaces.map(workspace => `  ${workspace.id} · ${workspace.health}`),
  ].join('\n')
}

function takeSocket(args: readonly string[], what: string): { socket: string | undefined; rest: readonly string[] } {
  const index = args.indexOf('--socket')
  if (index < 0) return { socket: undefined, rest: args }
  const socket = args[index + 1]
  if (!socket || socket.startsWith('--')) throw new Error(`--socket needs ${what}`)
  return { socket, rest: [...args.slice(0, index), ...args.slice(index + 2)] }
}

/** `--name value` flags (repeatable ones collect), boolean `--flag`s, and positionals. */
function parseFlags(args: readonly string[], valued: ReadonlySet<string>): { positional: string[]; values: Map<string, string[]>; flags: Set<string> } {
  const positional: string[] = []
  const values = new Map<string, string[]>()
  const flags = new Set<string>()
  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index]!
    if (!arg.startsWith('--')) { positional.push(arg); continue }
    const name = arg.slice(2)
    if (valued.has(name)) {
      const value = args[index + 1]
      if (value === undefined || value.startsWith('--')) throw new Error(`--${name} needs a value`)
      values.set(name, [...(values.get(name) ?? []), value])
      index += 1
    } else flags.add(name)
  }
  return { positional, values, flags }
}

/** Projects of the chats a local daemon has open, read with one summary call (no session is created). */
export async function daemonProjects(socketPath: string, timeoutMs = 8000): Promise<string[]> {
  return new Promise<string[]>((resolveProjects, reject) => {
    let buffer = ''
    let done = false
    const finish = (error: Error | undefined, value: string[] = []): void => {
      if (done) return
      done = true
      clearTimeout(timer)
      try { socket?.end() } catch { /* already closed */ }
      if (error) reject(error); else resolveProjects(value)
    }
    let socket: Bun.Socket<undefined> | undefined
    const timer = setTimeout(() => finish(new Error(`The daemon at ${socketPath} did not answer`)), timeoutMs)
    Bun.connect({ unix: socketPath, socket: {
      data(_socket, bytes) {
        buffer += bytes.toString()
        const newline = buffer.indexOf('\n')
        if (newline < 0) return
        try {
          const frame = JSON.parse(buffer.slice(0, newline)) as { result?: { sessions?: unknown } }
          const rows = Array.isArray(frame.result?.sessions) ? frame.result.sessions : []
          const dirs = rows.flatMap(row => (row !== null && typeof row === 'object' && typeof (row as { cwd?: unknown }).cwd === 'string' ? [(row as { cwd: string }).cwd] : []))
          finish(undefined, [...new Set(dirs.filter(dir => dir.startsWith('/')))].sort())
        } catch (error) { finish(error instanceof Error ? error : new Error(String(error))) }
      },
      error(_socket, error) { finish(error) },
      close() { finish(new Error(`The daemon at ${socketPath} closed the connection`)) },
    } }).then(connected => {
      socket = connected
      connected.write(JSON.stringify({ jsonrpc: '2.0', id: 1, method: 'session.active_list', params: { history_limit: 0 } }) + '\n')
    }, error => finish(error instanceof Error ? error : new Error(String(error))))
  })
}

/** Pin a Hub key on THIS machine's daemon for every project (plain local connection: the daemon accepts hub.enroll only from one). */
export async function enrollOnLocalDaemon(input: { readonly socketPath: string; readonly hubId: string; readonly publicKeyPem: string }): Promise<void> {
  const client = new HubDaemonClient({ socketPath: input.socketPath, projectDir: process.cwd(), retryBaseMs: 60_000, requestTimeoutMs: 10_000 })
  try {
    await client.connect()
    const result = await client.call('hub.enroll', { hub_id: input.hubId, public_key_pem: input.publicKeyPem, all_projects: true, replace: true })
    if (result.ok !== true) throw new Error(typeof result.error === 'string' ? result.error : 'unknown error')
  } finally {
    client.close()
  }
}

async function askLine(question: string): Promise<string | undefined> {
  const prompt = createInterface({ input: process.stdin, output: process.stdout })
  try { return await prompt.question(question) } catch { return undefined } finally { prompt.close() }
}

function withLabel(args: readonly string[], required: number): { values: readonly string[]; label?: string } {
  if (args.length === required) return { values: args }
  if (args.length === required + 2 && args[required] === '--label' && args[required + 1]?.trim()) {
    return { values: args.slice(0, required), label: args[required + 1]! }
  }
  throw new Error('Invalid workspace arguments')
}

/** Return an exit code rather than exiting, so embedding hosts/tests retain ownership. */
export async function runHubCli(args: readonly string[], options: HubCliOptions = {}): Promise<number> {
  const out = options.stdout ?? console.log
  const err = options.stderr ?? console.error
  const paths = options.paths ?? options.lifecycle?.paths ?? hubPaths(options.lifecycle?.environment)
  const lifecycle = { ...options.lifecycle, paths }
  const [command = 'run', ...rest] = args
  if ((command === '--help' || command === '-h' || command === 'help') && !rest.length) { out(HUB_HELP); return 0 }
  // Settings saved by `hub setup` fill in what the environment leaves unset.
  let baseEnvironment: Record<string, string | undefined>
  try {
    baseEnvironment = settingsEnvironment(options.lifecycle?.environment ?? process.env, await loadHubSettings(paths))
  } catch (error) {
    err(error instanceof Error ? error.message : String(error))
    return 1
  }
  if (IDENTITY_COMMANDS.has(command)) {
    return runHubIdentityCli(args, { paths, environment: baseEnvironment, out, err })
  }
  try {
    const executable = options.lifecycle?.executable ?? process.execPath
    const scriptPath = resolve(options.lifecycle?.scriptPath ?? process.argv[1] ?? 'xerxes')
    if (command === 'setup') {
      const parsed = parseFlags(rest, new Set(['name', 'port']))
      if (parsed.positional.length) throw new Error(`Unexpected argument: ${parsed.positional[0]}`)
      const portText = parsed.values.get('port')?.at(-1)
      const port = portText === undefined ? undefined : Number(portText)
      if (port !== undefined && (!Number.isInteger(port) || port < 1 || port > 65535)) throw new Error('--port must be an integer from 1 to 65535')
      const name = parsed.values.get('name')?.at(-1)
      await runHubSetup({
        paths, environment: baseEnvironment, executable, scriptPath,
        ...(name === undefined ? {} : { ownerName: name }), ...(port === undefined ? {} : { port }),
        service: !parsed.flags.has('no-service'), pair: !parsed.flags.has('no-pair'),
      }, {
        platform: process.platform, home: homedir(), run: spawnCommand, sleep: realSleep, uid: process.getuid?.() ?? 0, user: userInfo().username,
        tailscale: resolveTailscaleCli(process.platform),
        fetchHealth: async target => {
          try { return (await fetch(`http://127.0.0.1:${target}/healthz`, { signal: AbortSignal.timeout(2000) })).status === 200 } catch { return false }
        },
        ...(process.stdin.isTTY ? { ask: askLine } : {}),
        out, err,
      })
      return 0
    }
    if (command === 'link') {
      const parsed = parseFlags(rest, new Set(['project', 'machine', 'remote-xerxes', 'socket']))
      if (parsed.positional.length !== 1) throw new Error('usage: xerxes hub link <user@hub-host> [--machine <name>] [--socket <path>] [--json]')
      const json = parsed.flags.has('json')
      // In --json mode stdout carries exactly one JSON object; progress goes to stderr.
      const progress = json ? err : out
      if (parsed.values.has('project')) progress('Note: --project is no longer needed (the Hub discovers this machine\'s projects) and is ignored.')
      const target = parsed.positional[0]!
      const localSocket = resolve(parsed.values.get('socket')?.at(-1) ?? daemonPaths(process.cwd(), baseEnvironment).socketPath)
      const machine = parsed.values.get('machine')?.at(-1)
      const remoteXerxes = parsed.values.get('remote-xerxes')?.at(-1)
      const result = await runHubLink({
        target, localSocket, executable, scriptPath, environment: baseEnvironment,
        logPath: join(paths.logsDir, `link-${slug(target)}.log`),
        ...(machine === undefined ? {} : { machine }), ...(remoteXerxes === undefined ? {} : { remoteXerxes }),
        service: !parsed.flags.has('no-service'),
      }, {
        platform: process.platform, home: homedir(), run: spawnCommand, sleep: realSleep, uid: process.getuid?.() ?? 0, user: userInfo().username,
        ssh: Bun.which('ssh') ?? '/usr/bin/ssh', hostname: hostname(), socketExists: existsSync, out: progress,
        enrollLocal: enrollOnLocalDaemon,
      })
      if (json) {
        out(JSON.stringify({ machine_id: result.machineId, remote_socket: result.remoteSocket, ...(result.servicePath === undefined ? {} : { service_path: result.servicePath }), enrolled: true }))
      }
      return 0
    }
    if (command === 'remote') {
      // `hub remote <target> [--remote-xerxes <cmd>] <hub-args...>`: everything after the target is the remote hub command.
      const args = [...rest]
      let remoteXerxes: string | undefined
      const take = (): void => {
        if (args[0] === '--remote-xerxes') {
          const value = args[1]
          if (!value || value.startsWith('--')) throw new Error('--remote-xerxes needs a command')
          remoteXerxes = value
          args.splice(0, 2)
        }
      }
      take()
      const target = args.shift()
      if (!target) throw new Error('usage: xerxes hub remote <user@hub-host> [--remote-xerxes <cmd>] <hub-args...>')
      take()
      const result = await runHubRemote({ ssh: Bun.which('ssh') ?? '/usr/bin/ssh', target, args, ...(remoteXerxes === undefined ? {} : { remoteXerxes }) }, spawnCommand)
      if (result.stdout) out(result.stdout.replace(/\n$/, ''))
      if (result.stderr) err(result.stderr.replace(/\n$/, ''))
      if (result.timedOut) { err('The remote command timed out.'); return 124 }
      return result.code
    }
    if (command === 'info') {
      const parsed = parseFlags(rest, new Set())
      if (parsed.positional.length) throw new Error(`Unexpected argument: ${parsed.positional[0]}`)
      const signer = await loadOrCreateHubSigner({ paths })
      const origin = baseEnvironment.XERXES_HUB_PUBLIC_ORIGINS?.split(',').map(part => part.trim()).find(Boolean) ?? null
      const info = { hub_id: signer.hubId, fingerprint: signer.fingerprint, public_key_pem: signer.publicKeyPem, public_origin: origin, version: packageVersion }
      if (parsed.flags.has('json')) out(JSON.stringify(info))
      else out([`Hub ${info.hub_id} · Xerxes ${info.version}`, `Key fingerprint: ${info.fingerprint}`, `Public origin: ${origin ?? 'not set (run `xerxes hub setup`)'}`].join('\n'))
      return 0
    }
    if (command === 'machines') {
      const [action = 'list', ...parameters] = rest
      const parsed = parseFlags(parameters, new Set(['socket']))
      const json = parsed.flags.has('json')
      const registry = new WorkspaceRegistry(paths.workspacesPath)
      if (action === 'list' && parsed.positional.length === 0) {
        const snapshot = await registry.load()
        const status = await getHubStatus(lifecycle).catch(() => null)
        const live = new Map((status?.workspaces ?? []).map(row => [row.id, row]))
        const healthOf = (id: string): string => (status === null ? 'offline' : live.get(id)?.health ?? 'offline')
        const machines = snapshot.workspaces.filter(row => row.connection.discover === true).map(row => ({
          id: row.id, label: row.label, endpoint_kind: row.endpoint.kind, discover: true,
          projects: (status?.workspaces ?? []).filter(item => item.machine_id === row.id).map(item => ({
            workspace_id: item.id, project_dir: item.project_dir ?? '', label: item.label ?? item.id, health: item.health,
          })),
        }))
        // Older per-project entries still work; each is reported as a machine holding one project.
        const legacy = snapshot.workspaces.filter(row => row.connection.discover !== true).map(row => ({
          id: row.id, label: row.label, endpoint_kind: row.endpoint.kind, discover: false,
          projects: [{ workspace_id: row.id, project_dir: row.projectDir, label: row.label, health: healthOf(row.id) }],
        }))
        const all = [...machines, ...legacy]
        if (json) out(JSON.stringify({ machines: all }))
        else out(all.length === 0 ? 'No machines registered.' : all.flatMap(machine => [
          `${machine.id} · ${machine.label} · ${machine.endpoint_kind}${machine.discover ? ' · discovers projects' : ''}`,
          ...machine.projects.map(project => `    ${project.label} · ${project.project_dir} · ${project.health}`),
        ]).join('\n'))
        return 0
      }
      if (action === 'add-local' && parsed.positional.length === 1) {
        const socket = parsed.values.get('socket')?.at(-1)
        if (!socket) throw new Error('--socket <daemon-socket> is required')
        const saved = await registry.upsertMachine({ label: parsed.positional[0]!, socketPath: resolve(socket) })
        const result = { machine_id: saved.id, created: saved.created, removed_workspace_ids: saved.removed }
        out(json ? JSON.stringify(result) : `${saved.created ? 'Registered' : 'Updated'} machine ${saved.id}${saved.removed.length ? ` (replaced ${saved.removed.length} per-project entr${saved.removed.length === 1 ? 'y' : 'ies'})` : ''}.`)
        return 0
      }
      if (action === 'remove' && parsed.positional.length === 1) {
        const snapshot = await registry.load()
        await registry.remove(parsed.positional[0]!, snapshot.revision)
        out(json ? JSON.stringify({ removed: parsed.positional[0] }) : 'Machine removed. Its daemon was not stopped.')
        return 0
      }
    }
    if (command === 'link-tunnel') {
      // Internal: run by the `hub link` background service, which restarts it when it exits.
      const parsed = parseFlags(rest, new Set(['remote-socket', 'local-socket']))
      const remoteSocket = parsed.values.get('remote-socket')?.at(-1)
      const localSocket = parsed.values.get('local-socket')?.at(-1)
      if (parsed.positional.length !== 1 || !remoteSocket || !localSocket) throw new Error('usage: xerxes hub link-tunnel <target> --remote-socket <path> --local-socket <path>')
      return await runLinkTunnel({ ssh: Bun.which('ssh') ?? '/usr/bin/ssh', target: parsed.positional[0]!, remoteSocket, localSocket }, {
        run: spawnCommand,
        runAttached: async argv => Bun.spawn([...argv], { stdin: 'ignore', stdout: 'inherit', stderr: 'inherit' }).exited,
      })
    }
    if (command === 'run' && !rest.length) {
      const registry = await new WorkspaceRegistry(paths.workspacesPath).load()
      let service: HubService | undefined
      const handle = await runHub({
        ...lifecycle,
        workspaceHealth: () => service?.workspaceHealth() ?? registry.workspaces.map(workspace => ({ id: workspace.id, health: 'offline' })),
      })
      try {
        const signer = await loadOrCreateHubSigner({ paths })
        const environment = baseEnvironment
        const publicOrigin = environment.XERXES_HUB_PUBLIC_ORIGINS?.split(',').map(part => part.trim()).find(Boolean)
        const vapidSubject = environment.XERXES_HUB_VAPID_SUBJECT?.trim()
        const identity = await createHubIdentity({
          paths,
          signerFingerprint: signer.fingerprint,
          ...(publicOrigin === undefined ? {} : { publicOrigin }),
          ...(vapidSubject ? { vapidSubject } : {}),
          onListenerError: error => err(`Hub identity listener failed: ${error instanceof Error ? error.message : String(error)}`),
        })
        // Voice is opt-in (XERXES_HUB_VOICE_ENABLED=true plus provider keys). A bad
        // setting is reported and voice stays off; the rest of the Hub still runs.
        let voice: VoiceRuntimeConfig | undefined
        try {
          voice = createVoiceRuntimeConfig({ env: environment })
        } catch (error) {
          err(`Hub voice is off: ${error instanceof Error ? error.message : String(error)}`)
        }
        try {
          service = createHubService({
            paths, signer, identity, lifecycle: handle,
            exposure: exposureConfigFromEnvironment(environment),
            supervisor: supervisorConfigFromEnvironment(environment),
            ...(voice === undefined ? {} : { voice }),
            sendPush: (userId, message) => identity.admin.sendPush(userId, message),
            onError: (error, context) => err(`Hub ${context} failed: ${error instanceof Error ? error.message : String(error)}`),
          })
        } catch (error) {
          identity.close()
          throw error
        }
        let info: Awaited<ReturnType<HubService['start']>>
        try {
          info = await service.start()
        } catch (error) {
          identity.close()
          throw error
        }
        // Registered after service.start() so it runs after the server and router have stopped.
        handle.registerShutdownHook(() => identity.close())
        if (identity.admin.ownerUserId() === undefined) {
          err('No Hub owner exists yet. Run `xerxes hub owner init --name <name>` on this machine, then `xerxes hub pair`.')
        }
        const status = service.status()
        out(JSON.stringify({ event: 'hub.listening', port: info.port, hostname: info.hostname, identity: status.identity, fail_closed: status.fail_closed, ...(status.message === undefined ? {} : { message: status.message }) }))
      } catch (error) {
        await handle.stop().catch(() => undefined)
        throw error
      }
      out(JSON.stringify({ event: 'hub.ready', ...handle.status() }))
      await handle.closed
      out(JSON.stringify({ event: 'hub.stopped' }))
      return 0
    }
    if ((command === 'start' || command === 'restart') && !rest.length) {
      const result = await (command === 'start' ? startHub(lifecycle) : restartHub(lifecycle))
      out(`${result.alreadyRunning ? 'Hub already running.' : 'Hub started.'}\n${formatHubStatus(result.status)}`)
      if (result.logPath) out(`Log: ${result.logPath}`)
      return 0
    }
    if (command === 'stop' && !rest.length) {
      out((await stopHub(lifecycle)).stopped ? 'Hub stopped. Workspace daemons were not stopped.' : 'Hub is not running; no process was signalled.')
      return 0
    }
    if (command === 'status' && (!rest.length || (rest.length === 1 && rest[0] === '--json'))) {
      const status = await getHubStatus(lifecycle)
      out(rest[0] === '--json' ? JSON.stringify(status ? { running: true, ...status } : { running: false }, null, 2) : status ? formatHubStatus(status) : 'Xerxes Hub is not running.')
      return status ? 0 : 1
    }
    if (command === 'workspaces') {
      const [action = 'list', ...parameters] = rest
      const registry = new WorkspaceRegistry(paths.workspacesPath)
      const snapshot = await registry.load()
      if (action === 'list' && (!parameters.length || (parameters.length === 1 && parameters[0] === '--json'))) {
        if (parameters[0] === '--json') out(JSON.stringify(snapshot, null, 2))
        else out([`Hub workspaces (revision ${snapshot.revision})`, ...snapshot.workspaces.map(workspace => `${workspace.id} · ${workspace.label} · ${workspace.endpoint.kind} · ${workspace.projectDir}`), ...(snapshot.workspaces.length ? [] : ['No workspaces registered.'])].join('\n'))
        return 0
      }
      if (action === 'add-local') {
        // `--socket <path>` points at a daemon that does not use the per-project
        // socket (the desktop's global daemon, or a reverse SSH tunnel).
        const { socket: socketPath, rest } = takeSocket(parameters, 'the daemon socket path')
        const parsed = withLabel(rest, 1)
        const projectDir = resolve(parsed.values[0]!)
        const endpoint = { kind: 'local' as const, ...(socketPath === undefined ? {} : { socketPath: resolve(socketPath) }) }
        const label = parsed.label ?? (basename(projectDir) || projectDir)
        // Re-running `hub link` (or this command) for the same project and daemon updates
        // the existing entry instead of registering a duplicate.
        const existing = snapshot.workspaces.find(row => row.projectDir === projectDir && row.endpoint.kind === 'local'
          && (row.endpoint.socketPath ?? undefined) === endpoint.socketPath)
        if (existing) {
          const saved = await registry.update(existing.id, { label, connection: { autoConnect: true } }, snapshot.revision)
          out(JSON.stringify(saved.workspaces.find(row => row.id === existing.id), null, 2))
          return 0
        }
        // A workspace the operator adds by hand is meant to be used: connect it on start.
        const saved = await registry.add({
          label, projectDir, endpoint,
          connection: { autoConnect: true }, consent: { bootstrap: false, update: false },
        }, snapshot.revision)
        out(JSON.stringify(saved.workspaces.at(-1), null, 2))
        return 0
      }
      if (action === 'add-ssh') {
        // `--socket <remote-path>` may appear anywhere after the two positionals.
        const { socket: remoteSocketPath, rest } = takeSocket(parameters, 'the remote daemon socket path')
        const parsed = withLabel(rest, 2)
        const [sshAlias, projectDir] = parsed.values
        const saved = await registry.add({
          label: parsed.label ?? sshAlias, projectDir,
          endpoint: { kind: 'ssh', sshAlias, remoteProjectDir: projectDir, ...(remoteSocketPath === undefined ? {} : { remoteSocketPath }) },
          connection: { autoConnect: remoteSocketPath !== undefined }, consent: { bootstrap: false, update: false },
        }, snapshot.revision)
        if (remoteSocketPath === undefined) {
          err('No --socket given: this workspace stays "needs operator" until its remote daemon socket is registered (find it with `ls ~/.xerxes/daemon/*.sock` on the host).')
        }
        out(JSON.stringify(saved.workspaces.at(-1), null, 2))
        return 0
      }
      if (action === 'remove' && parameters.length === 1) {
        await registry.remove(parameters[0]!, snapshot.revision)
        out('Workspace removed. Its daemon was not stopped.')
        return 0
      }
      if (action === 'enroll' && (parameters.length === 1 || (parameters.length === 2 && parameters[1] === '--replace'))) {
        const workspace = snapshot.workspaces.find(row => row.id === parameters[0])
        if (!workspace) throw new Error(`No workspace with id ${JSON.stringify(parameters[0])}`)
        if (workspace.endpoint.kind !== 'local') {
          throw new Error('Enrolling an SSH workspace from the Hub is not supported yet: no remote daemon socket is registered for it. Run `xerxes hub workspaces enroll` against a local workspace on that host instead.')
        }
        const signer = await loadOrCreateHubSigner({ paths })
        const socketPath = workspace.endpoint.socketPath ?? daemonPaths(workspace.projectDir).socketPath
        // A plain local connection (no hub_auth): the daemon accepts hub.enroll only from such a caller.
        const client = new HubDaemonClient({ socketPath, projectDir: workspace.projectDir, retryBaseMs: 60_000, requestTimeoutMs: 10_000 })
        try {
          await client.connect()
          const result = await client.call('hub.enroll', {
            hub_id: signer.hubId, public_key_pem: signer.publicKeyPem, project_dirs: [workspace.projectDir],
            ...(parameters[1] === '--replace' ? { replace: true } : {}),
          })
          if (result.ok !== true) throw new Error(`The daemon refused enrollment: ${typeof result.error === 'string' ? result.error : 'unknown error'}`)
        } finally {
          client.close()
        }
        out(`Enrolled Hub ${signer.hubId} on ${workspace.label} (${workspace.projectDir}).\nKey fingerprint: ${signer.fingerprint}`)
        return 0
      }
      if (action === 'import-machines' && !parameters.length) {
        const saved = await registry.importMachines(snapshot.revision, paths.machinesPath)
        out(`Imported ${saved.workspaces.length - snapshot.workspaces.length} workspace(s). machines.json was not modified.`)
        return 0
      }
    }
    err(`Unknown or invalid Hub command.\n${HUB_HELP}`)
    return 1
  } catch (error) {
    err(`${error instanceof Error ? error.message : String(error)}\n${HUB_HELP}`)
    return 1
  }
}
