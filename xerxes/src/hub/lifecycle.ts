// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { chmod, lstat, open, readFile, rename, rm } from 'node:fs/promises'
import { createConnection, createServer, type Socket } from 'node:net'
import { hostname } from 'node:os'
import { dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { version as packageVersion } from '../../package.json' with { type: 'json' }
import { withFileLock } from '../session/daemonTranscript.js'
import { ensureHubDirectories, hubPaths, type HubPaths } from './paths.js'

export interface HubWorkspaceHealth {
  readonly id: string
  readonly health: string
  /** Set on workspaces found on a machine entry: the entry they came from, their folder and label. */
  readonly machine_id?: string
  readonly project_dir?: string
  readonly label?: string
  /** The verified runtime identity of the daemon behind this workspace, once known. */
  readonly runtime_id?: string
  /** The route the machine is reached through when it has fallback routes. */
  readonly route?: string
}
export interface HubStatus {
  readonly service: 'xerxes-hub'
  readonly pid: number
  readonly started_at: string
  readonly version: string
  readonly build_id: string
  readonly instance_id: string
  readonly hostname: string
  readonly listeners: { readonly control: true; readonly http: boolean }
  readonly workspaces: readonly HubWorkspaceHealth[]
}
export interface HubPidRecord {
  readonly pid: number
  readonly started_at: string
  readonly version: string
  readonly build_id: string
  readonly instance_id: string
  readonly hostname: string
  readonly control_socket: string
}
export interface HubClock { now(): number; sleep(milliseconds: number): Promise<void> }
export interface HubSpawnRequest {
  readonly argv: readonly string[]
  readonly environment: Record<string, string | undefined>
  readonly logPath: string
}
export type HubShutdownHook = () => void | Promise<void>
export interface HubLifecycleOptions {
  readonly paths?: HubPaths
  readonly clock?: HubClock
  readonly processProbe?: (pid: number) => boolean | Promise<boolean>
  readonly signal?: (pid: number, signal: 'SIGTERM') => void
  readonly spawn?: (request: HubSpawnRequest) => Promise<{ readonly pid: number }>
  readonly controlRequest?: (path: string, method: 'status' | 'stop', timeoutMs: number) => Promise<unknown>
  readonly timeoutMs?: number
  readonly executable?: string
  readonly scriptPath?: string
  readonly environment?: Record<string, string | undefined>
  readonly version?: string
  readonly buildId?: string
}
export interface HubRunOptions extends HubLifecycleOptions {
  readonly shutdownHooks?: readonly HubShutdownHook[]
  readonly workspaceHealth?: () => readonly HubWorkspaceHealth[]
  readonly httpReady?: () => boolean
  readonly installSignalHandlers?: boolean
}
export interface HubHandle {
  readonly admitting: boolean
  readonly closed: Promise<void>
  status(): HubStatus
  stop(): Promise<void>
  registerShutdownHook(hook: HubShutdownHook): () => void
  setHttpReady(ready: boolean): void
}
export class HubLifecycleError extends Error {
  constructor(readonly code: 'already_running' | 'start_failed' | 'stop_failed' | 'invalid_state', message: string) {
    super(message)
    this.name = 'HubLifecycleError'
  }
}

const nativeClock: HubClock = { now: Date.now, sleep: milliseconds => Bun.sleep(milliseconds) }
function isMissing(error: unknown): boolean { return !!error && typeof error === 'object' && 'code' in error && error.code === 'ENOENT' }
function isObject(value: unknown): value is Record<string, unknown> { return !!value && typeof value === 'object' && !Array.isArray(value) }
function positivePid(value: unknown): value is number { return typeof value === 'number' && Number.isSafeInteger(value) && value > 0 }
function alive(pid: number): boolean {
  try { process.kill(pid, 0); return true }
  catch (error) { return !!error && typeof error === 'object' && 'code' in error && error.code === 'EPERM' }
}
function statusValue(value: unknown): value is HubStatus {
  return isObject(value) && value.service === 'xerxes-hub' && positivePid(value.pid)
    && typeof value.started_at === 'string' && Number.isFinite(Date.parse(value.started_at))
    && typeof value.version === 'string' && typeof value.build_id === 'string'
    && typeof value.instance_id === 'string' && value.instance_id.length > 0 && typeof value.hostname === 'string'
    && isObject(value.listeners) && value.listeners.control === true && typeof value.listeners.http === 'boolean'
    && Array.isArray(value.workspaces) && value.workspaces.every(item => isObject(item) && typeof item.id === 'string' && typeof item.health === 'string')
}
async function readPid(paths: HubPaths): Promise<HubPidRecord | null> {
  let raw: string
  try {
    const info = await lstat(paths.pidPath)
    if (!info.isFile() || info.size > 16384) throw new HubLifecycleError('invalid_state', 'Hub pid record must be a small regular file')
    raw = await readFile(paths.pidPath, 'utf8')
  } catch (error) { if (isMissing(error)) return null; throw error }
  let value: unknown
  try { value = JSON.parse(raw) as unknown } catch { return null }
  if (!isObject(value) || !positivePid(value.pid) || typeof value.started_at !== 'string'
    || typeof value.version !== 'string' || typeof value.build_id !== 'string' || typeof value.hostname !== 'string'
    || typeof value.instance_id !== 'string' || value.control_socket !== paths.controlSocketPath) return null
  return { pid: value.pid, started_at: value.started_at, version: value.version, build_id: value.build_id, hostname: value.hostname, instance_id: value.instance_id, control_socket: paths.controlSocketPath }
}
function sameIdentity(status: HubStatus, record: HubPidRecord | HubStatus): boolean {
  return status.pid === record.pid && status.started_at === record.started_at && status.instance_id === record.instance_id
    && status.version === record.version && status.build_id === record.build_id && status.hostname === record.hostname
}

/** Bounded, one-request NDJSON exchange on the Hub-only Unix socket. */
export async function requestHubControl(path: string, method: 'status' | 'stop', timeoutMs = 1000): Promise<unknown> {
  return new Promise((resolve, reject) => {
    const socket = createConnection(path)
    let buffer = Buffer.alloc(0), done = false
    const finish = (error: Error | null, result?: unknown) => {
      if (done) return
      done = true
      clearTimeout(timer)
      socket.destroy()
      if (error) reject(error); else resolve(result)
    }
    const timer = setTimeout(() => finish(new Error('Hub control socket timed out')), timeoutMs)
    socket.on('connect', () => { socket.write(`${JSON.stringify({ method })}\n`) })
    socket.on('error', error => finish(error))
    socket.on('end', () => finish(new Error('Hub control socket closed without a response')))
    socket.on('data', chunk => {
      buffer = Buffer.concat([buffer, chunk])
      if (buffer.length > 64 * 1024) { finish(new Error('Hub control response exceeds 64 KiB')); return }
      const newline = buffer.indexOf(10)
      if (newline < 0) return
      try { finish(null, JSON.parse(buffer.subarray(0, newline).toString('utf8')) as unknown) }
      catch { finish(new Error('Invalid Hub control response')) }
    })
  })
}

async function probeSocket(options: HubLifecycleOptions): Promise<HubStatus | null> {
  const paths = options.paths ?? hubPaths(options.environment)
  let result: unknown
  try { result = await (options.controlRequest ?? requestHubControl)(paths.controlSocketPath, 'status', Math.min(options.timeoutMs ?? 1000, 1000)) }
  catch { return null }
  if (!statusValue(result) || !(await (options.processProbe ?? alive)(result.pid))) return null
  return result
}

/** A live PID alone is never sufficient: match the socket identity to the complete pid record. */
export async function getHubStatus(options: HubLifecycleOptions = {}): Promise<HubStatus | null> {
  const paths = options.paths ?? hubPaths(options.environment)
  const record = await readPid(paths)
  if (!record || !(await (options.processProbe ?? alive)(record.pid))) return null
  const status = await probeSocket({ ...options, paths })
  return status && sameIdentity(status, record) ? status : null
}

async function removeSocket(paths: HubPaths): Promise<void> {
  try {
    if (!(await lstat(paths.controlSocketPath)).isSocket()) throw new HubLifecycleError('invalid_state', 'Refusing to replace a non-socket at the Hub control path')
    await rm(paths.controlSocketPath)
  } catch (error) { if (!isMissing(error)) throw error }
}
async function writePid(paths: HubPaths, record: HubPidRecord): Promise<void> {
  const temporary = `${paths.pidPath}.${crypto.randomUUID()}.tmp`
  try {
    const file = await open(temporary, 'wx', 0o600)
    try { await file.writeFile(`${JSON.stringify(record)}\n`); await file.sync() }
    finally { await file.close() }
    await rename(temporary, paths.pidPath)
  } finally { await rm(temporary, { force: true }) }
}

/** Owns only Hub listeners/resources. Hooks must close relays, never shut down a daemon. */
export async function runHub(options: HubRunOptions = {}): Promise<HubHandle> {
  const paths = options.paths ?? hubPaths(options.environment)
  await ensureHubDirectories(paths)
  return withFileLock(paths.ownershipLockPath, async () => {
    await chmod(paths.ownershipLockPath, 0o600)
    // Even a damaged/missing pid file must not let us unlink an answering Hub socket.
    if (await getHubStatus({ ...options, paths }) || await probeSocket({ ...options, paths })) {
      throw new HubLifecycleError('already_running', 'Xerxes Hub is already running')
    }
    await readPid(paths) // Reject a non-regular pid path before replacing anything.
    await removeSocket(paths)
    const record: HubPidRecord = {
      pid: process.pid, started_at: new Date((options.clock ?? nativeClock).now()).toISOString(),
      version: options.version ?? packageVersion, build_id: options.buildId ?? options.version ?? packageVersion,
      instance_id: crypto.randomUUID(), hostname: hostname(), control_socket: paths.controlSocketPath,
    }
    let admitting = true, httpReady = false, stopped = false
    let stopping: Promise<void> | undefined
    const hooks = [...(options.shutdownHooks ?? [])]
    const sockets = new Set<Socket>()
    let resolveClosed!: () => void
    let rejectClosed!: (error: Error) => void
    const closed = new Promise<void>((resolve, reject) => { resolveClosed = resolve; rejectClosed = reject })
    // Some embedding callers observe stop() instead; retain the rejection without an unhandled promise.
    void closed.catch(() => undefined)
    const status = (): HubStatus => ({
      service: 'xerxes-hub', pid: record.pid, started_at: record.started_at, version: record.version,
      build_id: record.build_id, instance_id: record.instance_id, hostname: record.hostname,
      listeners: { control: true, http: options.httpReady?.() ?? httpReady }, workspaces: options.workspaceHealth?.() ?? [],
    })
    const server = createServer(socket => {
      if (!admitting) { socket.destroy(); return }
      sockets.add(socket)
      socket.on('close', () => { sockets.delete(socket) })
      socket.on('error', () => { socket.destroy() })
      socket.setTimeout(2000, () => socket.destroy())
      let buffer = Buffer.alloc(0), answered = false
      socket.on('data', chunk => {
        if (answered) return
        buffer = Buffer.concat([buffer, chunk])
        if (buffer.length > 4096) { socket.destroy(); return }
        const newline = buffer.indexOf(10)
        if (newline < 0) return
        answered = true
        let request: unknown
        try { request = JSON.parse(buffer.subarray(0, newline).toString('utf8')) as unknown }
        catch { socket.end('{"error":"invalid_request"}\n'); return }
        if (!isObject(request) || Object.keys(request).some(key => key !== 'method')) { socket.end('{"error":"invalid_request"}\n'); return }
        if (request.method === 'status') {
          try {
            const response = `${JSON.stringify(status())}\n`
            socket.end(Buffer.byteLength(response) <= 64 * 1024 ? response : '{"error":"status_too_large"}\n')
          } catch { socket.end('{"error":"status_unavailable"}\n') }
        } else if (request.method === 'stop') {
          socket.end('{"ok":true}\n', () => { void stop().catch(error => console.error(String(error))) })
        } else socket.end('{"error":"unknown_method"}\n')
      })
    })
    const onSignal = () => { void stop().catch(error => console.error(String(error))) }
    const stop = (): Promise<void> => {
      if (stopped) return Promise.resolve()
      if (stopping) return stopping
      admitting = false
      stopping = (async () => {
        const errors: unknown[] = []
        // Registration order is intentional: close clients, flush stores, then close runtime relays.
        for (const hook of hooks) { try { await hook() } catch (error) { errors.push(error) } }
        for (const socket of sockets) socket.destroy()
        await new Promise<void>(resolve => { server.close(error => { if (error) errors.push(error); resolve() }) })
        try {
          await withFileLock(paths.ownershipLockPath, async () => {
            await chmod(paths.ownershipLockPath, 0o600)
            const current = await readPid(paths)
            if (current?.instance_id === record.instance_id) {
              await rm(paths.pidPath, { force: true })
              await removeSocket(paths)
            }
          }, { label: 'Hub ownership', staleMs: 30000, waitMs: 5000 })
        } catch (error) { errors.push(error) }
        process.off('SIGTERM', onSignal)
        process.off('SIGINT', onSignal)
        stopped = true
        if (errors.length) {
          const error = new AggregateError(errors, `Hub shutdown failed: ${errors.map(error => error instanceof Error ? error.message : String(error)).join('; ')}`)
          rejectClosed(error)
          throw error
        }
        resolveClosed()
      })()
      return stopping
    }
    try {
      await new Promise<void>((resolve, reject) => {
        const startupError = (error: Error) => { reject(error) }
        server.once('error', startupError)
        server.listen(paths.controlSocketPath, () => { server.off('error', startupError); resolve() })
      })
      await chmod(paths.controlSocketPath, 0o600)
      await writePid(paths, record)
    } catch (error) {
      server.close()
      await removeSocket(paths)
      throw error
    }
    if (options.installSignalHandlers !== false) {
      process.on('SIGTERM', onSignal)
      process.on('SIGINT', onSignal)
    }
    return {
      get admitting() { return admitting }, closed, status, stop,
      registerShutdownHook(hook) {
        if (!admitting) throw new HubLifecycleError('invalid_state', 'Hub shutdown has already begun')
        hooks.push(hook)
        return () => { const index = hooks.indexOf(hook); if (index >= 0) hooks.splice(index, 1) }
      },
      setHttpReady(ready) { httpReady = ready },
    }
  }, { label: 'Hub ownership', staleMs: 30000, waitMs: 5000 })
}

async function spawnDetached(request: HubSpawnRequest): Promise<{ readonly pid: number }> {
  const log = await open(request.logPath, 'ax', 0o600)
  try {
    const child = Bun.spawn([...request.argv], { env: request.environment, stdin: 'ignore', stdout: log.fd, stderr: log.fd, detached: true, windowsHide: true })
    child.unref()
    return { pid: child.pid }
  } finally { await log.close() }
}
async function logTail(path: string): Promise<string> {
  try {
    const file = await open(path, 'r')
    try {
      const info = await file.stat()
      const buffer = Buffer.alloc(Math.min(info.size, 8192))
      await file.read(buffer, 0, buffer.length, Math.max(0, info.size - buffer.length))
      return buffer.toString('utf8').trim() || '(log is empty)'
    } finally { await file.close() }
  } catch (error) { if (isMissing(error)) return '(no log output)'; throw error }
}

export async function startHub(options: HubLifecycleOptions = {}): Promise<{ readonly alreadyRunning: boolean; readonly status: HubStatus; readonly logPath?: string }> {
  const paths = options.paths ?? hubPaths(options.environment)
  const timeout = options.timeoutMs ?? 10000
  const clock = options.clock ?? nativeClock
  await ensureHubDirectories(paths)
  return withFileLock(paths.startLockPath, async () => {
    await chmod(paths.startLockPath, 0o600)
    const running = await getHubStatus({ ...options, paths })
    if (running) return { alreadyRunning: true, status: running }
    const logPath = `${paths.logsDir}/hub-${clock.now()}-${crypto.randomUUID()}.log`
    const argv = [options.executable ?? process.execPath, options.scriptPath ?? process.argv[1] ?? fileURLToPath(new URL('../cli.ts', import.meta.url)), 'hub', 'run']
    const environment = { ...(options.environment ?? process.env), XERXES_HOME: dirname(paths.stateDir) }
    let child: { readonly pid: number }
    try { child = await (options.spawn ?? spawnDetached)({ argv, environment, logPath }) }
    catch (error) { throw new HubLifecycleError('start_failed', `Failed to spawn Hub: ${error instanceof Error ? error.message : String(error)}\nLog: ${logPath}\n${await logTail(logPath)}`) }
    const deadline = clock.now() + timeout
    do {
      const ready = await getHubStatus({ ...options, paths, timeoutMs: Math.min(timeout, 250) })
      if (ready && ready.pid === child.pid) return { alreadyRunning: false, status: ready, logPath }
      if (!(await (options.processProbe ?? alive)(child.pid))) break
      await clock.sleep(Math.min(50, Math.max(1, deadline - clock.now())))
    } while (clock.now() < deadline)
    throw new HubLifecycleError('start_failed', `Hub did not become ready within ${timeout}ms. Log: ${logPath}\n${await logTail(logPath)}`)
  }, { label: 'Hub start', staleMs: 30000, waitMs: timeout + 5000 })
}

/** SIGTERM fallback is allowed only after a fresh, complete Hub identity revalidation. */
export async function stopHub(options: HubLifecycleOptions = {}): Promise<{ readonly stopped: boolean }> {
  const paths = options.paths ?? hubPaths(options.environment)
  const status = await getHubStatus({ ...options, paths })
  if (!status) return { stopped: false }
  try { await (options.controlRequest ?? requestHubControl)(paths.controlSocketPath, 'stop', Math.min(options.timeoutMs ?? 1000, 1000)) }
  catch { /* A lost stop acknowledgement is resolved by observing ownership below, never replayed. */ }
  const clock = options.clock ?? nativeClock
  const timeout = options.timeoutMs ?? 5000
  const released = async () => {
    const record = await readPid(paths)
    return !record || record.instance_id !== status.instance_id || !(await (options.processProbe ?? alive)(status.pid))
  }
  const wait = async () => {
    const deadline = clock.now() + timeout
    do { if (await released()) return true; await clock.sleep(25) } while (clock.now() < deadline)
    return released()
  }
  if (await wait()) return { stopped: true }
  const verified = await getHubStatus({ ...options, paths })
  if (verified && sameIdentity(verified, status)) {
    ;(options.signal ?? ((pid, signal) => process.kill(pid, signal)))(verified.pid, 'SIGTERM')
    if (await wait()) return { stopped: true }
  }
  throw new HubLifecycleError('stop_failed', 'Hub did not stop; no unverified PID was signalled. Inspect the Hub log')
}

export async function restartHub(options: HubLifecycleOptions = {}): Promise<{ readonly alreadyRunning: boolean; readonly status: HubStatus; readonly logPath?: string }> {
  await stopHub(options)
  return startHub(options)
}
