// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { existsSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import { pathToFileURL } from 'node:url'

/**
 * Out-of-process execution for managed plugin modules.
 *
 * Each module runs in its own `bun --no-env-file` child process and talks JSON
 * over Bun's IPC channel. This was chosen over a Bun Worker after measuring
 * Bun 1.3.12: a Worker stuck in a busy loop kept a CPU core busy after
 * `terminate()`, and literal `process.env.X` reads inside a worker still saw
 * the daemon's environment. A child process stops on SIGKILL, and its
 * environment holds only the variables passed to it (`--no-env-file` stops Bun
 * from loading a `.env` beside the plugin).
 *
 * What this is NOT: a security sandbox. The child runs as the same user and can
 * use `node:fs`, `node:net`, `node:http`, `Bun.spawn` and `Bun.fetch` freely.
 * `fetch` is wrapped to refuse hosts the plugin did not declare — a guard rail
 * against mistakes, not a boundary against hostile code.
 */

// Plain JavaScript, written to a private temp file and run by the child: it
// must not import Xerxes sources, which do not exist as files in a bundle.
const HOST_SCRIPT = String.raw`
import { createRequire } from 'node:module'

const moduleCache = createRequire(import.meta.url).cache
const send = message => { try { process.send(message) } catch {} }
const errorText = error => error && typeof error.message === 'string' ? error.message : String(error)
const clone = value => {
  if (value === undefined) return undefined
  try { const text = JSON.stringify(value); return text === undefined ? undefined : JSON.parse(text) } catch { return undefined }
}

let allowedHosts = []
const realFetch = globalThis.fetch
const hostAllowed = host => allowedHosts.some(pattern => pattern.startsWith('*.') ? host.endsWith(pattern.slice(1)) : host === pattern)
const guardedFetch = function fetch(input, init) {
  let url
  try { url = new URL(typeof input === 'string' ? input : input instanceof URL ? input.href : String(input && input.url)) }
  catch { return Promise.reject(new TypeError('Plugin fetch needs an absolute URL')) }
  const host = url.hostname.toLowerCase().replace('[', '').replace(']', '')
  if (!hostAllowed(host)) return Promise.reject(new TypeError('Plugin fetch to ' + host + ' refused: the plugin did not declare it in permissions.network'))
  return realFetch(input, init)
}
Object.assign(guardedFetch, realFetch)
globalThis.fetch = guardedFetch

const calls = []
const tools = new Map()
const commands = new Map()
const names = []
let closed = false
let loaded = false
const open = () => { if (closed) throw new Error('Plugin registration is closed; capabilities must be registered before register() settles') }
const noteMeta = meta => { if (meta && typeof meta === 'object' && typeof meta.name === 'string' && !names.includes(meta.name)) names.push(meta.name) }
const recordTool = tool => {
  if (typeof tool === 'function') return { kind: 'positional' }
  if (!tool || typeof tool !== 'object') return { kind: 'invalid' }
  let parameters
  let parametersError = false
  try { const text = JSON.stringify(tool.parameters); parameters = text === undefined ? undefined : JSON.parse(text) } catch { parametersError = true }
  const timeoutMs = tool.timeoutMs === undefined ? undefined : typeof tool.timeoutMs === 'number' ? tool.timeoutMs : -1
  return { kind: 'structured', description: typeof tool.description === 'string' ? tool.description : undefined, parameters, parametersError, timeoutMs, hasExecute: typeof tool.execute === 'function' }
}
const registry = {
  get pluginNames() { return [...names] },
  inventory() { return [] },
  registerPlugin(meta) { open(); noteMeta(meta); calls.push({ op: 'plugin', meta: clone(meta) }); return { meta } },
  registerTool(name, tool, meta, pluginName) {
    open(); noteMeta(meta)
    calls.push({ op: 'tool', name: clone(name), tool: recordTool(tool), meta: clone(meta), pluginName: clone(pluginName) })
    if (typeof name === 'string' && !tools.has(name)) tools.set(name, tool)
  },
  registerCommand(name, handler, options, meta, pluginName) {
    open(); noteMeta(meta)
    calls.push({ op: 'command', name: clone(name), handler: typeof handler === 'function', options: clone(options), meta: clone(meta), pluginName: clone(pluginName) })
    const token = typeof name === 'string' ? name.trim().replace(/^\/+/, '').toLowerCase() : ''
    if (token && !commands.has(token)) commands.set(token, handler)
  },
  registerHook(name, _callback, meta, pluginName) { open(); calls.push({ op: 'hook', name: clone(name), meta: clone(meta), pluginName: clone(pluginName) }) },
  registerProvider(name, _provider, meta, pluginName) { open(); calls.push({ op: 'provider', name: clone(name), meta: clone(meta), pluginName: clone(pluginName) }) },
  registerChannel(name, _channel, meta, pluginName) { open(); calls.push({ op: 'channel', name: clone(name), meta: clone(meta), pluginName: clone(pluginName) }) },
  unregisterPlugin(name) { open(); calls.push({ op: 'unregister', name: clone(name) }) },
}
const declaredHosts = () => calls.flatMap(call => {
  const network = call.meta && call.meta.permissions && call.meta.permissions.network
  return Array.isArray(network) ? network.filter(entry => typeof entry === 'string').map(entry => entry.toLowerCase()) : []
})
const controllers = new Map()

process.on('message', async message => {
  if (!message || typeof message !== 'object') return
  if (message.t === 'load') {
    if (loaded) { send({ t: 'loadError', message: 'Plugin process already loaded its module' }); return }
    loaded = true
    try {
      const module = await import(message.href)
      if (typeof module.register === 'function') await module.register(registry)
      closed = true
      allowedHosts = declaredHosts()
      const folders = Array.isArray(message.folders) ? message.folders.filter(folder => typeof folder === 'string') : []
      const files = Object.keys(moduleCache).filter(file => folders.some(folder => file.startsWith(folder)) && !file.includes('/node_modules/') && !file.includes('\\node_modules\\'))
      send({ t: 'loaded', calls, files })
    } catch (error) {
      closed = true
      send({ t: 'loadError', message: errorText(error) })
    }
    return
  }
  if (message.t === 'abort') { const controller = controllers.get(message.id); if (controller) controller.abort(new Error('Plugin call cancelled')); return }
  if (message.t !== 'call' && message.t !== 'command') return
  const id = message.id
  const controller = new AbortController()
  controllers.set(id, controller)
  try {
    const context = message.context && typeof message.context === 'object' ? message.context : {}
    let value
    if (message.t === 'call') {
      const tool = tools.get(message.name)
      if (!tool) throw new Error('Plugin does not provide tool ' + message.name)
      value = typeof tool === 'function'
        ? await tool(...(Array.isArray(message.args) ? message.args : []))
        : await tool.execute(message.input && typeof message.input === 'object' && !Array.isArray(message.input) ? message.input : {}, { ...context, signal: controller.signal })
    } else {
      const handler = commands.get(message.name)
      if (typeof handler !== 'function') throw new Error('Plugin does not provide /' + message.name)
      value = await handler(typeof message.args === 'string' ? message.args : '', { ...context })
    }
    let encoded
    try { const text = JSON.stringify(value); encoded = text === undefined ? null : JSON.parse(text) }
    catch { throw new Error('Plugin result is not JSON-serializable') }
    send({ t: 'result', id, ok: true, value: encoded })
  } catch (error) {
    send({ t: 'result', id, ok: false, error: errorText(error) })
  } finally {
    controllers.delete(id)
  }
})
process.on('unhandledRejection', error => { process.stderr.write('Unhandled rejection in plugin: ' + errorText(error) + '\n') })
process.on('uncaughtException', error => { process.stderr.write('Uncaught exception in plugin: ' + errorText(error) + '\n') })
process.on('disconnect', () => process.exit(0))
`

let harnessPath: string | undefined

/** The host script on disk, rewritten if a temp cleaner removed it. */
function harness(): string {
  if (harnessPath && existsSync(harnessPath)) return harnessPath
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-plugin-host-'))
  const path = join(directory, 'host.mjs')
  writeFileSync(path, HOST_SCRIPT, { mode: 0o600 })
  process.once('exit', () => { try { rmSync(directory, { recursive: true, force: true }) } catch { /* best effort at exit */ } })
  harnessPath = path
  return path
}

/** What one module registered, as recorded by the child; validated by the caller. */
export interface PluginProcessLoad {
  readonly calls: readonly unknown[]
  /** Absolute paths of module files the child loaded from the plugin's folder. */
  readonly files: readonly string[]
}

export interface PluginProcessOptions {
  /** The module to import. Its folder is the child's working directory. */
  readonly path: string
  /** Folder prefixes (with a trailing separator) whose loaded files are reported. */
  readonly folders: readonly string[]
  /** The child's complete environment. Empty by default. */
  readonly env?: Readonly<Record<string, string>>
  readonly loadTimeoutMs?: number
  readonly executable?: string
}

export interface PluginInvokeOptions {
  readonly timeoutMs: number
  readonly signal?: AbortSignal
  /** Names the call in timeout, cancellation and failure messages. */
  readonly label: string
}

interface PendingCall {
  settled: boolean
  readonly resolve: (value: unknown) => void
  readonly reject: (error: Error) => void
  readonly cleanup: () => void
}

const ABORT_GRACE_MS = 2_000
const DEFAULT_LOAD_TIMEOUT_MS = 30_000
const STDERR_TAIL_CHARS = 2_000

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

/**
 * One child process serving one plugin module. A call that times out, or a
 * cancelled call that does not stop within a short grace period, kills the
 * process; every other call in flight then fails with the reason. The owner
 * starts a fresh process for the next call.
 */
export class PluginProcess {
  private child: Bun.Subprocess<'ignore', 'pipe', 'pipe'> | undefined
  private readonly pending = new Map<number, PendingCall>()
  private loading: { resolve(value: PluginProcessLoad): void; reject(error: Error): void } | undefined
  private nextId = 1
  private ended: string | undefined
  private stderr = ''

  constructor(private readonly options: PluginProcessOptions) {}

  get alive(): boolean {
    return this.child !== undefined && this.ended === undefined
  }

  /** Start the process, import the module and run its register(). */
  load(): Promise<PluginProcessLoad> {
    if (this.child) return Promise.reject(new Error('Plugin process already started'))
    const timeout = this.options.loadTimeoutMs ?? DEFAULT_LOAD_TIMEOUT_MS
    const child = Bun.spawn([this.options.executable ?? process.execPath, '--no-env-file', harness()], {
      cwd: dirname(this.options.path),
      env: { ...(this.options.env ?? {}) },
      stdin: 'ignore',
      stdout: 'pipe',
      stderr: 'pipe',
      serialization: 'json',
      ipc: message => this.receive(message),
      onExit: (_process, exitCode, signalCode) => this.exit(signalCode ? `was stopped (${String(signalCode)})` : `exited with code ${exitCode ?? 'unknown'}`),
    })
    this.child = child
    // The daemon must be able to exit while plugin processes idle; they exit
    // on their own when the IPC channel closes.
    child.unref()
    void this.drain(child.stdout, false)
    void this.drain(child.stderr, true)
    return new Promise<PluginProcessLoad>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.loading = undefined
        reject(new Error(`Plugin module did not finish loading within ${timeout} ms`))
        this.terminate('was stopped: loading timed out')
      }, timeout)
      this.loading = {
        resolve: value => { clearTimeout(timer); resolve(value) },
        reject: error => { clearTimeout(timer); reject(error) },
      }
      if (!this.post({ t: 'load', href: pathToFileURL(this.options.path).href, folders: this.options.folders })) {
        this.terminate('could not be reached')
      }
    })
  }

  /** Run a tool (`call`) or slash command (`command`) in the child. */
  invoke(kind: 'call' | 'command', payload: Readonly<Record<string, unknown>>, options: PluginInvokeOptions): Promise<unknown> {
    if (!this.alive) return Promise.reject(new Error(`Plugin process ${this.ended ?? 'is not running'}`))
    if (options.signal?.aborted) return Promise.reject(new Error(`${options.label} was cancelled`))
    const id = this.nextId++
    return new Promise<unknown>((resolve, reject) => {
      let timer: ReturnType<typeof setTimeout> | undefined
      let grace: ReturnType<typeof setTimeout> | undefined
      const onAbort = () => {
        if (entry.settled) return
        this.post({ t: 'abort', id })
        entry.reject(new Error(`${options.label} was cancelled`))
        clearTimeout(timer)
        // The caller is released at once; a plugin that ignores its signal
        // loses its process instead of running on unobserved.
        grace = setTimeout(() => {
          if (this.pending.has(id)) this.terminate(`was stopped: ${options.label} did not stop within ${ABORT_GRACE_MS} ms of being cancelled`)
        }, ABORT_GRACE_MS)
      }
      const entry: PendingCall = {
        settled: false,
        resolve: value => { if (!entry.settled) { entry.settled = true; resolve(value) } },
        reject: error => { if (!entry.settled) { entry.settled = true; reject(error) } },
        cleanup: () => {
          clearTimeout(timer)
          clearTimeout(grace)
          options.signal?.removeEventListener('abort', onAbort)
        },
      }
      this.pending.set(id, entry)
      timer = setTimeout(() => {
        entry.reject(new Error(`${options.label} timed out after ${options.timeoutMs} ms; its plugin process was stopped and starts again on the next call`))
        this.terminate(`was stopped after ${options.label} timed out`)
      }, options.timeoutMs)
      options.signal?.addEventListener('abort', onAbort, { once: true })
      if (!this.post({ ...payload, t: kind, id })) {
        this.pending.delete(id)
        entry.cleanup()
        entry.reject(new Error('Could not reach the plugin process'))
      }
    })
  }

  /** Kill the process; pending calls fail with the reason. */
  terminate(reason = 'was stopped'): void {
    if (this.child && this.ended === undefined) {
      try { this.child.kill('SIGKILL') } catch { /* already gone */ }
    }
    this.exit(reason)
  }

  private post(message: Readonly<Record<string, unknown>>): boolean {
    try {
      this.child?.send(message)
      return this.child !== undefined
    } catch {
      return false
    }
  }

  private receive(message: unknown): void {
    if (!isRecord(message)) return
    if (message.t === 'loaded' || message.t === 'loadError') {
      const loading = this.loading
      this.loading = undefined
      if (!loading) return
      if (message.t === 'loadError') {
        loading.reject(new Error(typeof message.message === 'string' && message.message ? message.message : 'Plugin module failed to load'))
        return
      }
      const files = Array.isArray(message.files) ? message.files.filter((file): file is string => typeof file === 'string') : []
      if (!Array.isArray(message.calls)) { loading.reject(new Error('Malformed plugin registration')); return }
      loading.resolve({ calls: message.calls, files })
      return
    }
    if (message.t !== 'result' || typeof message.id !== 'number') return
    const entry = this.pending.get(message.id)
    if (!entry) return
    this.pending.delete(message.id)
    entry.cleanup()
    if (message.ok === true) entry.resolve(message.value ?? null)
    else entry.reject(new Error(typeof message.error === 'string' && message.error ? message.error : 'Plugin call failed'))
  }

  private exit(reason: string): void {
    if (this.ended !== undefined) return
    this.ended = reason
    const tail = this.stderr.trim().slice(-500)
    const error = new Error(`Plugin process ${reason}${tail ? `: ${tail}` : ''}`)
    this.loading?.reject(error)
    this.loading = undefined
    for (const [id, entry] of this.pending) {
      this.pending.delete(id)
      entry.cleanup()
      entry.reject(error)
    }
  }

  private async drain(stream: ReadableStream<Uint8Array>, keep: boolean): Promise<void> {
    const decoder = new TextDecoder()
    const reader = stream.getReader()
    try {
      for (;;) {
        const { done, value } = await reader.read()
        if (done) break
        if (keep && value) this.stderr = (this.stderr + decoder.decode(value, { stream: true })).slice(-STDERR_TAIL_CHARS)
      }
    } catch {
      // The process is gone; its exit is reported through onExit.
    } finally {
      reader.releaseLock()
    }
  }
}
