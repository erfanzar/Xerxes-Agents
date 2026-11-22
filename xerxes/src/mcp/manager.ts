// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { JsonObject } from '../types/toolCalls.js'
import { MCPClient, type MCPToolCallOptions } from './client.js'
import { parseMcpServerConfig } from './config.js'
import {
  MCPReconnectError,
  mcpConfigSecrets,
  reconnectWithBackoff,
  scrubCredentials,
  type ReconnectWithBackoffOptions,
} from './reconnect.js'
import type {
  MCPCapabilityKind,
  MCPPrompt,
  MCPPromptResult,
  MCPResource,
  MCPResourceContentsResult,
  MCPServerConfig,
  MCPTool,
  MCPToolCallResult,
} from './types.js'

/**
 * The deliberately small client boundary used by the fleet manager.
 *
 * Hosts can substitute a remote transport or a test double without coupling
 * fleet lifecycle logic to Bun subprocess APIs.
 */
export interface MCPClientPort {
  readonly config: MCPServerConfig
  readonly prompts: readonly MCPPrompt[]
  readonly resources: readonly MCPResource[]
  readonly tools: readonly MCPTool[]
  readonly connected?: boolean
  connect(): Promise<void>
  disconnect(): Promise<void>
  callTool(name: string, arguments_?: JsonObject, options?: MCPToolCallOptions): Promise<MCPToolCallResult>
  readResource(uri: string): Promise<MCPResourceContentsResult>
  getPrompt(name: string, arguments_?: JsonObject): Promise<MCPPromptResult>
  onCapabilitiesChanged?(handler: (kind: MCPCapabilityKind) => void): () => void
}

/** Factory boundary for hosts that own MCP transports or authentication. */
export type MCPClientFactory = (config: MCPServerConfig) => MCPClientPort | Promise<MCPClientPort>

export type MCPServerLifecycleOperation = 'connect' | 'disconnect' | 'reconnect' | 'replace'

/** A redacted lifecycle failure retained for diagnostics. */
export interface MCPServerFailure {
  readonly attempt?: number
  readonly error: string
  readonly name: string
  readonly operation: MCPServerLifecycleOperation
}

export interface MCPServerStatus {
  readonly state?: 'disabled' | 'failed' | 'disconnected'
  readonly connected: boolean
  readonly lastError?: string
  readonly name: string
  readonly prompts: number
  readonly resources: number
  readonly tools: number
}

export interface MCPServerCapabilitiesSummary {
  readonly prompts: number
  readonly resources: number
  readonly tools: number
}

export interface MCPCapabilitiesChangedEvent {
  readonly kind: MCPCapabilityKind
  readonly name: string
}

export interface MCPManagerOptions {
  /** Creates one connected-client candidate for each start or reconnect attempt. */
  readonly clientFactory?: MCPClientFactory
  /** Maximum active plus queued operations retained for one server. */
  readonly maxPendingOperationsPerServer?: number
  /** Invoked when a live server refreshes tools, resources, or prompts via `list_changed`. */
  readonly onCapabilitiesChanged?: (event: MCPCapabilitiesChangedEvent) => void
  /** Receives already-redacted lifecycle failures. Observer errors are contained. */
  readonly onFailure?: (failure: MCPServerFailure) => void
  /** Retry policy and deterministic hooks used by reconnect. */
  readonly reconnect?: ReconnectWithBackoffOptions
}

/** Raised when one server exceeds its bounded active-plus-queued operation capacity. */
export class MCPServerQueueFullError extends Error {
  readonly serverName: string
  readonly limit: number

  constructor(serverName: string, limit: number) {
    super(`MCP server '${serverName}' has reached its pending operation limit (${limit})`)
    this.name = new.target.name
    this.serverName = serverName
    this.limit = limit
  }
}

/** Raised when a routed tool, resource, or prompt is unavailable from every active server. */
export class MCPCapabilityNotFoundError extends Error {
  readonly capability: 'prompt' | 'resource' | 'tool'
  readonly capabilityName: string

  constructor(capability: 'prompt' | 'resource' | 'tool', name: string) {
    super(capability + ' ' + name + ' not found in any connected MCP server')
    this.name = new.target.name
    this.capability = capability
    this.capabilityName = name
  }
}

/**
 * Own live MCP connections, present their capability union, and route calls
 * to the first server that published each capability.
 *
 * Lifecycle mutations are serialized so a simultaneous start, stop, or
 * reconnect cannot leave a half-registered client visible to discovery.
 */
export class MCPManager {
  static readonly DEFAULT_MAX_PENDING_OPERATIONS_PER_SERVER = 1_000
  private readonly clientFactory: MCPClientFactory
  private readonly failures = new Map<string, MCPServerFailure>()
  private readonly serverOperations = new Map<string, Promise<void>>()
  private readonly pendingServerOperations = new Map<string, number>()
  private readonly maxPendingOperationsPerServer: number
  private readonly capabilityListeners = new Set<(event: MCPCapabilitiesChangedEvent) => void>()
  private readonly onFailure: ((failure: MCPServerFailure) => void) | undefined
  private readonly reconnectOptions: ReconnectWithBackoffOptions | undefined
  private readonly configurations = new Map<string, MCPServerConfig>()
  private readonly registrationControllers = new Map<string, AbortController>()
  private readonly servers = new Map<string, MCPClientPort>()
  private readonly replacements = new Set<string>()
  private readonly replacementTokens = new Map<string, object>()

  constructor(options: MCPManagerOptions = {}) {
    this.clientFactory = options.clientFactory ?? (config => new MCPClient(config))
    this.maxPendingOperationsPerServer = positiveInteger(
      options.maxPendingOperationsPerServer ?? MCPManager.DEFAULT_MAX_PENDING_OPERATIONS_PER_SERVER,
      'maxPendingOperationsPerServer',
    )
    if (options.onCapabilitiesChanged) {
      this.capabilityListeners.add(options.onCapabilitiesChanged)
    }
    this.onFailure = options.onFailure
    this.reconnectOptions = options.reconnect
  }

  /** Subscribe to live capability updates across all active servers. */
  onCapabilitiesChanged(listener: (event: MCPCapabilitiesChangedEvent) => void): () => void {
    this.capabilityListeners.add(listener)
    return () => { this.capabilityListeners.delete(listener) }
  }

  /** Build, connect, and register a server. Disabled or duplicate configurations are skipped. */
  addServer(config: MCPServerConfig): Promise<boolean> {
    const normalized = normalizeConfig(config)
    return this.enqueueServer(normalized.name, async () => {
      if (this.servers.has(normalized.name)) return false
      this.registrationControllers.get(normalized.name)?.abort()
      this.registrationControllers.set(normalized.name, new AbortController())
      this.configurations.set(normalized.name, normalized)
      if (normalized.enabled === false) return false
      try {
        const client = await this.connectClient(normalized)
        this.servers.set(normalized.name, client)
        this.failures.delete(normalized.name)
        return true
      } catch (error) {
        this.recordFailure(normalized.name, 'connect', error, undefined, mcpConfigSecrets(normalized))
        return false
      }
    })
  }

  /** Semantic lifecycle alias for hosts that model MCP server registration as startup. */
  start(config: MCPServerConfig): Promise<boolean> {
    return this.addServer(config)
  }

  /** Validate and connect a replacement before retiring an existing registration.
   * This changes live state only; the settings host owns durable persistence.
   * Existing calls remain available during discovery. Concurrent lifecycle changes
   * invalidate the candidate instead of resurrecting removed or newer state.
   */
  async replaceServer(value: unknown, signal?: AbortSignal, commit?: () => undefined): Promise<boolean> {
    return this.stageSettings(value, false, signal, commit)
  }

  /** Admit a new name without publishing it until discovery and persistence succeed. */
  async createServer(value: unknown, signal?: AbortSignal, commit?: () => undefined): Promise<boolean> {
    return this.stageSettings(value, true, signal, commit)
  }

  private async stageSettings(value: unknown, create: boolean, signal?: AbortSignal, commit?: () => undefined): Promise<boolean> {
    const parsed = parseMcpServerConfig(value)
    if (!parsed.ok) throw new TypeError(parsed.error)
    const config = parsed.config
    const token = {}
    const baseline = await this.enqueueServer(config.name, async () => {
      const previousConfig = this.configurations.get(config.name)
      if (create && previousConfig) throw new Error('MCP server name already exists')
      if (!create && !previousConfig) throw new Error('MCP registration no longer exists; refresh settings')
      if (this.replacements.has(config.name)) throw new Error('MCP configuration replacement already in progress')
      this.replacements.add(config.name)
      this.replacementTokens.set(config.name, token)
      return { config: previousConfig, client: this.servers.get(config.name) }
    }, signal)
    const isCurrent = () => this.replacementTokens.get(config.name) === token && this.configurations.get(config.name) === baseline.config
      && this.servers.get(config.name) === baseline.client
    const secrets = [...mcpConfigSecrets(config), ...(baseline.config ? mcpConfigSecrets(baseline.config) : [])]
    let candidate: MCPClientPort | undefined
    let installed = false
    const checkCancelled = () => {
      if (signal?.aborted) throw new DOMException('MCP configuration replacement cancelled', 'AbortError')
    }
    try {
      checkCancelled()
      if (config.enabled !== false) candidate = await this.connectClient(config)
      checkCancelled()
      return await this.enqueueServer(config.name, async () => {
        checkCancelled()
        if (!isCurrent()) throw new Error('MCP registration changed; refresh settings before retrying')
        // The settings host commits synchronously after successful discovery and
        // before the in-memory swap. A failed commit leaves the old client intact.
        commit?.()
        this.registrationControllers.get(config.name)?.abort()
        this.registrationControllers.set(config.name, new AbortController())
        this.configurations.set(config.name, config)
        if (candidate) this.servers.set(config.name, candidate)
        else this.servers.delete(config.name)
        installed = true
        this.failures.delete(config.name)
        try { await baseline.client?.disconnect() } catch (error) {
          this.recordFailure(config.name, 'disconnect', error, undefined, secrets)
        }
        return true
      }, signal)
    } catch (error) {
      if (isCurrent()) this.recordFailure(config.name, 'replace', error, undefined, secrets)
      if (signal?.aborted) throw new DOMException('MCP configuration replacement cancelled', 'AbortError')
      throw new Error(scrubCredentials(errorMessage(error), secrets))
    } finally {
      if (candidate && !installed) {
        try { await candidate.disconnect() } catch (error) {
          if (isCurrent()) this.recordFailure(config.name, 'disconnect', error, undefined, secrets)
        }
      }
      this.replacements.delete(config.name)
      if (this.replacementTokens.get(config.name) === token) this.replacementTokens.delete(config.name)
    }
  }

  /**
   * Disconnect and drop one server. A teardown failure is retained for
   * diagnostics, but the server is removed so stale tools cannot be routed.
   */
  removeServer(name: string): Promise<boolean> {
    const normalized = normalizeName(name)
    return this.enqueueServer(normalized, async () => {
      this.replacementTokens.delete(normalized)
      const configured = this.configurations.delete(normalized)
      this.registrationControllers.get(normalized)?.abort()
      this.registrationControllers.delete(normalized)
      const client = this.servers.get(normalized)
      if (!client) {
        this.failures.delete(normalized)
        return configured
      }
      this.servers.delete(normalized)
      try {
        await client.disconnect()
        this.failures.delete(normalized)
      } catch (error) {
        this.recordFailure(normalized, 'disconnect', error, undefined, mcpConfigSecrets(client.config))
      }
      return true
    })
  }

  /** Semantic lifecycle alias for removing one active MCP server. */
  stop(name: string): Promise<boolean> {
    return this.removeServer(name)
  }

  /**
   * Connect an enabled configuration with a fresh client candidate, retrying failed
   * connection attempts according to the configured backoff policy.
   *
   * Only the registry delete and swap are serialized through the lifecycle
   * queue. The backoff sleep loop runs outside the queue so a long retry
   * schedule cannot block addServer, removeServer, or disconnectAll for
   * every other server.
   */
  async reconnect(name: string): Promise<boolean> {
    const normalized = normalizeName(name)
    const registration = await this.enqueueServer(normalized, () => {
      const config = this.configurations.get(normalized)
      if (!config || config.enabled === false) return Promise.resolve(undefined)
      const client = this.servers.get(normalized)
      if (client) {
        this.servers.delete(normalized)
      }
      return Promise.resolve({ client, config, signal: this.registrationControllers.get(normalized)!.signal })
    })
    if (!registration) {
      return false
    }
    const { client: previous, config, signal } = registration
    const isCurrent = () => this.configurations.get(normalized) === config
    const secrets = mcpConfigSecrets(config)
    try {
      await previous?.disconnect()
    } catch (error) {
      if (isCurrent()) this.recordFailure(normalized, 'disconnect', error, undefined, secrets)
    }

    let candidate: MCPClientPort
    try {
      candidate = await reconnectWithBackoff(
        () => {
          if (!isCurrent()) throw new Error('MCP registration was removed or replaced')
          return this.connectClient(config)
        },
        this.optionsForReconnect(normalized, secrets, isCurrent, signal),
      )
    } catch (error) {
      if (isCurrent()) this.recordFailure(
        normalized,
        'reconnect',
        error,
        error instanceof MCPReconnectError ? error.attempts : undefined,
        secrets,
      )
      return false
    }

    return this.enqueueServer(normalized, async () => {
      if (!isCurrent() || this.servers.has(normalized)) {
        // A concurrent registration claimed the name while backoff ran; keep
        // the newer client and tear down this superseded candidate.
        try {
          await candidate.disconnect()
        } catch (error) {
          if (isCurrent()) this.recordFailure(normalized, 'disconnect', error, undefined, secrets)
        }
        return isCurrent() && this.servers.has(normalized)
      }
      this.servers.set(normalized, candidate)
      this.failures.delete(normalized)
      return true
    })
  }

  /** Disconnect every server and clear the active capability registry. */
  async disconnectAll(): Promise<void> {
    const names = [...new Set([...this.listConfiguredServers(), ...this.replacements])]
    await Promise.all(names.map(name => this.removeServer(name)))
  }

  /** Semantic lifecycle alias for stopping every active MCP server. */
  stopAll(): Promise<void> {
    return this.disconnectAll()
  }

  /** Return a client only while it is active and eligible for capability routing. */
  getServer(name: string): MCPClientPort | undefined {
    return this.servers.get(normalizeName(name))
  }

  /** Return active server names in registration order. */
  listServers(): string[] {
    return [...this.servers.keys()]
  }

  /** Include enabled, disabled, and failed registrations in configuration order. */
  listConfiguredServers(): string[] {
    return [...this.configurations.keys()]
  }

  /** Return lifecycle status without exposing launch arguments, headers, or environment values. */
  status(name: string): MCPServerStatus | undefined {
    const normalized = normalizeName(name)
    const client = this.servers.get(normalized)
    const failure = this.failures.get(normalized)
    if (!client) {
      const config = this.configurations.get(normalized)
      if (!config) return undefined
      return { name: normalized, connected: false, tools: 0, resources: 0, prompts: 0,
        state: config.enabled === false ? 'disabled' : failure ? 'failed' : 'disconnected',
        ...(failure ? { lastError: failure.error } : {}) }
    }
    return {
      name: normalized,
      connected: client.connected ?? true,
      tools: client.tools.length,
      resources: client.resources.length,
      prompts: client.prompts.length,
      ...(failure === undefined ? {} : { lastError: failure.error }),
    }
  }

  /** Return statuses for every configured server in registration order. */
  listStatus(): MCPServerStatus[] {
    return this.listConfiguredServers().flatMap(name => {
      const status = this.status(name)
      return status === undefined ? [] : [status]
    })
  }

  /** Return a copy of each server's last redacted lifecycle failure. */
  lifecycleFailures(): MCPServerFailure[] {
    return [...this.failures.values()].map(failure => ({ ...failure }))
  }

  /** Return one server's last redacted lifecycle failure, if any. */
  lastFailure(name: string): MCPServerFailure | undefined {
    const failure = this.failures.get(normalizeName(name))
    return failure === undefined ? undefined : { ...failure }
  }

  /** Flatten discovered tools, retaining first-registration-wins for duplicate names. */
  getAllTools(): MCPTool[] {
    const tools: MCPTool[] = []
    const names = new Set<string>()
    for (const [serverName, client] of this.servers) {
      for (const tool of client.tools) {
        if (names.has(tool.name)) {
          continue
        }
        names.add(tool.name)
        tools.push({ ...tool, serverName })
      }
    }
    return tools
  }

  /** Flatten discovered resources from every active server. */
  getAllResources(): MCPResource[] {
    const resources: MCPResource[] = []
    for (const [serverName, client] of this.servers) {
      for (const resource of client.resources) {
        resources.push({ ...resource, serverName })
      }
    }
    return resources
  }

  /** Flatten discovered prompts from every active server. */
  getAllPrompts(): MCPPrompt[] {
    const prompts: MCPPrompt[] = []
    for (const [serverName, client] of this.servers) {
      for (const prompt of client.prompts) {
        prompts.push({ ...prompt, serverName })
      }
    }
    return prompts
  }

  /**
   * Route a tool call to the first active server that published its name.
   *
   * Calls on one server are sequenced with lifecycle changes for that server,
   * while independent servers remain fully concurrent.
   */
  async callTool(
    name: string,
    arguments_: JsonObject = {},
    options: MCPToolCallOptions = {},
  ): Promise<MCPToolCallResult> {
    const client = this.findTool(name)
    return this.callServerTool(client.config.name, client, name, arguments_, options)
  }

  /** Route a namespaced runtime tool to its exact discovery client, never a replacement. */
  async callServerTool(
    server: string,
    client: MCPClientPort,
    name: string,
    arguments_: JsonObject = {},
    options: MCPToolCallOptions = {},
  ): Promise<MCPToolCallResult> {
    const normalized = normalizeName(server)
    return this.enqueueServer(normalized, () => {
      options.signal?.throwIfAborted()
      if (this.servers.get(normalized) !== client || client.connected === false) {
        throw new Error(`MCP server '${server}' changed or disconnected; refresh the tool inventory`)
      }
      if (!client.tools.some(tool => tool.name === name)) throw new MCPCapabilityNotFoundError('tool', name)
      return client.callTool(name, arguments_, options)
    }, options.signal)
  }

  /** Route a resource read to the active server that published its URI. */
  async readResource(uri: string): Promise<MCPResourceContentsResult> {
    const client = this.findResource(uri)
    const name = normalizeName(client.config.name)
    return this.enqueueServer(name, () => {
      if (this.servers.get(name) !== client || client.connected === false) throw new Error('MCP server changed or disconnected; refresh the resource inventory')
      return client.readResource(uri)
    })
  }

  /** Route a prompt request to the first active server that published its name. */
  async getPrompt(name: string, arguments_: JsonObject = {}): Promise<MCPPromptResult> {
    const client = this.findPrompt(name)
    const server = normalizeName(client.config.name)
    return this.enqueueServer(server, () => {
      if (this.servers.get(server) !== client || client.connected === false) throw new Error('MCP server changed or disconnected; refresh the prompt inventory')
      return client.getPrompt(name, arguments_)
    })
  }

  /** Return Python-compatible per-server counts for live MCP capabilities. */
  getCapabilitiesSummary(): Record<string, MCPServerCapabilitiesSummary> {
    const summary: Record<string, MCPServerCapabilitiesSummary> = {}
    for (const [name, client] of this.servers) {
      summary[name] = {
        tools: client.tools.length,
        resources: client.resources.length,
        prompts: client.prompts.length,
      }
    }
    return summary
  }

  private async connectClient(config: MCPServerConfig): Promise<MCPClientPort> {
    const client = await this.clientFactory(config)
    try {
      await client.connect()
      client.onCapabilitiesChanged?.(kind => {
        if (this.servers.get(config.name) !== client) return
        for (const listener of [...this.capabilityListeners]) {
          try {
            listener({ name: config.name, kind })
          } catch {
            // Diagnostic capability observers must not alter transport state.
          }
        }
      })
      return client
    } catch (error) {
      try {
        await client.disconnect()
      } catch {
        // Preserve the connection failure; teardown can only add noise here.
      }
      throw error
    }
  }

  private optionsForReconnect(name: string, secrets: readonly string[], isCurrent: () => boolean, signal: AbortSignal): ReconnectWithBackoffOptions {
    const configured = this.reconnectOptions
    return {
      signal: configured?.signal ? AbortSignal.any([signal, configured.signal]) : signal,
      ...(configured?.policy === undefined ? {} : { policy: configured.policy }),
      ...(configured?.sleep === undefined ? {} : { sleep: configured.sleep }),
      onError: async (attempt, error) => {
        if (!isCurrent()) return
        this.recordFailure(name, 'reconnect', error, attempt, secrets)
        await configured?.onError?.(attempt, error)
      },
    }
  }

  private enqueueServer<T>(name: string, operation: () => Promise<T>, signal?: AbortSignal): Promise<T> {
    const cancelled = () => new DOMException('MCP operation cancelled before execution', 'AbortError')
    if (signal?.aborted) return Promise.reject(cancelled())
    const count = this.pendingServerOperations.get(name) ?? 0
    if (count >= this.maxPendingOperationsPerServer) {
      return Promise.reject(new MCPServerQueueFullError(name, this.maxPendingOperationsPerServer))
    }
    this.pendingServerOperations.set(name, count + 1)
    const previous = this.serverOperations.get(name) ?? Promise.resolve()
    let started = false
    let rejectWaiting: ((reason: unknown) => void) | undefined
    const abort = () => { if (!started) rejectWaiting?.(cancelled()) }
    const run = () => {
      started = true
      signal?.removeEventListener('abort', abort)
      if (signal?.aborted) throw cancelled()
      return operation()
    }
    const pending = previous.then(run, run)
    const settled = pending.then(
      () => undefined,
      () => undefined,
    )
    this.serverOperations.set(name, settled)
    void settled.finally(() => {
      const remaining = (this.pendingServerOperations.get(name) ?? 1) - 1
      if (remaining === 0) this.pendingServerOperations.delete(name)
      else this.pendingServerOperations.set(name, remaining)
      if (this.serverOperations.get(name) === settled) {
        this.serverOperations.delete(name)
      }
    })
    if (!signal) return pending
    // Cancel the caller's wait promptly, but retain the bounded queue slot until
    // it is skipped. Releasing it early would permit unlimited cancelled nodes.
    return new Promise<T>((resolve, reject) => {
      rejectWaiting = reject
      signal.addEventListener('abort', abort, { once: true })
      if (signal.aborted) abort()
      pending.then(value => {
        signal.removeEventListener('abort', abort)
        resolve(value)
      }, error => {
        signal.removeEventListener('abort', abort)
        reject(error)
      })
    })
  }

  private findTool(name: string): MCPClientPort {
    for (const client of this.servers.values()) {
      if (client.tools.some(tool => tool.name === name)) {
        return client
      }
    }
    throw new MCPCapabilityNotFoundError('tool', name)
  }

  private findResource(uri: string): MCPClientPort {
    for (const client of this.servers.values()) {
      if (client.resources.some(resource => resource.uri === uri)) {
        return client
      }
    }
    throw new MCPCapabilityNotFoundError('resource', uri)
  }

  private findPrompt(name: string): MCPClientPort {
    for (const client of this.servers.values()) {
      if (client.prompts.some(prompt => prompt.name === name)) {
        return client
      }
    }
    throw new MCPCapabilityNotFoundError('prompt', name)
  }

  private recordFailure(
    name: string,
    operation: MCPServerLifecycleOperation,
    error: unknown,
    attempt?: number,
    secrets: readonly string[] = [],
  ): void {
    const failure: MCPServerFailure = {
      name,
      operation,
      error: scrubCredentials(errorMessage(error), secrets),
      ...(attempt === undefined ? {} : { attempt }),
    }
    this.failures.set(name, failure)
    try {
      this.onFailure?.(failure)
    } catch {
      // Diagnostic observers must not change the lifecycle result.
    }
  }
}

function normalizeConfig(config: MCPServerConfig): MCPServerConfig {
  const name = normalizeName(config.name)
  // Each registration needs its own identity, even if a host reuses its config
  // object after removal while an older reconnect is still pending.
  return { ...config, name }
}

function normalizeName(name: string): string {
  const normalized = name.trim()
  if (!normalized) {
    throw new TypeError('MCP server name must not be empty')
  }
  return normalized
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

function positiveInteger(value: number, field: string): number {
  if (!Number.isInteger(value) || value < 1) throw new RangeError(`${field} must be a positive integer`)
  return value
}
