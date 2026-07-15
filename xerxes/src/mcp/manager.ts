// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { JsonObject } from '../types/toolCalls.js'
import { MCPClient } from './client.js'
import {
  MCPReconnectError,
  reconnectWithBackoff,
  scrubCredentials,
  type ReconnectWithBackoffOptions,
} from './reconnect.js'
import type {
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
  callTool(name: string, arguments_?: JsonObject): Promise<MCPToolCallResult>
  readResource(uri: string): Promise<MCPResourceContentsResult>
  getPrompt(name: string, arguments_?: JsonObject): Promise<MCPPromptResult>
}

/** Factory boundary for hosts that own MCP transports or authentication. */
export type MCPClientFactory = (config: MCPServerConfig) => MCPClientPort | Promise<MCPClientPort>

export type MCPServerLifecycleOperation = 'connect' | 'disconnect' | 'reconnect'

/** A redacted lifecycle failure retained for diagnostics. */
export interface MCPServerFailure {
  readonly attempt?: number
  readonly error: string
  readonly name: string
  readonly operation: MCPServerLifecycleOperation
}

export interface MCPServerStatus {
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

export interface MCPManagerOptions {
  /** Creates one connected-client candidate for each start or reconnect attempt. */
  readonly clientFactory?: MCPClientFactory
  /** Receives already-redacted lifecycle failures. Observer errors are contained. */
  readonly onFailure?: (failure: MCPServerFailure) => void
  /** Retry policy and deterministic hooks used by reconnect. */
  readonly reconnect?: ReconnectWithBackoffOptions
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
  private readonly clientFactory: MCPClientFactory
  private readonly failures = new Map<string, MCPServerFailure>()
  private lifecycle: Promise<void> = Promise.resolve()
  private readonly onFailure: ((failure: MCPServerFailure) => void) | undefined
  private readonly reconnectOptions: ReconnectWithBackoffOptions | undefined
  private readonly servers = new Map<string, MCPClientPort>()

  constructor(options: MCPManagerOptions = {}) {
    this.clientFactory = options.clientFactory ?? (config => new MCPClient(config))
    this.onFailure = options.onFailure
    this.reconnectOptions = options.reconnect
  }

  /** Build, connect, and register a server. Disabled or duplicate configurations are skipped. */
  addServer(config: MCPServerConfig): Promise<boolean> {
    const normalized = normalizeConfig(config)
    return this.enqueue(async () => {
      if (normalized.enabled === false || this.servers.has(normalized.name)) {
        return false
      }
      try {
        const client = await this.connectClient(normalized)
        this.servers.set(normalized.name, client)
        this.failures.delete(normalized.name)
        return true
      } catch (error) {
        this.recordFailure(normalized.name, 'connect', error)
        return false
      }
    })
  }

  /** Semantic lifecycle alias for hosts that model MCP server registration as startup. */
  start(config: MCPServerConfig): Promise<boolean> {
    return this.addServer(config)
  }

  /**
   * Disconnect and drop one server. A teardown failure is retained for
   * diagnostics, but the server is removed so stale tools cannot be routed.
   */
  removeServer(name: string): Promise<boolean> {
    const normalized = normalizeName(name)
    return this.enqueue(async () => {
      const client = this.servers.get(normalized)
      if (!client) {
        return false
      }
      this.servers.delete(normalized)
      try {
        await client.disconnect()
        this.failures.delete(normalized)
      } catch (error) {
        this.recordFailure(normalized, 'disconnect', error)
      }
      return true
    })
  }

  /** Semantic lifecycle alias for removing one active MCP server. */
  stop(name: string): Promise<boolean> {
    return this.removeServer(name)
  }

  /**
   * Replace an active server with a fresh client candidate, retrying failed
   * connection attempts according to the configured backoff policy.
   */
  reconnect(name: string): Promise<boolean> {
    const normalized = normalizeName(name)
    return this.enqueue(async () => {
      const previous = this.servers.get(normalized)
      if (!previous) {
        return false
      }
      const config = previous.config
      this.servers.delete(normalized)
      try {
        await previous.disconnect()
      } catch (error) {
        this.recordFailure(normalized, 'disconnect', error)
      }

      try {
        const client = await reconnectWithBackoff(
          () => this.connectClient(config),
          this.optionsForReconnect(normalized),
        )
        this.servers.set(normalized, client)
        this.failures.delete(normalized)
        return true
      } catch (error) {
        this.recordFailure(
          normalized,
          'reconnect',
          error,
          error instanceof MCPReconnectError ? error.attempts : undefined,
        )
        return false
      }
    })
  }

  /** Disconnect every server and clear the active capability registry. */
  disconnectAll(): Promise<void> {
    return this.enqueue(async () => {
      const servers = [...this.servers.entries()]
      this.servers.clear()
      for (const [name, client] of servers) {
        try {
          await client.disconnect()
          this.failures.delete(name)
        } catch (error) {
          this.recordFailure(name, 'disconnect', error)
        }
      }
    })
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

  /** Return lifecycle status without exposing launch arguments, headers, or environment values. */
  status(name: string): MCPServerStatus | undefined {
    const normalized = normalizeName(name)
    const client = this.servers.get(normalized)
    if (!client) {
      return undefined
    }
    const failure = this.failures.get(normalized)
    return {
      name: normalized,
      connected: client.connected ?? true,
      tools: client.tools.length,
      resources: client.resources.length,
      prompts: client.prompts.length,
      ...(failure === undefined ? {} : { lastError: failure.error }),
    }
  }

  /** Return statuses for every active server in registration order. */
  listStatus(): MCPServerStatus[] {
    return this.listServers().flatMap(name => {
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

  /** Route a tool call to the first active server that published its name. */
  async callTool(name: string, arguments_: JsonObject = {}): Promise<MCPToolCallResult> {
    const client = this.findTool(name)
    return client.callTool(name, arguments_)
  }

  /** Route a resource read to the active server that published its URI. */
  async readResource(uri: string): Promise<MCPResourceContentsResult> {
    const client = this.findResource(uri)
    return client.readResource(uri)
  }

  /** Route a prompt request to the first active server that published its name. */
  async getPrompt(name: string, arguments_: JsonObject = {}): Promise<MCPPromptResult> {
    const client = this.findPrompt(name)
    return client.getPrompt(name, arguments_)
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

  private optionsForReconnect(name: string): ReconnectWithBackoffOptions {
    const configured = this.reconnectOptions
    return {
      ...(configured?.policy === undefined ? {} : { policy: configured.policy }),
      ...(configured?.sleep === undefined ? {} : { sleep: configured.sleep }),
      onError: async (attempt, error) => {
        this.recordFailure(name, 'reconnect', error, attempt)
        await configured?.onError?.(attempt, error)
      },
    }
  }

  private enqueue<T>(operation: () => Promise<T>): Promise<T> {
    const pending = this.lifecycle.then(operation, operation)
    this.lifecycle = pending.then(
      () => undefined,
      () => undefined,
    )
    return pending
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
  ): void {
    const failure: MCPServerFailure = {
      name,
      operation,
      error: scrubCredentials(errorMessage(error)),
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
  return name === config.name ? config : { ...config, name }
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
