// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { Channel, InboundHandler } from './base.js'
import type { ChannelMessage } from './types.js'
import {
  ChannelRegistry,
  UnknownChannelError,
  type ChannelLifecycleFailure,
  type ChannelLifecycleOperation,
} from './registry.js'

export interface ManagedChannelStatus {
  readonly adapterName: string
  readonly enabled: boolean
  readonly lastError?: string
  readonly lastOperation?: ChannelLifecycleOperation
  readonly name: string
}

export interface ChannelManagerOptions {
  /** Named adapters owned and configured by the embedding host. */
  readonly channels?: Iterable<readonly [string, Channel]>
  /** The host decides how inbound platform messages enter agent sessions. */
  readonly onInbound?: InboundHandler
  /** Reuse an existing registry when the host owns broader channel wiring. */
  readonly registry?: ChannelRegistry
}

/** Raised when a daemon operation names an adapter absent from host configuration. */
export class ChannelNotConfiguredError extends Error {
  readonly channel: string

  constructor(channel: string) {
    super("channel '" + channel + "' is not configured")
    this.name = new.target.name
    this.channel = channel
  }
}

/** Raised when adapters exist but their host has not supplied an inbound route. */
export class ChannelInboundHandlerUnavailableError extends Error {
  constructor() {
    super('channel inbound handler is not configured by the host')
    this.name = new.target.name
  }
}

/**
 * Host-owned channel lifecycle boundary for the Bun daemon.
 *
 * It intentionally owns no credentials, transports, or fallback adapters.
 * A channel becomes usable only after its embedding host registers a concrete
 * adapter and supplies a real inbound route.
 */
export class ChannelManager {
  readonly registry: ChannelRegistry
  private inboundConfigured: boolean

  constructor(options: ChannelManagerOptions = {}) {
    this.registry = options.registry ?? new ChannelRegistry()
    this.inboundConfigured = options.onInbound !== undefined || this.registry.hasHandler()
    if (options.onInbound !== undefined) {
      this.registry.setHandler(options.onInbound)
    }
    for (const [name, channel] of options.channels ?? []) {
      this.register(name, channel)
    }
  }

  get hasConfiguredChannels(): boolean {
    return this.registry.names().length > 0
  }

  /** Install or replace the host callback used by subsequently enabled adapters. */
  setInboundHandler(handler: InboundHandler): void {
    this.registry.setHandler(handler)
    this.inboundConfigured = true
  }

  /** Register an adapter without starting it. Lifecycle changes are daemon RPC-driven. */
  register(name: string, channel: Channel): void {
    const normalized = normalizeName(name)
    this.registry.register(normalized, channel)
  }

  /** Return JSON-safe lifecycle facts without exposing adapter credentials. */
  list(): readonly ManagedChannelStatus[] {
    return this.registry.names()
      .sort((left, right) => left.localeCompare(right))
      .flatMap(name => {
        const status = this.status(name)
        return status === undefined ? [] : [status]
      })
  }

  status(name: string): ManagedChannelStatus | undefined {
    const normalized = name.trim()
    const channel = this.registry.get(normalized)
    if (!channel) {
      return undefined
    }
    const failure = latestFailure(this.registry.lifecycleFailures(), normalized)
    return {
      name: normalized,
      adapterName: channel.name,
      enabled: this.registry.isStarted(normalized),
      ...(failure === undefined
        ? {}
        : {
            lastOperation: failure.operation,
            lastError: errorMessage(failure.error),
          }),
    }
  }

  /** Enable exactly one configured adapter. It never synthesizes a fallback transport. */
  async enable(name: string): Promise<ManagedChannelStatus> {
    const normalized = this.requireConfigured(name)
    if (!this.inboundConfigured) {
      throw new ChannelInboundHandlerUnavailableError()
    }
    await this.registry.start(normalized)
    this.registry.clearLifecycleFailures(normalized)
    return this.requireStatus(normalized)
  }

  /** Disable exactly one configured adapter. Disabled adapters remain configured and listable. */
  async disable(name: string): Promise<ManagedChannelStatus> {
    const normalized = this.requireConfigured(name)
    if (this.registry.isStarted(normalized)) {
      await this.registry.stop(normalized)
      this.registry.clearLifecycleFailures(normalized)
    }
    return this.requireStatus(normalized)
  }

  /** Stop all active adapters while retaining host configuration for a later daemon start. */
  async stopAll(): Promise<void> {
    await this.registry.stopAll()
  }

  /** Deliver one concrete outbound message through a host-configured adapter. */
  async send(message: ChannelMessage): Promise<void> {
    await this.registry.send(message)
  }

  private requireConfigured(name: string): string {
    const normalized = name.trim()
    if (!normalized || !this.registry.get(normalized)) {
      throw new ChannelNotConfiguredError(normalized || name)
    }
    return normalized
  }

  private requireStatus(name: string): ManagedChannelStatus {
    const status = this.status(name)
    if (status === undefined) {
      throw new UnknownChannelError(name)
    }
    return status
  }
}

function latestFailure(
  failures: readonly ChannelLifecycleFailure[],
  name: string,
): ChannelLifecycleFailure | undefined {
  for (let index = failures.length - 1; index >= 0; index -= 1) {
    const failure = failures[index]
    if (failure?.channel === name) {
      return failure
    }
  }
  return undefined
}

function normalizeName(name: string): string {
  const normalized = name.trim()
  if (!normalized) {
    throw new TypeError('channel name must not be empty')
  }
  return normalized
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}
