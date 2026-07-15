// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { Channel, InboundHandler } from './base.js'
import type { ChannelMessage } from './types.js'

export type ChannelLifecycleOperation = 'start' | 'stop'

/** A contained channel lifecycle failure available to daemon diagnostics. */
export interface ChannelLifecycleFailure {
  readonly channel: string
  readonly error: unknown
  readonly operation: ChannelLifecycleOperation
}

export interface ChannelRegistryOptions {
  /** Receives failures that are intentionally isolated from other channels. */
  readonly onFailure?: (failure: ChannelLifecycleFailure) => void
}

/** Raised when an outbound message names a channel that is not registered. */
export class UnknownChannelError extends Error {
  readonly channel: string

  constructor(channel: string) {
    super(`unknown channel '${channel}'`)
    this.name = new.target.name
    this.channel = channel
  }
}

/**
 * Named collection of channels with lifecycle bookkeeping.
 *
 * A failed adapter never prevents healthy adapters from starting or stopping.
 * Failures are retained for status reporting and can be forwarded to the
 * daemon's logger through `onFailure`.
 */
export class ChannelRegistry {
  private readonly channels = new Map<string, Channel>()
  private readonly failures: ChannelLifecycleFailure[] = []
  private handler: InboundHandler | undefined
  private lifecycle: Promise<void> = Promise.resolve()
  private readonly onFailure:
    ((failure: ChannelLifecycleFailure) => void) | undefined
  private readonly started = new Map<string, Channel>()

  constructor(options: ChannelRegistryOptions = {}) {
    this.onFailure = options.onFailure
  }

  /** Add or replace a channel under its daemon-facing name. */
  register(name: string, channel: Channel): void {
    this.channels.set(name, channel)
  }

  /** Remove a channel without stopping it; callers control teardown explicitly. */
  unregister(name: string): void {
    this.channels.delete(name)
    this.started.delete(name)
  }

  get(name: string): Channel | undefined {
    return this.channels.get(name)
  }

  /** Return an isolated copy of the current name-to-channel mapping. */
  all(): Map<string, Channel> {
    return new Map(this.channels)
  }

  names(): string[] {
    return [...this.channels.keys()]
  }

  startedNames(): string[] {
    return [...this.started.keys()]
  }

  isStarted(name: string): boolean {
    return this.started.has(name)
  }

  /** Install the single inbound callback shared by every registered channel. */
  setHandler(handler: InboundHandler): void {
    this.handler = handler
  }

  hasHandler(): boolean {
    return this.handler !== undefined
  }

  /** Return a snapshot of contained lifecycle failures. */
  lifecycleFailures(): readonly ChannelLifecycleFailure[] {
    return [...this.failures]
  }

  clearLifecycleFailures(name?: string): void {
    if (name === undefined) {
      this.failures.length = 0
      return
    }
    let target = 0
    for (const failure of this.failures) {
      if (failure.channel !== name) {
        this.failures[target] = failure
        target += 1
      }
    }
    this.failures.length = target
  }

  /** Start one registered channel through the shared inbound handler. */
  async start(name: string): Promise<void> {
    return this.enqueue(async () => {
      const channel = this.requireChannel(name)
      const handler = this.requireHandler('start')
      await this.startOne(name, channel, handler, true)
    })
  }

  /** Stop one registered channel without affecting other enabled adapters. */
  async stop(name: string): Promise<void> {
    return this.enqueue(async () => {
      const channel = this.requireChannel(name)
      await this.stopOne(name, channel, true)
    })
  }

  /** Start every registered channel that is not already running. */
  async startAll(): Promise<void> {
    return this.enqueue(async () => {
      const handler = this.requireHandler('startAll')
      for (const [name, channel] of [...this.channels]) {
        await this.startOne(name, channel, handler, false)
      }
    })
  }

  /** Stop every running channel while isolating teardown failures. */
  async stopAll(): Promise<void> {
    return this.enqueue(async () => {
      for (const [name, channel] of [...this.started]) {
        await this.stopOne(name, channel, false)
      }
    })
  }

  /** Route an outbound message using its normalized channel name. */
  async send(message: ChannelMessage): Promise<void> {
    const channel = this.channels.get(message.channel)
    if (!channel) {
      throw new UnknownChannelError(message.channel)
    }
    await channel.send(message)
  }

  private enqueue(operation: () => Promise<void>): Promise<void> {
    const pending = this.lifecycle.then(operation, operation)
    this.lifecycle = pending.then(
      () => undefined,
      () => undefined,
    )
    return pending
  }

  private requireChannel(name: string): Channel {
    const channel = this.channels.get(name)
    if (!channel) {
      throw new UnknownChannelError(name)
    }
    return channel
  }

  private requireHandler(operation: 'start' | 'startAll'): InboundHandler {
    if (!this.handler) {
      throw new Error(operation === 'startAll'
        ? 'ChannelRegistry.setHandler must be called before startAll()'
        : 'ChannelRegistry.setHandler must be called before starting a channel')
    }
    return this.handler
  }

  private async startOne(
    name: string,
    channel: Channel,
    handler: InboundHandler,
    propagateFailure: boolean,
  ): Promise<void> {
    const running = this.started.get(name)
    if (running === channel) {
      return
    }
    if (running) {
      try {
        await running.stop()
        this.started.delete(name)
      } catch (error) {
        this.report({ channel: name, error, operation: 'stop' })
        if (propagateFailure) {
          throw error
        }
        return
      }
    }
    try {
      await channel.start(handler)
      this.started.set(name, channel)
    } catch (error) {
      this.report({ channel: name, error, operation: 'start' })
      if (propagateFailure) {
        throw error
      }
    }
  }

  private async stopOne(name: string, channel: Channel, propagateFailure: boolean): Promise<void> {
    try {
      await channel.stop()
    } catch (error) {
      this.report({ channel: name, error, operation: 'stop' })
      if (propagateFailure) {
        throw error
      }
    } finally {
      this.started.delete(name)
    }
  }

  private report(failure: ChannelLifecycleFailure): void {
    this.failures.push(failure)
    if (!this.onFailure) {
      return
    }
    try {
      this.onFailure(failure)
    } catch {
      // Reporting must not make an isolated adapter failure abort the batch.
    }
  }
}

/** Start multiple registries concurrently, matching the daemon's inbound fan-in. */
export async function gatherInbound(
  ...registries: readonly ChannelRegistry[]
): Promise<void> {
  await Promise.all(registries.map((registry) => registry.startAll()))
}
