// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { ChannelMessage } from './types.js'

/** Async callback invoked once for every normalized inbound message. */
export type InboundHandler = (message: ChannelMessage) => Promise<void>

/**
 * Transport boundary implemented by every messaging-channel adapter.
 *
 * Channels own their platform connection and relay parsed inbound messages to
 * the handler supplied at startup. The daemon only needs this contract; it
 * never needs to understand provider SDKs.
 */
export interface Channel {
  readonly name: string

  send(message: ChannelMessage): Promise<void>
  start(onInbound: InboundHandler): Promise<void>
  stop(): Promise<void>
}
