// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Per-client state shared by the daemon's Unix and WebSocket transports.
 *
 * A transport owns serialization and delivery. The daemon only changes the
 * selected session and emits JSON-RPC frames through this narrow interface.
 */
export interface DaemonTransportConnection {
  activeSessionKey: string
  send(frame: object): void
}
