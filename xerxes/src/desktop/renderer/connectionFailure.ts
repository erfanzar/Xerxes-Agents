// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** A rejected RPC proves the transport worked; retrying it cannot repair its input. */
export function connectionFailureKind(message: string | null): 'transport' | 'session' | 'configuration' {
  if (/session_id.*(?:different project|another workspace)/i.test(message ?? '')) return 'session'
  if (/rpc -\d+|validation error|authentication|unauthorized|forbidden|certificate|credentials|agent-preset|configuration/i.test(message ?? '')) return 'configuration'
  return 'transport'
}
