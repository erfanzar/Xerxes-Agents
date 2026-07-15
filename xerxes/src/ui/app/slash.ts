// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
//
// Pure slash-command classifier. The client owns a small set of commands that
// only make sense on the UI side (exit, transcript clear, clipboard, logs,
// detail toggle, queue); everything else falls through to the daemon's `slash`
// RPC so the daemon owns aliases, plugins, skills, and registry commands.
// Matches the routing model in xerxes/src/ui/PROTOCOL.md §2.

export type SlashAction =
  | { kind: 'message'; text: string }
  | { kind: 'shell'; command: string }
  | { kind: 'exit' }
  | { kind: 'clear' }
  | { kind: 'copy' }
  | { kind: 'logs' }
  | { kind: 'details'; arg: string }
  | { kind: 'queue' }
  | { kind: 'help' }
  | { kind: 'remote'; command: string }
  | { kind: 'noop' }

// Client-local command names → action. Aliases included.
const LOCAL: Record<string, (arg: string) => SlashAction> = {
  quit: () => ({ kind: 'exit' }),
  exit: () => ({ kind: 'exit' }),
  q: () => ({ kind: 'exit' }),
  clear: () => ({ kind: 'clear' }),
  copy: () => ({ kind: 'copy' }),
  logs: () => ({ kind: 'logs' }),
  details: arg => ({ kind: 'details', arg }),
  queue: () => ({ kind: 'queue' })
}

/**
 * Classify a raw input line into an action.
 *  - empty            → noop
 *  - `!cmd`           → shell
 *  - `/quit` etc.     → client-local action
 *  - `/help`          → help (rendered locally, may also query daemon)
 *  - other `/cmd`     → remote (daemon `slash`)
 *  - anything else    → message (a normal turn)
 */
export function classifyInput(raw: string): SlashAction {
  const text = raw.trim()
  if (!text) {
    return { kind: 'noop' }
  }

  // `!cmd` runs a shell command through the gateway (not queued).
  if (text.startsWith('!')) {
    const command = text.slice(1).trim()
    return command ? { kind: 'shell', command } : { kind: 'noop' }
  }

  if (!text.startsWith('/')) {
    return { kind: 'message', text }
  }

  const body = text.slice(1)
  const [nameRaw, ...rest] = body.split(/\s+/)
  const name = (nameRaw ?? '').toLowerCase()
  const arg = rest.join(' ').trim()

  if (name === 'help') {
    return { kind: 'help' }
  }

  const local = LOCAL[name]
  if (local) {
    return local(arg)
  }

  // Unknown slash → let the daemon resolve it (aliases/plugins/skills).
  return { kind: 'remote', command: text }
}

/** Commands the client resolves without touching the daemon (for completion/help). */
export const LOCAL_COMMAND_NAMES: readonly string[] = [
  'quit',
  'exit',
  'clear',
  'copy',
  'logs',
  'details',
  'queue',
  'help'
]

/** Whether a turn-style submission should be queued while the agent is busy. */
export function shouldQueue(action: SlashAction): boolean {
  // Only normal messages queue. Slash/shell/local actions run immediately even
  // mid-turn (they don't compete for the model).
  return action.kind === 'message'
}
