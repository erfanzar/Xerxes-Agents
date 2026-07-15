// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
export interface ExitHintInput {
  aliasCommand?: string
  bunCommand?: string
  sessionId?: null | string
}

export function formatExitHint({
  aliasCommand = 'xerxes',
  bunCommand = 'bun run xerxes',
  sessionId
}: ExitHintInput): string {
  const lines = ['Goodbye.']
  const id = sessionId?.trim()

  if (id) {
    lines.push(`Resume this session: ${aliasCommand} -r ${id}`)
    lines.push(`or: ${bunCommand} -r ${id}`)
  } else {
    lines.push(`Resume a saved session: ${aliasCommand} -r <session-id>`)
    lines.push(`or start again: ${bunCommand}`)
  }

  return lines.join('\n')
}
