// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** Parse the daemon's rendered goal text into card fields. */
export function parseGoal(
  text: string,
): { objective: string; phase: string; rounds: string; activation: string } | null {
  if (!text || text.startsWith('No goal')) return null
  const pick = (prefix: string): string => {
    for (const line of text.split('\n')) {
      const trimmed = line.trim()
      if (trimmed.startsWith(prefix)) return trimmed.slice(prefix.length).trim()
    }
    return ''
  }
  const objective = pick('Objective:')
  if (!objective) return null
  return {
    objective,
    phase: pick('Status:'),
    rounds: pick('Rounds:'),
    activation: pick('Activation:'),
  }
}
