// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
//
// Pure queue + interrupt semantics (mirrors the rules documented in Xerxes'
// TUI README): text typed while the agent is busy is queued; the queue drains
// after each turn; an empty Enter interrupts a running turn or sends the next
// queued item when idle.

export type SubmitDecision =
  | { kind: 'send'; text: string } // submit now as a turn
  | { kind: 'queue'; text: string } // hold until the current turn ends
  | { kind: 'interrupt' } // empty Enter while busy with a backlog
  | { kind: 'drain' } // empty Enter while idle with a backlog → send next
  | { kind: 'noop' }

/**
 * Decide what an Enter press means, given the trimmed draft, whether the agent
 * is mid-turn, and how many items are queued. Slash/shell commands bypass this
 * (they run immediately); this governs plain messages and the empty-Enter
 * interrupt/drain gesture.
 */
export function decideSubmit(draft: string, busy: boolean, queueLen: number): SubmitDecision {
  const text = draft.trim()
  if (text) {
    return busy ? { kind: 'queue', text } : { kind: 'send', text }
  }
  if (queueLen > 0) {
    return busy ? { kind: 'interrupt' } : { kind: 'drain' }
  }
  return { kind: 'noop' }
}

// Immutable queue ops — a plain string[] is the state.

export function enqueue(queue: readonly string[], text: string): string[] {
  const t = text.trim()
  return t ? [...queue, t] : [...queue]
}

export function dequeue(queue: readonly string[]): { next: string | undefined; rest: string[] } {
  if (queue.length === 0) {
    return { next: undefined, rest: [] }
  }
  const [next, ...rest] = queue
  return { next, rest }
}

/** Replace the most recently queued item (used when editing a queued draft). */
export function replaceLast(queue: readonly string[], text: string): string[] {
  if (queue.length === 0) {
    return enqueue(queue, text)
  }
  const copy = queue.slice(0, -1)
  return enqueue(copy, text)
}
