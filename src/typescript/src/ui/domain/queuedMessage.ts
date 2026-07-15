// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { Msg } from '../types.js'

/**
 * A follow-up keeps the text shown to the user separate from the payload sent
 * to the daemon. This mirrors Grok's queue model and matters for collapsed
 * paste blocks and shell interpolation: the transcript must show what the
 * user authored, while the provider receives the expanded value.
 */
export interface QueuedMessage {
  displayText: string
  submitText: string
}

export const queuedMessage = (displayText: string, submitText = displayText): QueuedMessage => ({
  displayText,
  submitText
})

export const queuedMessageDisplays = (queue: readonly QueuedMessage[]): string[] =>
  queue.map(message => message.displayText)

export const appendQueuedMessage = (queue: readonly QueuedMessage[], message: QueuedMessage): QueuedMessage[] => [
  ...queue,
  message
]

export const shiftQueuedMessage = (
  queue: readonly QueuedMessage[]
): { message: QueuedMessage | undefined; rest: QueuedMessage[] } => {
  const [message, ...rest] = queue

  return { message, rest }
}

/** The one canonical transcript shape for normal and mid-turn user input. */
export const queuedUserMessage = (message: QueuedMessage): Msg => ({
  role: 'user',
  text: message.displayText
})
