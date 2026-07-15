// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { describe, expect, it } from 'vitest'

import {
  appendQueuedMessage,
  queuedMessage,
  queuedMessageDisplays,
  queuedUserMessage,
  shiftQueuedMessage
} from './queuedMessage.js'

describe('queuedMessage', () => {
  it('keeps the authored user text separate from the expanded daemon payload', () => {
    const followUp = queuedMessage('review [Pasted 12 lines]', 'review\nconst answer = 42')

    expect(followUp).toEqual({
      displayText: 'review [Pasted 12 lines]',
      submitText: 'review\nconst answer = 42'
    })
    expect(queuedMessageDisplays([followUp])).toEqual(['review [Pasted 12 lines]'])
  })

  it('replaces a busy preview with exactly one ordinary user message when dispatched', () => {
    const followUp = queuedMessage('second request', 'second request')
    const busyQueue = appendQueuedMessage([], followUp)

    // While busy, the message exists only in the composer preview.
    expect(queuedMessageDisplays(busyQueue)).toEqual(['second request'])

    const { message, rest } = shiftQueuedMessage(busyQueue)
    const transcript = message ? [queuedUserMessage(message)] : []

    expect(rest).toEqual([])
    expect(transcript).toEqual([{ role: 'user', text: 'second request' }])
    expect(transcript.filter(item => item.role === 'user')).toHaveLength(1)
  })
})
