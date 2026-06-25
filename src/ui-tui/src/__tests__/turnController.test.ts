// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { afterEach, describe, expect, it } from 'vitest'

import { getTurnState } from '../app/turnStore.js'
import { turnController } from '../app/turnController.js'

describe('turnController', () => {
  afterEach(() => {
    turnController.fullReset()
  })

  it('keeps TodoWriteTool state pinned instead of archiving it into the transcript', () => {
    const todos = [{ content: 'verify the fix', id: '1', status: 'completed' as const }]

    turnController.fullReset()
    turnController.recordTodos(todos)

    const result = turnController.recordMessageComplete({ text: 'Done.' })

    expect(result.finalMessages.some(msg => msg.kind === 'trail' && Boolean(msg.todos?.length))).toBe(false)
    expect(result.finalMessages).toEqual([{ role: 'assistant', text: 'Done.' }])
    expect(getTurnState().todos).toEqual(todos)
  })
})
