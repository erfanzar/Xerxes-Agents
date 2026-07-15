// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { describe, expect, it } from 'vitest'

import { compactProgressRows } from './progressRows.js'

describe('compactProgressRows', () => {
  const state = {
    activity: [
      { id: 1, text: 'ambient info', tone: 'info' as const },
      { id: 2, text: 'permission warning', tone: 'warn' as const }
    ],
    outcome: 'approved (session)',
    todos: [
      { content: 'inspect renderer', id: '1', status: 'completed' as const },
      { content: 'run verification', id: '2', status: 'in_progress' as const }
    ],
    turnTrail: ['drafting tool…', 'analyzing tool output…', 'latest status']
  }

  it('keeps progress compact and bounded while retaining current work', () => {
    expect(compactProgressRows(state, { activityVisible: true, toolsVisible: true })).toEqual([
      { kind: 'todo', text: 'Tasks 1/2 · run verification', tone: 'info' },
      { kind: 'trail', text: 'analyzing tool output…', tone: 'info' },
      { kind: 'trail', text: 'latest status', tone: 'info' },
      { kind: 'activity', text: 'ambient info', tone: 'info' },
      { kind: 'activity', text: 'permission warning', tone: 'warn' },
      { kind: 'outcome', text: 'approved (session)', tone: 'success' }
    ])
  })

  it('retains warnings as a backstop when ordinary detail sections are hidden', () => {
    expect(compactProgressRows(state, { activityVisible: false, toolsVisible: false })).toEqual([
      { kind: 'todo', text: 'Tasks 1/2 · run verification', tone: 'info' },
      { kind: 'activity', text: 'permission warning', tone: 'warn' },
      { kind: 'outcome', text: 'approved (session)', tone: 'success' }
    ])
  })
})
