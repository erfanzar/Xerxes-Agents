// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, it } from 'vitest'
import { composeTabTitle, terminalActivityMarker } from '../domain/paths.js'

it('does not turn idle after cancellation, failure or normal completion into a success claim', () => {
  const title = () => composeTabTitle(terminalActivityMarker({ busy: false, waiting: false }), 'Saved task', 'selected-model', '/project')
  expect(title()).toBe('○ idle Saved task · selected-model · /project')
  expect(title()).not.toContain('✓')
})

it('keeps disconnected work distinct from busy and pending approval until connectivity returns', () => {
  const states = [
    { busy: true, waiting: false },
    { busy: true, waiting: true },
    { busy: true, waiting: true, disconnected: true },
    { busy: true, waiting: true, disconnected: false },
    { busy: true, waiting: false, disconnected: false },
    { busy: false, waiting: false, disconnected: false },
  ]
  expect(states.map(terminalActivityMarker)).toEqual(['⏳ working', '⚠ waiting', '⚠ disconnected', '⚠ waiting', '⏳ working', '○ idle'])
  expect(terminalActivityMarker({ busy: false, waiting: false, disconnected: true })).toBe('⚠ disconnected')
})
