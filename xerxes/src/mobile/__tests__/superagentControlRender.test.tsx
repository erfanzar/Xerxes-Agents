// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, it } from 'vitest'
import { liveState } from './helpers.js'
import { render } from './renderHelpers.js'

it('the phone offers Safety only to the Hub owner', () => {
  const initial = liveState()
  const owner = { ...initial, autopilot: { ...initial.autopilot, canEdit: true } }
  expect(render(owner, '/xerxes')).toContain('>Safety</button>')
  expect(render({ ...initial, autopilot: { ...initial.autopilot, canEdit: false }, watch: { ...initial.watch, canEdit: false } }, '/xerxes')).not.toContain('>Safety</button>')
})

it('emergency Safety stays available when the supervisor thread is unavailable', () => {
  const initial = liveState()
  const state = { ...initial, autopilot: { ...initial.autopilot, canEdit: true }, supervisor: { ...initial.supervisor, phase: 'error' as const, error: 'thread unavailable' } }
  expect(render(state, '/xerxes')).toContain('>Safety</button>')
})

it('the phone copy never claims XERXES decides human permissions', () => {
  const html = render(liveState(), '/xerxes')
  expect(html).toContain('Permissions stay human-only')
  expect(html).not.toContain('decides permission requests')
})
