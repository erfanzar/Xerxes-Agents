// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { ABILITY_SPECS, validateAbilityInput } from '../src/hub/supervisor/abilities/catalog.js'

test('ability schemas declare safety tiers separately from execution and reject extra authority fields', () => {
  expect(ABILITY_SPECS.find(spec => spec.providerName === 'workspace_read')?.tier).toBe('A')
  expect(ABILITY_SPECS.find(spec => spec.providerName === 'schedule_wakeup')?.tier).toBe('B')
  expect(validateAbilityInput('workspace.read', { workspace_id: 'w', path: 'src/a.ts' }).ok).toBe(true)
  expect(validateAbilityInput('workspace.read', { workspace_id: 'w', path: '../secret', authorized: true }).ok).toBe(false)
  expect(validateAbilityInput('ui.screenshot', { workspace_id: 'w', width: 999 }).ok).toBe(false)
  expect(validateAbilityInput('notebook.put', { workspace_id: 'w', key: 'plan', kind: 'plan', text: 'notes' }).ok).toBe(true)
})
