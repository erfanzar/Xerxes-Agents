// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { runSandboxCommand } from '../src/runtime/sandboxCommand.js'

test('sandbox status reports platform and fail-closed capabilities', async () => {
  const result = await runSandboxCommand({ action: 'status' })
  expect(result.ok).toBeTrue()
  expect(result.message).toContain('platform:')
  expect(result.message).toContain('failClosed: true')
})
