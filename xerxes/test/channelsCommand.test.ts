// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { runChannelsCommand } from '../src/runtime/channelsCommand.js'

test('channels list returns supported channels', async () => {
  const result = await runChannelsCommand('list')
  expect(result.ok).toBeTrue()
  expect(result.message).toContain('telegram')
  expect(result.message).toContain('discord')
  expect(result.message).toContain('slack')
})
