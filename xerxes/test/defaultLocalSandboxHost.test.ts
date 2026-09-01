// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { defaultLocalSandboxHost } from '../src/security/defaultLocalSandboxHost.js'

test('default local sandbox host reports platform and probe results', async () => {
  expect(defaultLocalSandboxHost.platform).toBe(process.platform)

  const seatbelt = await defaultLocalSandboxHost.probe('seatbelt')
  expect(typeof seatbelt.available).toBe('boolean')
  expect(seatbelt.enforcement).toMatch(/^(full|partial|none)$/)
  expect(typeof seatbelt.filesystemIsolation).toBe('boolean')
  expect(typeof seatbelt.networkIsolation).toBe('boolean')
  expect(typeof seatbelt.processIsolation).toBe('boolean')
  expect(Array.isArray(seatbelt.limitations)).toBeTrue()
})

test('default local sandbox host execute fails closed without an injected executor', async () => {
  await expect(
    defaultLocalSandboxHost.execute(
      'bubblewrap',
      { arguments: {}, toolName: 'echo', context: { agentId: 'test', metadata: {} } },
      { environment: {}, memoryLimitMb: 128, mountPaths: {}, mountReadonly: true, networkAccess: false, timeoutMs: 5_000 },
    ),
  ).rejects.toThrow('Default host does not execute commands')
})
