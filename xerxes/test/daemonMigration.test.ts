// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { releaseIdleProjectDaemon } from '../src/ui/lib/daemonMigration.js'

test('migration preserves busy daemons and waits for acknowledged idle shutdown', async () => {
  const calls: string[] = []
  expect(await releaseIdleProjectDaemon(async method => {
    calls.push(method)
    return method === 'runtime.status' ? { pid: 123 } : { ok: false, busy: true }
  }, () => { throw new Error('must not wait for or signal a busy process') })).toBe(false)
  expect(calls).toEqual(['runtime.status', 'runtime.restart_if_idle'])
  expect(await releaseIdleProjectDaemon(async method => method === 'runtime.status' ? { pid: 123 } : { ok: true }, () => false)).toBe(true)
})

test('legacy capability absence preserves ownership; other failures remain observable', async () => {
  expect(await releaseIdleProjectDaemon(async method => {
    if(method === 'runtime.status') return { pid: 123 }
    throw new Error('Unknown method runtime.restart_if_idle')
  })).toBe(false)
  await expect(releaseIdleProjectDaemon(async () => { throw new Error('authentication failed') })).rejects.toThrow('authentication failed')
  await expect(releaseIdleProjectDaemon(async method => method === 'runtime.status' ? { pid: 123 } : { ok: false, error: 'configuration invalid' })).rejects.toThrow('configuration invalid')
})
