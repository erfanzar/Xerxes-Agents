// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtempSync, rmSync } from 'node:fs'
import { join } from 'node:path'
import { tmpdir } from 'node:os'

import {
  DEFAULT_IDENTITY_SALT,
  IdentityResolver,
  ResetTrigger,
  StickerCache,
  createSessionResetPolicy,
  hashChat,
  hashUser,
  matchesUser,
  shouldReset,
} from '../src/channels/index.js'
import { SimpleStorage } from '../src/memory/storage.js'

test('identity resolver creates durable aliases and preserves the first known display name', () => {
  const storage = new SimpleStorage()
  const first = new IdentityResolver({
    storage,
    clock: () => new Date('2026-07-13T08:00:00.000Z'),
    userIdFactory: () => 'global-user-1',
  })
  const telegram = first.resolve('telegram', '123', 'Alice')
  const again = first.resolve('telegram', '123', 'Renamed Alice')
  first.link(telegram.userId, 'slack', 'U42')

  expect(again).toEqual(telegram)
  expect(first.channelsFor(telegram.userId).map(record => record.channel)).toEqual(['telegram', 'slack'])
  expect(storage.load('_identity_telegram::123')).toEqual({
    user_id: 'global-user-1',
    channel: 'telegram',
    channel_user_id: '123',
    display_name: 'Alice',
    first_seen: '2026-07-13T08:00:00.000Z',
  })

  const restored = new IdentityResolver({ storage })
  expect(restored.get('telegram', '123')).toEqual(telegram)
  expect(restored.get('slack', 'U42')?.userId).toBe('global-user-1')
})

test('identity hashing is salt-aware, platform-scoped, and safely comparable', () => {
  const options = { salt: 'test-salt' }
  const telegram = hashUser('telegram', 99, options)

  expect(telegram).toMatch(/^user_[a-f0-9]{16}$/)
  expect(hashUser('telegram', 99, options)).toBe(telegram)
  expect(hashUser('discord', 99, options)).not.toBe(telegram)
  expect(hashChat('slack', 'C1', options)).toMatch(/^slack:[a-f0-9]{16}$/)
  expect(matchesUser('telegram', 99, telegram, options)).toBe(true)
  expect(matchesUser('telegram', 100, telegram, options)).toBe(false)
  expect(DEFAULT_IDENTITY_SALT).toContain('xerxes-default')
})

test('session reset policy preserves manual, count, and timeout behavior', () => {
  const manual = createSessionResetPolicy()
  expect(shouldReset(manual, { messageCount: 10_000 })).toBe(false)
  expect(shouldReset(manual, { messageCount: 0, manualRequest: true })).toBe(true)

  const count = createSessionResetPolicy({ trigger: ResetTrigger.MESSAGE_COUNT, messageCount: 5 })
  expect(shouldReset(count, { messageCount: 4 })).toBe(false)
  expect(shouldReset(count, { messageCount: 5 })).toBe(true)
  expect(createSessionResetPolicy({ trigger: ResetTrigger.MSG_COUNT, msg_count: 3, timeout_minutes: 12 }))
    .toEqual({ trigger: 'msg_count', messageCount: 3, timeoutMinutes: 12 })

  const timeout = createSessionResetPolicy({ trigger: ResetTrigger.TIMEOUT, timeoutMinutes: 30 })
  const now = new Date('2026-07-13T08:00:00.000Z')
  expect(shouldReset(timeout, {
    messageCount: 0,
    lastMessageAt: new Date('2026-07-13T07:29:00.000Z'),
    now,
  })).toBe(true)
  expect(shouldReset(timeout, {
    messageCount: 0,
    lastMessageAt: new Date('2026-07-13T07:30:00.000Z'),
    now,
  })).toBe(false)
})

test('sticker cache persists a bounded LRU index without deleting media files', () => {
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-sticker-cache-'))
  try {
    let now = 100
    const cache = new StickerCache(directory, { clock: () => now, lruSize: 2 })
    cache.put('telegram', 'one', join(directory, 'one.webp'))
    now += 1
    cache.put('telegram', 'two', join(directory, 'two.webp'))
    expect(cache.get('telegram', 'one')?.fetchedAt).toBe(100)
    now += 1
    cache.put('telegram', 'three', join(directory, 'three.webp'))

    expect(cache.get('telegram', 'two')).toBeUndefined()
    expect(cache.get('telegram', 'one')?.localPath).toContain('one.webp')
    expect(cache.get('telegram', 'three')?.fetchedAt).toBe(102)

    const restored = new StickerCache(directory, { lruSize: 2 })
    expect(restored.size()).toBe(2)
    expect(restored.get('telegram', 'one')).toBeDefined()
    restored.clear()
    expect(restored.size()).toBe(0)
  } finally {
    rmSync(directory, { force: true, recursive: true })
  }
})
