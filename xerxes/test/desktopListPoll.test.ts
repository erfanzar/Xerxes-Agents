// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { createListPoll } from '../src/desktop/renderer/listPoll.js'

test('a poll tick while a slow list call runs is skipped, and the slow answer still lands', async () => {
  const pending: ((rows: string[]) => void)[] = []
  const applied: string[][] = []
  const poll = createListPoll<string[]>(() => new Promise(resolve => pending.push(resolve)), rows => applied.push(rows), () => {})
  expect(poll.request('s1')).toBe(true)
  // Two more 5 s ticks before the first reply: neither cancels nor duplicates it.
  expect(poll.request('s1')).toBe(false)
  expect(poll.request('s1')).toBe(false)
  expect(pending.length).toBe(1)
  pending[0]!(['plan', 'report'])
  await Bun.sleep(0)
  expect(applied).toEqual([['plan', 'report']])
  // Once settled, the next tick fetches again.
  expect(poll.request('s1')).toBe(true)
})

test('a reply for a session the panel has left is dropped', async () => {
  const pending: ((rows: string[]) => void)[] = []
  const applied: string[][] = []
  const poll = createListPoll<string[]>(() => new Promise(resolve => pending.push(resolve)), rows => applied.push(rows), () => {})
  poll.request('old')
  poll.request('new')
  pending[0]!(['old-row'])
  pending[1]!(['new-row'])
  await Bun.sleep(0)
  expect(applied).toEqual([['new-row']])
})
