// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { mkdtempSync, readFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { decideSubmit, dequeue, enqueue, replaceLast } from '../app/queue.js'
import { findSlashCommand } from '../app/slash/registry.js'
import { appendHistory, HistoryCursor, loadHistory } from '../lib/history.js'

describe('queue ops + decideSubmit', () => {
  it('enqueue trims and skips empties', () => {
    expect(enqueue(['a'], '  b ')).toEqual(['a', 'b'])
    expect(enqueue(['a'], '   ')).toEqual(['a'])
  })
  it('dequeue pulls the oldest', () => {
    expect(dequeue(['a', 'b'])).toEqual({ next: 'a', rest: ['b'] })
    expect(dequeue([])).toEqual({ next: undefined, rest: [] })
  })
  it('replaceLast swaps the newest', () => {
    expect(replaceLast(['a', 'b'], 'c')).toEqual(['a', 'c'])
  })
  it('decides send/queue/interrupt/drain', () => {
    expect(decideSubmit('hi', false, 0)).toEqual({ kind: 'send', text: 'hi' })
    expect(decideSubmit('hi', true, 0)).toEqual({ kind: 'queue', text: 'hi' })
    expect(decideSubmit('', true, 2).kind).toBe('interrupt')
    expect(decideSubmit('', false, 2).kind).toBe('drain')
    expect(decideSubmit('', false, 0).kind).toBe('noop')
  })
})

describe('slash registry', () => {
  it('/compact is not a client-local display toggle', () => {
    expect(findSlashCommand('compact')).toBeUndefined()
    expect(findSlashCommand('ui-compact')?.help).toBe('toggle compact transcript display')
  })
})

describe('input history', () => {
  let dir: string
  let path: string
  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), 'xerxes-hist-'))
    path = join(dir, '.tui_history')
  })
  afterEach(() => rmSync(dir, { recursive: true, force: true }))

  it('appends and loads, flattening newlines', () => {
    appendHistory('first', path)
    appendHistory('second\nline', path)
    expect(loadHistory(path)).toEqual(['first', 'second line'])
    expect(readFileSync(path, 'utf8')).toContain('second line')
  })
  it('cursor walks prev/next and returns to the live draft', () => {
    const cur = new HistoryCursor(['a', 'b', 'c'])
    expect(cur.atLive()).toBe(true)
    expect(cur.prev()).toBe('c')
    expect(cur.prev()).toBe('b')
    expect(cur.next()).toBe('c')
    expect(cur.next()).toBe('') // back on the live draft
    expect(cur.atLive()).toBe(true)
  })
  it('push de-dupes consecutive repeats and resets the cursor', () => {
    const cur = new HistoryCursor(['a'])
    cur.push('a') // dup, ignored
    cur.push('b')
    expect(cur.prev()).toBe('b')
    expect(cur.prev()).toBe('a')
  })
})

// keep vi import used even if a future test stubs timers
vi.useRealTimers?.()
