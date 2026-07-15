// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { describe, expect, it } from 'vitest'

import { activeToken, applyCompletion, cycleIndex, shouldRequestCompletion } from '../lib/completion.js'

describe('shouldRequestCompletion', () => {
  it('triggers on a slash command being typed', () => {
    expect(shouldRequestCompletion('/prov')).toBe(true)
    expect(shouldRequestCompletion('/help now')).toBe(false) // has a space → not the name
  })
  it('triggers on path-like last tokens', () => {
    expect(shouldRequestCompletion('open ./src/ap')).toBe(true)
    expect(shouldRequestCompletion('see ~/notes')).toBe(true)
    expect(shouldRequestCompletion('look @src/x')).toBe(true)
    expect(shouldRequestCompletion('cat /etc/ho')).toBe(true)
  })
  it('does not trigger on plain prose', () => {
    expect(shouldRequestCompletion('just a sentence')).toBe(false)
    expect(shouldRequestCompletion('')).toBe(false)
  })
})

describe('activeToken', () => {
  it('is the whole draft for a slash command', () => {
    expect(activeToken('/prov')).toBe('/prov')
  })
  it('is the last whitespace token otherwise', () => {
    expect(activeToken('open ./src/ap')).toBe('./src/ap')
  })
})

describe('applyCompletion', () => {
  it('replaces the whole draft for a slash command', () => {
    expect(applyCompletion('/prov', '/provider')).toBe('/provider')
  })
  it('replaces only the trailing token for paths', () => {
    expect(applyCompletion('open ./src/ap', './src/app/')).toBe('open ./src/app/')
    expect(applyCompletion('cat @src/m', '@src/main.ts')).toBe('cat @src/main.ts')
  })
  it('appends when the draft ends with whitespace', () => {
    expect(applyCompletion('edit ', 'foo.ts')).toBe('edit foo.ts')
  })
})

describe('cycleIndex', () => {
  it('wraps in both directions', () => {
    expect(cycleIndex(0, 3, 1)).toBe(1)
    expect(cycleIndex(2, 3, 1)).toBe(0)
    expect(cycleIndex(0, 3, -1)).toBe(2)
    expect(cycleIndex(0, 0, 1)).toBe(0)
  })
})
