// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, it } from 'vitest'

import { isUsableClipboardText } from '../lib/clipboard.js'

describe('native text clipboard parity', () => {
  it('accepts readable text while refusing empty and binary-like clipboard payloads', () => {
    expect(isUsableClipboardText('copied text\nwith a second line')).toBe(true)
    expect(isUsableClipboardText('')).toBe(false)
    expect(isUsableClipboardText(' \t\n')).toBe(false)
    expect(isUsableClipboardText('text\u0000payload')).toBe(false)
    expect(isUsableClipboardText('\u0001\u0002\u0003bad')).toBe(false)
  })
})
