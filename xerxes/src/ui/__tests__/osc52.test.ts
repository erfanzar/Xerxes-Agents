// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { describe, expect, it, vi } from 'vitest'

import type { CliRenderer } from '@opentui/core'

import { clearActiveRenderer, setActiveRenderer } from '../opentui/rendererSingleton.js'
import { inTmux, osc52Copy, osc52Sequence, writeOsc52Clipboard } from '../lib/osc52.js'

describe('osc52', () => {
  it('base64-encodes into an OSC 52 sequence', () => {
    const seq = osc52Sequence('hi')
    expect(seq).toBe('\x1b]52;c;aGk=\x07') // base64('hi') === 'aGk='
  })
  it('wraps for tmux passthrough', () => {
    const seq = osc52Sequence('hi', true)
    expect(seq.startsWith('\x1bPtmux;')).toBe(true)
    expect(seq.endsWith('\x1b\\')).toBe(true)
  })
  it('writes via the injected writer, skips empty', () => {
    let out = ''
    expect(osc52Copy('yo', s => (out += s))).toBe(true)
    expect(out).toContain('52;c;')
    out = ''
    expect(osc52Copy('', s => (out += s))).toBe(false)
    expect(out).toBe('')
  })
  it('routes live TUI clipboard writes through the native renderer', () => {
    const copyToClipboardOSC52 = vi.fn(() => true)
    const renderer = { copyToClipboardOSC52 } as unknown as CliRenderer

    setActiveRenderer(renderer)

    try {
      expect(writeOsc52Clipboard('native')).toBe(true)
      expect(copyToClipboardOSC52).toHaveBeenCalledWith('native')
      expect(writeOsc52Clipboard('')).toBe(false)
    } finally {
      clearActiveRenderer(renderer)
    }
  })
  it('detects tmux from env', () => {
    expect(inTmux({ TMUX: '/tmp/tmux-1/default,1,0' } as NodeJS.ProcessEnv)).toBe(true)
    expect(inTmux({ TERM: 'tmux-256color' } as NodeJS.ProcessEnv)).toBe(true)
    expect(inTmux({ TERM: 'xterm' } as NodeJS.ProcessEnv)).toBe(false)
  })
})
