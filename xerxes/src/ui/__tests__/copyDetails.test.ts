// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { describe, expect, it, vi } from 'vitest'

import type { CliRenderer } from '@opentui/core'

import { clearActiveRenderer, setActiveRenderer } from '../opentui/rendererSingleton.js'
import { inTmux, osc52Copy, osc52Sequence, writeOsc52Clipboard } from '../lib/osc52.js'
import { cycleDetails, filterTranscript, resolveDetails, showThinking } from '../lib/details.js'
import type { TranscriptRow } from '../app/gatewayState.js'

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

describe('details mode', () => {
  it('cycles expanded → collapsed → hidden → expanded', () => {
    expect(cycleDetails('expanded')).toBe('collapsed')
    expect(cycleDetails('collapsed')).toBe('hidden')
    expect(cycleDetails('hidden')).toBe('expanded')
  })
  it('resolves arguments and cycles on empty', () => {
    expect(resolveDetails('hidden', 'expanded')).toBe('hidden')
    expect(resolveDetails('cycle', 'expanded')).toBe('collapsed')
    expect(resolveDetails('', 'collapsed')).toBe('hidden')
    expect(resolveDetails('bogus', 'collapsed')).toBe('collapsed')
  })
  it('shows live thinking only when expanded', () => {
    expect(showThinking('expanded')).toBe(true)
    expect(showThinking('collapsed')).toBe(false)
    expect(showThinking('hidden')).toBe(false)
  })

  const rows: TranscriptRow[] = [
    { id: 1, role: 'user', text: 'hi' },
    { id: 2, role: 'tool', text: 'exec_command' },
    { id: 3, role: 'tool', text: '', blocks: [{ type: 'diff', diff: '+a', language: '' }] },
    { id: 4, role: 'assistant', text: 'done' }
  ]

  it('expanded keeps everything', () => {
    expect(filterTranscript(rows, 'expanded')).toBe(rows)
  })
  it('collapsed keeps tool one-liners, drops blocks', () => {
    const out = filterTranscript(rows, 'collapsed')
    expect(out).toHaveLength(4)
    expect(out[2]).toMatchObject({ role: 'tool', text: 'result ⋯' })
    expect(out[2]!.blocks).toBeUndefined()
  })
  it('hidden drops tool rows entirely', () => {
    const out = filterTranscript(rows, 'hidden')
    expect(out.map(r => r.role)).toEqual(['user', 'assistant'])
  })
})
