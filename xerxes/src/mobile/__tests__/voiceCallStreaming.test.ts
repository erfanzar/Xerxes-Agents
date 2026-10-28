// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, it } from 'vitest'

import { prepareCallSpeech } from '../../client/callSpeech.js'
import { voiceStateLine } from '../voice/call/presentation.js'
import type { VoiceCallSnapshot } from '../voice/call/service.js'

describe('streamed call speech', () => {
  it('speaks the whole reply, with a short first chunk broken at a comma', () => {
    const reply = 'Three agents are running right now, and two of them are waiting on tests in the hub service. The third is training. Nothing needs you.'
    const chunks = prepareCallSpeech(reply, 4000, 240, 60)
    expect(chunks[0]!.length).toBeLessThanOrEqual(60)
    expect(chunks[0]!.endsWith(',')).toBe(true)
    // Nothing is dropped: every word of the reply is spoken.
    expect(chunks.join(' ').replace(/\s+/g, ' ')).toBe(reply)
  })
  it('keeps full sentences after the first chunk', () => {
    expect(prepareCallSpeech('Short one. Second sentence here.', 4000, 240, 120)).toEqual(['Short one.', 'Second sentence here.'])
  })
})

describe('speech-engine restart state', () => {
  it('reads "Restoring speech", never a download, while the engine is replaced', () => {
    const state = {
      view: 'voice', tier: 'standard', connected: true, checking: false, downloadNoticeSeen: true, debugCaptureEnabled: false,
      phoneCache: { cachedBytes: 0, totalBytes: 264_000_000 },
      call: { phase: 'loading', tier: 'standard', muted: false, heard: '', reply: '', restoring: true, progress: { loaded: 95_000_000, total: 264_000_000 } },
    } as unknown as VoiceCallSnapshot
    expect(voiceStateLine(state)).toBe('Restoring speech')
  })
})
