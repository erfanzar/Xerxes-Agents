// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { describe, expect, it } from 'vitest'

import { DEFAULT_VOICE_RECORD_KEY } from './platform.js'
import {
  VoiceCapturePhase,
  applyVoiceCaptureStatus,
  beginVoicePlayback,
  completeVoiceTranscript,
  createVoiceCaptureState,
  failVoiceTranscript,
  finishVoicePlayback,
  matchesVoiceCaptureToggle,
  normalizeVoiceCaptureKey,
  setVoiceCaptureContinuous,
  toggleVoiceCapture
} from './voiceCaptureState.js'

describe('voice capture state', () => {
  it('transitions idle → recording → transcribing without doing audio I/O', () => {
    const idle = createVoiceCaptureState()
    const started = toggleVoiceCapture(idle)
    const stopped = toggleVoiceCapture(started.state)

    expect(started).toMatchObject({ command: 'start', state: { phase: VoiceCapturePhase.RECORDING }, submitText: null })
    expect(stopped).toMatchObject({
      command: 'stop',
      state: { phase: VoiceCapturePhase.TRANSCRIBING },
      submitText: null
    })
    expect(toggleVoiceCapture(stopped.state)).toEqual({ command: null, state: stopped.state, submitText: null })
  })

  it('uses daemon status as the authoritative recorder indicator', () => {
    const state = createVoiceCaptureState()

    expect(applyVoiceCaptureStatus(state, 'listening').state.phase).toBe(VoiceCapturePhase.RECORDING)
    expect(applyVoiceCaptureStatus(state, 'transcribing').state.phase).toBe(VoiceCapturePhase.TRANSCRIBING)
    expect(applyVoiceCaptureStatus(state, 'unexpected').state.phase).toBe(VoiceCapturePhase.IDLE)
  })

  it('submits a transcript and makes continuous restart an explicit command', () => {
    const transcribing = { continuous: true, phase: VoiceCapturePhase.TRANSCRIBING } as const

    expect(completeVoiceTranscript(transcribing, 'hello voice')).toEqual({
      command: 'start',
      state: { continuous: true, phase: VoiceCapturePhase.RECORDING },
      submitText: 'hello voice'
    })
    expect(completeVoiceTranscript(transcribing, '')).toEqual({
      command: null,
      state: { continuous: true, phase: VoiceCapturePhase.IDLE },
      submitText: null
    })
    expect(failVoiceTranscript(transcribing).state.phase).toBe(VoiceCapturePhase.IDLE)
    expect(setVoiceCaptureContinuous(transcribing, false)).toEqual({
      continuous: false,
      phase: VoiceCapturePhase.TRANSCRIBING
    })
  })

  it('reserves playing as a display-only state', () => {
    const playing = beginVoicePlayback(createVoiceCaptureState())

    expect(playing.state.phase).toBe(VoiceCapturePhase.PLAYING)
    expect(finishVoicePlayback(playing.state).state.phase).toBe(VoiceCapturePhase.IDLE)
  })
})

describe('voice capture keys', () => {
  it('reuses the established parser and matcher for normalization and validation', () => {
    const configured = normalizeVoiceCaptureKey('control+b')
    const rejected = normalizeVoiceCaptureKey('ctrl+c')

    expect(configured.label).toBe('Ctrl+B')
    expect(matchesVoiceCaptureToggle({ ctrl: true, meta: false }, 'b', configured.binding)).toBe(true)
    expect(rejected.binding).toEqual(DEFAULT_VOICE_RECORD_KEY)
  })
})
