// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { formatVoiceRecordKey, isVoiceToggleKey, parseVoiceRecordKey, type ParsedVoiceRecordKey } from './platform.js'

/**
 * Native terminal-facing voice capture state.
 *
 * Audio capture, WAV paths, and transcription belong to the daemon RPC layer;
 * this module intentionally does not emulate prompt_toolkit's recorder. It
 * only turns UI events into visible state, a submit value, and a start/stop
 * intent for that RPC layer.
 */

export const VoiceCapturePhase = {
  IDLE: 'idle',
  PLAYING: 'playing',
  RECORDING: 'recording',
  TRANSCRIBING: 'transcribing'
} as const

export type VoiceCapturePhase = (typeof VoiceCapturePhase)[keyof typeof VoiceCapturePhase]

export interface VoiceCaptureState {
  continuous: boolean
  phase: VoiceCapturePhase
}

export type VoiceCaptureCommand = 'start' | 'stop'

export interface VoiceCaptureTransition {
  command: null | VoiceCaptureCommand
  state: VoiceCaptureState
  submitText: null | string
}

export interface VoiceCaptureKeyConfig {
  binding: ParsedVoiceRecordKey
  label: string
}

export type VoiceCaptureKeyEvent = Parameters<typeof isVoiceToggleKey>[0]

export const createVoiceCaptureState = (continuous = false): VoiceCaptureState => ({
  continuous,
  phase: VoiceCapturePhase.IDLE
})

const transition = (
  state: VoiceCaptureState,
  phase: VoiceCapturePhase,
  command: null | VoiceCaptureCommand = null,
  submitText: null | string = null
): VoiceCaptureTransition => ({ command, state: { ...state, phase }, submitText })

/** Starts recording only when capture is idle. */
export const beginVoiceCapture = (state: VoiceCaptureState): VoiceCaptureTransition =>
  state.phase === VoiceCapturePhase.IDLE
    ? transition(state, VoiceCapturePhase.RECORDING, 'start')
    : transition(state, state.phase)

/** Requests a stop and immediately marks the UI as transcribing. */
export const stopVoiceCapture = (state: VoiceCaptureState): VoiceCaptureTransition =>
  state.phase === VoiceCapturePhase.RECORDING
    ? transition(state, VoiceCapturePhase.TRANSCRIBING, 'stop')
    : transition(state, state.phase)

/** Mirrors push-to-talk press/release semantics without performing I/O. */
export const toggleVoiceCapture = (state: VoiceCaptureState): VoiceCaptureTransition => {
  if (state.phase === VoiceCapturePhase.IDLE) {
    return beginVoiceCapture(state)
  }

  if (state.phase === VoiceCapturePhase.RECORDING) {
    return stopVoiceCapture(state)
  }

  return transition(state, state.phase)
}

/** Applies the daemon's authoritative capture state to the local indicator. */
export const applyVoiceCaptureStatus = (state: VoiceCaptureState, status: string): VoiceCaptureTransition => {
  if (status === 'listening') {
    return transition(state, VoiceCapturePhase.RECORDING)
  }

  if (status === 'transcribing') {
    return transition(state, VoiceCapturePhase.TRANSCRIBING)
  }

  return transition(state, VoiceCapturePhase.IDLE)
}

/**
 * Completes a transcription and returns the text for the caller to submit.
 *
 * Continuous mode restarts only after a non-empty transcript, matching the
 * previous push-to-talk behavior while leaving the actual restart RPC explicit.
 */
export const completeVoiceTranscript = (state: VoiceCaptureState, text: string): VoiceCaptureTransition => {
  if (!text) {
    return transition(state, VoiceCapturePhase.IDLE)
  }

  if (state.continuous) {
    return transition(state, VoiceCapturePhase.RECORDING, 'start', text)
  }

  return transition(state, VoiceCapturePhase.IDLE, null, text)
}

/** Returns to idle after a failed transcription without inventing a transcript. */
export const failVoiceTranscript = (state: VoiceCaptureState): VoiceCaptureTransition =>
  transition(state, VoiceCapturePhase.IDLE)

/** Updates continuous-listen preference without changing capture phase. */
export const setVoiceCaptureContinuous = (state: VoiceCaptureState, continuous: boolean): VoiceCaptureState => ({
  ...state,
  continuous: Boolean(continuous)
})

/** Reserves the optional playback state for TTS status indicators. */
export const beginVoicePlayback = (state: VoiceCaptureState): VoiceCaptureTransition =>
  transition(state, VoiceCapturePhase.PLAYING)

/** Leaves TTS playback and returns the indicator to idle. */
export const finishVoicePlayback = (state: VoiceCaptureState): VoiceCaptureTransition =>
  transition(state, VoiceCapturePhase.IDLE)

/** Delegates config validation to the UI's existing, platform-aware parser. */
export const normalizeVoiceCaptureKey = (raw: unknown): VoiceCaptureKeyConfig => {
  const binding = parseVoiceRecordKey(raw)

  return { binding, label: formatVoiceRecordKey(binding) }
}

/** Delegates matching to the same platform-aware key matcher used by terminal input. */
export const matchesVoiceCaptureToggle = (
  key: VoiceCaptureKeyEvent,
  ch: string,
  binding: ParsedVoiceRecordKey
): boolean => isVoiceToggleKey(key, ch, binding)
