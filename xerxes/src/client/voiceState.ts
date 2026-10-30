// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { StatusView } from './orbMapping.js'

/**
 * Pure voice-state reducer for the orb. Voice state, supervisor activity and
 * the agent count are kept apart: this module only knows the voice side plus a
 * count handed to it. Playback events are matched by response id AND
 * generation, so a late event from a response that was interrupted or replaced
 * can never move the orb.
 */

export type VoicePhase = 'idle' | 'connecting' | 'listening' | 'transcribing' | 'thinking' | 'buffering' | 'speaking' | 'interrupted' | 'failed'

export interface VoiceFailure {
  readonly code: string
  readonly message: string
  readonly recovery: string
}

export interface VoiceUiState {
  readonly phase: VoicePhase
  /** A live (hands-free) session is open: "resting" means listening, not idle. */
  readonly live: boolean
  /** Measured microphone level in [0, 1]; 0 outside `listening`. */
  readonly level: number
  readonly agentsWorking: number
  readonly responseId: string | undefined
  readonly generation: number | undefined
  /** The Hub finished sending audio for the current response. */
  readonly streamEnded: boolean
  /** Audio has actually started coming out of the speaker. */
  readonly playing: boolean
  /** Latest live transcript text, shown while it is being revised. */
  readonly caption: string | undefined
  readonly failure: VoiceFailure | undefined
  readonly notice: string | undefined
}

export const INITIAL_VOICE_STATE: VoiceUiState = Object.freeze({
  phase: 'idle',
  live: false,
  level: 0,
  agentsWorking: 0,
  responseId: undefined,
  generation: undefined,
  streamEnded: false,
  playing: false,
  caption: undefined,
  failure: undefined,
  notice: undefined,
})

export type VoiceEvent =
  | { readonly type: 'connecting' }
  | { readonly type: 'listening' }
  | { readonly type: 'level'; readonly level: number }
  | { readonly type: 'transcribing' }
  | { readonly type: 'transcript'; readonly text: string; readonly final: boolean; readonly live?: boolean }
  | { readonly type: 'submitted' }
  | { readonly type: 'agents'; readonly count: number }
  | { readonly type: 'reply'; readonly spoken: boolean }
  | { readonly type: 'server_state'; readonly state: string }
  | { readonly type: 'audio_start'; readonly responseId: string; readonly generation: number }
  | { readonly type: 'audio_end'; readonly responseId: string; readonly generation: number; readonly interrupted: boolean }
  | { readonly type: 'playback_started'; readonly responseId: string; readonly generation: number }
  | { readonly type: 'playback_ended'; readonly responseId: string; readonly generation: number }
  /** The response finished without any sound reaching the speaker (empty, or no output device). */
  | { readonly type: 'playback_empty'; readonly responseId: string; readonly generation: number }
  | { readonly type: 'interrupt' }
  | { readonly type: 'live'; readonly active: boolean }
  | { readonly type: 'failed'; readonly code: string; readonly message: string }
  | { readonly type: 'reset' }

const RECOVERY: Readonly<Record<string, string>> = {
  VOICE_UNAVAILABLE: 'Voice is not set up on this Hub. Type your message instead.',
  VOICE_PERMISSION: 'Allow the microphone for this site, or type instead.',
  VOICE_BUDGET: 'The voice spend limit was reached. Type instead, or try again later.',
  VOICE_BUSY: 'Someone else is speaking on this conversation. Wait a moment.',
  VOICE_OVERLOADED: 'The connection is too slow for audio. Read the reply on screen.',
  VOICE_FORMAT: 'This recording format is not supported. Type instead.',
  VOICE_LIMIT: 'That recording was too long. Try a shorter message.',
  VOICE_PROVIDER: 'The speech provider failed. Try again or type instead.',
  VOICE_BLOCKED: 'Tap the button once to let this page play sound.',
  VOICE_DISCONNECTED: 'The voice connection dropped. Hold the button to try again.',
  VOICE_DENIED: 'Allow the microphone for this site in your browser settings, or type instead.',
}

export function recoveryFor(code: string): string {
  return RECOVERY[code] ?? 'Try again, or type your message instead.'
}

function clamp01(value: number): number {
  return Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : 0
}

const noResponse = { responseId: undefined, generation: undefined, streamEnded: false, playing: false } as const

/** Where the orb rests when nothing is happening: listening in a live session, idle otherwise. */
function rest(state: VoiceUiState): VoicePhase {
  return state.live ? 'listening' : 'idle'
}

export function voiceReducer(state: VoiceUiState, event: VoiceEvent): VoiceUiState {
  switch (event.type) {
    case 'connecting':
      return { ...state, ...noResponse, phase: 'connecting', level: 0, caption: undefined, failure: undefined, notice: undefined }
    case 'listening':
      return { ...state, phase: 'listening', failure: undefined, notice: undefined }
    case 'level':
      return state.phase === 'listening' ? { ...state, level: clamp01(event.level) } : state
    case 'transcribing':
      return { ...state, phase: 'transcribing', level: 0 }
    case 'transcript': {
      const empty = event.text.trim() === ''
      // A live session keeps its phase (the Hub drives it); only the caption follows the revisions.
      if (event.live === true) return { ...state, caption: empty ? undefined : event.text }
      if (!event.final) return { ...state, caption: empty ? undefined : event.text }
      return empty
        ? { ...state, phase: 'idle', level: 0, caption: undefined, notice: 'Nothing was heard.' }
        : { ...state, phase: 'idle', level: 0, caption: undefined, notice: undefined }
    }
    case 'submitted':
      return { ...state, phase: 'thinking', caption: undefined, notice: undefined }
    case 'agents':
      return { ...state, agentsWorking: Math.max(0, Math.floor(event.count)) }
    case 'reply':
      // Only a reply that will be spoken moves the orb; text-only replies leave the screen to show it.
      return event.spoken
        ? { ...state, phase: state.playing ? state.phase : 'buffering' }
        : state.phase === 'thinking' ? { ...state, phase: rest(state) } : state
    case 'server_state':
      if (event.state === 'thinking' && (state.phase === 'idle' || state.phase === 'thinking' || (state.live && state.phase === 'listening'))) {
        return { ...state, phase: 'thinking' }
      }
      if (event.state === 'idle' && (state.phase === 'thinking' || (state.phase === 'buffering' && state.responseId === undefined))) {
        return { ...state, phase: rest(state) }
      }
      if (event.state === 'listening' && state.live && (state.phase === 'thinking' || state.phase === 'connecting' || state.phase === 'interrupted' || state.phase === 'idle')) {
        return { ...state, phase: 'listening' }
      }
      return state
    case 'audio_start':
      if (state.generation !== undefined && event.generation < state.generation) return state
      return { ...state, phase: 'buffering', responseId: event.responseId, generation: event.generation, streamEnded: false, playing: false, failure: undefined }
    case 'playback_started':
      if (event.responseId !== state.responseId || event.generation !== state.generation) return state
      return { ...state, phase: 'speaking', playing: true }
    case 'audio_end':
      if (event.responseId !== state.responseId || event.generation !== state.generation) return state
      if (event.interrupted) return { ...state, ...noResponse, phase: state.live ? 'listening' : 'interrupted' }
      // Queued audio may still be about to sound (its start is reported a moment later), so the orb
      // rests only when playback reports it ended, or that nothing played (`playback_empty`).
      return { ...state, streamEnded: true }
    case 'playback_ended':
    case 'playback_empty':
      if (event.responseId !== state.responseId || event.generation !== state.generation) return state
      return { ...state, ...noResponse, phase: rest(state) }
    case 'interrupt':
      return { ...state, ...noResponse, phase: state.live ? 'listening' : 'interrupted', level: 0 }
    case 'live':
      return event.active
        ? { ...INITIAL_VOICE_STATE, agentsWorking: state.agentsWorking, live: true, phase: 'connecting' }
        : { ...INITIAL_VOICE_STATE, agentsWorking: state.agentsWorking }
    case 'failed':
      return {
        ...state, ...noResponse, phase: 'failed', level: 0, live: false, caption: undefined,
        failure: { code: event.code, message: event.message, recovery: recoveryFor(event.code) },
      }
    case 'reset':
      return { ...INITIAL_VOICE_STATE, agentsWorking: state.agentsWorking }
  }
}

// ----------------------------------------------------------------------- view

export interface VoiceStatusView {
  readonly view: StatusView
  readonly level: number
  readonly recovery: string | undefined
  /** Whether the interrupt control applies: only while audio is being prepared or played. */
  readonly interruptible: boolean
}

function agentsLabel(count: number): string {
  return `${count} agent${count === 1 ? '' : 's'} working`
}

/** The orb for a voice state, following the plan's table. */
export function voiceStatusView(state: VoiceUiState): VoiceStatusView {
  const base = { level: state.phase === 'listening' ? state.level : 0, recovery: undefined, interruptible: false }
  switch (state.phase) {
    case 'idle':
      return { ...base, view: { kind: 'orb', state: 'breathing', live: false, label: state.notice ?? 'Idle', tone: 'calm' } }
    case 'connecting':
      return { ...base, view: { kind: 'orb', state: 'connecting', live: true, label: 'Connecting', tone: 'busy' } }
    case 'listening':
      return { ...base, view: { kind: 'orb', state: 'listening', live: true, label: 'Listening', tone: 'busy' } }
    case 'transcribing':
      return { ...base, view: { kind: 'orb', state: 'working', live: true, label: 'Transcribing', tone: 'busy' } }
    case 'thinking':
      return state.agentsWorking > 0
        ? { ...base, view: { kind: 'orb', state: 'connecting', live: true, label: agentsLabel(state.agentsWorking), tone: 'busy' } }
        : { ...base, view: { kind: 'orb', state: 'solving', live: true, label: 'Thinking', tone: 'busy' } }
    case 'buffering':
      return { ...base, interruptible: true, view: { kind: 'orb', state: 'working', live: true, label: 'Preparing audio', tone: 'busy' } }
    case 'speaking':
      return { ...base, interruptible: true, view: { kind: 'orb', state: 'composing', live: true, label: 'Speaking', tone: 'busy' } }
    case 'interrupted':
      return { ...base, view: { kind: 'mark', mark: 'unknown', label: 'Interrupted', detail: 'Playback stopped. Agents keep working.', tone: 'calm' } }
    case 'failed':
      return {
        ...base, recovery: state.failure?.recovery,
        view: { kind: 'mark', mark: 'failed', label: 'Voice failed', ...(state.failure === undefined ? {} : { detail: state.failure.message }), tone: 'problem' },
      }
  }
}

// ---------------------------------------------------------------------- store

export interface VoiceStateStore {
  subscribe(listener: () => void): () => void
  getSnapshot(): VoiceUiState
}

export interface VoiceStateController extends VoiceStateStore {
  dispatch(event: VoiceEvent): void
}

export function createVoiceStateStore(initial: VoiceUiState = INITIAL_VOICE_STATE): VoiceStateController {
  let state = initial
  const listeners = new Set<() => void>()
  return {
    getSnapshot: () => state,
    subscribe(listener) {
      listeners.add(listener)
      return () => { listeners.delete(listener) }
    },
    dispatch(event) {
      const next = voiceReducer(state, event)
      if (next === state) return
      state = next
      for (const listener of [...listeners]) listener()
    },
  }
}
