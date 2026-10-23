// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, it } from 'vitest'

import {
  INITIAL_VOICE_STATE, createVoiceStateStore, recoveryFor, voiceReducer, voiceStatusView, type VoiceEvent, type VoiceUiState,
} from '../../client/voiceState.js'

const run = (events: readonly VoiceEvent[], from: VoiceUiState = INITIAL_VOICE_STATE): VoiceUiState => events.reduce(voiceReducer, from)

function orbOf(state: VoiceUiState): { state: string; label: string; live: boolean } {
  const { view } = voiceStatusView(state)
  if (view.kind !== 'orb') throw new Error(`expected an orb, got a ${view.kind} (${view.label})`)
  return { state: view.state, label: view.label, live: view.live }
}

describe('voice orb states (plan table)', () => {
  it('idle breathes without motion', () => {
    expect(orbOf(INITIAL_VOICE_STATE)).toEqual({ state: 'breathing', label: 'Idle', live: false })
  })

  it('opening the microphone reads as connecting, not listening', () => {
    expect(orbOf(run([{ type: 'connecting' }]))).toMatchObject({ state: 'connecting', label: 'Connecting' })
  })

  it('listening shows the measured input level, only while listening', () => {
    const listening = run([{ type: 'connecting' }, { type: 'listening' }, { type: 'level', level: 0.4 }])
    expect(orbOf(listening)).toMatchObject({ state: 'listening', label: 'Listening' })
    expect(voiceStatusView(listening).level).toBe(0.4)
    expect(run([{ type: 'level', level: 0.9 }]).level).toBe(0)
    expect(run([{ type: 'listening' }, { type: 'level', level: 7 }]).level).toBe(1)
    expect(run([{ type: 'listening' }, { type: 'level', level: Number.NaN }]).level).toBe(0)
    expect(voiceStatusView(run([{ type: 'listening' }, { type: 'level', level: 0.5 }, { type: 'transcribing' }])).level).toBe(0)
  })

  it('transcribing is working with a label', () => {
    expect(orbOf(run([{ type: 'listening' }, { type: 'transcribing' }]))).toMatchObject({ state: 'working', label: 'Transcribing' })
  })

  it('thinking reports the real number of agents working, and never invents one', () => {
    const thinking = run([{ type: 'submitted' }])
    expect(orbOf(thinking)).toMatchObject({ label: 'Thinking', state: 'solving' })
    expect(orbOf(run([{ type: 'agents', count: 1 }], thinking)).label).toBe('1 agent working')
    expect(orbOf(run([{ type: 'agents', count: 3 }], thinking)).label).toBe('3 agents working')
    expect(run([{ type: 'agents', count: -2 }]).agentsWorking).toBe(0)
  })

  it('buffering says "Preparing audio" and can be interrupted', () => {
    const buffering = run([{ type: 'audio_start', responseId: 'r1', generation: 1 }])
    expect(orbOf(buffering)).toMatchObject({ state: 'working', label: 'Preparing audio' })
    expect(voiceStatusView(buffering).interruptible).toBe(true)
  })

  it('speaking lasts until playback ends, not until the Hub finishes sending', () => {
    let state = run([{ type: 'audio_start', responseId: 'r1', generation: 1 }, { type: 'playback_started', responseId: 'r1', generation: 1 }])
    expect(orbOf(state)).toMatchObject({ state: 'composing', label: 'Speaking' })
    expect(voiceStatusView(state).interruptible).toBe(true)
    state = run([{ type: 'audio_end', responseId: 'r1', generation: 1, interrupted: false }], state)
    expect(state.phase).toBe('speaking')
    expect(state.streamEnded).toBe(true)
    state = run([{ type: 'playback_ended', responseId: 'r1', generation: 1 }], state)
    expect(state.phase).toBe('idle')
    expect(state.responseId).toBeUndefined()
    expect(voiceStatusView(state).interruptible).toBe(false)
  })

  it('interrupted and failed are explicit marks with a recovery', () => {
    const interrupted = voiceStatusView(run([{ type: 'audio_start', responseId: 'r1', generation: 1 }, { type: 'interrupt' }]))
    expect(interrupted.view).toMatchObject({ kind: 'mark', label: 'Interrupted', detail: 'Playback stopped. Agents keep working.' })
    const failed = voiceStatusView(run([{ type: 'failed', code: 'VOICE_BUDGET', message: 'limit reached' }]))
    expect(failed.view).toMatchObject({ kind: 'mark', mark: 'failed', label: 'Voice failed', detail: 'limit reached', tone: 'problem' })
    expect(failed.recovery).toBe(recoveryFor('VOICE_BUDGET'))
    expect(recoveryFor('VOICE_DENIED')).toContain('microphone')
    expect(recoveryFor('SOMETHING_ELSE')).toContain('Try again')
  })
})

describe('voice reducer transitions', () => {
  it('shows the editable transcript caption, and an empty recognition says nothing was heard', () => {
    const partial = run([{ type: 'transcribing' }, { type: 'transcript', text: 'run the', final: false }])
    expect(partial.caption).toBe('run the')
    expect(partial.phase).toBe('transcribing')
    const done = run([{ type: 'transcript', text: 'run the tests', final: true }], partial)
    expect(done).toMatchObject({ phase: 'idle', caption: undefined, notice: undefined })
    const empty = run([{ type: 'transcribing' }, { type: 'transcript', text: '  ', final: true }])
    expect(empty).toMatchObject({ phase: 'idle', notice: 'Nothing was heard.' })
    expect(orbOf(empty).label).toBe('Nothing was heard.')
  })

  it('a spoken reply moves to buffering, a text-only reply releases thinking', () => {
    expect(run([{ type: 'submitted' }, { type: 'reply', spoken: true }]).phase).toBe('buffering')
    expect(run([{ type: 'submitted' }, { type: 'reply', spoken: false }]).phase).toBe('idle')
    expect(run([{ type: 'reply', spoken: false }]).phase).toBe('idle')
  })

  it('server state follows thinking and idle only where it is safe to', () => {
    expect(run([{ type: 'server_state', state: 'thinking' }]).phase).toBe('thinking')
    expect(run([{ type: 'submitted' }, { type: 'server_state', state: 'idle' }]).phase).toBe('idle')
    // The Hub saying "idle" never overrides audio that is playing.
    const speaking = run([{ type: 'audio_start', responseId: 'r', generation: 1 }, { type: 'playback_started', responseId: 'r', generation: 1 }])
    expect(run([{ type: 'server_state', state: 'idle' }], speaking).phase).toBe('speaking')
  })

  it('rejects stale playback events: wrong response, wrong generation, or after an interrupt', () => {
    const playing = run([{ type: 'audio_start', responseId: 'r2', generation: 5 }, { type: 'playback_started', responseId: 'r2', generation: 5 }])
    expect(run([{ type: 'playback_ended', responseId: 'r1', generation: 5 }], playing)).toBe(playing)
    expect(run([{ type: 'playback_ended', responseId: 'r2', generation: 4 }], playing)).toBe(playing)
    expect(run([{ type: 'audio_end', responseId: 'r1', generation: 5, interrupted: true }], playing)).toBe(playing)
    // An older response cannot replace the current one.
    expect(run([{ type: 'audio_start', responseId: 'r0', generation: 4 }], playing)).toBe(playing)

    const interrupted = run([{ type: 'interrupt' }], playing)
    expect(interrupted.phase).toBe('interrupted')
    expect(run([{ type: 'playback_started', responseId: 'r2', generation: 5 }], interrupted)).toBe(interrupted)
    expect(run([{ type: 'playback_ended', responseId: 'r2', generation: 5 }], interrupted)).toBe(interrupted)
  })

  it('a response that never produced sound ends the orb at audio_end', () => {
    const state = run([
      { type: 'audio_start', responseId: 'r', generation: 1 },
      { type: 'audio_end', responseId: 'r', generation: 1, interrupted: false },
    ])
    expect(state.phase).toBe('idle')
    const cut = run([{ type: 'audio_start', responseId: 'r', generation: 1 }, { type: 'audio_end', responseId: 'r', generation: 1, interrupted: true }])
    expect(cut.phase).toBe('interrupted')
  })

  it('keeps the agent count across resets', () => {
    const state = run([{ type: 'agents', count: 2 }, { type: 'failed', code: 'VOICE_PROVIDER', message: 'x' }, { type: 'reset' }])
    expect(state).toMatchObject({ phase: 'idle', agentsWorking: 2, failure: undefined })
  })

  it('a live session rests on listening, not idle, and a failure ends it', () => {
    const live = run([{ type: 'live', active: true }])
    expect(live).toMatchObject({ live: true, phase: 'connecting' })
    const listening = run([{ type: 'server_state', state: 'listening' }], live)
    expect(listening.phase).toBe('listening')
    const speaking = run([{ type: 'audio_start', responseId: 'r', generation: 2 }, { type: 'playback_started', responseId: 'r', generation: 2 }], listening)
    expect(run([{ type: 'interrupt' }], speaking).phase).toBe('listening')
    const ended = run([{ type: 'audio_end', responseId: 'r', generation: 2, interrupted: false }, { type: 'playback_ended', responseId: 'r', generation: 2 }], speaking)
    expect(ended.phase).toBe('listening')
    expect(run([{ type: 'server_state', state: 'thinking' }], listening).phase).toBe('thinking')
    // Live transcripts only move the caption; the Hub drives the phase.
    expect(run([{ type: 'transcript', text: 'status', final: false, live: true }], listening)).toMatchObject({ phase: 'listening', caption: 'status' })
    const failed = run([{ type: 'failed', code: 'VOICE_DISCONNECTED', message: 'dropped' }], listening)
    expect(failed).toMatchObject({ phase: 'failed', live: false })
    expect(run([{ type: 'live', active: false }], listening)).toMatchObject({ live: false, phase: 'idle' })
  })
})

describe('voice state store', () => {
  it('notifies subscribers only when the state changed, and unsubscribes', () => {
    const store = createVoiceStateStore()
    let calls = 0
    const off = store.subscribe(() => { calls += 1 })
    store.dispatch({ type: 'level', level: 0.5 })
    expect(calls).toBe(0)
    store.dispatch({ type: 'connecting' })
    store.dispatch({ type: 'listening' })
    expect(calls).toBe(2)
    expect(store.getSnapshot().phase).toBe('listening')
    off()
    store.dispatch({ type: 'transcribing' })
    expect(calls).toBe(2)
  })
})
