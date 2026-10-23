// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useState, useSyncExternalStore, type ReactElement } from 'react'

import type { VoiceControl } from '../../client/voiceControl.js'
import { INITIAL_VOICE_STATE, voiceStatusView } from '../../client/voiceState.js'
import { StatusBadge } from './StatusBadge.js'

const NO_SUBSCRIBE = (): (() => void) => () => undefined
const NO_SNAPSHOT = (): typeof INITIAL_VOICE_STATE => INITIAL_VOICE_STATE

/**
 * Voice orb, level meter, interrupt and optional live toggle. Interrupt only
 * flushes playback; stopping tasks is a separate, confirmed supervisor action.
 * Rendered only when a VoiceControl exists and has something to show.
 */
export function VoiceStatus({ voice }: { voice: VoiceControl | undefined }): ReactElement | null {
  const store = voice?.state
  const state = useSyncExternalStore(store?.subscribe ?? NO_SUBSCRIBE, store?.getSnapshot ?? NO_SNAPSHOT, store?.getSnapshot ?? NO_SNAPSHOT)
  const [liveError, setLiveError] = useState<string | undefined>(undefined)
  if (voice === undefined || store === undefined) return null
  const live = voice.live
  const view = voiceStatusView(state)
  if (state.phase === 'idle' && state.notice === undefined && live === undefined) return null

  const toggleLive = (): void => {
    if (live === undefined) return
    setLiveError(undefined)
    if (live.active) live.stop()
    else live.start().catch((reason: unknown) => setLiveError(reason instanceof Error ? reason.message : 'Live voice could not start.'))
  }

  return (
    <div className="voice-status" aria-live="polite">
      <StatusBadge view={view.view} />
      {state.phase === 'listening' ? (
        <progress className="voice-level" max={100} value={Math.round(view.level * 100)} aria-label="Microphone level" />
      ) : null}
      {state.caption === undefined ? null : <p className="voice-caption">{state.caption}</p>}
      {view.recovery === undefined ? null : <p role="alert" className="field-error">{view.recovery}</p>}
      {liveError === undefined ? null : <p role="alert" className="field-error">{liveError}</p>}
      <div className="composer-row">
        {view.interruptible && voice.interrupt !== undefined
          ? <button type="button" className="btn" onClick={() => voice.interrupt?.()}>Stop speaking</button>
          : null}
        {live === undefined
          ? null
          : <button type="button" className="btn" aria-pressed={live.active} onClick={toggleLive}>{live.active ? 'End live voice' : 'Start live voice'}</button>}
      </div>
    </div>
  )
}
