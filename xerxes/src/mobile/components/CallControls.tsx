// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useEffect, useSyncExternalStore, type ReactElement } from 'react'

import { useCallService, useHubState } from '../context.js'
import { voiceStateLine } from '../voice/call/presentation.js'
import { isCallActive } from '../voice/call/service.js'
import { CallView } from './CallView.js'

/** Mounted beside the app shell, never inside a routed tab. */
export function AppCallHost(): ReactElement | null {
  const call = useCallService()
  const state = useSyncExternalStore(call.subscribe, call.getSnapshot, call.getSnapshot)
  const hub = useHubState()
  useEffect(() => { call.setConnected(hub.connection.state === 'live') }, [call, hub.connection.state])
  useEffect(() => {
    const visibility = (): void => call.visibility(document.hidden)
    // pageshow: iOS can restore the page from its back-forward cache without a visibilitychange.
    const shown = (): void => call.visibility(false)
    document.addEventListener('visibilitychange', visibility)
    window.addEventListener('pageshow', shown)
    visibility()
    return () => {
      document.removeEventListener('visibilitychange', visibility)
      window.removeEventListener('pageshow', shown)
    }
  }, [call])
  return state.view === 'voice' ? <CallView /> : null
}

export function ComposerVoiceButton(): ReactElement {
  const call = useCallService()
  const state = useSyncExternalStore(call.subscribe, call.getSnapshot, call.getSnapshot)
  const hub = useHubState()
  const active = isCallActive(state.call)
  return (
    <button type="button" className={`composer-voice${active ? ' is-active' : ''}`} aria-label={active ? 'Return to voice mode' : 'Start voice mode'} title={active ? 'Return to voice' : 'Voice mode'} aria-haspopup="dialog" disabled={!active && (hub.connection.state !== 'live' || hub.supervisor.phase !== 'ready' || hub.supervisorConfig.status === 'unconfigured')} onClick={() => call.open()}>
      <svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" strokeWidth="2.6" strokeLinecap="round" aria-hidden="true"><path d="M4 10v4m4-7v10m4-13v16m4-13v10m4-7v4" /></svg>
      <span className="visually-hidden">Voice</span>
    </button>
  )
}

/** One persistent session, represented in chat; clicking either button never changes the chat transcript. */
export function CallPill(): ReactElement | null {
  const call = useCallService()
  const state = useSyncExternalStore(call.subscribe, call.getSnapshot, call.getSnapshot)
  if (state.view !== 'chat' || !isCallActive(state.call)) return null
  return (
    <div className="voice-pill" role="region" aria-label="Voice call in progress">
      <button type="button" className="voice-pill-return" aria-label={`Return to voice mode — ${voiceStateLine(state)}`} onClick={() => call.open()}><span className="voice-live-dot" aria-hidden="true" /><span><strong>Voice is on</strong><span className="voice-pill-state" role="status">{voiceStateLine(state)}</span></span><span className="voice-pill-open" aria-hidden="true">↗</span></button>
      <button type="button" className="voice-pill-end" aria-label="End voice call" onClick={() => call.end()}><span aria-hidden="true">×</span></button>
    </div>
  )
}
