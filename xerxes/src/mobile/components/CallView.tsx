// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useEffect, useRef, useSyncExternalStore, type CSSProperties, type KeyboardEvent, type ReactElement } from 'react'

import { tierDownloadBytes } from '../../client/voiceModelManifest.js'
import { useCallService, useHubState, useServices } from '../context.js'
import { shortVoiceCaption, voiceMegabytes, voiceOrbLevel, voiceStateLine } from '../voice/call/presentation.js'
import { isCallActive } from '../voice/call/service.js'
import { CallDebugCapture } from './CallDebugCapture.js'

/** Full-screen voice scene. Unmounting/minimizing this view NEVER owns or ends the call. */
export function CallView(): ReactElement {
  const call = useCallService()
  const state = useSyncExternalStore(call.subscribe, call.getSnapshot, call.getSnapshot)
  const hub = useHubState()
  const { router } = useServices()
  const panel = useRef<HTMLElement>(null)
  const current = state.call
  const active = isCallActive(current)
  const ready = state.connected && hub.supervisor.phase === 'ready' && hub.supervisorConfig.status !== 'unconfigured'
  const chat = (): void => { call.minimize(); router.navigate({ name: 'xerxes' }) }
  const end = (): void => { call.end(); router.navigate({ name: 'xerxes' }) }
  const style: CSSProperties & { '--voice-level': number } = { '--voice-level': voiceOrbLevel(state) }
  // Only bytes this browser does not already hold count as a download.
  const totalBytes = state.phoneCache?.totalBytes ?? tierDownloadBytes(active ? current.tier : state.tier)
  const cachedBytes = Math.min(totalBytes, Math.max(0, state.phoneCache?.cachedBytes ?? 0))
  const missingBytes = totalBytes - cachedBytes

  useEffect(() => {
    const previous = document.activeElement instanceof HTMLElement ? document.activeElement : null
    const overflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    panel.current?.focus()
    return () => { document.body.style.overflow = overflow; if (previous?.isConnected) previous.focus() }
  }, [])
  const onKey = (event: KeyboardEvent<HTMLElement>): void => {
    if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); chat(); return }
    if (event.key !== 'Tab' || !panel.current) return
    const items = [...panel.current.querySelectorAll<HTMLElement>('button:not([disabled]), summary, select:not([disabled]), input:not([disabled]), [tabindex="0"]')]
      .filter(item => !item.closest('details:not([open])') || item.tagName === 'SUMMARY')
    const first = items[0]
    const last = items.at(-1)
    if (!first || !last) return
    if (event.shiftKey && (document.activeElement === first || document.activeElement === panel.current)) { event.preventDefault(); last.focus() }
    else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus() }
  }

  return (
    <section ref={panel} className={`voice-screen voice-phase-${current.phase}`} role="dialog" aria-modal="true" aria-labelledby="voice-title" tabIndex={-1} onKeyDown={onKey}>
      <header className="voice-head">
        <h1 id="voice-title">XERXES <span>voice</span></h1>
        <details className="voice-settings">
          <summary aria-label="Voice settings" title="Voice settings"><span aria-hidden="true">⚙</span></summary>
          <div className="voice-settings-panel">
            <label className="field"><span>Voice quality / download size</span>
              <select value={active ? current.tier : state.tier} disabled={active} onChange={event => call.setTier(event.target.value === 'light' ? 'light' : 'standard')}>
                <option value="standard">Standard · {voiceMegabytes(state.models?.standard.bytes ?? tierDownloadBytes('standard'))} · Moonshine Streaming small + Kokoro</option>
                <option value="light">Light · {voiceMegabytes(state.models?.light.bytes ?? tierDownloadBytes('light'))} · Moonshine tiny + local phone voice</option>
              </select>
            </label>
            <p>English · speech runs on your phone. No Hub voice provider is needed.</p>
            <p>Light needs an installed local English voice. Change tiers before starting a call.</p>
            {state.models ? <p>{state.models[state.tier].ready ? 'Models are cached on the Hub; this phone may still need to download them.' : 'The Hub will fetch and verify models on the first call.'}</p> : null}
            <label className="voice-debug-toggle"><input type="checkbox" checked={state.debugCaptureEnabled} onChange={event => call.setDebugCaptureEnabled(event.target.checked)} /><span>Keep last utterance for debugging</span></label>
            <p>Off by default. Keeps only the last 30 seconds or less in memory on this phone. Audio is never uploaded; End or turning this off clears it. Only this setting is remembered.</p>
            {state.debugCaptureEnabled && current.debugCapture ? <CallDebugCapture capture={current.debugCapture} /> : null}
          </div>
        </details>
      </header>
      <div className="voice-stage">
        <div className="voice-orb-wrap" style={style}>
          <button type="button" className="voice-orb" disabled={current.phase !== 'speaking'} aria-label={current.phase === 'speaking' ? 'Interrupt XERXES' : 'Voice activity'} onClick={() => call.interrupt()}>
            <span className="voice-orb-aura" aria-hidden="true" /><span className="voice-orb-core" aria-hidden="true" /><span className="voice-orb-shine" aria-hidden="true" />
          </button>
        </div>
        <h2 className="voice-state" role="status" aria-live="polite">{voiceStateLine(state)}</h2>
        {current.phase === 'loading' && current.restoring !== true && missingBytes > 0 ? <progress className="voice-download" aria-label="Voice model bytes downloaded" max={missingBytes} value={Math.min(missingBytes, Math.max(0, (current.progress?.loaded ?? 0) - cachedBytes))} /> : null}
        <div className="voice-captions" aria-label="Live call captions" aria-live="polite" aria-atomic="true">
          {current.heard ? <p className="voice-heard"><span>You</span> {shortVoiceCaption(current.heard)}</p> : null}
          {current.spokenSentence ? <p className="voice-spoken"><span>XERXES</span> {shortVoiceCaption(current.spokenSentence)}</p> : null}
          {!current.heard && !current.spokenSentence ? <p className="voice-caption-hint">{active ? 'Go ahead. I’m listening.' : 'A conversation, without the keyboard.'}</p> : null}
        </div>
        {state.error || current.error ? <p role="alert" className="voice-error">{state.error ?? current.error}</p> : null}
        {!active && current.endReason ? <p role="alert" className="voice-error voice-end-reason">{current.endReason}</p> : null}
        {current.notice ? <p role="status" className="voice-notice">{current.notice}</p> : null}
      </div>
      {!active ? <div className="voice-start">
        {state.phoneCache !== undefined && missingBytes === 0
          ? <p className="voice-notice">Voice models are downloaded on this phone. Nothing to download.</p>
          : !state.downloadNoticeSeen || state.phoneCache !== undefined
            ? <p className="voice-notice">This call downloads ~{voiceMegabytes(missingBytes)} once{state.tier === 'standard' ? ` (Light needs ~${voiceMegabytes(tierDownloadBytes('light'))})` : ''}. Kept on this phone after that.</p>
            : null}
        {!ready ? <p className="voice-notice">Connect to the Hub and choose XERXES’s model to start.</p> : null}
        <button type="button" className="voice-start-button" disabled={!ready || !state.models || state.checking} onClick={() => { void call.start() }}>{current.endReason || current.phase === 'error' ? 'Restart voice' : 'Start voice'}</button>
        {state.error ? <button type="button" className="voice-retry" onClick={() => { void call.loadStatus() }}>Retry voice models</button> : null}
      </div> : null}
      <footer className="voice-controls" aria-label="Voice controls">
        <button type="button" disabled={!active || current.phase === 'loading'} aria-pressed={current.muted} className={current.muted ? 'voice-control is-muted' : 'voice-control'} onClick={() => call.mute()}><span aria-hidden="true">{current.muted ? '⊘' : '●'}</span><span>{current.muted ? 'Unmute' : 'Mute'}</span></button>
        <button type="button" className="voice-control" onClick={chat}><span aria-hidden="true">▤</span><span>Chat</span><span className="visually-hidden"> — keep voice running</span></button>
        <button type="button" className="voice-control voice-end" aria-label="End voice call" onClick={end}><span aria-hidden="true">×</span><span>End</span></button>
      </footer>
      <p className="voice-footer-note">{current.phase === 'speaking' ? 'Tap the orb or speak to interrupt.' : 'Chat keeps voice on. End stops it.'}</p>
    </section>
  )
}
