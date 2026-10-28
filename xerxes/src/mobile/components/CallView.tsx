// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useEffect, useState, useSyncExternalStore, type ReactElement } from 'react'

import { tierDownloadBytes, type VoiceModelTier } from '../../client/voiceModelManifest.js'
import { useHubState, useServices } from '../context.js'
import { browserCallEngine } from '../voice/call/browser.js'
import { type CallEngine, type CallSnapshot } from '../voice/call/engine.js'
import { chooseCallTier, fetchCallModelsStatus, type CallModelsStatus } from '../voice/call/logic.js'
import { Sheet } from './Sheet.js'

const INITIAL: CallSnapshot = { phase: 'idle', tier: 'standard', muted: false, heard: '', reply: '' }
const noopSubscribe = (): (() => void) => () => undefined
const initialSnapshot = (): CallSnapshot => INITIAL
const PHASE_LABEL = {
  idle: 'Ready to call', loading: 'Loading on-phone models…', listening: 'Listening', hearing: 'Hearing you…',
  transcribing: 'Transcribing on your phone…', thinking: 'XERXES is thinking…', speaking: 'XERXES is speaking', error: 'Call ended',
} as const
const mb = (bytes: number): string => `${Math.round(bytes / 1_000_000)} MB`

/** Independent of Hub voice providers: only ordinary supervisor messages leave the phone. */
export function CallView({ onClose }: { onClose: () => void }): ReactElement {
  const { controller, store, voice } = useServices()
  const hub = useHubState()
  const [engine, setEngine] = useState<CallEngine>()
  const snapshot = useSyncExternalStore(engine?.subscribe ?? noopSubscribe, engine?.getSnapshot ?? initialSnapshot, initialSnapshot)
  const [tier, setTier] = useState<VoiceModelTier>(() => chooseCallTier(typeof navigator === 'undefined' ? undefined : (navigator as Navigator & { deviceMemory?: number }).deviceMemory))
  const [status, setStatus] = useState<CallModelsStatus>()
  const [error, setError] = useState<string>()
  const [seen, setSeen] = useState(false)
  const active = snapshot.phase !== 'idle' && snapshot.phase !== 'error'
  const live = hub.connection.state === 'live'
  const ready = live && hub.supervisor.phase === 'ready' && hub.supervisorConfig.status !== 'unconfigured'

  useEffect(() => {
    const abort = new AbortController()
    voice?.cancel()
    voice?.live?.stop()
    voice?.interrupt?.()
    void voice?.armReply?.(false)
    try { setSeen(localStorage.getItem('xerxes-call-download-notice') === 'seen') } catch { /* private mode: show notice */ }
    void fetchCallModelsStatus(fetch, abort.signal).then(setStatus, (failure: unknown) => {
      if (!abort.signal.aborted) setError(failure instanceof Error ? failure.message : 'Call models are unavailable.')
    })
    return () => abort.abort()
  }, [voice])
  useEffect(() => {
    if (!engine) return
    const visibility = (): void => engine.visibility(document.hidden)
    visibility()
    document.addEventListener('visibilitychange', visibility)
    return () => { document.removeEventListener('visibilitychange', visibility); engine.hangup() }
  }, [engine])
  useEffect(() => { if (!live) engine?.connectionLost() }, [engine, live])

  const close = (): void => { engine?.hangup(); onClose() }
  const start = (): void => {
    if (!status || !ready) return
    try {
      const next = engine ?? browserCallEngine(controller, store)
      setEngine(next)
      setError(undefined)
      try { localStorage.setItem('xerxes-call-download-notice', 'seen') } catch { /* storage is optional */ }
      void next.start(tier)
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : 'Call mode is unavailable in this browser.')
    }
  }
  const progress = snapshot.progress
  return (
    <Sheet title="Call XERXES" onClose={close} fullScreen>
      <div className="call-view">
        <p className="block-meta">English · on-phone speech · {active ? snapshot.tier : tier} mode</p>
        <div className={`call-orb call-orb-${snapshot.phase}`} aria-hidden="true">X</div>
        <h3 role="status" aria-live="polite">{snapshot.muted && snapshot.phase === 'listening' ? 'Microphone muted' : PHASE_LABEL[snapshot.phase]}</h3>
        {snapshot.phase === 'loading' ? <div>
          <progress aria-label="Model bytes loaded" max={progress?.total ?? tierDownloadBytes(tier)} value={progress?.loaded ?? 0} />
          <p className="block-meta">{mb(progress?.loaded ?? 0)} / {mb(progress?.total ?? tierDownloadBytes(tier))} · first download and initialization can take several minutes</p>
        </div> : null}
        {snapshot.notice ? <p role="status" className="notice">{snapshot.notice}</p> : null}
        {error || snapshot.error ? <p role="alert" className="field-error">{error ?? snapshot.error}</p> : null}
        {!active ? <>
          <label className="field"><span>Voice quality / download size</span>
            <select value={tier} onChange={event => setTier(event.target.value === 'light' ? 'light' : 'standard')}>
              <option value="standard">Standard · {mb(status?.standard.bytes ?? tierDownloadBytes('standard'))} · Moonshine base + Kokoro</option>
              <option value="light">Light · {mb(status?.light.bytes ?? tierDownloadBytes('light'))} · Moonshine tiny + local phone voice</option>
            </select>
          </label>
          {!seen ? <p className="notice">First call downloads ~160 MB once in Standard mode (Light: ~32 MB). Models are cached on this phone; clearing browser storage downloads them again.</p> : null}
          <p className="block-meta">Audio stays on this phone. Transcripts go to XERXES as normal messages. No Hub voice provider is needed. Light needs an installed local English voice.</p>
          {status ? <p className="block-meta">{status[tier].ready ? 'Models are cached on the Hub; this phone may still need to download them.' : 'The Hub will fetch and verify models on the first call.'}</p> : !error ? <p role="status">Checking Hub call models…</p> : null}
          {!ready ? <p role="status">Connect to the Hub and configure XERXES’s model before starting a call.</p> : null}
          <button type="button" className="btn btn-primary" disabled={!status || !ready} onClick={start}>Start call</button>
        </> : null}
        <section className="call-captions" aria-label="Live call captions" aria-live="polite">
          {snapshot.heard ? <p><strong>You:</strong> {snapshot.heard}</p> : null}
          {snapshot.reply ? <p><strong>XERXES:</strong> {snapshot.reply}</p> : null}
        </section>
        <div className="call-controls">
          <button type="button" className="btn" disabled={!active || snapshot.phase === 'loading'} aria-pressed={snapshot.muted} onClick={() => engine?.mute(!snapshot.muted)}>{snapshot.muted ? 'Unmute' : 'Mute'}</button>
          <button type="button" className="btn btn-danger" onClick={close}>Hang up</button>
        </div>
        <p className="block-meta">Speak to interrupt XERXES. Hiding this tab for 60 seconds ends the call.</p>
      </div>
    </Sheet>
  )
}
