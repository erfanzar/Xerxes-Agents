// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useEffect, useRef, useState, type ReactElement } from 'react'

import { useCallService } from '../context.js'
import { playDebugCapture, saveDebugCapture, type DebugPlayback } from '../voice/call/debugAudio.js'
import type { CallDebugCapture as DebugCapture } from '../voice/call/debugCapture.js'
import { isCallActive } from '../voice/call/service.js'

/** The capture lives only in the engine snapshot; this panel never persists or submits it. */
export function CallDebugCapture({ capture }: { readonly capture: DebugCapture }): ReactElement {
  const call = useCallService()
  const playback = useRef<DebugPlayback | null>(null)
  const restoreMic = useRef(false)
  const [playing, setPlaying] = useState(false)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const restore = (): void => {
    const state = call.getSnapshot().call
    if (restoreMic.current && state.muted && isCallActive(state)) call.mute()
    restoreMic.current = false
  }
  useEffect(() => {
    setPlaying(false)
    setError('')
    return () => {
      playback.current?.stop()
      playback.current = null
      restore()
    }
  }, [call, capture.samples])
  const play = (): void => {
    if (playback.current) { playback.current.stop(); return }
    setError('')
    try {
      call.interrupt()
      const state = call.getSnapshot().call
      restoreMic.current = isCallActive(state) && !state.muted
      if (restoreMic.current) call.mute()
      const audio = playDebugCapture(capture)
      playback.current = audio
      setPlaying(true)
      void audio.finished.catch(() => {
        if (playback.current === audio) setError('Could not play captured audio. Check the phone’s audio permissions.')
      }).finally(() => {
        if (playback.current !== audio) return
        playback.current = null
        setPlaying(false)
        restore()
      })
    } catch {
      restore()
      setError('Could not play captured audio. This browser needs Web Audio support.')
    }
  }
  const save = (): void => {
    setSaving(true)
    setError('')
    void saveDebugCapture(capture).catch(() => setError('Could not save WAV. Try another browser or download location.')).finally(() => setSaving(false))
  }
  const seconds = (samples: number): string => (samples / capture.sampleRate).toFixed(3)
  return (
    <section className="voice-debug-capture" aria-label="Last utterance capture">
      <h3>Last utterance</h3>
      <p className="voice-debug-transcript"><strong>Transcript: </strong>{capture.transcript === null ? 'Transcribing…' : capture.transcript || '(No speech recognised)'}</p>
      <p>{capture.durationSeconds.toFixed(3)} s · 16 kHz mono · {capture.samples.length.toLocaleString('en-US')} samples</p>
      <p>Peak {capture.peak.toFixed(4)} · RMS {capture.rms.toFixed(4)} (full scale = 1)</p>
      <p>VAD start {seconds(capture.vad.startSample)} s · end {seconds(capture.vad.endSample)} s · pre-roll {seconds(capture.vad.preRollSamples)} s</p>
      <p>Offsets are relative to this clip, including pre-roll; VAD end excludes trailing silence. Play pauses the live mic.</p>
      <div className="voice-debug-actions">
        <button type="button" onClick={play} aria-pressed={playing}>{playing ? 'Stop' : 'Play'}</button>
        <button type="button" onClick={save} disabled={saving}>{saving ? 'Saving…' : 'Save WAV'}</button>
      </div>
      {error ? <p role="alert">{error}</p> : null}
    </section>
  )
}
