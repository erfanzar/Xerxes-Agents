// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { desktopError } from './desktopRpc.js'
import { useEffect, useRef, useState, type ReactElement } from 'react'
import { Icon } from './Icon.js'

/** Explicit push-to-talk. Tracks always stop on cancel, session switch, error, and unmount. */
export function Dictation({
  sessionKey,
  onText,
}: {
  sessionKey: string
  onText: (text: string) => void
}): ReactElement {
  const [state, setState] = useState<'idle' | 'starting' | 'recording' | 'transcribing'>('idle'),
    [error, setError] = useState('')
  const generation = useRef(0),
    recorder = useRef<MediaRecorder | null>(null),
    media = useRef<MediaStream | null>(null),
    timer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const stopTracks = () => {
    media.current?.getTracks().forEach((track) => track.stop())
    media.current = null
    if (timer.current) clearTimeout(timer.current)
  }
  const cancel = () => {
    generation.current++
    if (recorder.current?.state === 'recording') recorder.current.stop()
    recorder.current = null
    stopTracks()
    setState('idle')
    void window.xerxes.voice?.('cancel').catch((failure) => setError(desktopError(failure)))
  }
  useEffect(
    () => () => {
      generation.current++
      if (recorder.current?.state === 'recording') recorder.current.stop()
      stopTracks()
      void window.xerxes.voice?.('cancel').catch(() => {})
    },
    [sessionKey],
  )
  const start = async () => {
    const id = ++generation.current
    setError('')
    setState('starting')
    try {
      if (!window.xerxes.voice) throw new Error('Dictation requires the native desktop app.')
      await window.xerxes.voice('check')
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      if (id !== generation.current) {
        stream.getTracks().forEach((track) => track.stop())
        return
      }
      media.current = stream
      const mime = ['audio/webm;codecs=opus', 'audio/ogg;codecs=opus', 'audio/mp4'].find((type) =>
        MediaRecorder.isTypeSupported(type),
      )
      if (!mime) throw new Error('This desktop cannot record a supported audio format')
      const capture = new MediaRecorder(stream, { mimeType: mime })
      recorder.current = capture
      const chunks: Blob[] = []
      let size = 0,
        tooLarge = false
      capture.ondataavailable = (event) => {
        size += event.data.size
        if (size > 8 * 1024 * 1024) {
          tooLarge = true
          if (capture.state === 'recording') capture.stop()
        } else chunks.push(event.data)
      }
      capture.onerror = () => {
        if (id === generation.current) {
          setError('Microphone recording failed')
          cancel()
        }
      }
      capture.onstop = () => {
        stopTracks()
        if (id !== generation.current) return
        if (tooLarge) {
          setError('Recording exceeds 8 MiB. Try a shorter message.')
          setState('idle')
          return
        }
        setState('transcribing')
        void new Blob(chunks, { type: mime })
          .arrayBuffer()
          .then((bytes) =>
            window.xerxes.voice!('transcribe', { bytes: new Uint8Array(bytes), mediaType: mime }),
          )
          .then((result) => {
            if (id === generation.current && typeof result === 'string') onText(result)
          })
          .catch((failure) => {
            if (id === generation.current) setError(desktopError(failure))
          })
          .finally(() => {
            if (id === generation.current) setState('idle')
          })
      }
      capture.start(1000)
      setState('recording')
      timer.current = setTimeout(() => {
        if (capture.state === 'recording') capture.stop()
      }, 120000)
    } catch (failure) {
      stopTracks()
      if (id === generation.current) {
        setError(desktopError(failure))
        setState('idle')
      }
    }
  }
  return (
    <span className="dictation">
      <button
        className="cchip"
        title="Dictate into the draft"
        disabled={state === 'starting' || state === 'transcribing'}
        onClick={() => {
          if (state === 'recording') recorder.current?.stop()
          else void start()
        }}
      >
        {state === 'recording'
          ? <><Icon name="stop" size={12} /> Stop recording</>
          : state === 'transcribing'
            ? 'Transcribing…'
            : state === 'starting'
              ? 'Opening microphone…'
              : 'Dictate'}
      </button>
      {state !== 'idle' && (
        <button className="cchip" onClick={cancel}>
          Cancel
        </button>
      )}
      {error && (
        <span className="dictation-error" role="alert">
          <strong>Dictation unavailable</strong>
          <span>{error.includes("XERXES_DICTATION_") ? "Connect a transcription provider to use voice input. You can keep typing." : "Voice input could not finish. Your draft is unchanged."}</span>
          <details><summary>Technical details</summary><p>{error}</p></details>
          <button className="cchip" onClick={() => setError('')}>
            Dismiss
          </button>
        </span>
      )}
    </span>
  )
}
