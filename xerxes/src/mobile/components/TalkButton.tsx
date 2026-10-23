// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useRef, useState, type KeyboardEvent, type ReactElement } from 'react'

import type { VoiceControl } from '../../client/voiceControl.js'

type Phase = 'idle' | 'recording' | 'processing'

/**
 * Push-to-talk over an injected VoiceControl: hold to talk (pointer), or press
 * Enter/Space to start and again to stop (keyboard). The transcript only fills
 * an editable draft; it never sends, approves or confirms anything. Without a
 * VoiceControl nothing is rendered.
 */
export function TalkButton({ voice, onTranscript }: { voice: VoiceControl | undefined; onTranscript: (text: string) => void }): ReactElement | null {
  const [phase, setPhase] = useState<Phase>('idle')
  const [error, setError] = useState<string | undefined>(undefined)
  const starting = useRef<Promise<void> | undefined>(undefined)

  if (voice === undefined) return null

  const begin = (): void => {
    if (phase !== 'idle') return
    setError(undefined)
    setPhase('recording')
    starting.current = voice.start().catch((reason: unknown) => {
      starting.current = undefined
      setPhase('idle')
      setError(reason instanceof Error ? reason.message : 'The microphone is not available.')
    })
  }

  const end = (): void => {
    if (phase !== 'recording') return
    setPhase('processing')
    void (async () => {
      try {
        await starting.current
        if (starting.current === undefined) return
        const text = await voice.stop()
        if (text !== undefined && text.trim()) onTranscript(text.trim())
      } catch (reason) {
        setError(reason instanceof Error ? reason.message : 'Could not finish the recording.')
      } finally {
        starting.current = undefined
        setPhase('idle')
      }
    })()
  }

  const cancel = (): void => {
    if (phase !== 'recording') return
    voice.cancel()
    starting.current = undefined
    setPhase('idle')
  }

  const onKeyDown = (event: KeyboardEvent<HTMLButtonElement>): void => {
    if (event.key !== 'Enter' && event.key !== ' ') return
    event.preventDefault()
    if (event.repeat) return
    if (phase === 'recording') end()
    else begin()
  }

  const label = phase === 'recording' ? 'Release to finish' : phase === 'processing' ? 'Transcribing…' : 'Hold to talk'
  return (
    <span className="talk">
      <button
        type="button"
        className={`btn btn-talk${phase === 'recording' ? ' is-recording' : ''}`}
        aria-pressed={phase === 'recording'}
        disabled={phase === 'processing'}
        onPointerDown={event => { event.preventDefault(); begin() }}
        onPointerUp={end}
        onPointerCancel={cancel}
        onKeyDown={onKeyDown}
      >
        {label}
      </button>
      {error === undefined ? null : <span role="alert" className="field-error">{error}</span>}
    </span>
  )
}
