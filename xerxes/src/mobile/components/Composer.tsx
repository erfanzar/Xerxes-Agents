// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useRef, useState, type ChangeEvent, type ReactElement } from 'react'

import { IMAGE_LIMITS, browserImageEngine, checkImageFile, fitsWireBudget, prepareImage, type PreparedImage } from '../../client/imagePrep.js'
import type { ComposerMode } from '../../client/taskView.js'
import { useServices } from '../context.js'
import { TalkButton } from './TalkButton.js'

interface Preparing {
  readonly id: string
  readonly name: string
  readonly progress: number
}

export interface ComposerProps {
  readonly draftKey: string
  readonly mode: ComposerMode
  readonly running: boolean
  readonly canSend: boolean
  readonly sendMessage?: string | undefined
  /** The person changed the draft: a stale send error no longer applies. */
  readonly onEdited?: (() => void) | undefined
}

/**
 * What stays in the box after a send of `sent` was accepted: only what the
 * person typed while it was in flight. If they rewrote the sent text, keep it all.
 */
export function remainingAfterSend(current: string, sent: string): string {
  if (current === sent) return ''
  if (current.startsWith(sent)) return current.slice(sent.length).replace(/^\s+/, '')
  return current
}

/** Append a spoken transcript to whatever is in the box now. */
export function withTranscript(current: string, spoken: string): string {
  return current === '' ? spoken : `${current} ${spoken}`
}

/** Message box above the keyboard: Send or Steer (by task state), Stop, and image attachments. */
export function Composer({ draftKey, mode, running, canSend, sendMessage, onEdited }: ComposerProps): ReactElement {
  const services = useServices()
  const { controller, drafts } = services
  const [text, setText] = useState(() => drafts.get(draftKey))
  const [images, setImages] = useState<readonly PreparedImage[]>([])
  const [preparing, setPreparing] = useState<readonly Preparing[]>([])
  const [problem, setProblem] = useState<string | undefined>(undefined)
  const [busy, setBusy] = useState(false)
  const count = useRef(0)
  const attached = useRef<readonly PreparedImage[]>([])
  const typing = useRef(false)
  const latest = useRef(text)
  const sending = useRef(false)

  const update = (value: string): void => {
    latest.current = value
    setText(value)
    drafts.set(draftKey, value)
    onEdited?.()
    if (!typing.current && value !== '') {
      typing.current = true
      controller.setPresence('typing')
    }
  }
  const stoppedTyping = (): void => {
    if (typing.current) {
      typing.current = false
      controller.setPresence('viewing')
    }
  }

  const addFiles = async (event: ChangeEvent<HTMLInputElement>): Promise<void> => {
    const files = [...(event.target.files ?? [])]
    event.target.value = ''
    setProblem(undefined)
    const engine = services.imageEngine ?? browserImageEngine()
    for (const file of files) {
      const check = checkImageFile(file, attached.current.length + count.current)
      if (!check.ok) {
        setProblem(check.message)
        break
      }
      const id = `img${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`
      count.current += 1
      setPreparing(list => [...list, { id, name: file.name || 'Photo', progress: 0 }])
      try {
        const prepared = await prepareImage(file, engine, {
          id,
          onProgress: progress => setPreparing(list => list.map(item => (item.id === id ? { ...item, progress } : item))),
        })
        if (!fitsWireBudget([...attached.current, prepared])) {
          setProblem('Those images are too large to send together. Remove one or choose smaller images.')
        } else {
          attached.current = [...attached.current, prepared]
          setImages(attached.current)
        }
      } catch (error) {
        setProblem(error instanceof Error ? error.message : 'That image could not be prepared.')
      } finally {
        count.current -= 1
        setPreparing(list => list.filter(item => item.id !== id))
      }
    }
  }

  const removeImage = (id: string): void => {
    attached.current = attached.current.filter(image => image.id !== id)
    setImages(attached.current)
  }

  const send = async (): Promise<void> => {
    // The ref closes the gap between two taps before React re-renders `busy`.
    if (sending.current || busy || !canSend) return
    const sent = latest.current
    const sentImages = attached.current
    if (sent.trim() === '' && sentImages.length === 0) return
    sending.current = true
    setBusy(true)
    try {
      const ok = mode === 'steer' ? await controller.steer(sent.trim(), sentImages) : await controller.submit(sent.trim(), sentImages)
      // The draft is cleared only once the Hub accepted it, and only what was sent:
      // text typed and photos added while the send was in flight stay.
      if (ok) {
        const rest = remainingAfterSend(latest.current, sent)
        latest.current = rest
        setText(rest)
        if (rest === '') drafts.clear(draftKey)
        else drafts.set(draftKey, rest)
        const sentIds = new Set(sentImages.map(image => image.id))
        attached.current = attached.current.filter(image => !sentIds.has(image.id))
        setImages(attached.current)
        if (rest === '') stoppedTyping()
      }
    } finally {
      sending.current = false
      setBusy(false)
    }
  }

  const stop = async (): Promise<void> => {
    if (busy || !canSend) return
    setBusy(true)
    try {
      await controller.stopTurn()
    } finally {
      setBusy(false)
    }
  }

  const empty = text.trim() === '' && images.length === 0
  const verb = mode === 'steer' ? 'Steer' : 'Send'
  return (
    <form className="composer" aria-label="Message" onSubmit={event => { event.preventDefault(); void send() }}>
      {images.length === 0 && preparing.length === 0 ? null : (
        <ul className="attachments" aria-label="Attached images">
          {images.map(image => (
            <li key={image.id} className="attachment">
              <img src={image.previewUrl} alt={`Attached image ${image.width} by ${image.height}`} width={56} height={56} />
              <button type="button" className="btn btn-quiet" onClick={() => removeImage(image.id)}>Remove</button>
            </li>
          ))}
          {preparing.map(item => (
            <li key={item.id} className="attachment">
              <progress max={1} value={item.progress} aria-label={`Preparing ${item.name}`} />
            </li>
          ))}
        </ul>
      )}
      {images.length > 0 && !controller.imagesSupported ? <p className="notice">This Hub does not accept images yet. Remove them to send your message.</p> : null}
      {problem === undefined ? null : <p role="alert" className="field-error">{problem}</p>}
      {sendMessage === undefined ? null : <p role="alert" className="field-error">{sendMessage}</p>}
      <label className="field">
        <span className="visually-hidden">{mode === 'steer' ? 'Steer the task' : 'Message to the task'}</span>
        <textarea
          rows={2}
          value={text}
          placeholder={mode === 'steer' ? 'Steer this task…' : 'Message Xerxes…'}
          onChange={event => update(event.target.value)}
          onBlur={stoppedTyping}
        />
      </label>
      <div className="composer-row">
        <label className="btn btn-quiet">
          Camera
          <input type="file" accept="image/*" capture="environment" className="visually-hidden" disabled={!canSend || images.length + preparing.length >= IMAGE_LIMITS.maxImages} onChange={event => { void addFiles(event) }} />
        </label>
        <label className="btn btn-quiet">
          Photos
          <input type="file" accept="image/jpeg,image/png,image/webp,image/gif" multiple className="visually-hidden" disabled={!canSend || images.length + preparing.length >= IMAGE_LIMITS.maxImages} onChange={event => { void addFiles(event) }} />
        </label>
        <TalkButton voice={services.voice} onTranscript={spoken => update(withTranscript(latest.current, spoken))} />
        {running ? <button type="button" className="btn btn-danger" disabled={!canSend || busy} onClick={() => { void stop() }}>Stop</button> : null}
        <button type="submit" className="btn btn-primary" disabled={!canSend || busy || empty || preparing.length > 0}>{busy ? 'Sending…' : verb}</button>
      </div>
    </form>
  )
}
