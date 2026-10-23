// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useEffect, useState, type ReactElement } from 'react'

import { useServices } from '../context.js'
import type { PushState } from '../pwa/push.js'

/** Opt-in lives behind a button: the permission prompt only ever follows a tap. */
export function PushOptIn(): ReactElement | null {
  const { push } = useServices()
  const [state, setState] = useState<PushState | 'checking'>('checking')
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState<string | undefined>(undefined)

  useEffect(() => {
    let live = true
    if (push === undefined) return undefined
    void push.state().then(next => { if (live) setState(next) })
    return () => { live = false }
  }, [push])

  if (push === undefined || state === 'checking' || state === 'unsupported') return null

  const enable = (): void => {
    setBusy(true)
    setMessage(undefined)
    void push.enable().then(outcome => {
      setBusy(false)
      if (outcome.ok) setState('enabled')
      else setMessage(outcome.message)
    })
  }

  return (
    <section className="card card-plain" aria-labelledby="push-title">
      <h2 id="push-title" className="card-title">Notifications</h2>
      {state === 'enabled'
        ? <p>This device will be notified when a task needs you. Notifications open the task; they never approve anything.</p>
        : state === 'blocked'
          ? <p>Notifications are blocked for Xerxes. Allow them in your browser or system settings to turn this on.</p>
          : (
            <>
              <p>Get a notification when a task needs you, fails or finishes. It opens the task; decisions are always made in the app.</p>
              <button type="button" className="btn btn-primary" disabled={busy} onClick={enable}>{busy ? 'Turning on…' : 'Turn on notifications'}</button>
            </>
          )}
      {message === undefined ? null : <p role="alert" className="field-error">{message}</p>}
    </section>
  )
}
