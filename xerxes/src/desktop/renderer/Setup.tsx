// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useEffect, useRef, useState, type ReactElement } from 'react'
import { store, type Snapshot } from './store.js'

import { setupReadiness } from './setupReadiness.js'
import { Icon } from './Icon.js'

const SETUP_KEY = 'xerxes.desktop.setup.v1'
/** A resumable checklist; configuring providers uses the existing credential editor. */
export function FirstRunSetup({ snap }: { snap: Snapshot }): ReactElement | null {
  const [visible, setVisible] = useState(false)
  const dialog = useRef<HTMLElement>(null)
  const state = setupReadiness(snap)
  // A checklist whose every step is already ✓ is a speed bump, not an
  // onboarding. It is also gated on having a workspace, so the genuine
  // first run never saw it — you picked a folder and only THEN were told
  // to "set up before your first task". Show it only when there is
  // something to do; mark it done otherwise so it never comes back.
  const shown = visible && !snap.noWorkspace && !snap.settingsOpen && !state.ready
  useEffect(() => {
    try {
      setVisible(localStorage.getItem(SETUP_KEY) !== 'done')
    } catch {
      setVisible(true)
    }
    const reopen = () => setVisible(true)
    window.addEventListener('xerxes:setup', reopen)
    return () => window.removeEventListener('xerxes:setup', reopen)
  }, [])
  const finish = () => {
    try {
      localStorage.setItem(SETUP_KEY, 'done')
    } catch {
      /* This window still dismisses in restricted storage environments. */
    }
    setVisible(false)
  }
  useEffect(() => {
    if (!shown) return
    const previous = document.activeElement
    dialog.current?.focus()
    const handle = (event: KeyboardEvent) => {
      if (event.key === 'Escape') { event.preventDefault(); event.stopImmediatePropagation(); finish(); return }
      if (event.key !== 'Tab') return
      const buttons = Array.from(dialog.current?.querySelectorAll<HTMLButtonElement>('button:not(:disabled)') ?? [])
      const first = buttons[0], last = buttons.at(-1)
      if (event.shiftKey && (document.activeElement === first || document.activeElement === dialog.current)) { event.preventDefault(); last?.focus() }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus() }
    }
    document.addEventListener('keydown', handle, true)
    return () => { document.removeEventListener('keydown', handle, true); if (previous instanceof HTMLElement && previous.isConnected) previous.focus() }
  }, [shown])
  if (!shown) return null
  return (
    <div className="setup-backdrop">
    <section ref={dialog} tabIndex={-1} role="dialog" aria-modal="true" className="setup-card" aria-label="Get started with Xerxes">
      <div className="setup-heading">
        <div>
          <strong>Make yourself at home.</strong>
          {/* Only the unfinished steps are ever shown now, so naming them
              beats claiming there are "a few things" when there is one. */}
          <p>{[!state.workspace && 'a folder', !state.runtime && 'a runtime connection', !state.model && 'a model'].filter(Boolean).join(' and ') || 'Almost there'} — then you can start.</p>
        </div>
        <button onClick={finish}>Later</button>
      </div>
      <div className="setup-steps">
        <div>
          <span>{state.workspace ? <Icon name="check" size={12} /> : '1'}</span>
          <strong>Your workspace</strong>
          <p>{snap.cwd || 'Choose a project folder.'}</p>
          <button onClick={() => store.chooseWorkspace()}>Change folder</button>
        </div>
        <div>
          <span>{state.runtime ? <Icon name="check" size={12} /> : '2'}</span>
          <strong>Runtime</strong>
          <p>{state.runtime ? 'Connected and ready.' : state.runtimeStalled ? (snap.error || 'This workspace needs attention before the runtime can start.') : 'Connecting to the shared runtime…'}</p>
          {!state.runtime && (
            state.runtimeStalled
              ? <button onClick={() => store.openSettings()}>Open settings</button>
              : <button onClick={() => store.retryConnection()}>Retry connection</button>
          )}
        </div>
        <div>
          <span>{state.model ? <Icon name="check" size={12} /> : '3'}</span>
          <strong>Your model</strong>
          <p>{snap.model || 'Use your own provider account or local endpoint.'}</p>
          <button disabled={!state.runtime} onClick={() => store.openSettings('models')}>
            {state.model ? 'Change model' : 'Connect a provider'}
          </button>
        </div>
      </div>
      <footer>
        <small>
          Provider credentials are managed in Preferences. No test prompt is sent during setup.
        </small>
        <button disabled={!state.runtime} onClick={() => store.openSettings('permissions')}>
          Review tool permissions
        </button>
        <button className="studio-primary" disabled={!state.ready} onClick={finish}>
          Start working <Icon name="arrow" size={12} />
        </button>
      </footer>
    </section>
    </div>
  )
}
