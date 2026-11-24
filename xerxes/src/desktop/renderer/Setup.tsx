// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useEffect, useRef, useState, type ReactElement } from 'react'
import { store, type Snapshot } from './store.js'

import { setupReadiness } from './setupReadiness.js'
import { localMachineLabel } from './runModeView.js'
import { Icon } from './Icon.js'

const SETUP_KEY = 'xerxes.desktop.setup.v1'

/** What is left to do, as the card's opening sentence: "Choose how Xerxes runs — then you can start." */
export function setupTodo(state: { readonly workspace: boolean; readonly runtime: boolean; readonly model: boolean }): string {
  const left = [!state.workspace && 'a folder', !state.runtime && 'a runtime connection', !state.model && 'how Xerxes runs'].filter(Boolean)
  return left.length ? `Choose ${left.join(' and ')}` : 'Almost there'
}
/** Whether the setup checklist should render, separating first-run auto-show from explicit "Open…" requests. */
export function shouldShowSetup(input: {
  readonly visible: boolean
  readonly manualOpen: boolean
  readonly noWorkspace: boolean
  readonly settingsOpen: boolean
  readonly ready: boolean
}): boolean {
  return !input.noWorkspace && !input.settingsOpen && (input.manualOpen || (input.visible && !input.ready))
}
/** A resumable checklist; configuring providers uses the existing credential editor. */
export function FirstRunSetup({ snap }: { snap: Snapshot }): ReactElement | null {
  const [visible, setVisible] = useState(false)
  const [manualOpen, setManualOpen] = useState(false)
  const dialog = useRef<HTMLElement>(null)
  const state = setupReadiness(snap)
  // First-run auto-show only appears when something still needs setup, while
  // explicit Settings → Setup checklist → Open… always reopens it to inspect status.
  const shown = shouldShowSetup({
    visible,
    manualOpen,
    noWorkspace: Boolean(snap.noWorkspace),
    settingsOpen: snap.settingsOpen,
    ready: state.ready,
  })
  useEffect(() => {
    try {
      setVisible(localStorage.getItem(SETUP_KEY) !== 'done')
    } catch {
      setVisible(true)
    }
    const reopen = () => {
      setVisible(true)
      setManualOpen(true)
    }
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
    setManualOpen(false)
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
  const machine = localMachineLabel()
  return (
    <div className="setup-backdrop" onMouseDown={event => { if (event.target === event.currentTarget) finish() }}>
    <section ref={dialog} tabIndex={-1} role="dialog" aria-modal="true" className="setup-card" aria-label="Get started with Xerxes">
      <div className="setup-heading">
        <div>
          <strong>{state.ready ? 'Setup checklist' : 'Make yourself at home.'}</strong>
          {/* Only the unfinished steps are ever shown now, so naming them
              beats claiming there are "a few things" when there is one. */}
          <p>{state.ready ? 'Workspace, runtime, and model are all ready.' : `${setupTodo(state)} — then you can start.`}</p>
        </div>
        <button onClick={finish}>{state.ready ? 'Close' : 'Later'}</button>
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
          <strong>How Xerxes runs</strong>
          <p>{snap.model || `Private: a local model on ${machine}. Hybrid: a cloud model leads, a local model does the file work here. Cloud: a cloud model.`}</p>
          <button disabled={!state.runtime} onClick={() => store.openSettings('models')}>
            {state.model ? 'Change mode' : 'Choose a mode'}
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
          {state.ready ? 'Done' : <>Start working <Icon name="arrow" size={12} /></>}
        </button>
      </footer>
    </section>
    </div>
  )
}
