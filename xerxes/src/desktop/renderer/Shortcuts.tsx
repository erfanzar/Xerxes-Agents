// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * The keyboard contract, in one place.
 *
 * Shortcuts used to exist as three independent literals — a `<kbd>` in
 * JSX, a `hint` string on a palette row, and a branch in `GlobalKeys` —
 * with nothing keeping them in agreement, and the app had no shortcut
 * reference at all. This table is the reference; the accelerators the
 * native menu owns are marked so it stays honest about who handles what.
 */

import { useEffect, useRef, useState, type ReactElement } from 'react'

import { useDialogFocus } from './dialogFocus.js'

interface Binding {
  readonly keys: string
  readonly label: string
}

const GROUPS: readonly { readonly title: string; readonly items: readonly Binding[] }[] = [
  {
    title: 'Tasks',
    items: [
      { keys: '⌘N', label: 'New task' },
      { keys: '⌥⌘N', label: 'New task, choosing preset, worktree and model' },
      { keys: '⇧⌘N', label: 'New window' },
      { keys: '⇧⌘O', label: 'Open another workspace in a new window' },
      { keys: '⌘E', label: 'Export this transcript as markdown' },
    ],
  },
  {
    title: 'Finding things',
    items: [
      { keys: '⌘K', label: 'Command palette' },
      { keys: '⌘F', label: 'Find in this conversation' },
      { keys: '⇧⌘F', label: 'Search the messages of every task' },
      { keys: '⌘,', label: 'Settings' },
      { keys: '⌘/', label: 'This list' },
    ],
  },
  {
    title: 'While a task runs',
    items: [
      { keys: '⏎', label: 'Send — or queue as steering while the agent acts' },
      { keys: '⇧⏎', label: 'Newline' },
      { keys: 'esc', label: 'Close what is open, or stop the running task' },
    ],
  },
  {
    title: 'When the agent asks',
    items: [
      { keys: '1', label: 'Allow once — only while the approval card has focus' },
      { keys: '2', label: 'Allow for this task' },
      { keys: '3', label: 'Deny' },
      { keys: '1…9', label: 'Choose an answer on a question or plan card' },
    ],
  },
]

export function Shortcuts(): ReactElement | null {
  const [open, setOpen] = useState(false)
  const card = useRef<HTMLDivElement>(null)
  useDialogFocus(card, open)
  useEffect(() => {
    const show = (): void => setOpen(true)
    window.addEventListener('xerxes:shortcuts', show)
    return () => window.removeEventListener('xerxes:shortcuts', show)
  }, [])
  useEffect(() => {
    if (!open) return
    // Capture: closing this must not also stop the running task.
    const onKey = (event: KeyboardEvent): void => {
      if (event.key !== 'Escape') return
      event.preventDefault()
      event.stopImmediatePropagation()
      setOpen(false)
    }
    document.addEventListener('keydown', onKey, true)
    return () => document.removeEventListener('keydown', onKey, true)
  }, [open])
  if (!open) return null
  return (
    <div className="backdrop" onMouseDown={event => { if (event.target === event.currentTarget) setOpen(false) }}>
      <div ref={card} className="modal shortcuts" role="dialog" aria-modal="true" aria-label="Keyboard shortcuts">
        <div className="modal__main">
          <h2 className="modal__title">Keyboard shortcuts</h2>
          <p className="modal__sub">On Windows and Linux, ⌘ is Ctrl and ⌥ is Alt.</p>
          {GROUPS.map(group => (
            <section key={group.title} className="shortcuts__group">
              <h3>{group.title}</h3>
              <dl>
                {group.items.map(item => (
                  <div key={item.keys + item.label}>
                    <dt><kbd>{item.keys}</kbd></dt>
                    <dd>{item.label}</dd>
                  </div>
                ))}
              </dl>
            </section>
          ))}
          <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
            <button className="btn" onClick={() => setOpen(false)}>Close</button>
          </div>
        </div>
      </div>
    </div>
  )
}
