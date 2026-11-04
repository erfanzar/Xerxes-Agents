// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useRef, type ReactElement, type ReactNode } from 'react'

import { useDialogFocus } from './dialogFocus.js'

/**
 * XERXES's controls (Autopilot, Watch, Model) as one dialog shaped like
 * Settings: tabs on the left, the chosen tab on the right. It floats over the
 * conversation instead of pushing it down; Escape, Close or a click on the
 * backdrop closes it.
 */

export type ControlsTab = 'autopilot' | 'watch' | 'model'

export interface ControlsTabEntry {
  readonly id: ControlsTab
  readonly label: string
  /** One short line under the tab name, e.g. "On · 2 workspaces". */
  readonly hint?: string
}

export interface SupervisorControlsProps {
  readonly tab: ControlsTab
  readonly tabs: readonly ControlsTabEntry[]
  readonly onTab: (tab: ControlsTab) => void
  readonly onClose: () => void
  readonly children: ReactNode
}

export function SupervisorControls({ tab, tabs, onTab, onClose, children }: SupervisorControlsProps): ReactElement {
  const dialog = useRef<HTMLDivElement>(null)
  useDialogFocus(dialog)
  const current = tabs.find(entry => entry.id === tab)
  return (
    <div
      className="backdrop"
      onMouseDown={event => { if (event.target === event.currentTarget) onClose() }}
      onKeyDown={event => { if (event.key === 'Escape') { event.stopPropagation(); onClose() } }}
    >
      <div ref={dialog} className="modal xcontrols" role="dialog" aria-modal="true" aria-label="XERXES controls" tabIndex={-1}>
        <nav className="modal__side" aria-label="XERXES controls">
          <div className="cap">XERXES</div>
          {tabs.map(entry => (
            <button
              key={entry.id}
              type="button"
              className={`mtab xcontrols__tab${entry.id === tab ? ' is-on' : ''}`}
              aria-current={entry.id === tab ? 'page' : undefined}
              onClick={() => onTab(entry.id)}
            >
              <span>{entry.label}</span>
              {entry.hint === undefined ? null : <span className="xcontrols__hint">{entry.hint}</span>}
            </button>
          ))}
        </nav>
        <div className="modal__main">
          <div className="xcontrols__head">
            <h2 className="modal__title">{current?.label ?? ''}</h2>
            <button type="button" className="btn" onClick={onClose}>Close</button>
          </div>
          {children}
        </div>
      </div>
    </div>
  )
}
