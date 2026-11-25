// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { useId, useRef, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { useDialogFocus } from './dialogFocus.js'
import { Icon } from './Icon.js'

/** Human confirmation stays in the viewport, independent of its source list's scroll position. */
export function ActionDialog({ title, children, confirmLabel, busyLabel = 'Working…', busy = false, disabled = false, done = false, destructive = false, error, onConfirm, onCancel }: {
  title: string; children: ReactNode; confirmLabel: string; busyLabel?: string
  busy?: boolean; disabled?: boolean; done?: boolean; destructive?: boolean; error?: string
  onConfirm: () => void; onCancel: () => void
}) {
  const root = useRef<HTMLDivElement>(null)
  const titleId = useId(), descriptionId = useId()
  useDialogFocus(root)
  const content = <div className="atelier action-dialog-layer" onKeyDown={event => {
    if (event.key === 'Escape') { event.stopPropagation(); event.preventDefault(); if (!busy) onCancel() }
  }}>
    <div className="action-dialog-scrim" aria-hidden="true" onClick={() => { if (!busy) onCancel() }} />
    <div ref={root} className="action-dialog" role="dialog" aria-modal="true" aria-labelledby={titleId} aria-describedby={descriptionId} aria-busy={busy} tabIndex={-1}>
      <header><h2 id={titleId}>{title}</h2><button type="button" className="action-dialog__close" aria-label="Close confirmation" disabled={busy} onClick={onCancel}><Icon name="close" size={14} /></button></header>
      <div id={descriptionId} className="action-dialog__body">{children}</div>
      {error && <p role="alert" className="studio-error">{error}</p>}
      <footer>
        {!done && <button type="button" className="btn" disabled={busy} onClick={onCancel}>Cancel</button>}
        <button type="button" className={destructive ? 'btn btn--danger' : 'btn btn--solid'} disabled={busy || disabled} onClick={onConfirm}>{busy ? busyLabel : confirmLabel}</button>
      </footer>
    </div>
  </div>
  return typeof document === 'undefined' ? content : createPortal(content, document.body)
}
