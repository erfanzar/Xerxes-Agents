// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useEffect, useId, useRef, type KeyboardEvent, type ReactElement, type ReactNode } from 'react'

const FOCUSABLE = 'button:not([disabled]), [href], input:not([disabled]), textarea:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])'

/**
 * A modal sheet: labelled dialog, focus moves in, Tab stays inside, Escape and
 * the Close button dismiss it, and focus returns to what opened it.
 */
export function Sheet({ title, onClose, children, tall = false }: { title: string; onClose: () => void; children: ReactNode; tall?: boolean }): ReactElement {
  const titleId = useId()
  const panel = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const previous = document.activeElement instanceof HTMLElement ? document.activeElement : null
    panel.current?.focus()
    return () => { previous?.focus() }
  }, [])

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>): void => {
    if (event.key === 'Escape') {
      event.stopPropagation()
      onClose()
      return
    }
    if (event.key !== 'Tab' || panel.current === null) return
    const items = [...panel.current.querySelectorAll<HTMLElement>(FOCUSABLE)]
    const first = items[0]
    const last = items.at(-1)
    if (first === undefined || last === undefined) return
    if (event.shiftKey && (document.activeElement === first || document.activeElement === panel.current)) {
      event.preventDefault()
      last.focus()
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault()
      first.focus()
    }
  }

  return (
    <div className="sheet-layer">
      <button type="button" className="sheet-backdrop" aria-label="Close" tabIndex={-1} onClick={onClose} />
      <div ref={panel} role="dialog" aria-modal="true" aria-labelledby={titleId} tabIndex={-1} className={`sheet${tall ? ' sheet-tall' : ''}`} onKeyDown={onKeyDown}>
        <header className="sheet-head">
          <h2 id={titleId}>{title}</h2>
          <button type="button" className="btn btn-quiet" onClick={onClose}>Close</button>
        </header>
        <div className="sheet-body">{children}</div>
      </div>
    </div>
  )
}
