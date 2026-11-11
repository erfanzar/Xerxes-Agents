// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { useEffect, useRef, type ReactElement } from 'react'

/** Put async action feedback at its source and reveal errors inside scrollable panels. */
export function ActionFeedback({ busy = false, error, message = 'Saving…' }: { busy?: boolean; error?: string | undefined; message?: string }): ReactElement | null {
  const root = useRef<HTMLParagraphElement>(null)
  useEffect(() => {
    if (!root.current || (!busy && !error)) return
    if (error) root.current.focus({ preventScroll: true })
    root.current.scrollIntoView({ block: 'nearest' })
  }, [busy, error])
  if (!busy && !error) return null
  return <p ref={root} tabIndex={-1} role={error ? 'alert' : 'status'} className={error ? 'action-card__error' : 'xap__meta'}>{error || message}</p>
}
