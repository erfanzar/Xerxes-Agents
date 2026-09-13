// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useEffect, type RefObject } from 'react'

/** Contain keyboard traversal and return focus to the control that opened a dialog. */
export function useDialogFocus(ref: RefObject<HTMLElement | null>, enabled = true): void {
  useEffect(() => {
    if (!enabled || !ref.current) return
    const root = ref.current.closest<HTMLElement>('[role="dialog"]') ?? ref.current
    const previous = document.activeElement
    const focusable = () => Array.from(root.querySelectorAll<HTMLElement>('button:not(:disabled),input:not(:disabled),select:not(:disabled),textarea:not(:disabled),a[href],summary,[tabindex="0"]')).filter(element => element.getClientRects().length > 0)
    ;(ref.current.matches('input,textarea') ? ref.current : focusable()[0] ?? root).focus()
    const key = (event: KeyboardEvent) => {
      if (event.key !== 'Tab') return
      const elements = focusable(), first = elements[0], last = elements.at(-1)
      if (!first) { event.preventDefault(); root.focus(); return }
      if (!root.contains(document.activeElement) || (event.shiftKey && document.activeElement === first) || (!event.shiftKey && document.activeElement === last)) {
        event.preventDefault()
        ;(event.shiftKey ? last : first)?.focus()
      }
    }
    root.addEventListener('keydown', key)
    return () => { root.removeEventListener('keydown', key); if (previous instanceof HTMLElement && previous.isConnected) previous.focus() }
  }, [enabled, ref])
}
