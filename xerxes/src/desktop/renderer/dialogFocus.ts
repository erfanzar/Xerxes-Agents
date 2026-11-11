// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useEffect, type RefObject } from 'react'

const focusOwners: Array<{ readonly root: HTMLElement }> = []

/** A document capture handler must leave a nested portal's Escape untouched. */
export function handleDialogEscape(event: Pick<KeyboardEvent, 'key' | 'preventDefault' | 'stopImmediatePropagation'>, close: () => void, ownsFocus: boolean): void {
  if (event.key !== 'Escape' || !ownsFocus) return
  event.preventDefault()
  event.stopImmediatePropagation()
  close()
}

/** Contain keyboard traversal and return focus to the control that opened a dialog. */
export function useDialogFocus(ref: RefObject<HTMLElement | null>, enabled = true): void {
  useEffect(() => {
    if (!enabled || !ref.current) return
    const root = ref.current.closest<HTMLElement>('[role="dialog"]') ?? ref.current
    const previous = document.activeElement
    const owner = { root }
    focusOwners.push(owner)
    const ownsTabIndex = !root.hasAttribute('tabindex')
    if (ownsTabIndex) root.tabIndex = -1
    let active = true
    const containFocus = (event: FocusEvent): void => {
      // A native Tab/click clears activeElement before moving to relatedTarget.
      // Do not interrupt a move between controls belonging to this dialog.
      if (event.type === 'focusout' && event.relatedTarget instanceof Node && root.contains(event.relatedTarget)) return
      // Disabling a control has no related target: reclaim focus after the
      // native blur rather than during React layout.
      queueMicrotask(() => {
        if (active && focusOwners.at(-1) === owner && root.isConnected && !root.contains(document.activeElement)) root.focus()
      })
    }
    const focusable = () => Array.from(root.querySelectorAll<HTMLElement>('button:not(:disabled),input:not(:disabled),select:not(:disabled),textarea:not(:disabled),a[href],summary,[tabindex="0"]')).filter(element => element.getClientRects().length > 0)
    ;(ref.current.matches('input,textarea') ? ref.current : focusable()[0] ?? root).focus()
    const key = (event: KeyboardEvent) => {
      if (event.key !== 'Tab' || focusOwners.at(-1) !== owner) return
      const elements = focusable(), first = elements[0], last = elements.at(-1)
      if (!first) { event.preventDefault(); root.focus(); return }
      if (!root.contains(document.activeElement) || (event.shiftKey && document.activeElement === first) || (!event.shiftKey && document.activeElement === last)) {
        event.preventDefault()
        ;(event.shiftKey ? last : first)?.focus()
      }
    }
    root.addEventListener('keydown', key)
    document.addEventListener('focusin', containFocus, true)
    document.addEventListener('focusout', containFocus, true)
    return () => {
      active = false
      const ownerIndex = focusOwners.indexOf(owner)
      if (ownerIndex >= 0) focusOwners.splice(ownerIndex, 1)
      root.removeEventListener('keydown', key)
      document.removeEventListener('focusin', containFocus, true)
      document.removeEventListener('focusout', containFocus, true)
      if (ownsTabIndex) root.removeAttribute('tabindex')
      if (previous instanceof HTMLElement && previous.isConnected) previous.focus()
    }
  }, [enabled, ref])
}
