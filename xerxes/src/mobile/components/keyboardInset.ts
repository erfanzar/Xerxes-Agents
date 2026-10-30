// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * iOS keeps the layout viewport full height while the keyboard is up, so a
 * sticky composer sits behind the keyboard. The visual viewport shrinks; the
 * difference is how far the composer has to rise (`--keyboard-inset`).
 */

export interface VisualViewportLike {
  readonly height: number
  readonly offsetTop: number
}

export function keyboardInset(innerHeight: number, viewport: VisualViewportLike | null | undefined): number {
  if (viewport === null || viewport === undefined) return 0
  const inset = innerHeight - viewport.height - viewport.offsetTop
  return Number.isFinite(inset) && inset > 1 ? Math.round(inset) : 0
}

export interface KeyboardInsetHost {
  readonly innerHeight: number
  readonly visualViewport: (VisualViewportLike & Pick<EventTarget, 'addEventListener' | 'removeEventListener'>) | null
}

/** Keeps `--keyboard-inset` on `root` in step with the visual viewport; returns the detach function. */
export function trackKeyboardInset(host: KeyboardInsetHost, root: { readonly style: Pick<CSSStyleDeclaration, 'setProperty' | 'removeProperty'> }): () => void {
  const viewport = host.visualViewport
  if (viewport === null) return () => {}
  const apply = (): void => { root.style.setProperty('--keyboard-inset', `${keyboardInset(host.innerHeight, viewport)}px`) }
  apply()
  viewport.addEventListener('resize', apply)
  viewport.addEventListener('scroll', apply)
  return () => {
    viewport.removeEventListener('resize', apply)
    viewport.removeEventListener('scroll', apply)
    root.style.removeProperty('--keyboard-inset')
  }
}
