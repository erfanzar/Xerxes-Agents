// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useEffect, useState, type ReactElement } from 'react'

import { Screens } from './Screens.js'
import { INITIAL_STATE, type DesktopState } from './viewModel.js'

/** Colour updates per second on the idle screen's animated wordmark. */
const WORDMARK_FRAME_MS = 80
const WORDMARK_FRAMES = 80

export function App(): ReactElement {
  const [state, setState] = useState<DesktopState>({
    ...INITIAL_STATE,
    w: typeof window === 'undefined' ? 1600 : window.innerWidth,
  })
  const set = (patch: Partial<DesktopState>) => setState(previous => ({ ...previous, ...patch }))

  // The layout drops panels by width — notes before the rail, rail before the
  // centre pane — so the view model needs the real window width, not a guess.
  useEffect(() => {
    const onResize = () => setState(previous => ({ ...previous, w: window.innerWidth }))
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [])

  // Only the idle screen animates, and only while it is the one on screen: an
  // idle window should not ask the machine for work.
  const animating = state.screen === 'session' && state.turn === 'idle'
  useEffect(() => {
    if (!animating) return
    const timer = setInterval(
      () => setState(previous => ({ ...previous, frame: (previous.frame + 1) % WORDMARK_FRAMES })),
      WORDMARK_FRAME_MS,
    )
    return () => clearInterval(timer)
  }, [animating])

  // The generated stylesheet keys dark/light off `data-theme` on the root, and
  // treats an absent attribute as "follow the system".
  useEffect(() => {
    const root = document.documentElement
    if (state.theme === 'auto') root.removeAttribute('data-theme')
    else root.setAttribute('data-theme', state.theme)
  }, [state.theme])

  return <Screens state={state} set={set} />
}
