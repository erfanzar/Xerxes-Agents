// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { atom, computed } from 'nanostores'

import type { OverlayState } from './interfaces.js'

const buildOverlayState = (): OverlayState => ({
  agents: false,
  agentsInitialHistoryIndex: 0,
  approval: null,
  clarify: null,
  confirm: null,
  modelPicker: false,
  pager: null,
  pluginsHub: false,
  secret: null,
  sessions: false,
  skillsHub: false,
  sudo: null
})

export const $overlayState = atom<OverlayState>(buildOverlayState())

export const $isBlocked = computed(
  $overlayState,
  ({ agents, approval, clarify, confirm, modelPicker, pager, pluginsHub, secret, sessions, skillsHub, sudo }) =>
    Boolean(
      agents ||
      approval ||
      clarify ||
      confirm ||
      modelPicker ||
      pager ||
      pluginsHub ||
      secret ||
      sessions ||
      skillsHub ||
      sudo
    )
)

export const getOverlayState = () => $overlayState.get()

export const patchOverlayState = (next: Partial<OverlayState> | ((state: OverlayState) => OverlayState)) =>
  $overlayState.set(typeof next === 'function' ? next($overlayState.get()) : { ...$overlayState.get(), ...next })

/** Close one approval without erasing a newer request that replaced it. */
export const clearApprovalOverlay = (requestId: string): boolean => {
  const current = $overlayState.get()

  if (current.approval?.requestId !== requestId) {
    return false
  }

  patchOverlayState({ approval: null })

  return true
}

/**
 * Close one clarify prompt without accidentally dismissing a newer prompt
 * emitted by the daemon while the previous answer was still in flight.
 */
export const clearClarifyOverlay = (requestId: string): boolean => {
  const current = $overlayState.get()

  if (current.clarify?.requestId !== requestId) {
    return false
  }

  patchOverlayState({ clarify: null })

  return true
}

/** Full reset — used by session/turn teardown and tests. */
export const resetOverlayState = () => $overlayState.set(buildOverlayState())

/**
 * Soft reset: drop FLOW-scoped overlays (approval / clarify / confirm / sudo
 * / secret / pager) but PRESERVE user-toggled ones — agents dashboard, model
 * picker, skills hub, sessions overlay.  Those are opened deliberately and
 * shouldn't vanish when a turn ends.  Called from turnController.idle() on
 * every turn completion / interrupt; the old "reset everything" behaviour
 * silently closed /agents the moment delegation finished.
 */
export const resetFlowOverlays = () =>
  $overlayState.set({
    ...buildOverlayState(),
    agents: $overlayState.get().agents,
    agentsInitialHistoryIndex: $overlayState.get().agentsInitialHistoryIndex,
    modelPicker: $overlayState.get().modelPicker,
    pluginsHub: $overlayState.get().pluginsHub,
    sessions: $overlayState.get().sessions,
    skillsHub: $overlayState.get().skillsHub
  })
