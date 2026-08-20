// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { atom, computed } from 'nanostores'

import type { OverlayState } from './interfaces.js'

const buildOverlayState = (): OverlayState => ({
  agents: false,
  agentsInitialHistoryIndex: 0,
  agentsInspectId: null,
  approval: null,
  clarify: null,
  confirm: null,
  copyPicker: null,
  diff: false,
  modelPicker: false,
  pager: null,
  pluginsHub: false,
  reasoningPicker: false,
  secret: null,
  sessions: false,
  skillsHub: false,
  sudo: null,
  terminals: false
})

export const $overlayState = atom<OverlayState>(buildOverlayState())

export const $isBlocked = computed(
  $overlayState,
  ({ agents, approval, clarify, confirm, copyPicker, diff, modelPicker, pager, pluginsHub, reasoningPicker, secret, sessions, skillsHub, sudo, terminals }) =>
    Boolean(
      agents ||
      approval ||
      clarify ||
      confirm ||
      copyPicker ||
      diff ||
      modelPicker ||
      pager ||
      pluginsHub ||
      reasoningPicker ||
      secret ||
      sessions ||
      skillsHub ||
      sudo ||
      terminals
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
 * picker, reasoning picker, skills hub, sessions overlay. Those are opened
 * deliberately and shouldn't vanish when a turn ends. Called from
 * turnController.idle() on every turn completion / interrupt; the old "reset
 * everything" behaviour silently closed /agents the moment delegation
 * finished.
 */
export const resetFlowOverlays = () =>
  $overlayState.set({
    ...buildOverlayState(),
    agents: $overlayState.get().agents,
    agentsInitialHistoryIndex: $overlayState.get().agentsInitialHistoryIndex,
    agentsInspectId: $overlayState.get().agentsInspectId,
    copyPicker: $overlayState.get().copyPicker,
    diff: $overlayState.get().diff,
    modelPicker: $overlayState.get().modelPicker,
    pluginsHub: $overlayState.get().pluginsHub,
    reasoningPicker: $overlayState.get().reasoningPicker,
    sessions: $overlayState.get().sessions,
    skillsHub: $overlayState.get().skillsHub,
    terminals: $overlayState.get().terminals
  })
