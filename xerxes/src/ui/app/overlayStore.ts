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
  machinePicker: false,
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

/**
 * Overlay flags that gate UI behaviour. `agentsInitialHistoryIndex` and
 * `agentsInspectId` are metadata carried alongside the `agents` flag, never
 * gates on their own — excluding them here means a new boolean-ish overlay
 * added to `OverlayState` automatically joins every policy table below.
 */
type OverlayFlagKey = Exclude<keyof OverlayState, 'agentsInitialHistoryIndex' | 'agentsInspectId'>

const OVERLAY_FLAG_KEYS = [
  'agents',
  'approval',
  'clarify',
  'confirm',
  'copyPicker',
  'diff',
  'machinePicker',
  'modelPicker',
  'pager',
  'pluginsHub',
  'reasoningPicker',
  'secret',
  'sessions',
  'skillsHub',
  'sudo',
  'terminals'
] as const satisfies readonly OverlayFlagKey[]

export const $isBlocked = computed($overlayState, state =>
  OVERLAY_FLAG_KEYS.some(key => Boolean(state[key]))
)

/**
 * Per-overlay hotkey policy: does this overlay, while open, suppress the
 * background global hotkeys (F6/F7/F8 panels, session-tab chords)?
 *
 * `agents` is deliberately false — F6 must stay live to CLOSE the agents
 * overlay, the same chord that opened it. Every other overlay blocks: a
 * background hotkey firing underneath a modal prompt would mutate state the
 * user cannot see.
 *
 * Structured as a complete Record so adding a new overlay to OverlayState
 * without a policy entry is a compile error HERE, not a silent key leak at
 * the hotkey call sites.
 */
export const OVERLAY_BLOCKS_BACKGROUND_HOTKEYS: Record<OverlayFlagKey, boolean> = {
  agents: false,
  approval: true,
  clarify: true,
  confirm: true,
  copyPicker: true,
  diff: true,
  machinePicker: true,
  modelPicker: true,
  pager: true,
  pluginsHub: true,
  reasoningPicker: true,
  secret: true,
  sessions: true,
  skillsHub: true,
  sudo: true,
  terminals: true
}

/**
 * True while any hotkey-blocking overlay is open. Single source of truth for
 * the `disabled` prop of the background hotkey components — call sites must
 * never re-enumerate overlay flags by hand.
 */
export const overlayBlocksBackgroundHotkeys = (state: OverlayState): boolean =>
  OVERLAY_FLAG_KEYS.some(key => OVERLAY_BLOCKS_BACKGROUND_HOTKEYS[key] && Boolean(state[key]))

export const $backgroundHotkeysBlocked = computed($overlayState, overlayBlocksBackgroundHotkeys)

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
