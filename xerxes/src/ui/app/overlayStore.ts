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
  goal: false,
  learning: false,
  loops: false,
  schedules: false,
  agentSettings: false,
  lspSettings: false,
  mcpSettings: false,
  contextInspector: false,
  usage: false,
  machinePicker: false,
  customAgentEditor: false,
  presetEditor: false,
  forge: false,
  activity: false,
  monitors: false,
  runs: false,
  sideQuestion: false,
  snapshots: false,
  swarm: false,
  workspaces: false,
  modelPicker: false,
  pager: null,
  pluginsHub: false,
  reasoningPicker: false,
  secret: null,
  sessions: false,
  capabilities: false,
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
export type OverlayFlagKey = Exclude<keyof OverlayState, 'agentsInitialHistoryIndex' | 'agentsInspectId'>

export const OVERLAY_FLAG_KEYS = [
  'agents',
  'approval',
  'clarify',
  'confirm',
  'copyPicker',
  'diff',
  'goal',
  'learning',
  'loops',
  'schedules',
  'agentSettings',
  'lspSettings',
  'mcpSettings',
  'contextInspector',
  'usage',
  'machinePicker',
  'customAgentEditor',
  'presetEditor',
  'forge',
  'activity',
  'monitors',
  'runs',
  'sideQuestion',
  'snapshots',
  'swarm',
  'workspaces',
  'modelPicker',
  'pager',
  'pluginsHub',
  'reasoningPicker',
  'secret',
  'sessions',
  'capabilities',
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
  goal: true,
  learning: true,
  loops: true,
  schedules: true,
  agentSettings: true,
  lspSettings: true,
  mcpSettings: true,
  contextInspector: true,
  usage: true,
  machinePicker: true,
  customAgentEditor: true,
  presetEditor: true,
  forge: true,
  activity: true,
  monitors: true,
  runs: true,
  sideQuestion: true,
  snapshots: true,
  swarm: true,
  workspaces: true,
  modelPicker: true,
  pager: true,
  pluginsHub: true,
  reasoningPicker: true,
  secret: true,
  sessions: true,
  capabilities: true,
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

/**
 * Exhaustive closed value for every overlay key in `OVERLAY_FLAG_KEYS`.
 * Compile-time checked so adding a new overlay automatically supports Ctrl+C
 * dismissal instead of silently swallowing the keypress.
 */
export const OVERLAY_CLOSED_VALUES: Pick<OverlayState, OverlayFlagKey> = {
  agents: false,
  approval: null,
  clarify: null,
  confirm: null,
  copyPicker: null,
  diff: false,
  goal: false,
  learning: false,
  loops: false,
  schedules: false,
  agentSettings: false,
  lspSettings: false,
  mcpSettings: false,
  contextInspector: false,
  usage: false,
  machinePicker: false,
  customAgentEditor: false,
  presetEditor: false,
  forge: false,
  activity: false,
  monitors: false,
  runs: false,
  sideQuestion: false,
  snapshots: false,
  swarm: false,
  workspaces: false,
  modelPicker: false,
  pager: null,
  pluginsHub: false,
  reasoningPicker: false,
  secret: null,
  sessions: false,
  capabilities: false,
  skillsHub: false,
  sudo: null,
  terminals: false
}

/**
 * Return the state patch that dismisses the first active overlay in
 * `OVERLAY_FLAG_KEYS`, or null if none is open. Callers that need RPC cancel
 * hooks (clarify, approval, sudo, secret) handle those first.
 */
export function dismissOpenOverlayPatch(state: OverlayState): null | Partial<OverlayState> {
  for (const key of OVERLAY_FLAG_KEYS) {
    if (!state[key]) {
      continue
    }
    if (key === 'agents') {
      return { agents: false, agentsInspectId: null }
    }
    return { [key]: OVERLAY_CLOSED_VALUES[key] } as Partial<OverlayState>
  }
  return null
}

/**
 * Gate clickable chrome actions so mouse clicks never stack a new overlay on
 * top of an active approval, dialog, or other blocking modal.
 */
export function openOverlayIfUnblocked(patch: Partial<OverlayState>): boolean {
  const current = $overlayState.get()
  if (overlayBlocksBackgroundHotkeys(current) || current.agents) {
    return false
  }
  patchOverlayState(patch)
  return true
}

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

type FlowOverlayKey = 'approval' | 'clarify' | 'confirm' | 'pager' | 'secret' | 'sudo'

const FLOW_OVERLAY_RESET: Pick<OverlayState, FlowOverlayKey> = {
  approval: null,
  clarify: null,
  confirm: null,
  pager: null,
  secret: null,
  sudo: null
}

/**
 * Soft reset: drop FLOW-scoped overlays (approval / clarify / confirm / sudo
 * / secret / pager) but PRESERVE user-toggled ones — agents dashboard, usage,
 * model picker, reasoning picker, skills hub, sessions overlay, etc. Those are
 * opened deliberately and shouldn't vanish when a turn ends. Called from
 * turnController.idle() on every turn completion / interrupt / error.
 */
export const resetFlowOverlays = () =>
  $overlayState.set({
    ...$overlayState.get(),
    ...FLOW_OVERLAY_RESET
  })
