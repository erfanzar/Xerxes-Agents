// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

// JSX-free on purpose: the runtime tsconfig includes src/**/*.ts but sets no --jsx,
// so nothing here may import a .tsx module. Render helpers live in renderHelpers.tsx.

import type { HubController } from '../../client/hubController.js'
import { initialHubState, initialSupervisorConfig, type HubState, type SupervisorConfigState } from '../../client/hubStore.js'
import type { MachineView, SupervisorOptions, TaskSummary, WorkspaceView } from '../../client/hubTypes.js'
import { NO_ACTIVITY } from '../../client/taskTypes.js'
import type { HubEvent, WorkspaceHealth } from '../../hub/protocol.js'

export const NOW = Date.parse('2026-10-21T12:00:00Z')
export const minutesAgo = (minutes: number): string => new Date(NOW - minutes * 60_000).toISOString()

export function stubController(overrides: Partial<HubController> = {}): HubController {
  const noop = async (): Promise<void> => {}
  return {
    imagesSupported: false,
    start() {}, dispose() {}, refreshInbox: noop, openSession() {}, closeSession() {}, loadOlder: noop,
    submit: async () => true, steer: async () => true, stopTurn: async () => true,
    decide: noop, answerQuestion: noop, setModel: noop, setGoal: noop, loadUsage: noop, setPresence() {},
    supervisorOpen: noop, supervisorSend: async () => true, supervisorRespond: noop,
    machinesList: noop, supervisorConfigGet: noop, supervisorConfigOptions: noop, supervisorConfigSet: async () => true,
    loadAutopilot: noop, setAutopilot: async () => true, autopilotThreadOpen: noop, autopilotThreadRespond: noop,
    loadWatches: noop, setWatch: async () => true,
    ...overrides,
  }
}

export function workspace(id: string, label: string, health: WorkspaceHealth = 'online'): WorkspaceView {
  return { id, label, health, models: [], efforts: [] }
}

export function summary(partial: Partial<TaskSummary> & { id: string }): TaskSummary {
  return {
    workspaceId: 'w1', title: partial.id, status: 'idle', model: '', goalSummary: '', pendingCount: 0, interactionIds: [],
    agents: { total: 0, active: 0 }, updatedAt: minutesAgo(10), stale: false, interrupted: false, activity: NO_ACTIVITY,
    ...partial,
  }
}

export function liveState(tasks: readonly TaskSummary[] = [], workspaces: readonly WorkspaceView[] = [workspace('w1', 'Mac')]): HubState {
  const map: Record<string, WorkspaceView> = {}
  for (const w of workspaces) map[w.id] = w
  const taskMap: Record<string, TaskSummary> = {}
  for (const t of tasks) taskMap[`${t.workspaceId}\u001f${t.id}`] = t
  return {
    ...initialHubState(),
    connection: { state: 'live' },
    workspaces: map,
    workspaceOrder: workspaces.map(w => w.id),
    tasks: taskMap,
    inboxLoaded: true,
  }
}

export function machine(id: string, label: string, workspaceIds: readonly string[], extra: Partial<MachineView> = {}): MachineView {
  return { id, label, health: 'online', readOnly: false, modelsSupported: true, workspaceIds, ...extra }
}

export const OPTIONS: SupervisorOptions = {
  machineId: 'm1',
  profiles: [{ name: 'work', label: 'Work', provider: 'anthropic', defaultModel: 'sonnet' }, { name: 'home', label: 'Home', provider: 'openai', defaultModel: 'gpt' }],
  profile: 'work',
  models: [{ id: 'sonnet', label: 'Sonnet' }, { id: 'opus', label: 'Opus' }],
  model: 'sonnet',
  efforts: [{ effort: 'low' }, { effort: 'high', description: 'Slowest, most careful' }],
  defaultEffort: 'low',
}

export function withSupervisorConfig(state: HubState, patch: Partial<SupervisorConfigState>): HubState {
  return { ...state, supervisor: { phase: 'ready', messages: [], confirmations: [], busy: false }, supervisorConfig: { ...initialSupervisorConfig(), phase: 'ready', ...patch } }
}

let sequence = 0
export function hubEvent(type: string, payload: Record<string, unknown>, extra: Partial<HubEvent> = {}): HubEvent {
  sequence += 1
  return {
    event_id: `e${sequence}`, sequence, cursor: `c${sequence}`, workspace_id: 'w1', session_id: 's1', type, payload,
    occurred_at: new Date(NOW + sequence * 1000).toISOString(), ...extra,
  }
}
