// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { UsageReportView } from '../auth/usageView.js'
import type { HubEvent, WorkspaceHealth } from '../hub/protocol.js'
import type { ConnectionStatus } from './bridge.js'
import { foldSummaryEvent, presenceChips, type PresenceEntry, type SupervisorConfirmation, type SupervisorMessage } from './hubParsers.js'
import {
  taskKey, type MachineView, type PresenceChip, type SupervisorChoice, type SupervisorConfigStatus, type SupervisorConfigView, type SupervisorOptions,
  type TaskSummary, type WorkspaceView,
} from './hubTypes.js'
import { EMPTY_TAIL, foldAll, foldSessionEvent, type Block, type SessionTail } from './sessionFold.js'
import type { TaskStatus } from './taskTypes.js'

/**
 * Framework-agnostic observable state for the phone app. `reduceHub` is pure
 * and immutable; `createHubStore` only holds the current value and notifies.
 * Compatible with React's useSyncExternalStore (stable getSnapshot).
 */

export const OLDER_BLOCKS_LIMIT = 400
export const SUPERVISOR_MESSAGE_LIMIT = 100

export type ConfigPhase = 'idle' | 'requested' | 'accepted' | 'rejected' | 'unknown'

/**
 * A setting has three distinct values: what the person requested, whether the
 * Hub accepted the request, and what the session reports as effective. A
 * request is never shown as applied until the session reports it.
 */
export interface ConfigField {
  readonly effective?: string
  readonly requested?: string
  readonly phase: ConfigPhase
  readonly message?: string
}

export type DecisionKind = 'once' | 'deny' | 'session'

export interface DecisionState {
  readonly phase: 'sending' | 'failed' | 'unknown' | 'resolved'
  readonly decision: DecisionKind
  readonly message?: string
  /** Who got there first, when the decision was already resolved. */
  readonly by?: string
  /** True when someone else resolved the interaction before this decision landed. */
  readonly already?: boolean
}

export interface SendState {
  readonly phase: 'idle' | 'sending' | 'failed' | 'unknown'
  readonly message?: string
  readonly commandId?: string
}

export interface SelectedSession {
  readonly workspaceId: string
  readonly sessionId: string
  readonly phase: 'loading' | 'ready' | 'error'
  readonly error?: string
  readonly tail: SessionTail
  readonly older: readonly Block[]
  readonly olderCursor?: string | number
  readonly hasOlder: boolean
  readonly loadingOlder: boolean
  readonly olderError?: string
  readonly model: ConfigField
  readonly effort: ConfigField
  readonly goal: ConfigField & { readonly revision?: number }
  readonly decisions: Readonly<Record<string, DecisionState>>
  readonly send: SendState
}

export type SupervisorPhase = 'idle' | 'loading' | 'ready' | 'unsupported' | 'error'

export interface SupervisorState {
  readonly phase: SupervisorPhase
  readonly threadId?: string
  readonly messages: readonly SupervisorMessage[]
  readonly confirmations: readonly SupervisorConfirmation[]
  /** The submit request is in flight. */
  readonly busy: boolean
  /**
   * XERXES is running a turn on the Hub. The submit request returns as soon
   * as the message is stored; the reply arrives later as a thread event, so
   * `busy` alone ended the spinner before XERXES had started.
   */
  readonly working?: boolean
  readonly error?: string
}

/** Hub-level totals that sit beside the per-model rows. */
export interface UsageExtras {
  /** Billable spend; plan-paid usage is never added into it. */
  readonly billableCostUsd?: number
  readonly subscriptionLabel?: string
  readonly inputTokens: number
  readonly outputTokens: number
  /** Workspaces whose usage could not be read. */
  readonly unavailable: readonly string[]
}

export type UsageState =
  | { readonly phase: 'idle' }
  | { readonly phase: 'loading'; readonly report?: UsageReportView; readonly extras?: UsageExtras }
  | { readonly phase: 'ready'; readonly report: UsageReportView; readonly extras?: UsageExtras }
  | { readonly phase: 'error'; readonly message: string; readonly report?: UsageReportView; readonly extras?: UsageExtras }

export type LoadPhase = 'idle' | 'loading' | 'ready' | 'error'

/** Machines the Hub reaches (`hub.machines.list`). */
export interface MachinesState {
  readonly phase: LoadPhase
  readonly list: readonly MachineView[]
  readonly error?: string
}

/** The chooser's option lists for one machine (`supervisor.config.options`). */
export interface SupervisorOptionsState {
  readonly phase: LoadPhase
  readonly options?: SupervisorOptions
  readonly error?: string
}

/**
 * The supervisor's model choice. `status` is `unknown` until the Hub has
 * answered; `canEdit` is false until the Hub says the person may edit.
 */
export interface SupervisorConfigState {
  readonly phase: LoadPhase
  readonly status: SupervisorConfigStatus | 'unknown'
  readonly config?: SupervisorChoice
  readonly canEdit: boolean
  readonly message?: string
  readonly error?: string
  readonly options: SupervisorOptionsState
  readonly saving: boolean
  readonly saveError?: string
}

export interface HubState {
  readonly connection: ConnectionStatus
  readonly selfUserId?: string
  readonly workspaces: Readonly<Record<string, WorkspaceView>>
  readonly workspaceOrder: readonly string[]
  readonly tasks: Readonly<Record<string, TaskSummary>>
  readonly inboxLoaded: boolean
  readonly inboxError?: string
  readonly presence: Readonly<Record<string, readonly PresenceEntry[]>>
  readonly selected?: SelectedSession
  readonly usage: UsageState
  readonly supervisor: SupervisorState
  readonly machines: MachinesState
  readonly supervisorConfig: SupervisorConfigState
}

export function initialSupervisorConfig(): SupervisorConfigState {
  return { phase: 'idle', status: 'unknown', canEdit: false, options: { phase: 'idle' }, saving: false }
}

export const INITIAL_CONFIG: ConfigField = Object.freeze({ phase: 'idle' as const })

export function initialHubState(): HubState {
  return {
    connection: { state: 'connecting' },
    workspaces: {},
    workspaceOrder: [],
    tasks: {},
    inboxLoaded: false,
    presence: {},
    usage: { phase: 'idle' },
    supervisor: { phase: 'idle', messages: [], confirmations: [], busy: false },
    machines: { phase: 'idle', list: [] },
    supervisorConfig: initialSupervisorConfig(),
  }
}

export type HubAction =
  | { readonly type: 'connection'; readonly status: ConnectionStatus }
  | { readonly type: 'self'; readonly userId: string }
  | { readonly type: 'workspaces'; readonly list: readonly WorkspaceView[] }
  | { readonly type: 'workspaceHealth'; readonly workspaceId: string; readonly health: WorkspaceHealth }
  | { readonly type: 'summaries'; readonly workspaceId: string; readonly tasks: readonly TaskSummary[] }
  | { readonly type: 'inboxLoaded' }
  | { readonly type: 'inboxError'; readonly message: string | undefined }
  /** `tail: false` folds only summaries (the Inbox stream), never the open task's transcript. */
  | { readonly type: 'events'; readonly events: readonly HubEvent[]; readonly tail?: boolean }
  | { readonly type: 'presence'; readonly workspaceId: string; readonly sessionId?: string; readonly entries: readonly PresenceEntry[] }
  | { readonly type: 'select'; readonly workspaceId: string; readonly sessionId: string }
  | { readonly type: 'deselect' }
  /** The replay of the open task reached the live edge. */
  | { readonly type: 'sessionReady'; readonly workspaceId: string; readonly sessionId: string }
  /** The stream was reset (gap, expired cursor): drop the folded tail and replay again. */
  | { readonly type: 'sessionReset'; readonly workspaceId: string; readonly sessionId: string }
  | { readonly type: 'sessionError'; readonly workspaceId: string; readonly sessionId: string; readonly message: string }
  | { readonly type: 'olderLoading' }
  | { readonly type: 'olderLoaded'; readonly blocks: readonly Block[]; readonly cursor?: string | number; readonly hasMore: boolean }
  | { readonly type: 'olderFailed'; readonly message: string }
  | { readonly type: 'decision'; readonly interactionId: string; readonly state: DecisionState | undefined }
  | { readonly type: 'send'; readonly state: SendState }
  | { readonly type: 'config'; readonly field: 'model' | 'effort' | 'goal'; readonly value: Partial<ConfigField> & { readonly revision?: number } }
  | { readonly type: 'usage'; readonly state: UsageState }
  | { readonly type: 'supervisor'; readonly patch: Partial<SupervisorState> }
  | { readonly type: 'supervisorMessages'; readonly messages: readonly SupervisorMessage[] }
  | { readonly type: 'supervisorConfirmations'; readonly confirmations: readonly SupervisorConfirmation[] }
  | { readonly type: 'supervisorConfirmation'; readonly id: string; readonly patch: Partial<SupervisorConfirmation> }
  | { readonly type: 'machines'; readonly state: MachinesState }
  /** A fresh answer from `supervisor.config.get`/`.set` or a `supervisor.config_changed` event. */
  | { readonly type: 'supervisorConfigView'; readonly view: SupervisorConfigView }
  /** Everything but the Hub's answer: loading, errors, saving. */
  | { readonly type: 'supervisorConfig'; readonly patch: Partial<Pick<SupervisorConfigState, 'phase' | 'error' | 'saving' | 'saveError' | 'status'>> }
  | { readonly type: 'supervisorConfigOptions'; readonly state: SupervisorOptionsState }

export function presenceKey(workspaceId: string, sessionId?: string): string {
  return `${workspaceId}\u001f${sessionId ?? ''}`
}

function bounded<T>(items: readonly T[], limit: number): readonly T[] {
  return items.length > limit ? items.slice(items.length - limit) : items
}

function freshSelected(workspaceId: string, sessionId: string): SelectedSession {
  return {
    workspaceId,
    sessionId,
    phase: 'loading',
    tail: EMPTY_TAIL,
    older: [],
    hasOlder: false,
    loadingOlder: false,
    model: INITIAL_CONFIG,
    effort: INITIAL_CONFIG,
    goal: INITIAL_CONFIG,
    decisions: {},
    send: { phase: 'idle' },
  }
}

/** Reconcile requested values with what the session now reports as effective. */
function syncConfig(selected: SelectedSession): SelectedSession {
  const effective = selected.tail.effective
  const sync = <F extends ConfigField>(field: F, value: string | undefined): F => {
    if (value === undefined) return field
    const applied = field.requested !== undefined && field.requested === value
    const { requested: _requested, message: _message, ...rest } = field
    return applied && (field.phase === 'accepted' || field.phase === 'requested' || field.phase === 'unknown')
      ? ({ ...rest, effective: value, phase: 'idle' } as F)
      : ({ ...field, effective: value } as F)
  }
  const model = sync(selected.model, effective.model)
  const effort = sync(selected.effort, effective.effort)
  const goal = sync(selected.goal, effective.goal)
  return model === selected.model && effort === selected.effort && goal === selected.goal ? selected : { ...selected, model, effort, goal }
}

function applyEvents(state: HubState, events: readonly HubEvent[], foldTail: boolean): HubState {
  let tasks = state.tasks
  let selected = state.selected
  for (const event of events) {
    const sessionId = event.session_id
    if (sessionId !== undefined) {
      const key = taskKey(event.workspace_id, sessionId)
      const next = foldSummaryEvent(event.workspace_id, tasks[key], event)
      if (next !== tasks[key]) {
        const copy = { ...tasks }
        if (next === undefined) delete copy[key]
        else copy[key] = next
        tasks = copy
      }
      if (foldTail && selected && selected.workspaceId === event.workspace_id && selected.sessionId === sessionId) {
        const tail = foldSessionEvent(selected.tail, { id: event.event_id, type: event.type, payload: event.payload, ...(event.actor ? { actor: event.actor } : {}) })
        if (tail !== selected.tail) {
          // A resolved interaction clears any decision UI that was waiting on it.
          const decisions: Record<string, DecisionState> = { ...selected.decisions }
          for (const id of Object.keys(selected.decisions)) {
            if (!tail.pending.some(item => item.id === id) && selected.tail.pending.some(item => item.id === id)) {
              const resolution = tail.resolved[id]
              const mine = selected.decisions[id]
              if (mine && mine.phase === 'sending') delete decisions[id]
              else if (mine) decisions[id] = { ...mine, phase: 'resolved', ...(resolution?.by ? { by: resolution.by } : {}) }
            }
          }
          selected = syncConfig({ ...selected, tail, decisions })
        }
      }
    }
  }
  return tasks === state.tasks && selected === state.selected ? state : { ...state, tasks, ...(selected ? { selected } : {}) }
}

export function reduceHub(state: HubState, action: HubAction): HubState {
  switch (action.type) {
    case 'connection': return { ...state, connection: action.status }
    case 'self': return { ...state, selfUserId: action.userId }
    case 'workspaces': {
      const map: Record<string, WorkspaceView> = {}
      for (const w of action.list) map[w.id] = w
      return { ...state, workspaces: map, workspaceOrder: action.list.map(w => w.id) }
    }
    case 'workspaceHealth': {
      const found = state.workspaces[action.workspaceId]
      if (!found || found.health === action.health) return state
      return { ...state, workspaces: { ...state.workspaces, [found.id]: { ...found, health: action.health } } }
    }
    case 'summaries': {
      const tasks: Record<string, TaskSummary> = {}
      for (const [key, task] of Object.entries(state.tasks)) if (task.workspaceId !== action.workspaceId) tasks[key] = task
      for (const task of action.tasks) tasks[taskKey(task.workspaceId, task.id)] = task
      return { ...state, tasks }
    }
    case 'inboxLoaded': {
      if (state.inboxLoaded && state.inboxError === undefined) return state
      const { inboxError: _inboxError, ...rest } = state
      return { ...rest, inboxLoaded: true }
    }
    case 'inboxError': {
      if (action.message === undefined) {
        if (state.inboxError === undefined) return state
        const { inboxError: _inboxError, ...rest } = state
        return rest
      }
      return state.inboxError === action.message ? state : { ...state, inboxError: action.message }
    }
    case 'events': return applyEvents(state, action.events, action.tail !== false)
    case 'presence': {
      const key = presenceKey(action.workspaceId, action.sessionId)
      return { ...state, presence: { ...state.presence, [key]: action.entries } }
    }
    case 'select': {
      if (state.selected && state.selected.workspaceId === action.workspaceId && state.selected.sessionId === action.sessionId) return state
      return { ...state, selected: freshSelected(action.workspaceId, action.sessionId) }
    }
    case 'deselect': {
      if (!state.selected) return state
      const { selected: _selected, ...rest } = state
      return rest
    }
    case 'sessionReady': {
      const selected = state.selected
      if (!selected || selected.workspaceId !== action.workspaceId || selected.sessionId !== action.sessionId) return state
      if (selected.phase === 'ready') return state
      const { error: _error, ...rest } = selected
      // Earlier turns come from hub.history; the replayed tail only covers what the journal kept.
      return { ...state, selected: { ...rest, phase: 'ready', hasOlder: true } }
    }
    case 'sessionReset': {
      const selected = state.selected
      if (!selected || selected.workspaceId !== action.workspaceId || selected.sessionId !== action.sessionId) return state
      const { error: _error, olderCursor: _oc, olderError: _oe, ...rest } = selected
      return { ...state, selected: { ...rest, phase: 'loading', tail: EMPTY_TAIL, older: [], hasOlder: false, loadingOlder: false } }
    }
    case 'sessionError': {
      const selected = state.selected
      if (!selected || selected.workspaceId !== action.workspaceId || selected.sessionId !== action.sessionId) return state
      return { ...state, selected: { ...selected, phase: selected.phase === 'ready' ? 'ready' : 'error', error: action.message } }
    }
    case 'olderLoading': return state.selected ? { ...state, selected: { ...state.selected, loadingOlder: true } } : state
    case 'olderLoaded': {
      const selected = state.selected
      if (!selected) return state
      const { olderError: _oe, olderCursor: _oc, ...rest } = selected
      const older = bounded([...action.blocks, ...selected.older], OLDER_BLOCKS_LIMIT)
      return { ...state, selected: { ...rest, older, loadingOlder: false, hasOlder: action.hasMore, ...(action.cursor ? { olderCursor: action.cursor } : {}) } }
    }
    case 'olderFailed': return state.selected ? { ...state, selected: { ...state.selected, loadingOlder: false, olderError: action.message } } : state
    case 'decision': {
      const selected = state.selected
      if (!selected) return state
      const decisions = { ...selected.decisions }
      if (action.state === undefined) delete decisions[action.interactionId]
      else decisions[action.interactionId] = action.state
      return { ...state, selected: { ...selected, decisions } }
    }
    case 'send': return state.selected ? { ...state, selected: { ...state.selected, send: action.state } } : state
    case 'config': {
      const selected = state.selected
      if (!selected) return state
      const current = selected[action.field]
      const next = { ...current, ...action.value }
      if (action.value.phase !== undefined && action.value.message === undefined) delete (next as { message?: string }).message
      return { ...state, selected: { ...selected, [action.field]: next } }
    }
    case 'usage': return { ...state, usage: action.state }
    case 'supervisor': return { ...state, supervisor: { ...state.supervisor, ...action.patch } }
    case 'supervisorMessages': {
      const known = new Set(state.supervisor.messages.map(m => m.id))
      const added = action.messages.filter(m => !known.has(m.id))
      if (added.length === 0) return state
      return { ...state, supervisor: { ...state.supervisor, messages: bounded([...state.supervisor.messages, ...added], SUPERVISOR_MESSAGE_LIMIT) } }
    }
    case 'supervisorConfirmations': {
      const byId = new Map(state.supervisor.confirmations.map(c => [c.id, c]))
      for (const c of action.confirmations) byId.set(c.id, c)
      return { ...state, supervisor: { ...state.supervisor, confirmations: bounded([...byId.values()], 50) } }
    }
    case 'supervisorConfirmation': {
      return {
        ...state,
        supervisor: { ...state.supervisor, confirmations: state.supervisor.confirmations.map(c => (c.id === action.id ? { ...c, ...action.patch } : c)) },
      }
    }
    case 'machines': return { ...state, machines: action.state }
    case 'supervisorConfigView': {
      const { config: _config, message: _message, error: _error, saveError: _saveError, ...rest } = state.supervisorConfig
      const view = action.view
      return {
        ...state,
        supervisorConfig: {
          ...rest,
          phase: 'ready',
          status: view.status,
          canEdit: view.canEdit,
          saving: false,
          ...(view.config ? { config: view.config } : {}),
          ...(view.message ? { message: view.message } : {}),
        },
      }
    }
    case 'supervisorConfig': {
      const next = { ...state.supervisorConfig, ...action.patch }
      // A fresh phase without an error clears the old one.
      if (action.patch.phase !== undefined && action.patch.error === undefined) delete (next as { error?: string }).error
      if (action.patch.saving === true && action.patch.saveError === undefined) delete (next as { saveError?: string }).saveError
      return { ...state, supervisorConfig: next }
    }
    case 'supervisorConfigOptions': return { ...state, supervisorConfig: { ...state.supervisorConfig, options: action.state } }
  }
}

// ── selectors ─────────────────────────────────────────────────────────

export type InboxStatusFilter = 'all' | 'needs_you' | 'working' | 'failed' | 'idle'

export interface InboxFilter {
  readonly workspaceId?: string
  readonly status?: InboxStatusFilter
}

const RANK: Readonly<Record<TaskStatus, number>> = { needs_you: 0, working: 1, failed: 2, unknown: 3, idle: 4 }

export function workspaceAvailable(workspace: WorkspaceView | undefined): boolean {
  return workspace !== undefined && workspace.health !== 'offline' && workspace.health !== 'needs_operator'
}

/** Needs you first, then working, then the rest by recency. Unavailable workspaces sink to the end. */
export function orderTasks(tasks: readonly TaskSummary[], workspaces: Readonly<Record<string, WorkspaceView>>): readonly TaskSummary[] {
  return [...tasks].sort((a, b) => {
    const au = workspaceAvailable(workspaces[a.workspaceId]) ? 0 : 1
    const bu = workspaceAvailable(workspaces[b.workspaceId]) ? 0 : 1
    if (au !== bu) return au - bu
    const rank = RANK[a.status] - RANK[b.status]
    if (rank !== 0) return rank
    const time = Date.parse(b.updatedAt) - Date.parse(a.updatedAt)
    if (time !== 0 && !Number.isNaN(time)) return time
    return a.id < b.id ? -1 : a.id > b.id ? 1 : 0
  })
}

export function inboxTasks(state: HubState, filter: InboxFilter = {}): readonly TaskSummary[] {
  const all = Object.values(state.tasks).filter(task => {
    if (filter.workspaceId !== undefined && task.workspaceId !== filter.workspaceId) return false
    const status = filter.status ?? 'all'
    return status === 'all' || task.status === status
  })
  return orderTasks(all, state.workspaces)
}

export interface InboxCounts {
  readonly needsYou: number
  readonly working: number
  readonly failed: number
  readonly total: number
  readonly unavailable: number
}

/** Counts exclude unavailable workspaces: their status is outdated and cannot be acted on. */
export function inboxCounts(state: HubState): InboxCounts {
  let needsYou = 0
  let working = 0
  let failed = 0
  let unavailable = 0
  const tasks = Object.values(state.tasks)
  for (const task of tasks) {
    if (!workspaceAvailable(state.workspaces[task.workspaceId])) { unavailable += 1; continue }
    if (task.status === 'needs_you') needsYou += 1
    else if (task.status === 'working') working += 1
    else if (task.status === 'failed') failed += 1
  }
  return { needsYou, working, failed, total: tasks.length, unavailable }
}

export function presenceFor(state: HubState, workspaceId: string, sessionId?: string): readonly PresenceChip[] {
  return presenceChips(state.presence[presenceKey(workspaceId, sessionId)] ?? [], state.selfUserId)
}

/** Mutations need a live link, a reachable workspace, and a runtime that verifies signed commands. */
export function canMutate(state: HubState, workspaceId: string): boolean {
  const workspace = state.workspaces[workspaceId]
  return state.connection.state === 'live' && workspaceAvailable(workspace) && workspace?.health !== 'stale' && workspace?.readOnly !== true
}

export interface ConnectionBanner {
  readonly tone: 'busy' | 'problem'
  readonly text: string
}

export function connectionBanner(status: ConnectionStatus): ConnectionBanner | undefined {
  switch (status.state) {
    case 'live': return undefined
    case 'connecting': return { tone: 'busy', text: 'Connecting…' }
    case 'replaying': return { tone: 'busy', text: 'Catching up…' }
    case 'reconnecting': return { tone: 'problem', text: 'Reconnecting — status may be outdated' }
    case 'revoked': return { tone: 'problem', text: status.reason }
    case 'closed': return { tone: 'problem', text: 'Disconnected' }
  }
}

// ── store ─────────────────────────────────────────────────────────────

export interface HubStore {
  getSnapshot(): HubState
  subscribe(listener: () => void): () => void
  dispatch(action: HubAction): void
}

export interface HubStoreOptions {
  /** Coalesce notifications: 0 (default) notifies synchronously. */
  readonly coalesceMs?: number
}

export function createHubStore(initial: HubState = initialHubState(), options: HubStoreOptions = {}): HubStore {
  let state = initial
  const listeners = new Set<() => void>()
  let timer: ReturnType<typeof setTimeout> | undefined
  const flush = (): void => {
    timer = undefined
    for (const listener of [...listeners]) listener()
  }
  return {
    getSnapshot: () => state,
    subscribe(listener) {
      listeners.add(listener)
      return () => { listeners.delete(listener) }
    },
    dispatch(action) {
      const next = reduceHub(state, action)
      if (next === state) return
      state = next
      const wait = options.coalesceMs ?? 0
      if (wait <= 0) flush()
      else timer ??= setTimeout(flush, wait)
    },
  }
}
