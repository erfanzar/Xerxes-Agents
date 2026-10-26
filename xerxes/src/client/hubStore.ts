// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { UsageReportView } from '../auth/usageView.js'
import type { HubEvent, WorkspaceHealth } from '../hub/protocol.js'
import type { ConnectionStatus } from './bridge.js'
import { foldSummaryEvent, presenceChips, type HistoryPage, type PresenceEntry, type SupervisorConfirmation, type SupervisorMessage } from './hubParsers.js'
import {
  taskKey, type AutopilotRow, type AutopilotView, type MachineView, type PresenceChip, type SupervisorChoice, type SupervisorConfigStatus, type SupervisorConfigView, type SupervisorOptions,
  type TaskSummary, type WatchRow, type WatchView, type WorkspaceView,
} from './hubTypes.js'
import { EMPTY_TAIL, foldAll, foldSessionEvent, type Block, type PendingInteraction, type SessionTail } from './sessionFold.js'
import type { TaskStatus } from './taskTypes.js'

/**
 * Framework-agnostic observable state for the phone app. `reduceHub` is pure
 * and immutable; `createHubStore` only holds the current value and notifies.
 * Compatible with React's useSyncExternalStore (stable getSnapshot).
 */

// Together with the 40-block live tail, keep at most 400 transcript blocks.
export const OLDER_BLOCKS_LIMIT = 360
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
  /** Authoritative history, chronological. The live tail follows it. */
  readonly older: readonly Block[]
  readonly historyActionIds: readonly string[]
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

/**
 * Autopilot per workspace (`supervisor.autopilot.get`). `canEdit` is false
 * until the Hub says the person is the owner; `rows` is empty until it answers.
 */
export interface AutopilotState {
  readonly phase: LoadPhase
  readonly canEdit: boolean
  readonly rows: readonly AutopilotRow[]
  readonly error?: string
  /** A save is in flight for this workspace id. */
  readonly savingId?: string
  readonly saveError?: string
  /** True once this Hub answered `unsupported`: the control is hidden. */
  readonly unsupported: boolean
}

/** `savingId: undefined` explicitly ends a save; an absent key leaves it alone. */
export interface AutopilotPatch {
  readonly phase?: LoadPhase
  readonly error?: string
  readonly savingId?: string | undefined
  readonly saveError?: string
  readonly unsupported?: boolean
}

export function initialAutopilot(): AutopilotState {
  return { phase: 'idle', canEdit: false, rows: [], unsupported: false }
}

export type AutopilotThreadPhase = 'idle' | 'loading' | 'ready' | 'hidden' | 'error'

/**
 * The owner's private Autopilot thread (`autopilot-<ownerUserId>`). Its
 * messages and cards are kept apart from the shared thread: events are routed
 * by `thread_id`. `hidden` means the Hub said the thread is not ours (or not
 * offered); it is never asked for again on this connection.
 */
export interface AutopilotThreadState {
  readonly phase: AutopilotThreadPhase
  readonly threadId?: string
  readonly messages: readonly SupervisorMessage[]
  readonly confirmations: readonly SupervisorConfirmation[]
  readonly working?: boolean
  readonly error?: string
}

export function initialAutopilotThread(): AutopilotThreadState {
  return { phase: 'idle', messages: [], confirmations: [] }
}

/** Cards in the Autopilot thread that wait for the owner's Confirm. */
export function autopilotPendingCount(state: Pick<HubState, 'autopilotThread'>): number {
  return state.autopilotThread.confirmations.filter(card => card.state === 'pending' && card.auto !== true).length
}

/**
 * Watch per task (`supervisor.watch.list`). `canEdit` is false until the Hub
 * says the person is the owner; `rows` is empty until it answers. Rows are
 * unique per workspace id + task id.
 */
export interface WatchState {
  readonly phase: LoadPhase
  readonly canEdit: boolean
  readonly rows: readonly WatchRow[]
  readonly error?: string
  /** A save is in flight for this `watchKey`. */
  readonly savingKey?: string
  readonly saveError?: string
  /** True once this Hub said it does not know Watch: the section is hidden. */
  readonly unsupported: boolean
}

/** `savingKey: undefined` explicitly ends a save; an absent key leaves it alone. */
export interface WatchPatch {
  readonly phase?: LoadPhase
  readonly error?: string
  readonly savingKey?: string | undefined
  readonly saveError?: string
  readonly unsupported?: boolean
}

export function initialWatch(): WatchState {
  return { phase: 'idle', canEdit: false, rows: [], unsupported: false }
}

/** The key of one watched task in `WatchState.rows` and `savingKey`. */
export function watchKey(workspaceId: string, taskId: string): string {
  return `${workspaceId}\u001f${taskId}`
}

/** Watches that are on right now (the Hub's verdict). */
export function activeWatchRows(state: Pick<HubState, 'watch'>): readonly WatchRow[] {
  return state.watch.rows.filter(row => row.active)
}

/** The row for one task, if the Hub listed it. */
export function watchRowFor(state: Pick<HubState, 'watch'>, workspaceId: string, taskId: string): WatchRow | undefined {
  return state.watch.rows.find(row => row.workspaceId === workspaceId && row.taskId === taskId)
}

/**
 * Whether this person is the Hub owner. Autopilot's `can_edit` answers it once
 * the Autopilot list has loaded. While that list is unknown (never loaded,
 * failed, or this Hub has no Autopilot) Watch's `can_edit` is the signal, since
 * the Hub gives both to the owner only. A loaded Autopilot answer always wins.
 */
export function isHubOwner(state: Pick<HubState, 'autopilot' | 'watch'>): boolean {
  if (state.autopilot.canEdit) return true
  const autopilotKnown = state.autopilot.phase === 'ready' && !state.autopilot.unsupported
  return !autopilotKnown && state.watch.canEdit
}

/**
 * Whether the Autopilot thread view is offered: only the Hub owner (see
 * `isHubOwner`), while a workspace has Autopilot on, or a task is being
 * watched (watch wakes land in this thread too), or cards still wait for a
 * Confirm; and never after the Hub refused to open it.
 */
export function autopilotThreadAvailable(state: Pick<HubState, 'autopilot' | 'autopilotThread' | 'selfUserId' | 'watch'>): boolean {
  if (!isHubOwner(state) || state.selfUserId === undefined || state.autopilotThread.phase === 'hidden') return false
  return state.autopilot.rows.some(row => row.active) || state.watch.rows.some(row => row.active) || autopilotPendingCount(state) > 0
}

/** Rows that are on right now (the Hub's verdict). */
export function activeAutopilotRows(state: Pick<HubState, 'autopilot'>): readonly AutopilotRow[] {
  return state.autopilot.rows.filter(row => row.active)
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
  readonly autopilot: AutopilotState
  readonly autopilotThread: AutopilotThreadState
  readonly watch: WatchState
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
    autopilot: initialAutopilot(),
    autopilotThread: initialAutopilotThread(),
    watch: initialWatch(),
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
  | { readonly type: 'historyLoaded'; readonly workspaceId: string; readonly sessionId: string; readonly page: HistoryPage; readonly throughIds: readonly string[] }
  | { readonly type: 'pendingRestored'; readonly workspaceId: string; readonly sessionId: string; readonly pending: readonly PendingInteraction[] }
  | { readonly type: 'sessionReady'; readonly workspaceId: string; readonly sessionId: string }
  /** The stream was reset (gap, expired cursor): refetch history, never replay the journal. */
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
  /** A fresh `supervisor.autopilot.get` answer replaces every row. */
  | { readonly type: 'autopilotView'; readonly view: AutopilotView }
  /** One row from `supervisor.autopilot.set` or a `supervisor.autopilot_changed` event. */
  | { readonly type: 'autopilotRow'; readonly row: AutopilotRow }
  /** Loading, errors, saving and unsupported. */
  | { readonly type: 'autopilot'; readonly patch: AutopilotPatch }
  /** Phase, id, working flag and errors of the owner's Autopilot thread. */
  | { readonly type: 'autopilotThread'; readonly patch: Partial<AutopilotThreadState> }
  | { readonly type: 'autopilotThreadMessages'; readonly messages: readonly SupervisorMessage[] }
  | { readonly type: 'autopilotThreadConfirmations'; readonly confirmations: readonly SupervisorConfirmation[] }
  | { readonly type: 'autopilotThreadConfirmation'; readonly id: string; readonly patch: Partial<SupervisorConfirmation> }
  /** A fresh `supervisor.watch.list` answer replaces every row. */
  | { readonly type: 'watchView'; readonly view: WatchView }
  /** One row from `supervisor.watch.set` or a `supervisor.watch_update` event. */
  | { readonly type: 'watchRow'; readonly row: WatchRow }
  /** Loading, errors, saving and unsupported. */
  | { readonly type: 'watch'; readonly patch: WatchPatch }

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
    historyActionIds: [],
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
        const input = { id: event.event_id, type: event.type, payload: event.payload, ...(event.actor ? { actor: event.actor } : {}) }
        const folded = foldSessionEvent(selected.tail, input)
        const historyTools = new Set(selected.older.flatMap(block => block.kind === 'tool' ? [block.callId] : []))
        const tail = event.type === 'tool_call' ? { ...folded, blocks: folded.blocks.filter(block => block.kind !== 'tool' || !historyTools.has(block.callId)) } : folded
        const older = event.type === 'tool_result' ? foldSessionEvent({ ...EMPTY_TAIL, blocks: selected.older }, input).blocks : selected.older
        if (tail !== selected.tail || older !== selected.older) {
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
          selected = syncConfig({ ...selected, tail, older, decisions })
        }
      }
    }
  }
  return tasks === state.tasks && selected === state.selected ? state : { ...state, tasks, ...(selected ? { selected } : {}) }
}

/** History replaces its whole overlapping window, including blocks whose contents changed. */
function reconcileHistory(selected: SelectedSession, page: HistoryPage, throughIds: readonly string[]): SelectedSession {
  const pageIds = new Set(page.actionIds)
  const firstOverlap = selected.historyActionIds.findIndex(id => pageIds.has(id))
  const prefixIds = new Set(firstOverlap < 0 ? [] : selected.historyActionIds.slice(0, firstOverlap))
  const prefix = selected.older.filter(block => block.historyActionId !== undefined && prefixIds.has(block.historyActionId))
  const history = [...prefix, ...page.blocks]
  const tools = new Set(history.flatMap(block => block.kind === 'tool' ? [block.callId] : []))
  const through = new Set(throughIds)
  let covered = -1
  selected.tail.blocks.forEach((block, index) => {
    if (block.kind === 'tool' && tools.has(block.callId)) covered = index
    else if (block.kind === 'user' && through.has(block.id) && page.blocks.some(item => item.kind === 'user' && item.text === block.text && item.images === block.images)) covered = index
  })
  const live = selected.tail.blocks.slice(covered + 1).filter(block => block.kind !== 'tool' || !tools.has(block.callId))
  const { error: _error, olderCursor: _cursor, ...rest } = selected
  const keepCursor = prefixIds.size > 0
  const cursor = keepCursor ? selected.olderCursor : page.cursor
  return {
    ...rest, phase: 'ready',
    older: bounded(history, OLDER_BLOCKS_LIMIT),
    historyActionIds: bounded([...prefixIds, ...page.actionIds], OLDER_BLOCKS_LIMIT),
    tail: { ...selected.tail, blocks: live },
    hasOlder: keepCursor ? selected.hasOlder : page.hasMore,
    ...(cursor === undefined ? {} : { olderCursor: cursor }),
  }
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
      const fresh = freshSelected(action.workspaceId, action.sessionId)
      const task = state.tasks[taskKey(action.workspaceId, action.sessionId)]
      const selected = task ? syncConfig({ ...fresh, tail: { ...fresh.tail, title: task.title, status: task.status, effective: { model: task.model, goal: task.goalSummary } } }) : fresh
      return { ...state, selected }
    }
    case 'deselect': {
      if (!state.selected) return state
      const { selected: _selected, ...rest } = state
      return rest
    }
    case 'pendingRestored': {
      const selected = state.selected
      if (!selected || selected.workspaceId !== action.workspaceId || selected.sessionId !== action.sessionId) return state
      const known = new Set(selected.tail.pending.map(item => item.id))
      const added = action.pending.filter(item => !known.has(item.id) && selected.tail.resolved[item.id] === undefined)
      if (added.length === 0) return state
      return { ...state, selected: { ...selected, tail: { ...selected.tail, pending: [...selected.tail.pending, ...added] } } }
    }
    case 'historyLoaded': {
      const selected = state.selected
      if (!selected || selected.workspaceId !== action.workspaceId || selected.sessionId !== action.sessionId) return state
      return { ...state, selected: reconcileHistory(selected, action.page, action.throughIds) }
    }
    case 'sessionReady': {
      const selected = state.selected
      if (!selected || selected.workspaceId !== action.workspaceId || selected.sessionId !== action.sessionId) return state
      if (selected.phase === 'ready') return state
      const { error: _error, ...rest } = selected
      return { ...state, selected: { ...rest, phase: 'ready' } }
    }
    case 'sessionReset': {
      const selected = state.selected
      if (!selected || selected.workspaceId !== action.workspaceId || selected.sessionId !== action.sessionId) return state
      const { error: _error, olderCursor: _oc, olderError: _oe, ...rest } = selected
      return { ...state, selected: { ...rest, phase: 'loading', tail: EMPTY_TAIL, older: [], historyActionIds: [], hasOlder: false, loadingOlder: false } }
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
      const known = new Set(selected.older.map(block => block.id))
      const incoming = action.blocks.filter(block => !known.has(block.id))
      const combined = [...incoming, ...selected.older]
      const older = bounded(combined, OLDER_BLOCKS_LIMIT)
      const historyActionIds = [...new Set(older.flatMap(block => block.historyActionId === undefined ? [] : [block.historyActionId]))]
      return { ...state, selected: { ...rest, older, historyActionIds, loadingOlder: false, hasOlder: action.hasMore && combined.length < OLDER_BLOCKS_LIMIT, ...(action.cursor === undefined ? {} : { olderCursor: action.cursor }) } }
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
    case 'autopilotView': {
      const { error: _error, saveError: _saveError, savingId: _savingId, ...rest } = state.autopilot
      return { ...state, autopilot: { ...rest, phase: 'ready', canEdit: action.view.canEdit, rows: action.view.workspaces, unsupported: false } }
    }
    case 'autopilotRow': {
      const rows = state.autopilot.rows
      const index = rows.findIndex(row => row.workspaceId === action.row.workspaceId)
      const next = index === -1 ? [...rows, action.row] : rows.map((row, i) => (i === index ? action.row : row))
      return { ...state, autopilot: { ...state.autopilot, rows: next } }
    }
    case 'autopilotThread': {
      const next: AutopilotThreadState = { ...state.autopilotThread, ...action.patch }
      // A fresh phase without an error clears the old one.
      if (action.patch.phase !== undefined && action.patch.error === undefined) delete (next as { error?: string }).error
      return { ...state, autopilotThread: next }
    }
    case 'autopilotThreadMessages': {
      const known = new Set(state.autopilotThread.messages.map(m => m.id))
      const added = action.messages.filter(m => !known.has(m.id))
      if (added.length === 0) return state
      return { ...state, autopilotThread: { ...state.autopilotThread, messages: bounded([...state.autopilotThread.messages, ...added], SUPERVISOR_MESSAGE_LIMIT) } }
    }
    case 'autopilotThreadConfirmations': {
      const byId = new Map(state.autopilotThread.confirmations.map(c => [c.id, c]))
      for (const c of action.confirmations) byId.set(c.id, c)
      return { ...state, autopilotThread: { ...state.autopilotThread, confirmations: bounded([...byId.values()], 50) } }
    }
    case 'autopilotThreadConfirmation': {
      return {
        ...state,
        autopilotThread: { ...state.autopilotThread, confirmations: state.autopilotThread.confirmations.map(c => (c.id === action.id ? { ...c, ...action.patch } : c)) },
      }
    }
    case 'autopilot': {
      const { savingId: patchSaving, ...patch } = action.patch
      const next: AutopilotState = { ...state.autopilot, ...patch }
      const mutable = next as { error?: string; saveError?: string; savingId?: string }
      // A fresh phase without an error clears the old one; a new save clears the old save error.
      if (patch.phase !== undefined && patch.error === undefined) delete mutable.error
      if ('savingId' in action.patch) {
        if (patchSaving === undefined) delete mutable.savingId
        else {
          mutable.savingId = patchSaving
          if (patch.saveError === undefined) delete mutable.saveError
        }
      }
      return { ...state, autopilot: next }
    }
    case 'watchView': {
      const { error: _error, saveError: _saveError, savingKey: _savingKey, ...rest } = state.watch
      return { ...state, watch: { ...rest, phase: 'ready', canEdit: action.view.canEdit, rows: action.view.watches, unsupported: false } }
    }
    case 'watchRow': {
      const rows = state.watch.rows
      const index = rows.findIndex(row => row.workspaceId === action.row.workspaceId && row.taskId === action.row.taskId)
      const next = index === -1 ? [...rows, action.row] : rows.map((row, i) => (i === index ? action.row : row))
      return { ...state, watch: { ...state.watch, rows: next } }
    }
    case 'watch': {
      const { savingKey: patchSaving, ...patch } = action.patch
      const next: WatchState = { ...state.watch, ...patch }
      const mutable = next as { error?: string; saveError?: string; savingKey?: string }
      // A fresh phase without an error clears the old one; a new save clears the old save error.
      if (patch.phase !== undefined && patch.error === undefined) delete mutable.error
      if ('savingKey' in action.patch) {
        if (patchSaving === undefined) delete mutable.savingKey
        else {
          mutable.savingKey = patchSaving
          if (patch.saveError === undefined) delete mutable.saveError
        }
      }
      return { ...state, watch: next }
    }
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
