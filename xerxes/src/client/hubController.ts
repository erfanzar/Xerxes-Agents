// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { HubCommandResult, HubMethod } from '../hub/protocol.js'
import { AUTOPILOT_THREAD_PREFIX, autopilotThreadId } from './autopilot.js'
import { newCommandId } from './bridge.js'
import { HubRequestError, type HubConnection, type SubscriptionHandle, type Timers } from './hubConnection.js'
import {
  parseAgentDetail, parseAgentList, parseAutopilotRow, parseAutopilotView, parseConfirmation, parseHistoryPage, parseMachines, parsePresence, parseSummaries, parseSupervisorConfig, parseSupervisorMessage,
  parseSupervisorOptions, parseWatchRow, parseWatchView, parseWorkspaces, snapshotSessions, type SupervisorConfirmation,
} from './hubParsers.js'
import { IDLE_AGENT_DETAIL, canMutate, isHubOwner, watchKey, type DecisionKind, type HubStore } from './hubStore.js'
import { taskKey, type WorkspaceView } from './hubTypes.js'
import type { PreparedImage } from './imagePrep.js'
import { arr, rec, str } from './parse.js'
import { usageExtrasFromResult, usageFromResult } from './usageMerge.js'
import { EMPTY_TAIL, foldSessionEvent } from './sessionFold.js'

/**
 * Wires one HubConnection to one HubStore: loads the Inbox (watermark, then
 * snapshot, then a subscription resumed from the watermark), opens the
 * selected task as a scoped replaying subscription, and turns user intent into
 * `hub.command` calls. Every mutation is sent once; an unknown outcome is
 * shown as unknown and never replayed.
 */

export type PresenceState = 'viewing' | 'typing' | 'idle'

export interface HubController {
  /** Whether the connected Hub accepts image attachments on turn.submit / turn.steer. */
  readonly imagesSupported: boolean
  start(): void
  dispose(): void
  refreshInbox(): Promise<void>
  openSession(workspaceId: string, sessionId: string): void
  /** Leave the open task: its subscription closes and presence becomes `idle`. */
  closeSession(): void
  loadOlder(): Promise<void>
  /** Loads `agents.list` for the open task into `selected.agents`. */
  loadAgents(): Promise<void>
  /** Opens one subagent's steps (`agents.inspect`) into `selected.agentDetail`. */
  inspectAgent(agentId: string): Promise<void>
  closeAgent(): void
  /**
   * The app came back to the foreground (visibilitychange, pageshow, online):
   * probe the link, reconnect at once when it is dead, otherwise re-read the
   * Inbox, the open task and the XERXES threads.
   */
  resync(): Promise<void>
  /** Drop the link and reconnect at once (the connection banner's Retry). */
  reconnectNow(): void
  /** Resolves true only when the Hub accepted the message, so the caller clears the draft. */
  submit(text: string, images: readonly PreparedImage[]): Promise<boolean>
  steer(text: string, images: readonly PreparedImage[]): Promise<boolean>
  stopTurn(): Promise<boolean>
  decide(interactionId: string, decision: DecisionKind): Promise<void>
  answerQuestion(interactionId: string, answers: Readonly<Record<string, string>>): Promise<void>
  setModel(profile: string | undefined, model: string): Promise<void>
  setGoal(objective: string): Promise<void>
  loadUsage(): Promise<void>
  setPresence(state: PresenceState): void
  supervisorOpen(): Promise<void>
  supervisorSend(text: string, source?: 'text' | 'voice', messageId?: string): Promise<boolean>
  supervisorRespond(confirmationId: string, decision: 'confirm' | 'cancel'): Promise<void>
  /** Loads `hub.machines.list` into `state.machines`. */
  machinesList(): Promise<void>
  /** Loads `supervisor.config.get` into `state.supervisorConfig`. */
  supervisorConfigGet(): Promise<void>
  /** Loads the chooser lists of one machine (and profile) into `state.supervisorConfig.options`. */
  supervisorConfigOptions(input: SupervisorOptionsInput): Promise<void>
  /** Resolves true only when the Hub saved the choice. A refusal is left in `supervisorConfig.saveError`. */
  supervisorConfigSet(input: SupervisorConfigInput): Promise<boolean>
  /** Loads `supervisor.autopilot.get` into `state.autopilot`. */
  loadAutopilot(): Promise<void>
  /** Resolves true only when the Hub saved the row. A refusal is left in `autopilot.saveError`. */
  setAutopilot(input: AutopilotInput): Promise<boolean>
  /**
   * Opens the owner's private `autopilot-<selfUserId>` thread. Does nothing for
   * anyone but the Hub owner, and never asks again after the Hub said not_found.
   */
  autopilotThreadOpen(): Promise<void>
  /** Confirm or cancel a card held in the Autopilot thread. */
  autopilotThreadRespond(confirmationId: string, decision: 'confirm' | 'cancel'): Promise<void>
  /** Loads `supervisor.watch.list` into `state.watch`. */
  loadWatches(): Promise<void>
  /** Resolves true only when the Hub saved the row. A refusal is left in `watch.saveError`. Never retried. */
  setWatch(input: WatchInput): Promise<boolean>
}

export interface WatchInput {
  readonly workspaceId: string
  readonly taskId: string
  readonly enabled: boolean
  /** Omitted: keep the saved interval. null: no review. Otherwise whole minutes, 5 to 120. */
  readonly reviewIntervalMinutes?: number | null
  /** Omitted: keep the saved expiry. null: no expiry. Otherwise an ISO time in the future. */
  readonly expiresAt?: string | null
}

export const WATCH_REVIEW_MIN_MINUTES = 5
export const WATCH_REVIEW_MAX_MINUTES = 120
/** At most one full re-list per this long when `supervisor.watch_update` payloads are unreadable. */
export const WATCH_RELIST_MS = 2_000

export interface AutopilotInput {
  readonly workspaceId: string
  readonly enabled: boolean
  /** Omitted: keep the saved instruction. At most 2000 characters. */
  readonly instruction?: string
  /** Omitted: keep the saved expiry. null: no expiry. Otherwise an ISO time in the future. */
  readonly expiresAt?: string | null
}

export const AUTOPILOT_INSTRUCTION_LIMIT = 2000

export interface SupervisorOptionsInput {
  readonly machineId: string
  readonly profile?: string
  readonly model?: string
}

export interface SupervisorConfigInput {
  readonly machineId: string
  readonly profile: string
  readonly model: string
  /** null: leave thinking effort to the provider default. */
  readonly effort: string | null
}

export interface HubControllerOptions {
  readonly connection: HubConnection
  readonly store: HubStore
  readonly now?: () => number
  /** The Hub's command router currently takes `text` only. Set true once it accepts images. */
  readonly imagesSupported?: boolean
  /** Injected by offline tests; production uses browser timers. */
  readonly timers?: Timers
}

export const HISTORY_PAGE_SIZE = 30
/** History is re-read this often while a task works, so replies show even when no tool runs. */
export const HISTORY_REFRESH_MS = 4_000
/** Agents refresh at most once per this long while `subagent_event`s arrive. */
export const AGENTS_REFRESH_MS = 2_000
/** The open agent's steps refresh at most once per this long, and only while its sheet is open. */
export const AGENT_DETAIL_REFRESH_MS = 5_000
export const AGENT_STEPS_LIMIT = 100
/** A failed subscribe is retried after 1 s, 2 s, 4 s … up to 30 s. */
export const SUBSCRIBE_RETRY_BASE_MS = 1_000
export const SUBSCRIBE_RETRY_MAX_MS = 30_000
/** `typing` falls back to `viewing` once nothing was typed for this long. */
export const TYPING_IDLE_MS = 8_000
/** How long a foreground probe may take before the link counts as dead. */
export const RESYNC_PROBE_MS = 3_000
export const AGENTS_UNSUPPORTED = "This Hub cannot list a task's agents yet. Update the Hub to see them."
const RETRIABLE_SUBSCRIBE: ReadonlySet<string> = new Set(['timeout', 'unavailable', 'rate_limited', 'internal'])
const TRANSIENT_CARD: ReadonlySet<string> = new Set(['offline', 'unavailable', 'busy', 'rate_limited'])

const DECISION_RESPONSE: Readonly<Record<DecisionKind, string>> = { once: 'approve', deny: 'reject', session: 'approve_for_session' }
const UNKNOWN_OUTCOME = 'Could not confirm whether this was applied. Check the task before trying again.'
/** Shown when the Hub refuses a message because XERXES has no model yet. */
export const UNCONFIGURED_PROMPT = "Choose XERXES's model first. Open the Model settings and save a choice."
export const IMAGES_UNSUPPORTED = 'This Hub does not accept image attachments yet. Remove the images to send your message.'

function failureMessage(error: unknown): string {
  return error instanceof Error ? error.message : 'Something went wrong.'
}

function imageArgs(images: readonly PreparedImage[]): readonly Record<string, unknown>[] {
  return images.map(image => ({ media_type: image.media_type, data: image.data, width: image.width, height: image.height }))
}

export function createHubController(options: HubControllerOptions): HubController {
  const { connection, store } = options
  const now = options.now ?? Date.now
  const iso = (): string => new Date(now()).toISOString()
  const imagesSupported = options.imagesSupported === true
  const disposers: (() => void)[] = []
  let inboxSub: SubscriptionHandle | undefined
  let inboxKey = ''
  let sessionSub: SubscriptionHandle | undefined
  let interactionReplay: SubscriptionHandle | undefined
  let openToken = 0
  let startingSelected = false
  let historyRefresh: Promise<void> | undefined
  let refreshAgain = false
  let historyTimer: unknown
  const timers = options.timers ?? {
    setTimeout: (callback: () => void, ms: number) => globalThis.setTimeout(callback, ms),
    clearTimeout: (handle: unknown) => globalThis.clearTimeout(handle as ReturnType<typeof setTimeout>),
  }
  let supervisorThread: string | undefined

  let optionsToken = 0
  // Subscribe retries (a failed subscribe is never left silently dead).
  let inboxRetryTimer: unknown
  let inboxRetries = 0
  let sessionRetryTimer: unknown
  let sessionRetries = 0
  // Agents of the open task.
  let agentsTimer: unknown
  let agentsLastAt = Number.NEGATIVE_INFINITY
  let agentsInFlight: Promise<void> | undefined
  let detailToken = 0
  let detailTimer: unknown
  let detailLastAt = Number.NEGATIVE_INFINITY
  // Presence of this device on the open task.
  let presenceState: PresenceState = 'idle'
  let typingTimer: unknown
  // Supervisor messages whose outcome is unknown: a retry of the same text reuses the id, so the Hub dedupes it.
  const pendingMessageIds = new Map<string, string>()
  // Confirmation cards whose outcome could not be read yet; re-checked after a reconnect.
  const uncheckedCards = new Map<string, 'shared' | 'autopilot'>()
  let wasLive = false
  // Set once the Hub answered `hub.history` with `pending_interactions`: no journal replay is needed.
  let historyServesPending = false

  const retryDelay = (attempt: number): number => Math.min(SUBSCRIBE_RETRY_MAX_MS, SUBSCRIBE_RETRY_BASE_MS * 2 ** Math.max(0, attempt - 1))
  const clearTimer = (handle: unknown): undefined => {
    if (handle !== undefined) timers.clearTimeout(handle)
    return undefined
  }

  const live = (): boolean => connection.status().state === 'live'
  const selected = () => store.getSnapshot().selected
  // The cockpit methods are newer than some Hubs; the Hub answers `unsupported`/`invalid_request` for the ones it lacks.
  const call = (method: string, params: Readonly<Record<string, unknown>> = {}): Promise<Readonly<Record<string, unknown>>> =>
    connection.request(method as HubMethod, params)

  // ── inbox ──────────────────────────────────────────────────────────

  async function loadSummaries(list: readonly WorkspaceView[]): Promise<void> {
    if (list.length === 0) return
    const snapshot = await connection.request('hub.snapshot', { workspace_ids: list.map(workspace => workspace.id), limit: 100 })
    for (const workspace of list) {
      store.dispatch({
        type: 'summaries',
        workspaceId: workspace.id,
        tasks: parseSummaries(workspace.id, snapshotSessions(snapshot, workspace.id), iso(), store.getSnapshot().tasks),
      })
    }
  }

  async function refreshInbox(): Promise<void> {
    try {
      const list = parseWorkspaces(await connection.request('hub.workspaces.list'))
      store.dispatch({ type: 'workspaces', list })
      const ids = list.map(workspace => workspace.id)
      const key = [...ids].sort().join('\u001f')
      const needSubscription = ids.length > 0 && (key !== inboxKey || inboxSub === undefined)
      // Position first, snapshot second: anything that happens in between is replayed from the position.
      const mark = needSubscription ? await connection.watermark({ workspaceIds: ids }) : undefined
      await loadSummaries(list)
      store.dispatch({ type: 'inboxLoaded' })
      if (needSubscription) subscribeInbox(list, key, mark)
    } catch (error) {
      store.dispatch({ type: 'inboxError', message: failureMessage(error) })
    }
  }

  function subscribeInbox(list: readonly WorkspaceView[], key: string, mark: string | undefined): void {
    inboxSub?.dispose()
    inboxKey = key
    const ids = list.map(workspace => workspace.id)
    inboxSub = connection.subscribe(
      { workspaceIds: ids },
      {
        apply: event => {
          if (event.type === 'hub.workspace_status') {
            const found = parseWorkspaces({ workspaces: [{ id: event.workspace_id, ...event.payload }] })[0]
            if (found) store.dispatch({ type: 'workspaceHealth', workspaceId: found.id, health: found.health })
            return
          }
          inboxRetries = 0
          store.dispatch({ type: 'events', events: [event], tail: false })
        },
        reset: async () => {
          const fresh = parseWorkspaces(await connection.request('hub.workspaces.list'))
          store.dispatch({ type: 'workspaces', list: fresh })
          const cursor = await connection.watermark({ workspaceIds: ids })
          await loadSummaries(fresh)
          return { cursor }
        },
        error: error => onInboxSubscribeFailed(error),
      },
      mark === undefined ? undefined : { cursor: mark },
    )
  }

  /**
   * A failed Inbox subscribe never leaves the app silently without live
   * updates: the person sees "Live updates paused", a retriable failure is
   * retried with backoff, any other one on the next reconnect or resync.
   */
  function onInboxSubscribeFailed(error: HubRequestError): void {
    // A lost socket restarts every subscription itself once it reconnects.
    if (error.code === 'disconnected' || error.code === 'offline') return
    store.dispatch({ type: 'inboxError', message: `Live updates paused: ${error.message}` })
    inboxRetryTimer = clearTimer(inboxRetryTimer)
    inboxSub?.dispose()
    inboxSub = undefined
    if (!RETRIABLE_SUBSCRIBE.has(error.code)) return
    inboxRetries += 1
    inboxRetryTimer = timers.setTimeout(() => {
      inboxRetryTimer = undefined
      if (live() && inboxSub === undefined) void refreshInbox()
    }, retryDelay(inboxRetries))
  }

  // ── selected task ──────────────────────────────────────────────────

  function cancelHistoryRefresh(): void {
    if (historyTimer !== undefined) timers.clearTimeout(historyTimer)
    historyTimer = undefined
  }

  const currentToken = (token: number): boolean => token === openToken && selected() !== undefined

  async function readLatest(token: number): Promise<void> {
    const current = selected()
    if (!current || !currentToken(token)) return
    const throughIds = current.tail.blocks.map(block => block.id)
    const pendingThrough = current.tail.pending.map(item => item.id)
    const result = await connection.request('hub.history', {
      workspace_id: current.workspaceId, session_id: current.sessionId, limit: HISTORY_PAGE_SIZE,
    })
    const page = parseHistoryPage(result)
    if (!currentToken(token)) return
    store.dispatch({ type: 'historyLoaded', workspaceId: current.workspaceId, sessionId: current.sessionId, page, throughIds })
    // A Hub that serves the waiting interactions with history needs no journal replay to find them.
    if (Array.isArray(result.pending_interactions)) {
      // A partial list (`pending_complete: false`) still needs the journal replay to find the rest.
      historyServesPending = result.pending_complete !== false
      let folded = EMPTY_TAIL
      result.pending_interactions.forEach((raw, index) => {
        const frame = rec(raw)
        const type = str(frame.type)
        if (type) folded = foldSessionEvent(folded, { id: `pending:${index}`, type, payload: rec(frame.payload) })
      })
      store.dispatch({ type: 'pendingRestored', workspaceId: current.workspaceId, sessionId: current.sessionId, pending: folded.pending.slice(-50), complete: historyServesPending, throughIds: pendingThrough })
    }
  }

  function refreshLatest(): Promise<void> {
    cancelHistoryRefresh()
    if (historyRefresh) {
      // In particular, a turn_end during a read must get a final read after it.
      refreshAgain = true
      return historyRefresh
    }
    const token = openToken
    const current = selected()
    if (!current) return Promise.resolve()
    const pending = readLatest(token).catch(error => {
      if (currentToken(token)) store.dispatch({ type: 'sessionError', workspaceId: current.workspaceId, sessionId: current.sessionId, message: failureMessage(error) })
    }).finally(() => {
      if (!currentToken(token)) return
      historyRefresh = undefined
      if (refreshAgain) {
        refreshAgain = false
        void refreshLatest()
      } else if (selected()?.tail.status === 'working' && live()) {
        // A working turn keeps being re-read, so a text-only reply shows even when no tool event arrives.
        scheduleHistoryRefresh()
      }
    })
    historyRefresh = pending
    return pending
  }

  function scheduleHistoryRefresh(): void {
    // Coalesce new tool events without starving a long, continuously working turn.
    if (historyTimer !== undefined) return
    historyTimer = timers.setTimeout(() => {
      historyTimer = undefined
      void refreshLatest()
    }, HISTORY_REFRESH_MS)
  }

  function restorePending(workspaceId: string, sessionId: string, token: number): void {
    interactionReplay?.dispose()
    interactionReplay = undefined
    const task = store.getSnapshot().tasks[taskKey(workspaceId, sessionId)]
    // A Hub that serves the waiting interactions with history already restored them.
    if (!task?.pendingCount || historyServesPending) return
    // History has no pending-interaction snapshot. Only for a waiting task,
    // recover that metadata in the background; never put replayed conversation
    // or tool blocks into the visible list, nor delay the history-first view.
    let recovered = EMPTY_TAIL
    const finish = (): void => {
      interactionReplay?.dispose()
      interactionReplay = undefined
    }
    const metadata = new Set(['approval_request', 'permission_request', 'question_request', 'approval_response', 'permission_response', 'permission_resolved', 'question_response', 'question_resolved', 'interaction_resolved', 'turn_end'])
    interactionReplay = connection.subscribe({ workspaceIds: [workspaceId], sessionIds: [sessionId] }, {
      apply: event => {
        if (currentToken(token) && metadata.has(event.type)) recovered = foldSessionEvent(recovered, { id: event.event_id, type: event.type, payload: event.payload, ...(event.actor ? { actor: event.actor } : {}) })
      },
      caughtUp: () => {
        if (!currentToken(token)) return
        store.dispatch({ type: 'pendingRestored', workspaceId, sessionId, pending: recovered.pending.slice(-50) })
        finish()
      },
      reset: () => { if (currentToken(token)) finish(); return undefined },
      error: error => {
        if (!currentToken(token) || error.code === 'disconnected' || error.code === 'offline') return
        finish()
        // Looked for and not found: the app says the details are unavailable instead of showing nothing to tap.
        store.dispatch({ type: 'pendingRestored', workspaceId, sessionId, pending: [] })
        store.dispatch({ type: 'sessionError', workspaceId, sessionId, message: error.message })
      },
    })
  }

  /** A failed task subscribe is shown and retried with backoff; a lost socket restarts it on reconnect. */
  function onSessionSubscribeFailed(workspaceId: string, sessionId: string, token: number, error: HubRequestError): void {
    if (!currentToken(token) || error.code === 'disconnected' || error.code === 'offline') return
    store.dispatch({ type: 'sessionError', workspaceId, sessionId, message: `Live updates paused: ${error.message}` })
    sessionSub?.dispose()
    sessionSub = undefined
    sessionRetryTimer = clearTimer(sessionRetryTimer)
    if (!RETRIABLE_SUBSCRIBE.has(error.code)) return
    sessionRetries += 1
    sessionRetryTimer = timers.setTimeout(() => {
      sessionRetryTimer = undefined
      if (currentToken(token) && live() && sessionSub === undefined && !startingSelected) void startSelected(workspaceId, sessionId)
    }, retryDelay(sessionRetries))
  }

  async function startSelected(workspaceId: string, sessionId: string): Promise<void> {
    interactionReplay?.dispose()
    interactionReplay = undefined
    sessionSub?.dispose()
    sessionSub = undefined
    startingSelected = true
    const token = ++openToken
    cancelHistoryRefresh()
    historyRefresh = undefined
    refreshAgain = false
    const scope = { workspaceIds: [workspaceId], sessionIds: [sessionId] }
    try {
      // History is requested first; capture the live cursor while it loads. A
      // second read after subscribing closes the history/watermark race without
      // ever replaying the journal from its beginning.
      const [_, cursor] = await Promise.all([readLatest(token), connection.watermark(scope)])
      if (!currentToken(token)) return
      sessionSub = connection.subscribe(scope, {
        apply: event => {
          if (!currentToken(token)) return
          sessionRetries = 0
          store.dispatch({ type: 'events', events: [event] })
          if (event.type === 'turn_begin' || event.type === 'turn_end') {
            interactionReplay?.dispose()
            interactionReplay = undefined
          }
          if (event.type === 'turn_end') void refreshLatest()
          else if (event.type === 'tool_call' || event.type === 'tool_result' || event.type === 'turn_begin') scheduleHistoryRefresh()
          if (event.type === 'subagent_event') scheduleAgentsRefresh()
        },
        reset: async () => {
          if (!currentToken(token)) return undefined
          cancelHistoryRefresh()
          await historyRefresh
          try {
            const cursor = await connection.watermark(scope)
            if (!currentToken(token)) return { cursor }
            store.dispatch({ type: 'sessionReset', workspaceId, sessionId })
            try {
              await readLatest(token)
              if (currentToken(token)) restorePending(workspaceId, sessionId, token)
            }
            catch (error) {
              if (currentToken(token)) store.dispatch({ type: 'sessionError', workspaceId, sessionId, message: failureMessage(error) })
            }
            return { cursor }
          } catch (error) {
            if (currentToken(token)) {
              // With no safe boundary, stop instead of silently replaying the
              // entire journal. The visible error offers an explicit retry.
              sessionSub?.dispose()
              sessionSub = undefined
              store.dispatch({ type: 'sessionError', workspaceId, sessionId, message: failureMessage(error) })
            }
            return undefined
          }
        },
        error: error => onSessionSubscribeFailed(workspaceId, sessionId, token, error),
      }, { cursor })
      void refreshLatest()
      restorePending(workspaceId, sessionId, token)
      if (presenceState !== 'typing') presenceState = 'viewing'
      sendPresence(workspaceId, sessionId, presenceState)
      void loadAgents()
    } catch (error) {
      if (currentToken(token)) store.dispatch({ type: 'sessionError', workspaceId, sessionId, message: failureMessage(error) })
    } finally {
      if (currentToken(token)) startingSelected = false
    }
  }

  /** Everything tied to the open task stops: subscriptions, timers, agent refreshes. */
  function stopSelected(): void {
    openToken += 1
    cancelHistoryRefresh()
    historyRefresh = undefined
    refreshAgain = false
    startingSelected = false
    interactionReplay?.dispose()
    interactionReplay = undefined
    sessionSub?.dispose()
    sessionSub = undefined
    sessionRetryTimer = clearTimer(sessionRetryTimer)
    sessionRetries = 0
    agentsTimer = clearTimer(agentsTimer)
    agentsInFlight = undefined
    agentsLastAt = Number.NEGATIVE_INFINITY
    detailTimer = clearTimer(detailTimer)
    detailLastAt = Number.NEGATIVE_INFINITY
    detailToken += 1
  }

  /** Tell the Hub this device left the task, so nobody sees a stale "watching" or "typing". */
  function leavePresence(): void {
    typingTimer = clearTimer(typingTimer)
    const current = selected()
    if (current && presenceState !== 'idle') sendPresence(current.workspaceId, current.sessionId, 'idle')
    presenceState = 'idle'
  }

  function openSession(workspaceId: string, sessionId: string): void {
    const current = selected()
    if (current && current.workspaceId === workspaceId && current.sessionId === sessionId && current.phase !== 'error' && current.error === undefined) return
    if (current && (current.workspaceId !== workspaceId || current.sessionId !== sessionId)) leavePresence()
    stopSelected()
    store.dispatch({ type: 'deselect' })
    store.dispatch({ type: 'select', workspaceId, sessionId })
    if (live()) void startSelected(workspaceId, sessionId)
  }

  function closeSession(): void {
    leavePresence()
    stopSelected()
    store.dispatch({ type: 'deselect' })
  }

  // ── agents of the open task ────────────────────────────────────────

  const agentsFailure = (error: unknown): string =>
    error instanceof HubRequestError && (error.code === 'unsupported' || (error.code === 'invalid_request' && /unknown hub method/.test(error.message)))
      ? AGENTS_UNSUPPORTED
      : failureMessage(error)

  function loadAgents(): Promise<void> {
    if (agentsInFlight) return agentsInFlight
    const current = selected()
    if (!current || !live()) return Promise.resolve()
    const { workspaceId, sessionId } = current
    const token = openToken
    agentsLastAt = now()
    const rows = current.agents.rows
    store.dispatch({ type: 'agents', workspaceId, sessionId, state: { phase: 'loading', rows } })
    const run = (async () => {
      try {
        const listed = parseAgentList(await call('agents.list', { workspace_id: workspaceId, session_id: sessionId }))
        if (token !== openToken) return
        store.dispatch({
          type: 'agents', workspaceId, sessionId,
          state: listed === undefined ? { phase: 'error', rows, error: 'The Hub sent an agent list this app cannot read.' } : { phase: 'ready', rows: listed },
        })
      } catch (error) {
        if (token !== openToken) return
        store.dispatch({ type: 'agents', workspaceId, sessionId, state: { phase: 'error', rows: selected()?.agents.rows ?? rows, error: agentsFailure(error) } })
      } finally {
        if (token === openToken) agentsInFlight = undefined
      }
    })()
    agentsInFlight = run
    return run
  }

  /** At most one agents refresh (list and the open agent) per AGENTS_REFRESH_MS; the last event is never lost. */
  function scheduleAgentsRefresh(): void {
    if (agentsTimer !== undefined) return
    const run = (): void => {
      agentsTimer = undefined
      void loadAgents()
      scheduleDetailRefresh()
    }
    const wait = Math.max(0, agentsLastAt + AGENTS_REFRESH_MS - now())
    if (wait === 0 && agentsInFlight === undefined) run()
    else agentsTimer = timers.setTimeout(run, Math.max(wait, agentsInFlight === undefined ? 0 : AGENTS_REFRESH_MS))
  }

  /** The open agent's steps are re-read at most once per AGENT_DETAIL_REFRESH_MS, and only while its sheet is open. */
  function openAgentId(): string | undefined {
    const detail = selected()?.agentDetail
    return detail?.agentId !== undefined && detail.phase !== 'idle' ? detail.agentId : undefined
  }

  function scheduleDetailRefresh(): void {
    if (detailTimer !== undefined || openAgentId() === undefined) return
    const wait = Math.max(0, detailLastAt + AGENT_DETAIL_REFRESH_MS - now())
    const run = (): void => {
      detailTimer = undefined
      const agentId = openAgentId()
      if (agentId !== undefined) void inspectAgent(agentId)
    }
    if (wait === 0) run()
    else detailTimer = timers.setTimeout(run, wait)
  }

  async function inspectAgent(agentId: string): Promise<void> {
    const current = selected()
    if (!current) return
    const { workspaceId, sessionId } = current
    detailLastAt = now()
    const token = (detailToken += 1)
    const open = openToken
    const previous = current.agentDetail.agentId === agentId ? current.agentDetail : undefined
    const known = previous?.agent ?? current.agents.rows.find(row => row.id === agentId)
    store.dispatch({
      type: 'agentDetail', workspaceId, sessionId,
      state: { phase: 'loading', agentId, steps: previous?.steps ?? [], truncated: previous?.truncated ?? false, ...(known ? { agent: known } : {}) },
    })
    if (!live()) {
      store.dispatch({ type: 'agentDetail', workspaceId, sessionId, state: { phase: 'error', agentId, steps: previous?.steps ?? [], truncated: false, error: 'Offline. Reconnect to see this agent.', ...(known ? { agent: known } : {}) } })
      return
    }
    try {
      const detail = parseAgentDetail(await call('agents.inspect', { workspace_id: workspaceId, session_id: sessionId, agent_id: agentId, limit: AGENT_STEPS_LIMIT }))
      if (token !== detailToken || open !== openToken) return
      store.dispatch({
        type: 'agentDetail', workspaceId, sessionId,
        state: detail === undefined
          ? { phase: 'error', agentId, steps: previous?.steps ?? [], truncated: false, error: 'The Hub sent agent details this app cannot read.', ...(known ? { agent: known } : {}) }
          : { phase: 'ready', agentId, agent: detail.agent, steps: detail.steps, truncated: detail.truncated },
      })
    } catch (error) {
      if (token !== detailToken || open !== openToken) return
      const message = error instanceof HubRequestError && error.code === 'not_found' ? 'This agent is no longer known to the task.' : agentsFailure(error)
      store.dispatch({ type: 'agentDetail', workspaceId, sessionId, state: { phase: 'error', agentId, steps: previous?.steps ?? [], truncated: false, error: message, ...(known ? { agent: known } : {}) } })
    }
  }

  function closeAgent(): void {
    detailTimer = clearTimer(detailTimer)
    detailToken += 1
    const current = selected()
    if (current) store.dispatch({ type: 'agentDetail', workspaceId: current.workspaceId, sessionId: current.sessionId, state: IDLE_AGENT_DETAIL })
  }

  async function loadOlder(): Promise<void> {
    const current = selected()
    if (!current || current.phase !== 'ready' || current.loadingOlder || !current.hasOlder) return
    const token = openToken
    store.dispatch({ type: 'olderLoading' })
    try {
      const result = await connection.request('hub.history', {
        workspace_id: current.workspaceId,
        session_id: current.sessionId,
        limit: HISTORY_PAGE_SIZE,
        ...(current.olderCursor === undefined ? {} : { before: current.olderCursor }),
      })
      if (!currentToken(token)) return
      if (selected()?.olderCursor !== current.olderCursor) {
        store.dispatch({ type: 'olderFailed', message: 'History changed. Try loading earlier history again.' })
        return
      }
      const page = parseHistoryPage(result)
      store.dispatch({
        type: 'olderLoaded', blocks: page.blocks, hasMore: page.hasMore && page.cursor !== current.olderCursor,
        ...(page.cursor === undefined ? {} : { cursor: page.cursor }),
        ...(page.todos !== undefined && page.todosCallId !== undefined ? { todos: page.todos, todosCallId: page.todosCallId } : {}),
      })
    } catch (error) {
      if (currentToken(token)) store.dispatch({ type: 'olderFailed', message: failureMessage(error) })
    }
  }

  // ── commands ───────────────────────────────────────────────────────

  async function run(action: 'turn.submit' | 'turn.steer' | 'turn.cancel', args: Record<string, unknown>): Promise<HubCommandResult | undefined> {
    const current = selected()
    if (!current) return undefined
    const scope = { workspaceId: current.workspaceId, sessionId: current.sessionId }
    const commandId = newCommandId()
    store.dispatch({ type: 'send', state: { phase: 'sending', commandId }, ...scope })
    if (!canMutate(store.getSnapshot(), current.workspaceId)) {
      store.dispatch({ type: 'send', state: { phase: 'failed', message: 'Offline. Nothing was sent.' }, ...scope })
      return undefined
    }
    const result = await connection.command({ ...scope, action, args, commandId })
    // The answer belongs to the task that sent it, even when the person opened another one meanwhile.
    if (result.state === 'accepted') store.dispatch({ type: 'send', state: { phase: 'idle' }, ...scope })
    else if (result.state === 'unknown') store.dispatch({ type: 'send', state: { phase: 'unknown', message: UNKNOWN_OUTCOME, commandId }, ...scope })
    else store.dispatch({ type: 'send', state: { phase: 'failed', message: result.error?.message ?? 'The Hub did not accept that.' }, ...scope })
    return result
  }

  function textArgs(text: string, images: readonly PreparedImage[]): Record<string, unknown> | undefined {
    if (images.length === 0) return { text }
    if (!imagesSupported) {
      store.dispatch({ type: 'send', state: { phase: 'failed', message: IMAGES_UNSUPPORTED } })
      return undefined
    }
    return { text, images: imageArgs(images) }
  }

  async function submit(text: string, images: readonly PreparedImage[]): Promise<boolean> {
    const args = textArgs(text, images)
    const ok = args !== undefined && (await run('turn.submit', args))?.state === 'accepted'
    if (ok) afterSend()
    return ok
  }

  async function steer(text: string, images: readonly PreparedImage[]): Promise<boolean> {
    const args = textArgs(text, images)
    const ok = args !== undefined && (await run('turn.steer', args))?.state === 'accepted'
    if (ok) afterSend()
    return ok
  }

  /** A message went out: the person stopped typing, and the reply is polled for even with no tool events. */
  function afterSend(): void {
    if (presenceState === 'typing') setPresence('viewing')
    scheduleHistoryRefresh()
  }

  async function stopTurn(): Promise<boolean> {
    return (await run('turn.cancel', {}))?.state === 'accepted'
  }

  async function respond(interactionId: string, decision: DecisionKind, action: 'permission.respond' | 'question.respond', args: Record<string, unknown>): Promise<void> {
    const current = selected()
    if (!current) return
    const scope = { workspaceId: current.workspaceId, sessionId: current.sessionId }
    store.dispatch({ type: 'decision', interactionId, state: { phase: 'sending', decision }, ...scope })
    const result = await connection.command({ ...scope, action, args })
    if (result.state === 'accepted') {
      store.dispatch({ type: 'decision', interactionId, state: { phase: 'resolved', decision }, ...scope })
    } else if (result.error?.code === 'interaction_already_resolved') {
      // The Hub maps every refused interaction response to this code, so the message is shown as is.
      store.dispatch({ type: 'decision', interactionId, state: { phase: 'resolved', decision, already: true, message: result.error.message }, ...scope })
    } else if (result.state === 'unknown') {
      store.dispatch({ type: 'decision', interactionId, state: { phase: 'unknown', decision, message: UNKNOWN_OUTCOME }, ...scope })
    } else {
      store.dispatch({ type: 'decision', interactionId, state: { phase: 'failed', decision, message: result.error?.message ?? 'The decision was not applied.' }, ...scope })
    }
  }

  const decide = (interactionId: string, decision: DecisionKind): Promise<void> =>
    respond(interactionId, decision, 'permission.respond', { request_id: interactionId, response: DECISION_RESPONSE[decision] })

  const answerQuestion = (interactionId: string, answers: Readonly<Record<string, string>>): Promise<void> =>
    respond(interactionId, 'once', 'question.respond', { request_id: interactionId, answers })

  async function configure(field: 'model' | 'goal', requested: string, action: 'model.set' | 'goal.set', args: Record<string, unknown>, expectedRevision?: number): Promise<void> {
    const current = selected()
    if (!current) return
    const scope = { workspaceId: current.workspaceId, sessionId: current.sessionId }
    store.dispatch({ type: 'config', field, value: { requested, phase: 'requested' }, ...scope })
    const result = await connection.command({
      ...scope,
      action,
      args,
      ...(expectedRevision === undefined ? {} : { expectedRevision }),
    })
    if (result.state === 'accepted') store.dispatch({ type: 'config', field, value: { phase: 'accepted' }, ...scope })
    else if (result.state === 'unknown') store.dispatch({ type: 'config', field, value: { phase: 'unknown', message: UNKNOWN_OUTCOME }, ...scope })
    else store.dispatch({ type: 'config', field, value: { phase: 'rejected', message: result.error?.message ?? 'The change was not accepted.' }, ...scope })
  }

  const setModel = (profile: string | undefined, model: string): Promise<void> =>
    configure('model', model, 'model.set', { model, ...(profile === undefined ? {} : { provider_profile: profile }) })
  const setGoal = (objective: string): Promise<void> => configure('goal', objective, 'goal.set', { input: objective }, selected()?.goal.revision)

  // ── usage, presence ────────────────────────────────────────────────

  async function loadUsage(): Promise<void> {
    const previous = store.getSnapshot().usage
    const report = previous.phase === 'idle' ? undefined : previous.report
    const extras = previous.phase === 'idle' ? undefined : previous.extras
    const carry = { ...(report ? { report } : {}), ...(extras ? { extras } : {}) }
    store.dispatch({ type: 'usage', state: { phase: 'loading', ...carry } })
    try {
      const result = await connection.request('hub.usage')
      const fresh = usageExtrasFromResult(result)
      store.dispatch({ type: 'usage', state: { phase: 'ready', report: usageFromResult(result), ...(fresh ? { extras: fresh } : {}) } })
    } catch (error) {
      store.dispatch({ type: 'usage', state: { phase: 'error', message: failureMessage(error), ...carry } })
    }
  }

  function sendPresence(workspaceId: string, sessionId: string, state: PresenceState): void {
    if (!live()) return
    void connection.request('hub.presence.update', { workspace_id: workspaceId, session_id: sessionId, state }).catch(() => undefined)
  }

  /**
   * Presence of this device on the open task. `typing` is re-armed by every
   * call and falls back to `viewing` after TYPING_IDLE_MS without one, so it
   * never sticks while the keyboard stays open. The latest state is
   * re-announced after a reconnect.
   */
  function setPresence(state: PresenceState): void {
    const current = selected()
    if (!current) return
    typingTimer = clearTimer(typingTimer)
    if (state === 'typing') {
      typingTimer = timers.setTimeout(() => {
        typingTimer = undefined
        if (presenceState === 'typing') setPresence('viewing')
      }, TYPING_IDLE_MS)
    }
    if (state === presenceState) return
    presenceState = state
    sendPresence(current.workspaceId, current.sessionId, state)
  }

  // ── supervisor ─────────────────────────────────────────────────────

  async function supervisorOpen(): Promise<void> {
    const phase = store.getSnapshot().supervisor.phase
    // `unsupported` is a property of this Hub, not a transient failure: do not ask again on every tab visit.
    if (phase === 'ready' || phase === 'loading' || phase === 'unsupported') return
    store.dispatch({ type: 'supervisor', patch: { phase: 'loading' } })
    try {
      const result = await connection.request('supervisor.thread.open', {})
      supervisorThread = str(result.thread_id)
      store.dispatch({ type: 'supervisor', patch: { phase: 'ready', working: result.busy === true, ...(supervisorThread ? { threadId: supervisorThread } : {}) } })
      store.dispatch({ type: 'supervisorMessages', messages: arr(result.messages).flatMap(raw => { const m = parseSupervisorMessage(raw); return m ? [m] : [] }) })
      store.dispatch({ type: 'supervisorConfirmations', confirmations: arr(result.confirmations).flatMap(raw => { const c = parseConfirmation(raw); return c ? [c] : [] }) })
      // thread.open reports the model status too; the full answer (who may edit, which model) comes from config.get.
      const reported = str(result.config_status)
      if (reported !== undefined && store.getSnapshot().supervisorConfig.phase === 'idle') {
        const known = parseSupervisorConfig({ status: reported })
        if (known.status !== 'invalid') store.dispatch({ type: 'supervisorConfig', patch: { status: known.status } })
      }
      void supervisorConfigGet()
      void loadAutopilot()
      void loadWatches()
    } catch (error) {
      if (error instanceof HubRequestError && error.code === 'unsupported') store.dispatch({ type: 'supervisor', patch: { phase: 'unsupported' } })
      else store.dispatch({ type: 'supervisor', patch: { phase: 'error', error: failureMessage(error) } })
    }
  }

  /**
   * Send one message to XERXES. A retry of the same text after an uncertain
   * failure reuses the first `message_id`, so the Hub stores it once and
   * starts no second turn (it answers `duplicate: true`).
   */
  async function supervisorSend(text: string, source: 'text' | 'voice' = 'text', explicitId?: string): Promise<boolean> {
    const key = text.trim()
    const messageId = explicitId ?? pendingMessageIds.get(key) ?? newCommandId()
    store.dispatch({ type: 'supervisor', patch: { busy: true } })
    try {
      const result = await connection.request('supervisor.message.submit', { message_id: messageId, text, source, ...(supervisorThread ? { thread_id: supervisorThread } : {}) })
      pendingMessageIds.delete(key)
      const messages = arr(result.messages).flatMap(raw => { const m = parseSupervisorMessage(raw); return m ? [m] : [] })
      store.dispatch({ type: 'supervisorMessages', messages: messages.length > 0 ? messages : [{ id: messageId, role: 'user', text, at: iso(), source }] })
      store.dispatch({ type: 'supervisorConfirmations', confirmations: arr(result.confirmations).flatMap(raw => { const c = parseConfirmation(raw); return c ? [c] : [] }) })
      // A stored message always starts a turn on the Hub; show it as working
      // until the Hub says idle or the reply lands, so there is no gap
      // between this answer and the Hub's own `activity: working` event.
      // A duplicate started nothing new: keep whatever the Hub reported.
      store.dispatch({ type: 'supervisor', patch: result.duplicate === true ? { busy: false, error: undefined } : { busy: false, working: true } })
      return true
    } catch (error) {
      if (error instanceof HubRequestError && (error.code as string) === 'unconfigured') {
        pendingMessageIds.delete(key)
        // XERXES has no model yet: show the chooser, not a generic failure.
        store.dispatch({ type: 'supervisorConfig', patch: { status: 'unconfigured' } })
        store.dispatch({ type: 'supervisor', patch: { busy: false, error: UNCONFIGURED_PROMPT } })
        return false
      }
      const uncertain = error instanceof HubRequestError && error.uncertain
      if (uncertain) {
        // Remember the id: the next send of this text is the same message.
        pendingMessageIds.set(key, messageId)
        while (pendingMessageIds.size > 20) pendingMessageIds.delete(pendingMessageIds.keys().next().value as string)
      } else pendingMessageIds.delete(key)
      store.dispatch({ type: 'supervisor', patch: { busy: false, error: uncertain ? UNKNOWN_OUTCOME : failureMessage(error) } })
      return false
    }
  }

  const parseMessages = (value: unknown) => arr(value).flatMap(raw => { const m = parseSupervisorMessage(raw); return m ? [m] : [] })
  const parseCards = (value: unknown) => arr(value).flatMap(raw => { const c = parseConfirmation(raw); return c ? [c] : [] })

  /**
   * After a reconnect or a return to the foreground: re-read the shared XERXES
   * thread and (when open) the Autopilot thread past the phase guards, so
   * replies sent while the phone slept show and a stale `working` clears.
   * `unsupported` and `hidden` stay as the Hub answered them.
   */
  async function refreshThreads(): Promise<void> {
    const snapshot = store.getSnapshot()
    const jobs: Promise<void>[] = []
    if (snapshot.supervisor.phase === 'ready' || snapshot.supervisor.phase === 'error') {
      jobs.push((async () => {
        try {
          const result = await connection.request('supervisor.thread.open', {})
          supervisorThread = str(result.thread_id) ?? supervisorThread
          store.dispatch({
            type: 'threadRefreshed', thread: 'shared', messages: parseMessages(result.messages), confirmations: parseCards(result.confirmations),
            working: result.busy === true, ...(supervisorThread ? { threadId: supervisorThread } : {}),
          })
        } catch (error) {
          if (error instanceof HubRequestError && error.code === 'unsupported') store.dispatch({ type: 'supervisor', patch: { phase: 'unsupported' } })
          // Otherwise keep what is shown; the next reconnect tries again.
        }
      })())
    }
    const self = snapshot.selfUserId
    const apPhase = snapshot.autopilotThread.phase
    if (self !== undefined && isHubOwner(snapshot) && (apPhase === 'ready' || apPhase === 'error')) {
      const threadId = autopilotThreadId(self)
      jobs.push((async () => {
        try {
          const result = await connection.request('supervisor.thread.open', { thread_id: threadId })
          store.dispatch({
            type: 'threadRefreshed', thread: 'autopilot', messages: parseMessages(result.messages), confirmations: parseCards(result.confirmations),
            working: result.busy === true, threadId: str(result.thread_id) ?? threadId,
          })
        } catch (error) {
          if (error instanceof HubRequestError && (error.code === 'not_found' || error.code === 'forbidden' || error.code === 'unsupported' || error.code === 'invalid_request')) {
            store.dispatch({ type: 'autopilotThread', patch: { phase: 'hidden' } })
          }
        }
      })())
    }
    await Promise.all(jobs)
    // Cards whose outcome could not be read before are asked about again (a read: safe to repeat).
    for (const [id, thread] of [...uncheckedCards]) void recheckCard(id, thread)
  }

  // ── machines and the supervisor's model ────────────────────────────

  async function machinesList(): Promise<void> {
    const previous = store.getSnapshot().machines
    store.dispatch({ type: 'machines', state: { phase: 'loading', list: previous.list } })
    try {
      const list = parseMachines(await call('hub.machines.list'))
      store.dispatch({ type: 'machines', state: { phase: 'ready', list } })
    } catch (error) {
      // An older Hub has no machine list: workspaces are shown ungrouped, which is the old behavior.
      if (error instanceof HubRequestError && (error.code === 'unsupported' || error.code === 'invalid_request')) {
        store.dispatch({ type: 'machines', state: { phase: 'ready', list: [] } })
      } else store.dispatch({ type: 'machines', state: { phase: 'error', list: previous.list, error: failureMessage(error) } })
    }
  }

  async function supervisorConfigGet(): Promise<void> {
    store.dispatch({ type: 'supervisorConfig', patch: { phase: 'loading' } })
    try {
      store.dispatch({ type: 'supervisorConfigView', view: parseSupervisorConfig(await call('supervisor.config.get')) })
    } catch (error) {
      if (error instanceof HubRequestError && (error.code === 'unsupported' || error.code === 'invalid_request')) {
        // This Hub predates runtime model choice: nothing to edit here.
        store.dispatch({ type: 'supervisorConfig', patch: { phase: 'ready' } })
      } else store.dispatch({ type: 'supervisorConfig', patch: { phase: 'error', error: failureMessage(error) } })
    }
  }

  async function supervisorConfigOptions(input: SupervisorOptionsInput): Promise<void> {
    const token = (optionsToken += 1)
    const previous = store.getSnapshot().supervisorConfig.options.options
    store.dispatch({ type: 'supervisorConfigOptions', state: { phase: 'loading', ...(previous ? { options: previous } : {}) } })
    try {
      const result = await call('supervisor.config.options', {
        machine_id: input.machineId,
        ...(input.profile === undefined ? {} : { profile: input.profile }),
        ...(input.model === undefined ? {} : { model: input.model }),
      })
      if (token !== optionsToken) return // a newer question is in flight
      const options = parseSupervisorOptions(result)
      store.dispatch({
        type: 'supervisorConfigOptions',
        state: options ? { phase: 'ready', options } : { phase: 'error', error: 'The Hub sent options this app cannot read.' },
      })
    } catch (error) {
      if (token !== optionsToken) return
      store.dispatch({ type: 'supervisorConfigOptions', state: { phase: 'error', error: failureMessage(error) } })
    }
  }

  async function supervisorConfigSet(input: SupervisorConfigInput): Promise<boolean> {
    store.dispatch({ type: 'supervisorConfig', patch: { saving: true } })
    try {
      const result = await call('supervisor.config.set', { machine_id: input.machineId, profile: input.profile, model: input.model, effort: input.effort })
      const view = parseSupervisorConfig(result)
      if (view.status === 'invalid') {
        store.dispatch({ type: 'supervisorConfig', patch: { saving: false, saveError: view.message ?? 'The Hub did not accept that choice.' } })
        return false
      }
      store.dispatch({ type: 'supervisorConfigView', view })
      return true
    } catch (error) {
      if (error instanceof HubRequestError && error.uncertain) {
        // Never send it again: read back what the Hub holds now.
        store.dispatch({ type: 'supervisorConfig', patch: { saving: false, saveError: UNKNOWN_OUTCOME } })
        void supervisorConfigGet()
        return false
      }
      store.dispatch({ type: 'supervisorConfig', patch: { saving: false, saveError: failureMessage(error) } })
      return false
    }
  }

  // ── autopilot ──────────────────────────────────────────────────────

  async function loadAutopilot(): Promise<void> {
    store.dispatch({ type: 'autopilot', patch: { phase: 'loading' } })
    try {
      const view = parseAutopilotView(await call('supervisor.autopilot.get'))
      if (view === undefined) store.dispatch({ type: 'autopilot', patch: { phase: 'error', error: 'The Hub sent an Autopilot answer this app cannot read.' } })
      else {
        store.dispatch({ type: 'autopilotView', view })
        void maybeOpenAutopilotThread()
      }
    } catch (error) {
      if (error instanceof HubRequestError && (error.code === 'unsupported' || error.code === 'invalid_request')) {
        // This Hub predates Autopilot: the control stays hidden.
        store.dispatch({ type: 'autopilot', patch: { phase: 'ready', unsupported: true } })
      } else store.dispatch({ type: 'autopilot', patch: { phase: 'error', error: failureMessage(error) } })
    }
  }

  async function setAutopilot(input: AutopilotInput): Promise<boolean> {
    if (input.instruction !== undefined && input.instruction.length > AUTOPILOT_INSTRUCTION_LIMIT) {
      store.dispatch({ type: 'autopilot', patch: { savingId: undefined, saveError: `The instruction can be at most ${AUTOPILOT_INSTRUCTION_LIMIT} characters.` } })
      return false
    }
    store.dispatch({ type: 'autopilot', patch: { savingId: input.workspaceId } })
    try {
      const result = await call('supervisor.autopilot.set', {
        workspace_id: input.workspaceId,
        enabled: input.enabled,
        ...(input.instruction === undefined ? {} : { instruction: input.instruction }),
        ...(input.expiresAt === undefined ? {} : { expires_at: input.expiresAt }),
      })
      const row = parseAutopilotRow(result.workspace)
      if (row === undefined) {
        // Never show a guess: read back what the Hub holds.
        store.dispatch({ type: 'autopilot', patch: { savingId: undefined, saveError: 'The Hub sent an answer this app cannot read. Checking what it saved…' } })
        void loadAutopilot()
        return false
      }
      store.dispatch({ type: 'autopilotRow', row })
      store.dispatch({ type: 'autopilot', patch: { savingId: undefined } })
      void maybeOpenAutopilotThread()
      return true
    } catch (error) {
      if (error instanceof HubRequestError && error.uncertain) {
        // Never send it again: read back what the Hub holds now.
        store.dispatch({ type: 'autopilot', patch: { savingId: undefined, saveError: UNKNOWN_OUTCOME } })
        void loadAutopilot()
        return false
      }
      store.dispatch({ type: 'autopilot', patch: { savingId: undefined, saveError: failureMessage(error) } })
      return false
    }
  }

  // ── watch ──────────────────────────────────────────────────────────

  async function loadWatches(): Promise<void> {
    store.dispatch({ type: 'watch', patch: { phase: 'loading' } })
    try {
      const view = parseWatchView(await call('supervisor.watch.list'))
      if (view === undefined) store.dispatch({ type: 'watch', patch: { phase: 'error', error: 'The Hub sent a Watch answer this app cannot read.' } })
      else {
        store.dispatch({ type: 'watchView', view })
        void maybeOpenAutopilotThread()
      }
    } catch (error) {
      if (error instanceof HubRequestError && (error.code === 'unsupported' || error.code === 'invalid_request')) {
        // This Hub predates Watch: the section stays hidden.
        store.dispatch({ type: 'watch', patch: { phase: 'ready', unsupported: true } })
      } else store.dispatch({ type: 'watch', patch: { phase: 'error', error: failureMessage(error) } })
    }
  }

  async function setWatch(input: WatchInput): Promise<boolean> {
    const interval = input.reviewIntervalMinutes
    if (interval !== undefined && interval !== null && (!Number.isInteger(interval) || interval < WATCH_REVIEW_MIN_MINUTES || interval > WATCH_REVIEW_MAX_MINUTES)) {
      store.dispatch({ type: 'watch', patch: { savingKey: undefined, saveError: `The review interval must be a whole number of minutes from ${WATCH_REVIEW_MIN_MINUTES} to ${WATCH_REVIEW_MAX_MINUTES}.` } })
      return false
    }
    store.dispatch({ type: 'watch', patch: { savingKey: watchKey(input.workspaceId, input.taskId) } })
    try {
      const result = await call('supervisor.watch.set', {
        workspace_id: input.workspaceId,
        task_id: input.taskId,
        enabled: input.enabled,
        ...(interval === undefined ? {} : { review_interval_minutes: interval }),
        ...(input.expiresAt === undefined ? {} : { expires_at: input.expiresAt }),
      })
      const row = parseWatchRow(result.watch)
      if (row === undefined) {
        // Never show a guess: read back what the Hub holds.
        store.dispatch({ type: 'watch', patch: { savingKey: undefined, saveError: 'The Hub sent an answer this app cannot read. Checking what it saved…' } })
        void loadWatches()
        return false
      }
      store.dispatch({ type: 'watchRow', row })
      store.dispatch({ type: 'watch', patch: { savingKey: undefined } })
      void maybeOpenAutopilotThread()
      return true
    } catch (error) {
      if (error instanceof HubRequestError && error.uncertain) {
        // Never send it again: read back what the Hub holds now.
        store.dispatch({ type: 'watch', patch: { savingKey: undefined, saveError: UNKNOWN_OUTCOME } })
        await loadWatches()
        // The fresh list clears save errors; the person still has to be told the outcome was unclear.
        store.dispatch({ type: 'watch', patch: { saveError: UNKNOWN_OUTCOME } })
        return false
      }
      store.dispatch({ type: 'watch', patch: { savingKey: undefined, saveError: failureMessage(error) } })
      return false
    }
  }

  /**
   * An unreadable `supervisor.watch_update` asks for a fresh list. A burst of
   * them must not become a burst of full lists: the first goes out at once,
   * later ones within the window collapse into one trailing list that is
   * always sent, so the last bad update is never lost.
   */
  let relistTimer: ReturnType<typeof setTimeout> | undefined
  let lastRelistAt = Number.NEGATIVE_INFINITY

  function requestWatchRelist(): void {
    if (relistTimer !== undefined) return // the trailing list is already scheduled
    const wait = Math.max(0, lastRelistAt + WATCH_RELIST_MS - now())
    if (wait === 0) {
      lastRelistAt = now()
      void loadWatches()
      return
    }
    relistTimer = setTimeout(() => {
      relistTimer = undefined
      lastRelistAt = now()
      void loadWatches()
    }, wait)
  }

  // ── the owner's Autopilot thread ───────────────────────────────────

  const isAutopilotThread = (threadId: unknown): boolean => {
    const self = store.getSnapshot().selfUserId
    return typeof threadId === 'string' && self !== undefined && threadId === autopilotThreadId(self)
  }

  /** Opens the thread when the owner has an active workspace or an active watch (watch wakes land here). Anyone else never asks. */
  async function maybeOpenAutopilotThread(): Promise<void> {
    const snapshot = store.getSnapshot()
    if (!isHubOwner(snapshot) || snapshot.selfUserId === undefined) return
    if (!snapshot.autopilot.rows.some(row => row.active) && !snapshot.watch.rows.some(row => row.active)) return
    await autopilotThreadOpen()
  }

  async function autopilotThreadOpen(): Promise<void> {
    const snapshot = store.getSnapshot()
    const self = snapshot.selfUserId
    const phase = snapshot.autopilotThread.phase
    // Only the Hub owner has this thread. `hidden` is the Hub's answer; asking again would loop.
    if (!isHubOwner(snapshot) || self === undefined) return
    if (phase === 'loading' || phase === 'ready' || phase === 'hidden') return
    const threadId = autopilotThreadId(self)
    store.dispatch({ type: 'autopilotThread', patch: { phase: 'loading', threadId } })
    try {
      const result = await connection.request('supervisor.thread.open', { thread_id: threadId })
      store.dispatch({ type: 'autopilotThread', patch: { phase: 'ready', threadId: str(result.thread_id) ?? threadId, working: result.busy === true } })
      store.dispatch({ type: 'autopilotThreadMessages', messages: arr(result.messages).flatMap(raw => { const m = parseSupervisorMessage(raw); return m ? [m] : [] }) })
      store.dispatch({ type: 'autopilotThreadConfirmations', confirmations: arr(result.confirmations).flatMap(raw => { const c = parseConfirmation(raw); return c ? [c] : [] }) })
    } catch (error) {
      if (error instanceof HubRequestError && (error.code === 'not_found' || error.code === 'forbidden' || error.code === 'unsupported' || error.code === 'invalid_request')) {
        store.dispatch({ type: 'autopilotThread', patch: { phase: 'hidden' } })
      } else store.dispatch({ type: 'autopilotThread', patch: { phase: 'error', error: failureMessage(error) } })
    }
  }

  const patchCardIn = (thread: 'shared' | 'autopilot') => (id: string, patch: Partial<SupervisorConfirmation>): void =>
    store.dispatch(thread === 'shared' ? { type: 'supervisorConfirmation', id, patch } : { type: 'autopilotThreadConfirmation', id, patch })

  const autopilotThreadRespond = (confirmationId: string, decision: 'confirm' | 'cancel'): Promise<void> =>
    respondToCard(
      store.getSnapshot().selfUserId === undefined ? undefined : autopilotThreadId(store.getSnapshot().selfUserId!),
      'autopilot',
      confirmationId,
      decision,
    )

  const supervisorRespond = (confirmationId: string, decision: 'confirm' | 'cancel'): Promise<void> =>
    respondToCard(supervisorThread, 'shared', confirmationId, decision)

  /** Read what happened to a card (a read: safe to repeat). True once the outcome is known. */
  async function readCardOutcome(confirmationId: string, thread: 'shared' | 'autopilot'): Promise<boolean> {
    const status = await connection.request('supervisor.action.status', { confirmation_id: confirmationId })
    const known = parseConfirmation({ id: confirmationId, ...rec(status.confirmation ?? status) })
    if (!known || known.state === 'confirming') return false
    uncheckedCards.delete(confirmationId)
    patchCardIn(thread)(confirmationId, { state: known.state, ...(known.message ? { message: known.message } : {}) })
    return true
  }

  async function recheckCard(confirmationId: string, thread: 'shared' | 'autopilot'): Promise<void> {
    try { await readCardOutcome(confirmationId, thread) } catch { /* still unknown; asked again after the next reconnect */ }
  }

  async function respondToCard(
    threadId: string | undefined,
    thread: 'shared' | 'autopilot',
    confirmationId: string,
    decision: 'confirm' | 'cancel',
  ): Promise<void> {
    const patchCard = patchCardIn(thread)
    const cards = thread === 'shared' ? store.getSnapshot().supervisor.confirmations : store.getSnapshot().autopilotThread.confirmations
    const before = cards.find(card => card.id === confirmationId)?.state ?? 'pending'
    patchCard(confirmationId, { state: 'confirming' })
    const params = { confirmation_id: confirmationId, decision, response_id: newCommandId(), ...(threadId ? { thread_id: threadId } : {}) }
    try {
      const result = await connection.request('supervisor.confirmation.respond', params)
      uncheckedCards.delete(confirmationId)
      const updated = parseConfirmation(result.confirmation ?? { id: confirmationId, state: decision === 'confirm' ? 'confirmed' : 'cancelled' })
      patchCard(confirmationId, updated ? { state: updated.state, ...(updated.message ? { message: updated.message } : {}) } : { state: decision === 'confirm' ? 'confirmed' : 'cancelled' })
    } catch (error) {
      if (error instanceof HubRequestError && error.uncertain) {
        // Ask what happened; never send the confirmation again.
        try {
          if (await readCardOutcome(confirmationId, thread)) return
        } catch { /* the status read failed too */ }
        // Not "failed": the outcome is unknown. Keep checking after the next reconnect.
        uncheckedCards.set(confirmationId, thread)
        patchCard(confirmationId, { state: 'confirming', message: 'Checking what happened…' })
        return
      }
      if (error instanceof HubRequestError && TRANSIENT_CARD.has(error.code)) {
        // Nothing was applied: the buttons come back.
        patchCard(confirmationId, { state: before === 'confirming' ? 'pending' : before, message: `${failureMessage(error)} Try again.` })
        return
      }
      patchCard(confirmationId, { state: error instanceof HubRequestError && error.code === 'conflict' ? 'expired' : 'failed', message: failureMessage(error) })
    }
  }

  // ── foreground resync ──────────────────────────────────────────────

  /**
   * Re-read everything a person sees, after a reconnect (`reconnected`) or a
   * foreground return whose probe confirmed the existing link.
   */
  function refreshAll(reconnected: boolean): void {
    void refreshInbox()
    if (store.getSnapshot().usage.phase !== 'idle') void loadUsage()
    if (store.getSnapshot().machines.phase !== 'loading') void machinesList()
    if (store.getSnapshot().supervisorConfig.phase !== 'idle') void supervisorConfigGet()
    if (store.getSnapshot().autopilot.phase !== 'idle' && !store.getSnapshot().autopilot.unsupported) void loadAutopilot()
    if (store.getSnapshot().watch.phase !== 'idle' && !store.getSnapshot().watch.unsupported) void loadWatches()
    void refreshThreads()
    const current = selected()
    // A send that failed on the old link is not news once a new link exists (UI-15). A foreground
    // probe of the same link changes nothing about it: the message stays until a retry.
    if (reconnected && current && current.send.phase === 'failed') store.dispatch({ type: 'send', state: { phase: 'idle' }, workspaceId: current.workspaceId, sessionId: current.sessionId })
    if (current && !sessionSub && !startingSelected) void startSelected(current.workspaceId, current.sessionId)
    else if (current && sessionSub) {
      void refreshLatest()
      void loadAgents()
      // The Hub forgot this device's presence with the old socket: say again where it is.
      if (presenceState !== 'idle') sendPresence(current.workspaceId, current.sessionId, presenceState)
    }
  }

  /**
   * Foreground return. A broken link reconnects at once (its `live`
   * transition resyncs); a link that reads live is probed, and a confirmed
   * link resyncs through `onLinkVerified`. A dead socket is replaced.
   */
  async function resync(): Promise<void> {
    const state = connection.status().state
    if (state === 'revoked' || state === 'closed') return
    if (state !== 'live') {
      connection.resume()
      return
    }
    await connection.verifyLink(RESYNC_PROBE_MS, { notify: true })
  }

  // ── lifecycle ──────────────────────────────────────────────────────

  function start(): void {
    disposers.push(connection.onStatus(status => {
      store.dispatch({ type: 'connection', status })
      if (status.state !== 'live') {
        if (status.state !== 'replaying') wasLive = false
        return
      }
      // Only a transition into `live` resyncs; a repeated `live` report is not a reconnect.
      if (wasLive) return
      wasLive = true
      const userId = connection.helloInfo().userId
      if (userId) store.dispatch({ type: 'self', userId })
      inboxRetryTimer = clearTimer(inboxRetryTimer)
      inboxRetries = 0
      refreshAll(true)
    }))
    // A foreground probe confirmed the existing link: no `live` transition, so re-read here.
    // Heartbeat probes do not notify (a quiet visible phone must not re-read everything each beat).
    const verified = connection.onLinkVerified?.(() => { if (live()) refreshAll(false) })
    if (verified) disposers.push(verified)
    disposers.push(connection.onNotice((type, payload) => {
      if (type === 'hub.presence') {
        const presence = parsePresence(payload)
        if (presence) store.dispatch({ type: 'presence', workspaceId: presence.workspaceId, ...(presence.sessionId ? { sessionId: presence.sessionId } : {}), entries: presence.entries })
      } else if ((type === 'supervisor.thread_event' || type === 'supervisor.confirmation') && isAutopilotThread(payload.thread_id)) {
        // The owner's private Autopilot thread: its own state, never the shared thread's.
        const messages = [payload.message, ...arr(payload.messages)].flatMap(raw => { const m = parseSupervisorMessage(raw); return m ? [m] : [] })
        if (messages.length > 0) store.dispatch({ type: 'autopilotThreadMessages', messages })
        if (payload.activity === 'working') store.dispatch({ type: 'autopilotThread', patch: { working: true } })
        else if (payload.activity === 'idle' || messages.some(message => message.role === 'xerxes')) store.dispatch({ type: 'autopilotThread', patch: { working: false } })
        const confirmation = parseConfirmation(type === 'supervisor.confirmation' ? payload.confirmation ?? payload : payload.confirmation)
        if (confirmation) store.dispatch({ type: 'autopilotThreadConfirmations', confirmations: [confirmation] })
      } else if ((type === 'supervisor.thread_event' || type === 'supervisor.confirmation')
        && typeof payload.thread_id === 'string' && payload.thread_id.startsWith(AUTOPILOT_THREAD_PREFIX)) {
        // Another owner's Autopilot thread, or ours before `selfUserId` is known. Never mix it into
        // the shared thread: opening the Autopilot thread fetches its full state anyway.
      } else if (type === 'supervisor.thread_event') {
        const messages = [payload.message, ...arr(payload.messages)].flatMap(raw => { const m = parseSupervisorMessage(raw); return m ? [m] : [] })
        if (messages.length > 0) store.dispatch({ type: 'supervisorMessages', messages })
        // The Hub reports when a turn starts and ends; a XERXES reply also ends it.
        if (payload.activity === 'working') store.dispatch({ type: 'supervisor', patch: { working: true } })
        // "XERXES stopped: …" is the Hub's notice for a failed turn; an older Hub sends no `idle`.
        else if (payload.activity === 'idle' || messages.some(message => message.role === 'xerxes' || (message.role === 'system' && message.text.startsWith('XERXES stopped')))) store.dispatch({ type: 'supervisor', patch: { working: false } })
        const confirmation = parseConfirmation(payload.confirmation)
        if (confirmation) store.dispatch({ type: 'supervisorConfirmations', confirmations: [confirmation] })
      } else if ((type as string) === 'supervisor.config_changed') {
        // Same shape as config.get. An unreadable payload is a reason to ask again, not to guess.
        if (parseSupervisorConfig(payload).status === 'invalid' && payload.status !== 'invalid') void supervisorConfigGet()
        else store.dispatch({ type: 'supervisorConfigView', view: parseSupervisorConfig(payload) })
      } else if ((type as string) === 'supervisor.autopilot_changed') {
        // One row. An unreadable row is a reason to ask again, not to guess.
        const row = parseAutopilotRow(payload.workspace)
        if (row) {
          store.dispatch({ type: 'autopilotRow', row })
          void maybeOpenAutopilotThread()
        } else void loadAutopilot()
      } else if ((type as string) === 'supervisor.watch_update') {
        // One row, replaced whole (the Hub sends the latest stats, at most one per task per second).
        // An unreadable row is a reason to ask again, not to guess.
        const row = parseWatchRow(payload.watch)
        if (row) {
          store.dispatch({ type: 'watchRow', row })
          void maybeOpenAutopilotThread()
        } else requestWatchRelist()
      } else if (type === 'supervisor.confirmation') {
        const confirmation = parseConfirmation(payload.confirmation ?? payload)
        if (confirmation) store.dispatch({ type: 'supervisorConfirmations', confirmations: [confirmation] })
      }
    }))
    store.dispatch({ type: 'connection', status: connection.status() })
    connection.connect()
  }

  function dispose(): void {
    stopSelected()
    typingTimer = clearTimer(typingTimer)
    inboxRetryTimer = clearTimer(inboxRetryTimer)
    for (const stop of disposers.splice(0)) stop()
    if (relistTimer !== undefined) clearTimeout(relistTimer)
    relistTimer = undefined
    inboxSub?.dispose()
    interactionReplay?.dispose()
    interactionReplay = undefined
    sessionSub?.dispose()
    inboxSub = undefined
    sessionSub = undefined
    connection.close()
  }

  return {
    imagesSupported,
    start, dispose, refreshInbox, openSession, closeSession, loadOlder, loadAgents, inspectAgent, closeAgent, resync,
    reconnectNow: () => { connection.reconnectNow() },
    submit, steer, stopTurn, decide, answerQuestion,
    setModel, setGoal, loadUsage, setPresence, supervisorOpen, supervisorSend, supervisorRespond,
    machinesList, supervisorConfigGet, supervisorConfigOptions, supervisorConfigSet, loadAutopilot, setAutopilot,
    autopilotThreadOpen, autopilotThreadRespond, loadWatches, setWatch,
  }
}
