// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { HubCommandResult, HubMethod } from '../hub/protocol.js'
import { AUTOPILOT_THREAD_PREFIX, autopilotThreadId } from './autopilot.js'
import { newCommandId } from './bridge.js'
import { HubRequestError, type HubConnection, type SubscriptionHandle, type Timers } from './hubConnection.js'
import {
  parseAutopilotRow, parseAutopilotView, parseConfirmation, parseHistoryPage, parseMachines, parsePresence, parseSummaries, parseSupervisorConfig, parseSupervisorMessage,
  parseSupervisorOptions, parseWatchRow, parseWatchView, parseWorkspaces, snapshotSessions, type SupervisorConfirmation,
} from './hubParsers.js'
import { canMutate, isHubOwner, watchKey, type DecisionKind, type HubStore } from './hubStore.js'
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
  closeSession(): void
  loadOlder(): Promise<void>
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
export const HISTORY_REFRESH_MS = 4_000

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
          store.dispatch({ type: 'events', events: [event], tail: false })
        },
        reset: async () => {
          const fresh = parseWorkspaces(await connection.request('hub.workspaces.list'))
          store.dispatch({ type: 'workspaces', list: fresh })
          const cursor = await connection.watermark({ workspaceIds: ids })
          await loadSummaries(fresh)
          return { cursor }
        },
        error: error => store.dispatch({ type: 'inboxError', message: error.message }),
      },
      mark === undefined ? undefined : { cursor: mark },
    )
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
    const page = parseHistoryPage(await connection.request('hub.history', {
      workspace_id: current.workspaceId, session_id: current.sessionId, limit: HISTORY_PAGE_SIZE,
    }))
    if (currentToken(token)) store.dispatch({ type: 'historyLoaded', workspaceId: current.workspaceId, sessionId: current.sessionId, page, throughIds })
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
    if (!task?.pendingCount) return
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
        if (!currentToken(token)) return
        finish()
        store.dispatch({ type: 'sessionError', workspaceId, sessionId, message: error.message })
      },
    })
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
          store.dispatch({ type: 'events', events: [event] })
          if (event.type === 'turn_begin' || event.type === 'turn_end') {
            interactionReplay?.dispose()
            interactionReplay = undefined
          }
          if (event.type === 'turn_end') void refreshLatest()
          else if (event.type === 'tool_call' || event.type === 'tool_result') scheduleHistoryRefresh()
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
        error: error => {
          if (currentToken(token)) store.dispatch({ type: 'sessionError', workspaceId, sessionId, message: error.message })
        },
      }, { cursor })
      void refreshLatest()
      restorePending(workspaceId, sessionId, token)
      void connection.request('hub.presence.update', { workspace_id: workspaceId, session_id: sessionId, state: 'viewing' }).catch(() => undefined)
    } catch (error) {
      if (currentToken(token)) store.dispatch({ type: 'sessionError', workspaceId, sessionId, message: failureMessage(error) })
    } finally {
      if (currentToken(token)) startingSelected = false
    }
  }

  function openSession(workspaceId: string, sessionId: string): void {
    const current = selected()
    if (current && current.workspaceId === workspaceId && current.sessionId === sessionId && current.phase !== 'error' && current.error === undefined) return
    interactionReplay?.dispose()
    interactionReplay = undefined
    sessionSub?.dispose()
    sessionSub = undefined
    openToken += 1
    store.dispatch({ type: 'deselect' })
    store.dispatch({ type: 'select', workspaceId, sessionId })
    cancelHistoryRefresh()
    historyRefresh = undefined
    startingSelected = false
    if (live()) void startSelected(workspaceId, sessionId)
  }

  function closeSession(): void {
    openToken += 1
    cancelHistoryRefresh()
    historyRefresh = undefined
    startingSelected = false
    interactionReplay?.dispose()
    interactionReplay = undefined
    sessionSub?.dispose()
    sessionSub = undefined
    store.dispatch({ type: 'deselect' })
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
      store.dispatch({ type: 'olderLoaded', blocks: page.blocks, hasMore: page.hasMore && page.cursor !== current.olderCursor, ...(page.cursor === undefined ? {} : { cursor: page.cursor }) })
    } catch (error) {
      if (currentToken(token)) store.dispatch({ type: 'olderFailed', message: failureMessage(error) })
    }
  }

  // ── commands ───────────────────────────────────────────────────────

  async function run(action: 'turn.submit' | 'turn.steer' | 'turn.cancel', args: Record<string, unknown>): Promise<HubCommandResult | undefined> {
    const current = selected()
    if (!current) return undefined
    const commandId = newCommandId()
    store.dispatch({ type: 'send', state: { phase: 'sending', commandId } })
    if (!canMutate(store.getSnapshot(), current.workspaceId)) {
      store.dispatch({ type: 'send', state: { phase: 'failed', message: 'Offline. Nothing was sent.' } })
      return undefined
    }
    const result = await connection.command({ workspaceId: current.workspaceId, sessionId: current.sessionId, action, args, commandId })
    if (result.state === 'accepted') store.dispatch({ type: 'send', state: { phase: 'idle' } })
    else if (result.state === 'unknown') store.dispatch({ type: 'send', state: { phase: 'unknown', message: UNKNOWN_OUTCOME, commandId } })
    else store.dispatch({ type: 'send', state: { phase: 'failed', message: result.error?.message ?? 'The Hub did not accept that.' } })
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
    return args !== undefined && (await run('turn.submit', args))?.state === 'accepted'
  }

  async function steer(text: string, images: readonly PreparedImage[]): Promise<boolean> {
    const args = textArgs(text, images)
    return args !== undefined && (await run('turn.steer', args))?.state === 'accepted'
  }

  async function stopTurn(): Promise<boolean> {
    return (await run('turn.cancel', {}))?.state === 'accepted'
  }

  async function respond(interactionId: string, decision: DecisionKind, action: 'permission.respond' | 'question.respond', args: Record<string, unknown>): Promise<void> {
    const current = selected()
    if (!current) return
    store.dispatch({ type: 'decision', interactionId, state: { phase: 'sending', decision } })
    const result = await connection.command({ workspaceId: current.workspaceId, sessionId: current.sessionId, action, args })
    if (result.state === 'accepted') {
      store.dispatch({ type: 'decision', interactionId, state: { phase: 'resolved', decision } })
    } else if (result.error?.code === 'interaction_already_resolved') {
      // The Hub maps every refused interaction response to this code, so the message is shown as is.
      store.dispatch({ type: 'decision', interactionId, state: { phase: 'resolved', decision, already: true, message: result.error.message } })
    } else if (result.state === 'unknown') {
      store.dispatch({ type: 'decision', interactionId, state: { phase: 'unknown', decision, message: UNKNOWN_OUTCOME } })
    } else {
      store.dispatch({ type: 'decision', interactionId, state: { phase: 'failed', decision, message: result.error?.message ?? 'The decision was not applied.' } })
    }
  }

  const decide = (interactionId: string, decision: DecisionKind): Promise<void> =>
    respond(interactionId, decision, 'permission.respond', { request_id: interactionId, response: DECISION_RESPONSE[decision] })

  const answerQuestion = (interactionId: string, answers: Readonly<Record<string, string>>): Promise<void> =>
    respond(interactionId, 'once', 'question.respond', { request_id: interactionId, answers })

  async function configure(field: 'model' | 'goal', requested: string, action: 'model.set' | 'goal.set', args: Record<string, unknown>, expectedRevision?: number): Promise<void> {
    const current = selected()
    if (!current) return
    store.dispatch({ type: 'config', field, value: { requested, phase: 'requested' } })
    const result = await connection.command({
      workspaceId: current.workspaceId,
      sessionId: current.sessionId,
      action,
      args,
      ...(expectedRevision === undefined ? {} : { expectedRevision }),
    })
    if (result.state === 'accepted') store.dispatch({ type: 'config', field, value: { phase: 'accepted' } })
    else if (result.state === 'unknown') store.dispatch({ type: 'config', field, value: { phase: 'unknown', message: UNKNOWN_OUTCOME } })
    else store.dispatch({ type: 'config', field, value: { phase: 'rejected', message: result.error?.message ?? 'The change was not accepted.' } })
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

  function setPresence(state: PresenceState): void {
    const current = selected()
    if (!current || !live()) return
    void connection.request('hub.presence.update', { workspace_id: current.workspaceId, session_id: current.sessionId, state }).catch(() => undefined)
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

  async function supervisorSend(text: string, source: 'text' | 'voice' = 'text', messageId: string = newCommandId()): Promise<boolean> {
    store.dispatch({ type: 'supervisor', patch: { busy: true } })
    try {
      const result = await connection.request('supervisor.message.submit', { message_id: messageId, text, source, ...(supervisorThread ? { thread_id: supervisorThread } : {}) })
      const messages = arr(result.messages).flatMap(raw => { const m = parseSupervisorMessage(raw); return m ? [m] : [] })
      store.dispatch({ type: 'supervisorMessages', messages: messages.length > 0 ? messages : [{ id: messageId, role: 'user', text, at: iso(), source }] })
      store.dispatch({ type: 'supervisorConfirmations', confirmations: arr(result.confirmations).flatMap(raw => { const c = parseConfirmation(raw); return c ? [c] : [] }) })
      // A stored message always starts a turn on the Hub; show it as working
      // until the Hub says idle or the reply lands, so there is no gap
      // between this answer and the Hub's own `activity: working` event.
      store.dispatch({ type: 'supervisor', patch: { busy: false, working: true } })
      return true
    } catch (error) {
      if (error instanceof HubRequestError && (error.code as string) === 'unconfigured') {
        // XERXES has no model yet: show the chooser, not a generic failure.
        store.dispatch({ type: 'supervisorConfig', patch: { status: 'unconfigured' } })
        store.dispatch({ type: 'supervisor', patch: { busy: false, error: UNCONFIGURED_PROMPT } })
        return false
      }
      store.dispatch({ type: 'supervisor', patch: { busy: false, error: error instanceof HubRequestError && error.uncertain ? UNKNOWN_OUTCOME : failureMessage(error) } })
      return false
    }
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

  const autopilotThreadRespond = (confirmationId: string, decision: 'confirm' | 'cancel'): Promise<void> =>
    respondToCard(
      store.getSnapshot().selfUserId === undefined ? undefined : autopilotThreadId(store.getSnapshot().selfUserId!),
      (id, patch) => store.dispatch({ type: 'autopilotThreadConfirmation', id, patch }),
      confirmationId,
      decision,
    )

  const supervisorRespond = (confirmationId: string, decision: 'confirm' | 'cancel'): Promise<void> =>
    respondToCard(supervisorThread, (id, patch) => store.dispatch({ type: 'supervisorConfirmation', id, patch }), confirmationId, decision)

  async function respondToCard(
    threadId: string | undefined,
    patchCard: (id: string, patch: Partial<SupervisorConfirmation>) => void,
    confirmationId: string,
    decision: 'confirm' | 'cancel',
  ): Promise<void> {
    patchCard(confirmationId, { state: 'confirming' })
    const params = { confirmation_id: confirmationId, decision, response_id: newCommandId(), ...(threadId ? { thread_id: threadId } : {}) }
    try {
      const result = await connection.request('supervisor.confirmation.respond', params)
      const updated = parseConfirmation(result.confirmation ?? { id: confirmationId, state: decision === 'confirm' ? 'confirmed' : 'cancelled' })
      patchCard(confirmationId, updated ? { state: updated.state, ...(updated.message ? { message: updated.message } : {}) } : { state: decision === 'confirm' ? 'confirmed' : 'cancelled' })
    } catch (error) {
      if (error instanceof HubRequestError && error.uncertain) {
        // Ask what happened; never send the confirmation again.
        try {
          const status = await connection.request('supervisor.action.status', { confirmation_id: confirmationId })
          const known = parseConfirmation({ id: confirmationId, ...rec(status.confirmation ?? status) })
          if (known && known.state !== 'confirming' && known.state !== 'pending') {
            patchCard(confirmationId, { state: known.state })
            return
          }
        } catch { /* fall through to the honest unknown */ }
        patchCard(confirmationId, { state: 'failed', message: UNKNOWN_OUTCOME })
        return
      }
      patchCard(confirmationId, { state: error instanceof HubRequestError && error.code === 'conflict' ? 'expired' : 'failed', message: failureMessage(error) })
    }
  }

  // ── lifecycle ──────────────────────────────────────────────────────

  function start(): void {
    disposers.push(connection.onStatus(status => {
      store.dispatch({ type: 'connection', status })
      if (status.state !== 'live') return
      const userId = connection.helloInfo().userId
      if (userId) store.dispatch({ type: 'self', userId })
      void refreshInbox()
      if (store.getSnapshot().usage.phase !== 'idle') void loadUsage()
      void machinesList()
      if (store.getSnapshot().supervisorConfig.phase !== 'idle') void supervisorConfigGet()
      if (store.getSnapshot().autopilot.phase !== 'idle' && !store.getSnapshot().autopilot.unsupported) void loadAutopilot()
      if (store.getSnapshot().watch.phase !== 'idle' && !store.getSnapshot().watch.unsupported) void loadWatches()
      const current = selected()
      if (current && !sessionSub && !startingSelected) void startSelected(current.workspaceId, current.sessionId)
      else if (current && sessionSub) void refreshLatest()
    }))
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
    openToken += 1
    cancelHistoryRefresh()
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
    start, dispose, refreshInbox, openSession, closeSession, loadOlder, submit, steer, stopTurn, decide, answerQuestion,
    setModel, setGoal, loadUsage, setPresence, supervisorOpen, supervisorSend, supervisorRespond,
    machinesList, supervisorConfigGet, supervisorConfigOptions, supervisorConfigSet, loadAutopilot, setAutopilot,
    autopilotThreadOpen, autopilotThreadRespond, loadWatches, setWatch,
  }
}
