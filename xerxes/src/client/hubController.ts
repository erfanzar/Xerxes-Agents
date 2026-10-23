// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { HubCommandResult, HubMethod } from '../hub/protocol.js'
import { newCommandId } from './bridge.js'
import { HubRequestError, type HubConnection, type SubscriptionHandle } from './hubConnection.js'
import {
  parseConfirmation, parseHistoryPage, parseMachines, parsePresence, parseSummaries, parseSupervisorConfig, parseSupervisorMessage,
  parseSupervisorOptions, parseWorkspaces, snapshotSessions,
} from './hubParsers.js'
import { canMutate, type DecisionKind, type HubStore } from './hubStore.js'
import type { WorkspaceView } from './hubTypes.js'
import type { PreparedImage } from './imagePrep.js'
import { arr, rec, str } from './parse.js'
import { usageExtrasFromResult, usageFromResult } from './usageMerge.js'

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
  supervisorSend(text: string, source?: 'text' | 'voice'): Promise<boolean>
  supervisorRespond(confirmationId: string, decision: 'confirm' | 'cancel'): Promise<void>
  /** Loads `hub.machines.list` into `state.machines`. */
  machinesList(): Promise<void>
  /** Loads `supervisor.config.get` into `state.supervisorConfig`. */
  supervisorConfigGet(): Promise<void>
  /** Loads the chooser lists of one machine (and profile) into `state.supervisorConfig.options`. */
  supervisorConfigOptions(input: SupervisorOptionsInput): Promise<void>
  /** Resolves true only when the Hub saved the choice. A refusal is left in `supervisorConfig.saveError`. */
  supervisorConfigSet(input: SupervisorConfigInput): Promise<boolean>
}

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
}

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
  let openToken = 0
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

  function startSelected(workspaceId: string, sessionId: string): void {
    sessionSub?.dispose()
    // The Hub has no per-session snapshot: the task is rebuilt by replaying its
    // journal (bounded by retention), then followed live. `caughtUp` marks the end of the replay.
    sessionSub = connection.subscribe(
      { workspaceIds: [workspaceId], sessionIds: [sessionId] },
      {
        apply: event => store.dispatch({ type: 'events', events: [event] }),
        caughtUp: () => store.dispatch({ type: 'sessionReady', workspaceId, sessionId }),
        reset: () => {
          store.dispatch({ type: 'sessionReset', workspaceId, sessionId })
          return { fromStart: true }
        },
        error: error => store.dispatch({ type: 'sessionError', workspaceId, sessionId, message: error.message }),
      },
    )
    void connection.request('hub.presence.update', { workspace_id: workspaceId, session_id: sessionId, state: 'viewing' }).catch(() => undefined)
  }

  function openSession(workspaceId: string, sessionId: string): void {
    const current = selected()
    if (current && current.workspaceId === workspaceId && current.sessionId === sessionId && current.phase !== 'error') return
    sessionSub?.dispose()
    sessionSub = undefined
    openToken += 1
    store.dispatch({ type: 'deselect' })
    store.dispatch({ type: 'select', workspaceId, sessionId })
    if (live()) startSelected(workspaceId, sessionId)
  }

  function closeSession(): void {
    openToken += 1
    sessionSub?.dispose()
    sessionSub = undefined
    store.dispatch({ type: 'deselect' })
  }

  async function loadOlder(): Promise<void> {
    const current = selected()
    if (!current || current.loadingOlder || !current.hasOlder) return
    store.dispatch({ type: 'olderLoading' })
    try {
      const result = await connection.request('hub.history', {
        workspace_id: current.workspaceId,
        session_id: current.sessionId,
        limit: 20,
        ...(current.olderCursor === undefined ? {} : { before: current.olderCursor }),
      })
      const page = parseHistoryPage(result)
      store.dispatch({ type: 'olderLoaded', blocks: page.blocks, hasMore: page.hasMore, ...(page.cursor === undefined ? {} : { cursor: page.cursor }) })
    } catch (error) {
      store.dispatch({ type: 'olderFailed', message: failureMessage(error) })
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
      store.dispatch({ type: 'supervisor', patch: { phase: 'ready', ...(supervisorThread ? { threadId: supervisorThread } : {}) } })
      store.dispatch({ type: 'supervisorMessages', messages: arr(result.messages).flatMap(raw => { const m = parseSupervisorMessage(raw); return m ? [m] : [] }) })
      store.dispatch({ type: 'supervisorConfirmations', confirmations: arr(result.confirmations).flatMap(raw => { const c = parseConfirmation(raw); return c ? [c] : [] }) })
      // thread.open reports the model status too; the full answer (who may edit, which model) comes from config.get.
      const reported = str(result.config_status)
      if (reported !== undefined && store.getSnapshot().supervisorConfig.phase === 'idle') {
        const known = parseSupervisorConfig({ status: reported })
        if (known.status !== 'invalid') store.dispatch({ type: 'supervisorConfig', patch: { status: known.status } })
      }
      void supervisorConfigGet()
    } catch (error) {
      if (error instanceof HubRequestError && error.code === 'unsupported') store.dispatch({ type: 'supervisor', patch: { phase: 'unsupported' } })
      else store.dispatch({ type: 'supervisor', patch: { phase: 'error', error: failureMessage(error) } })
    }
  }

  async function supervisorSend(text: string, source: 'text' | 'voice' = 'text'): Promise<boolean> {
    const messageId = newCommandId()
    store.dispatch({ type: 'supervisor', patch: { busy: true } })
    try {
      const result = await connection.request('supervisor.message.submit', { message_id: messageId, text, source, ...(supervisorThread ? { thread_id: supervisorThread } : {}) })
      const messages = arr(result.messages).flatMap(raw => { const m = parseSupervisorMessage(raw); return m ? [m] : [] })
      store.dispatch({ type: 'supervisorMessages', messages: messages.length > 0 ? messages : [{ id: messageId, role: 'user', text, at: iso(), source }] })
      store.dispatch({ type: 'supervisorConfirmations', confirmations: arr(result.confirmations).flatMap(raw => { const c = parseConfirmation(raw); return c ? [c] : [] }) })
      store.dispatch({ type: 'supervisor', patch: { busy: false } })
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

  async function supervisorRespond(confirmationId: string, decision: 'confirm' | 'cancel'): Promise<void> {
    store.dispatch({ type: 'supervisorConfirmation', id: confirmationId, patch: { state: 'confirming' } })
    const params = { confirmation_id: confirmationId, decision, response_id: newCommandId(), ...(supervisorThread ? { thread_id: supervisorThread } : {}) }
    try {
      const result = await connection.request('supervisor.confirmation.respond', params)
      const updated = parseConfirmation(result.confirmation ?? { id: confirmationId, state: decision === 'confirm' ? 'confirmed' : 'cancelled' })
      store.dispatch({ type: 'supervisorConfirmation', id: confirmationId, patch: updated ? { state: updated.state, ...(updated.message ? { message: updated.message } : {}) } : { state: decision === 'confirm' ? 'confirmed' : 'cancelled' } })
    } catch (error) {
      if (error instanceof HubRequestError && error.uncertain) {
        // Ask what happened; never send the confirmation again.
        try {
          const status = await connection.request('supervisor.action.status', { confirmation_id: confirmationId })
          const known = parseConfirmation({ id: confirmationId, ...rec(status.confirmation ?? status) })
          if (known && known.state !== 'confirming' && known.state !== 'pending') {
            store.dispatch({ type: 'supervisorConfirmation', id: confirmationId, patch: { state: known.state } })
            return
          }
        } catch { /* fall through to the honest unknown */ }
        store.dispatch({ type: 'supervisorConfirmation', id: confirmationId, patch: { state: 'failed', message: UNKNOWN_OUTCOME } })
        return
      }
      store.dispatch({ type: 'supervisorConfirmation', id: confirmationId, patch: { state: error instanceof HubRequestError && error.code === 'conflict' ? 'expired' : 'failed', message: failureMessage(error) } })
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
      const current = selected()
      if (current && !sessionSub) startSelected(current.workspaceId, current.sessionId)
    }))
    disposers.push(connection.onNotice((type, payload) => {
      if (type === 'hub.presence') {
        const presence = parsePresence(payload)
        if (presence) store.dispatch({ type: 'presence', workspaceId: presence.workspaceId, ...(presence.sessionId ? { sessionId: presence.sessionId } : {}), entries: presence.entries })
      } else if (type === 'supervisor.thread_event') {
        const messages = [payload.message, ...arr(payload.messages)].flatMap(raw => { const m = parseSupervisorMessage(raw); return m ? [m] : [] })
        if (messages.length > 0) store.dispatch({ type: 'supervisorMessages', messages })
        const confirmation = parseConfirmation(payload.confirmation)
        if (confirmation) store.dispatch({ type: 'supervisorConfirmations', confirmations: [confirmation] })
      } else if ((type as string) === 'supervisor.config_changed') {
        // Same shape as config.get. An unreadable payload is a reason to ask again, not to guess.
        if (parseSupervisorConfig(payload).status === 'invalid' && payload.status !== 'invalid') void supervisorConfigGet()
        else store.dispatch({ type: 'supervisorConfigView', view: parseSupervisorConfig(payload) })
      } else if (type === 'supervisor.confirmation') {
        const confirmation = parseConfirmation(payload.confirmation ?? payload)
        if (confirmation) store.dispatch({ type: 'supervisorConfirmations', confirmations: [confirmation] })
      }
    }))
    store.dispatch({ type: 'connection', status: connection.status() })
    connection.connect()
  }

  function dispose(): void {
    for (const stop of disposers.splice(0)) stop()
    inboxSub?.dispose()
    sessionSub?.dispose()
    inboxSub = undefined
    sessionSub = undefined
    connection.close()
  }

  return {
    imagesSupported,
    start, dispose, refreshInbox, openSession, closeSession, loadOlder, submit, steer, stopTurn, decide, answerQuestion,
    setModel, setGoal, loadUsage, setPresence, supervisorOpen, supervisorSend, supervisorRespond,
    machinesList, supervisorConfigGet, supervisorConfigOptions, supervisorConfigSet,
  }
}
