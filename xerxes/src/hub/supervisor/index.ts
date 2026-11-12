// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash, randomUUID } from 'node:crypto'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { Database } from 'bun:sqlite'

import { canSeeWorkspace, grantedSessionIds, hasScope, type HubGrant, type HubPrincipal } from '../../security/principal.js'
import type { AuthenticatedContext } from '../identityPort.js'
import { HubProtocolError } from '../protocol.js'
import type { CommandBackend } from '../commandRouter.js'
import type { HubIdentityPort } from '../identityPort.js'
import type { HubActor } from '../protocol.js'
import type { HubWorkspaceView } from '../server.js'
import type { SupervisorThreadStatus, SupervisorVoiceBridge } from '../voice/supervisorBridge.js'
import { ActionService, SupervisorDenied, SupervisorGrantOverlay, type ActionExecutor, type ActionTargets, type AutopilotGrant, type RiskActionHost } from './actions.js'
import { AttentionQueue, attentionToWire, type AttentionItem } from './attention.js'
import { SUPERVISOR_DISPLAY_NAME, SUPERVISOR_PRINCIPAL_ID, SupervisorAudit, supervisorActor, supervisorRequestHash } from './audit.js'
import {
  AutopilotActionBudget, AutopilotSettings, AutopilotWaker, autopilotOwnerPrincipal, displayOption, scanPendingInteractions, type PendingInteractions,
} from './autopilot.js'
import { DigestService, readStatusSnapshot, type DigestEventSource, type StatusSnapshot } from './digests.js'
import { createSupervisorHandlers, type SupervisorHandlers } from './handlers.js'
import {
  DEFAULT_SUPERVISOR_MAX_TOKENS_PER_WINDOW, SupervisorRunner, createModelDigestSummarizer, type SupervisorModelPort, type SupervisorQuotas,
  type SupervisorStatusInput,
} from './runner.js'
import { createSupervisorToolRegistry, type SupervisorToolDeps } from './tools.js'
import { SuperagentAbilities } from './abilities/service.js'
import { SuperagentControl } from './abilities/control.js'
import { SubscriptionEvents } from './abilities/events.js'
import { WatchService, type WatchEvent } from './watch.js'
import {
  SupervisorThreads, autopilotThreadId, delegatedScopes, messageToWire,
  type MessageSource, type ScopeReader, type SupervisorEventSink, type ThreadMessage, type UserPrincipal,
} from './threads.js'

/**
 * Composition of the XERXES supervisor: threads, digests, confirmed actions,
 * attention watchers, the dedicated tool registry and the runner, all on one
 * SQLite connection to the Hub database (WAL), plus the facade the voice
 * subsystem uses (`SupervisorVoiceBridge`).
 */

export { SupervisorGrantOverlay } from './actions.js'
export { createUnsupportedSupervisorHandlers } from './handlers.js'
export { createLlmModelPort, createModelDigestSummarizer } from './runner.js'
export { AUTOPILOT_ENV, autopilotCheckInFromEnvironment } from './autopilot.js'
export { WATCH_ENV, watchStallFromEnvironment } from './watch.js'
export type { SupervisorModelPort, SupervisorQuotas } from './runner.js'

/**
 * The operator's configured quotas, unchanged. XERXES is the owner's personal agent, so no spend
 * bound is added by default; `DEFAULT_SUPERVISOR_MAX_TOKENS_PER_WINDOW` is only a suggested value.
 */
export function withDefaultSpendBound(quotas: Partial<SupervisorQuotas> | undefined): Partial<SupervisorQuotas> {
  return quotas ?? {}
}
void DEFAULT_SUPERVISOR_MAX_TOKENS_PER_WINDOW

/** Env keys read by `xerxes hub`. */
export const SUPERVISOR_ENV = Object.freeze({
  profile: 'XERXES_HUB_SUPERVISOR_PROFILE',
  model: 'XERXES_HUB_SUPERVISOR_MODEL',
})

export const SUPERVISOR_SETUP_HINT =
  `The XERXES supervisor is not configured on this Hub. Set ${SUPERVISOR_ENV.profile} (a provider profile name) and ${SUPERVISOR_ENV.model} (a cheap, tool-capable model id) and restart the Hub. There is no default model.`

/** The persona prompt shipped next to the agent presets. Throws when it cannot be found. */
export function loadSupervisorPersona(): string {
  // Source tree (src/hub/supervisor -> src/agents/default), then the bundle:
  // build:runtime copies agents/default to dist/default beside dist/cli.js.
  for (const relative of ['../../agents/default/supervisor.md', './default/supervisor.md', './agents/default/supervisor.md']) {
    try {
      const text = readFileSync(fileURLToPath(new URL(relative, import.meta.url)), 'utf8')
      if (text.trim() !== '') return text
    } catch {
      // try the next location
    }
  }
  throw new Error('the supervisor persona (agents/default/supervisor.md) was not found in this build')
}

export interface PushLike {
  (userId: string, message: { readonly title: string; readonly body: string; readonly url: string; readonly dedupeKey?: string }): Promise<unknown>
}

export interface HubSupervisorOptions {
  readonly databasePath: string
  readonly identity: Pick<HubIdentityPort, 'grantsFor' | 'audit'> & Partial<Pick<HubIdentityPort, 'onRevoked'>>
  readonly overlay: SupervisorGrantOverlay
  /** Runs a confirmed command through `HubCommandRouter.execute`. */
  readonly execute: ActionExecutor
  /** Optional typed risky operations. Ownership always comes from the live identity service, never this port or model text. */
  readonly risk?: Omit<RiskActionHost, 'isOwner'>
  readonly workspaces: Pick<HubWorkspaceView, 'list' | 'projection' | 'health'>
  readonly backend: Pick<CommandBackend, 'call'>
  /** Explicit opt-in to native disk sampling; no reads without an authorized disk subscription. */
  readonly diskPolling?: { readonly intervalMs?: number }
  /** Journal for digests; observers feed attention. */
  readonly events?: DigestEventSource & { onEvents?(observer: (events: readonly WatchEvent[]) => void): () => void }
  /** Bounded reverse reads of retained task records; separate from summary/digest event fakes. */
  readonly journal?: SupervisorToolDeps['journal']
  readonly model: SupervisorModelPort
  /** Whether XERXES has a model to run on; `unconfigured` makes `supervisor.message.submit` refuse. */
  readonly configStatus?: () => { readonly status: string; readonly message?: string }
  readonly persona?: string
  readonly quotas?: Partial<SupervisorQuotas>
  readonly sendPush?: PushLike
  /** Only true when the runtime guarantees `model.set` never touches defaults. Off by default. */
  readonly modelSetSessionOnly?: boolean
  /**
   * Live check that a user is the Hub owner (XERXES Autopilot stays active only while its
   * enabler is). Absent: the owner flag last stored for that person in the shared thread.
   */
  readonly isHubOwner?: (userId: string) => boolean
  /** Sends `supervisor.autopilot_changed` to the people who can see the workspace. */
  readonly broadcastAutopilot?: (workspaceId: string, payload: Readonly<Record<string, unknown>>) => void
  /** Autopilot timings: check-in period (0 disables; default 15 min), per-reason cooldown (10 min), wakes per workspace per hour (12). */
  readonly autopilot?: {
    readonly checkInMs?: number; readonly cooldownMs?: number; readonly maxWakesPerHour?: number
    /** Autonomous actions (runs without Confirm, answered questions) per workspace per hour. Default 30. */
    readonly maxActionsPerHour?: number
  }
  /** Sends `supervisor.watch_update` to the people who can read the task. */
  readonly broadcastWatch?: (workspaceId: string, taskId: string, payload: Readonly<Record<string, unknown>>) => void
  /** XERXES Watch timings: stall threshold (default 10 min, at least 60 s), wake budgets per task per hour (6 anomaly, 12 review), update throttle (1 s). */
  readonly watch?: {
    readonly stallMs?: number; readonly maxAnomalyWakesPerHour?: number; readonly maxReviewWakesPerHour?: number; readonly updateIntervalMs?: number
    /** Watch wakes across all tasks per hour (default 60) and the per (task, anomaly kind) cooldown (default 30 min). */
    readonly maxHubWakesPerHour?: number; readonly anomalyCooldownMs?: number
  }
  readonly clock?: () => number
  readonly onError?: (error: unknown, context: string) => void
}

export interface HubSupervisor {
  readonly handlers: SupervisorHandlers
  readonly voice: SupervisorVoiceBridge
  readonly threads: SupervisorThreads
  readonly actions: ActionService
  readonly attention: AttentionQueue
  readonly digests: DigestService
  readonly runner: SupervisorRunner
  readonly autopilot: AutopilotSettings
  readonly waker: AutopilotWaker
  readonly watch: WatchService
  attachSink(sink: SupervisorEventSink | undefined): void
  close(): void
}

function sha(text: string): string {
  return createHash('sha256').update(text).digest('hex').slice(0, 24)
}

export function createHubSupervisor(options: HubSupervisorOptions): HubSupervisor {
  const clock = options.clock ?? Date.now
  const report = (error: unknown, context: string): void => options.onError?.(error, context)
  const db = new Database(options.databasePath, { create: true, strict: true })
  db.exec('PRAGMA journal_mode = WAL; PRAGMA synchronous = FULL; PRAGMA busy_timeout = 5000;')

  const { workspaces, identity } = options
  const eligible = (principal: UserPrincipal, grants: Parameters<typeof canSeeWorkspace>[1]): boolean =>
    workspaces.list().some(workspace =>
      canSeeWorkspace(principal, grants, workspace.id, clock()) || grantedSessionIds(principal, grants, workspace.id, clock()).length > 0)

  const threads = new SupervisorThreads({ db, identity, clock, eligible, onError: report,
    ...(options.isHubOwner === undefined ? {} : { isHubOwner: options.isHubOwner }) })
  threads.ensureShared()

  const autopilot = new AutopilotSettings({
    db,
    workspaces,
    // Fail closed without a live owner check: the owner flag stored for that person in the shared thread.
    isHubOwner: options.isHubOwner ?? (userId => threads.principalOrStored(userId)?.hubOwner === true),
    audit: record => identity.audit(record),
    ...(options.broadcastAutopilot === undefined ? {} : { broadcast: options.broadcastAutopilot }),
    // An automatic disable is told in the shared thread (people who can see the workspace) and pushed to its enabler.
    onChange: change => {
      if (change.reason === 'set' || change.ownerId === undefined) return
      const why = change.reason === 'expired' ? 'it expired' : 'the person who turned it on is no longer the Hub owner'
      postWorkspace(change.workspaceId, change.ownerId, `Autopilot needs you: Autopilot is off in ${change.label} because ${why}.`)
      void options.sendPush?.(change.ownerId, { title: 'XERXES', body: 'Autopilot turned off.', url: '/xerxes' })
        ?.catch(error => report(error, 'autopilot off push'))
    },
    clock,
    onError: report,
  })

  const actionBudget = new AutopilotActionBudget({
    ...(options.autopilot?.maxActionsPerHour === undefined ? {} : { limit: options.autopilot.maxActionsPerHour }),
    clock,
  })

  const isOwner = options.isHubOwner ?? ((userId: string) => threads.principalOrStored(userId)?.hubOwner === true)
  // New privileged controls require a live identity port, never a cached participant flag.
  const controlOwner = options.isHubOwner ?? (() => false)
  const control = new SuperagentControl({ db, clock, isOwner: controlOwner,
    audit: event => identity.audit({ actor: event.ownerId === undefined ? supervisorActor(SUPERVISOR_PRINCIPAL_ID)
      : { principal_kind: 'user', principal_id: event.ownerId, display_name: threads.principalOrStored(event.ownerId)?.displayName ?? 'Hub owner' },
      action: `superagent.control.${event.kind}`, requestHash: event.hash, decision: 'allowed', outcome: 'n/a' }) })
  const controlledModel = control.model(options.model)

  /**
   * XERXES is the Hub owner's personal agent. When the owner is the one talking to it (or a Hub
   * wake-up runs in the owner's own thread), its actions and answers run at once, with no Confirm
   * tap, whether or not Autopilot is on. Anyone else in the conversation, or a non-owner requester,
   * keeps Confirm cards: their messages never borrow the owner's authority.
   */
  const autopilotGrant = (input: {
    readonly workspaceId: string; readonly threadId: string; readonly requester: UserPrincipal
    readonly trigger: boolean; readonly othersInContext: boolean
  }): AutopilotGrant | undefined => {
    if (!input.requester.hubOwner || !isOwner(input.requester.id)) return undefined
    const owner: UserPrincipal = { ...input.requester, hubOwner: true }
    if (input.trigger) {
      // Watch-only Autopilot: a wake-up never acts by itself; its actions become Confirm cards.
      if (autopilot.active(input.workspaceId)?.mode === 'observe') return undefined
      // A wake turn counts only in the owner's own Autopilot thread.
      return input.threadId === autopilotThreadId(owner.id) ? { owner, wake: true } : undefined
    }
    if (input.othersInContext) return undefined
    return { owner, wake: false }
  }
  void autopilotOwnerPrincipal

  /**
   * Watch-only Autopilot: a wake-up's answer to a question or permission request is
   * recorded in the shared thread as what XERXES would have answered, and nothing is sent.
   * Undefined when the turn is not a watch-only wake-up, so the normal path applies.
   */
  const observedAnswer = (input: {
    readonly workspaceId: string; readonly sessionId: string; readonly requestId: string
    readonly requester: UserPrincipal; readonly trigger: boolean
  }, kind: 'question' | 'permission request', answer: string): Record<string, unknown> | undefined => {
    if (!input.trigger) return undefined
    const active = autopilot.active(input.workspaceId)
    if (active?.mode !== 'observe') return undefined
    const rawTitle = workspaces.projection(input.workspaceId)?.sessions[input.sessionId]?.title ?? input.sessionId
    const title = rawTitle.replace(/\s+/g, ' ').trim().slice(0, 80)
    const text = answer.replace(/\s+/g, ' ').trim()
    const shown = text.length <= 300 ? text : `${text.slice(0, 299)}…`
    try {
      postShared(input.workspaceId, input.sessionId, active.ownerId, `Autopilot (watch only) would answer the ${kind} in “${title}”: ${shown}`)
    } catch (error) {
      report(error, 'autopilot observe notice')
    }
    return {
      state: 'recorded', request_id: input.requestId, mode: 'observe',
      note: `Watch-only mode: nothing was sent. The ${kind} still waits for a person.`,
    }
  }

  const digests = new DigestService({
    workspaces,
    ...(options.events === undefined ? {} : { events: options.events }),
    summarizer: createModelDigestSummarizer(controlledModel),
    clock,
  })

  const targets: ActionTargets = {
    instanceId: id => workspaces.health(id)?.instanceId,
    revision: (workspaceId, sessionId) => {
      const row = workspaces.projection(workspaceId)?.sessions[sessionId]
      return row === undefined ? undefined : sha(`${row.status}|${row.goal_summary}|${row.model}`)
    },
    hasGoal: (workspaceId, sessionId) => {
      const row = workspaces.projection(workspaceId)?.sessions[sessionId]
      return row === undefined ? undefined : row.goal_summary !== ''
    },
    online: id => workspaces.health(id)?.state === 'online',
    label: id => workspaces.list().find(workspace => workspace.id === id)?.label,
    title: (workspaceId, sessionId) => workspaces.projection(workspaceId)?.sessions[sessionId]?.title,
    status: (workspaceId, sessionId) => workspaces.projection(workspaceId)?.sessions[sessionId]?.status,
  }

  const actions = new ActionService({
    db,
    threads,
    identity,
    targets,
    execute: options.execute,
    isHubOwner: isOwner,
    dispatchFence: () => control.assertActive(),
    dispatchSignal: () => control.signal,
    beforeDispatch: async action => { await control.reserve(action.tool === 'risk.execute' ? 'risk' : 'command', action.id); control.assertActive() },
    ...(options.risk === undefined ? {} : { risk: { ...options.risk, isOwner: controlOwner,
      execute: (action, owner) => { control.assertActive(); return options.risk!.execute(action, owner, control.signal) } } }),
    overlay: options.overlay,
    access: (threadId, principal) => { threads.access(threadId, principal) },
    clock,
    modelSetSessionOnly: options.modelSetSessionOnly === true,
    autopilot: input => autopilotGrant(input),
    autopilotBudget: actionBudget,
    autopilotNotice: ({ workspaceId, sessionId, ownerId, text }) => postShared(workspaceId, sessionId, ownerId, text),
    ...(options.sendPush === undefined ? {} : { push: (userId: string, message: { title: string; body: string; url: string }) => options.sendPush!(userId, message) }),
    // Declared below; a card can only be proposed once a turn runs, long after construction.
    onWatchCard: action => watch.cardProposed(action),
    onError: report,
  })

  /** People in the shared thread who may read this task (or, without a task, the workspace). */
  const readersOf = (workspaceId: string, sessionId?: string): string[] => {
    const shared = threads.get('shared')
    if (!shared) return []
    const target = sessionId === undefined ? { workspaceId } : { workspaceId, sessionId }
    return threads.participants(shared.id).flatMap(person => {
      const principal = threads.principalOrStored(person.id)
      if (!principal) return []
      return hasScope(principal, identity.grantsFor(person.id), target, 'session.read', clock()) ? [person.id] : []
    })
  }
  const readerOf = (item: AttentionItem): string[] => readersOf(item.workspaceId, item.sessionId)

  /** A system notice in the shared thread, only for readers of the target and the owner. */
  const postShared = (workspaceId: string, sessionId: string | undefined, ownerId: string, text: string): void => {
    const audience = [...new Set([...readersOf(workspaceId, sessionId), ownerId])]
    const { message } = threads.append({ threadId: 'shared', role: 'system', text, audience })
    threads.emit('shared', 'supervisor.thread_event', { message: messageToWire(message) }, audience)
  }
  const postWorkspace = (workspaceId: string, ownerId: string, text: string): void => {
    try { postShared(workspaceId, undefined, ownerId, text) } catch (error) { report(error, 'autopilot notice') }
  }

  const pendingOf = (workspaceId: string, sessionId: string): PendingInteractions | undefined =>
    options.events === undefined ? undefined : scanPendingInteractions(options.events, workspaceId, sessionId)

  const supervisorAudit = new SupervisorAudit(identity, report)

  /** Autopilot: answer one waiting QUESTION for the enabling owner. Never a permission. */
  const answerQuestion = async (input: {
    readonly threadId: string; readonly workspaceId: string; readonly sessionId: string; readonly requestId: string
    readonly answers: Readonly<Record<string, string>>; readonly requester: UserPrincipal
    readonly reader: ScopeReader; readonly trigger: boolean; readonly othersInContext: boolean
    readonly target?: { readonly workspaceId: string; readonly sessionId: string }
  }): Promise<Record<string, unknown>> => {
    control.assertActive()
    const signal = control.signal
    const { workspaceId, sessionId, requestId } = input
    const watched = observedAnswer(input, 'question', Object.values(input.answers).join('; '))
    if (watched !== undefined) return watched
    const grant = autopilotGrant(input)
    if (grant === undefined) {
      throw new SupervisorDenied('XERXES answers a task\'s question only for the Hub owner, in a conversation with nobody else in it. Ask the owner to answer it.')
    }
    // The same shared-scope check createIntent applies: everyone this thread speaks for must be able to answer.
    const target = { workspaceId, sessionId }
    if (!input.reader.canRead(target) || !input.reader.scopes(target).has('question.delegated')) {
      throw new SupervisorDenied('question.delegated is required on this task for everyone in this thread, so XERXES cannot answer it here.')
    }
    if (requestId.length === 0 || requestId.length > 200) throw new Error('request_id must be 1 to 200 characters')
    const pending = pendingOf(workspaceId, sessionId)
    if (pending === undefined) throw new Error('waiting questions cannot be read on this Hub, so XERXES will not answer blind')
    if (pending.permissionIds.includes(requestId)) {
      throw new Error('That is a permission request, not a question; the human owner must decide it in the task approval UI.')
    }
    const question = pending.questions.find(row => row.request_id === requestId)
    if (question === undefined) throw new Error('no such waiting question in this task; read pending_interactions again')
    const answers: Record<string, string> = {}
    for (const [key, value] of Object.entries(input.answers)) {
      const item = question.questions.find(row => row.id === key)
      if (item === undefined) throw new Error(`question ${JSON.stringify(key.slice(0, 60))} is not part of that request`)
      // Options are compared in full; the clipped form the model was shown maps back to its full text.
      const exact = item.options.find(option => option === value) ?? item.options.find(option => displayOption(option) === value)
      if (exact === undefined && !item.allow_free_form) throw new Error(`the answer to ${JSON.stringify(key.slice(0, 60))} must be one of its options`)
      answers[key] = exact ?? value
    }
    const owner = grant.owner
    let scopes = delegatedScopes(owner, identity.grantsFor(owner.id), target, clock())
    if (!scopes.has('question.delegated')) throw new SupervisorDenied(`${owner.displayName} cannot answer questions in this task, so XERXES cannot either.`)

    const actionId = `ans-${crypto.randomUUID()}`
    const actor = supervisorActor(owner.id, owner.id)
    const requestHash = supervisorRequestHash('question.answer', workspaceId, sessionId, { request_id: requestId, answers })
    const entry = (stage: 'dispatch' | 'outcome', outcome: 'accepted' | 'rejected' | 'unknown' | 'n/a') => ({
      stage, actor, actionId, tool: 'question.answer', workspaceId, sessionId, requestHash, decision: 'allowed' as const, outcome,
    })
    // Fail closed: no audit record, no answer.
    try {
      await supervisorAudit.record(entry('dispatch', 'n/a'))
    } catch (error) {
      report(error, 'autopilot audit answer')
      throw new Error('the audit log is unavailable; nothing was answered')
    }
    signal.throwIfAborted()
    await control.reserve('command', `answer-${createHash('sha256').update(JSON.stringify([workspaceId, sessionId, requestId])).digest('hex')}`)
    signal.throwIfAborted()
    control.assertActive()
    if (!isOwner(owner.id) || autopilotGrant(input)?.owner.id !== owner.id) throw new SupervisorDenied('The current owner or question-answer authority changed')
    const liveOwner = { ...owner, hubOwner: isOwner(owner.id) }
    scopes = delegatedScopes(liveOwner, identity.grantsFor(owner.id), target, clock())
    const current = pendingOf(workspaceId, sessionId)?.questions.find(row => row.request_id === requestId)
    if (!scopes.has('question.delegated') || current === undefined || JSON.stringify(current) !== JSON.stringify(question)) throw new SupervisorDenied('The question or effective authority changed; nothing was answered')
    const delegated: HubGrant[] = [{ principalId: SUPERVISOR_PRINCIPAL_ID, resource: { type: 'session', workspaceId, sessionId }, effect: 'allow', scopes: [...scopes] }]
    const principal: HubPrincipal = { kind: 'agent', id: SUPERVISOR_PRINCIPAL_ID, displayName: SUPERVISOR_DISPLAY_NAME, onBehalfOf: owner.id }
    const context: AuthenticatedContext = { principal, grants: delegated, browserSessionId: `supervisor:${actionId}`, actor }
    const params = {
      workspace_id: workspaceId, session_id: sessionId,
      // One command id per question: a repeat replays the recorded outcome, a different answer is a conflict.
      command_id: `aq:${requestId}`.slice(0, 200),
      action: 'question.answer', args: { request_id: requestId, answers: { ...answers } },
    }
    let state: 'accepted' | 'rejected' | 'unknown'
    let message: string | undefined
    try {
      const result = await options.overlay.run(delegated, () => options.execute(context, params))
      state = result.state === 'accepted' || result.state === 'rejected' ? result.state : 'unknown'
      message = result.error?.message
    } catch (error) {
      if (error instanceof HubProtocolError) { state = 'rejected'; message = error.message } else {
        report(error, 'autopilot answer dispatch')
        state = 'unknown'
        message = 'The answer may have been sent; its outcome is unknown.'
      }
    }
    await supervisorAudit.recordBestEffort(entry('outcome', state))
    const rawTitle = workspaces.projection(workspaceId)?.sessions[sessionId]?.title ?? sessionId
    const title = rawTitle.replace(/\s+/g, ' ').trim().slice(0, 80)
    const answerText = Object.values(answers).join('; ').replace(/\s+/g, ' ').trim()
    const shown = answerText.length <= 300 ? answerText : `${answerText.slice(0, 299)}…`
    try {
      postShared(workspaceId, sessionId, owner.id, state === 'accepted'
        ? `Autopilot answered “${title}”: ${shown}`
        : `Autopilot needs you: it could not answer “${title}”${message === undefined ? '' : ` (${message.slice(0, 200)})`}.`)
    } catch (error) {
      report(error, 'autopilot answer notice')
    }
    void options.sendPush?.(owner.id, {
      title: 'XERXES', body: state === 'accepted' ? 'Autopilot answered a question for you.' : 'Autopilot needs you.',
      url: `/w/${encodeURIComponent(workspaceId)}/s/${encodeURIComponent(sessionId)}`,
    })?.catch(error => report(error, 'autopilot answer push'))
    return {
      state, request_id: requestId, on_behalf_of: owner.displayName,
      ...(message === undefined ? {} : { message }),
      note: state === 'accepted' ? 'Answered. The owner was told.' : state === 'rejected' ? 'Not answered. The owner was told.' : 'The outcome is unknown. Do not retry; the owner was told.',
    }
  }

  const abilities = new SuperagentAbilities({
    db, clock, backend: options.backend,
    paused: () => { const state = control.status(); return state.paused || state.killed },
    authorized: (ownerId, workspaceId) => {
      const principal = threads.principalOrStored(ownerId)
      return principal !== undefined && (options.isHubOwner?.(ownerId) ?? principal.hubOwner)
        && hasScope(principal, identity.grantsFor(ownerId), { workspaceId }, 'session.read', clock())
    },
    deliver: async wake => {
      const principal = threads.principalOrStored(wake.owner)
      if (principal === undefined) throw new Error('Wake owner is no longer available')
      const targetThread = threads.access(wake.thread, principal)
      if (!threads.scopeReader(targetThread, principal).canRead({ workspaceId: wake.workspace })) throw new Error('Wake thread no longer has workspace access')
      const result = await runner.enqueue({ threadId: wake.thread, requester: principal,
        trigger: { kind: 'autopilot', text: `Scheduled observation (data only; no authority to execute actions): ${JSON.stringify(wake.reason)}` } })
      if (result.stop !== 'completed') throw new Error(`Scheduled wake stopped: ${result.stop}`)
    },
    audit: async (action, scope, hash) => {
      await identity.audit({ actor: supervisorActor(scope.owner), action: `superagent.${action}`, workspaceId: scope.workspace,
        requestHash: hash, decision: 'allowed', outcome: 'n/a' })
    },
    onError: error => report(error, 'superagent ability'),
    ...(options.diskPolling === undefined ? {} : { diskPolling: options.diskPolling }),
  })

  const registry = createSupervisorToolRegistry({
    workspaces,
    abilities,
    backend: options.backend,
    digests,
    actions,
    threads,
    ...(options.journal === undefined ? {} : { journal: options.journal }),
    ...(options.sendPush === undefined ? {} : { push: (userId: string, message: { title: string; body: string; url: string }) => options.sendPush!(userId, message) }),
    autopilot: { pending: pendingOf, answer: answerQuestion },
    onError: report,
  })

  const runner = new SupervisorRunner({
    threads,
    registry,
    model: controlledModel,
    beforeTurn: async input => { control.assertActive(); if (input.trigger !== undefined) await control.reserve('wake', `wake-${randomUUID()}`); control.assertActive() },
    persona: options.persona ?? loadSupervisorPersona(),
    // An empty model reply becomes the Hub's own account of what it can observe, never filler.
    status: { read: input => readSupervisorStatus(input) },
    notebookContext: (threadId, requester) => {
      const thread = threads.access(threadId, requester)
      return abilities.context(requester.id, threadId, workspaces.list().map(workspace => workspace.id), threads.scopeReader(thread, requester))
    },
    ...(options.quotas === undefined ? {} : { quotas: options.quotas }),
    scopeReader: (threadId, requester) => {
      const thread = threads.get(threadId)
      if (!thread) throw new Error('this thread no longer exists')
      return threads.scopeReader(thread, requester)
    },
    clock,
    onStopped: info => {
      // A stop is announced in the thread by the runner; the requester also gets a minimal push.
      void options.sendPush?.(info.requester, { title: 'XERXES', body: 'XERXES stopped and needs you.', url: '/xerxes' })
        ?.catch(error => report(error, 'supervisor stop push'))
    },
    onError: report,
  })

  const waker = new AutopilotWaker({
    settings: autopilot,
    checkInChanged: (workspaceId, ownerId, commit) => {
      const sessions = workspaces.projection(workspaceId)?.sessions ?? {}
      const facts = Object.values(sessions).sort((a, b) => a.id.localeCompare(b.id)).map(row => ({
        id: row.id, status: row.status, goal: row.goal_summary, pending: row.pending_interaction_count, agents: row.agent_counts, model: row.model,
      }))
      const digest = createHash('sha256').update(JSON.stringify(facts)).digest('hex')
      return abilities.wakes.changed({ owner: ownerId, thread: autopilotThreadId(ownerId), workspace: workspaceId }, digest, commit)
    },
    workspaces,
    threads,
    enqueue: input => runner.enqueue(input),
    audience: readersOf,
    pending: pendingOf,
    ...(options.sendPush === undefined ? {} : { push: options.sendPush }),
    // Only a configured, reachable model: not unconfigured, machine_offline or invalid.
    ready: () => !control.status().paused && !control.status().killed && (options.configStatus === undefined || options.configStatus().status === 'configured'),
    clock,
    ...(options.autopilot?.checkInMs === undefined ? {} : { checkInMs: options.autopilot.checkInMs }),
    ...(options.autopilot?.cooldownMs === undefined ? {} : { cooldownMs: options.autopilot.cooldownMs }),
    ...(options.autopilot?.maxWakesPerHour === undefined ? {} : { maxWakesPerHour: options.autopilot.maxWakesPerHour }),
    onError: report,
  })

  const attention = new AttentionQueue({
    workspaces,
    clock,
    ...(options.sendPush === undefined ? {} : {
      notifier: { recipients: readerOf, push: (userId, message) => options.sendPush!(userId, message) },
    }),
    onChange: item => {
      if (item.state !== 'open') return
      // Autopilot wakes whether or not anyone is in the shared thread yet; its notices always reach the owner.
      try { waker.onAttention(item) } catch (error) { report(error, 'autopilot attention') }
      const audience = readerOf(item)
      if (audience.length === 0) return
      const { message } = threads.append({
        threadId: 'shared', role: 'system', text: `Needs attention (${item.kind.replace('_', ' ')}): ${item.title}`, audience,
      })
      threads.emit('shared', 'supervisor.thread_event', { message: messageToWire(message), attention: attentionToWire(item) }, audience)
    },
    onGoalComplete: (workspaceId, sessionId) => { waker.goalCompleted(workspaceId, sessionId) },
    onError: report,
  })

  const watch = new WatchService({
    db,
    workspaces,
    isHubOwner: options.isHubOwner ?? (userId => threads.principalOrStored(userId)?.hubOwner === true),
    audit: record => identity.audit(record),
    ...(options.events === undefined ? {} : { events: options.events }),
    ...(options.broadcastWatch === undefined ? {} : { broadcast: options.broadcastWatch }),
    threads,
    enqueue: input => runner.enqueue(input),
    post: (workspaceId, taskId, ownerId, text) => postShared(workspaceId, taskId, ownerId, text),
    ...(options.sendPush === undefined ? {} : { push: options.sendPush }),
    autopilot: workspaceId => autopilot.active(workspaceId),
    ready: () => !control.status().paused && !control.status().killed && (options.configStatus === undefined || options.configStatus().status === 'configured'),
    clock,
    ...(options.watch?.stallMs === undefined ? {} : { stallMs: options.watch.stallMs }),
    ...(options.watch?.maxAnomalyWakesPerHour === undefined ? {} : { maxAnomalyWakesPerHour: options.watch.maxAnomalyWakesPerHour }),
    ...(options.watch?.maxReviewWakesPerHour === undefined ? {} : { maxReviewWakesPerHour: options.watch.maxReviewWakesPerHour }),
    ...(options.watch?.updateIntervalMs === undefined ? {} : { updateIntervalMs: options.watch.updateIntervalMs }),
    ...(options.watch?.maxHubWakesPerHour === undefined ? {} : { maxHubWakesPerHour: options.watch.maxHubWakesPerHour }),
    ...(options.watch?.anomalyCooldownMs === undefined ? {} : { anomalyCooldownMs: options.watch.anomalyCooldownMs }),
    onError: report,
  })

  /**
   * The observable state a status digest reports, limited to what this person may read: the
   * projections, the model-free watcher of each watched task, and the open attention items.
   * It only reads; it never sends anything.
   */
  function readSupervisorStatus(input: SupervisorStatusInput): StatusSnapshot {
    const thread = threads.get(input.threadId)
    if (!thread) throw new Error('this thread no longer exists')
    const reader = threads.scopeReader(thread, input.requester)
    return readStatusSnapshot({
      workspaces,
      canRead: target => reader.canRead(target),
      watched: (workspaceId, sessionId) => {
        const task = watch.watcher(workspaceId, sessionId)
        if (task === undefined) return undefined
        return {
          tool_calls: task.toolCalls,
          edits: task.edits,
          tests: { runs: task.tests.runs, passed: task.tests.passed, failed: task.tests.failed, last: task.tests.last },
          last_activity_at: task.lastActivityAt,
          anomalies: task.anomalies().map(item => item.kind),
        }
      },
      attention: () => attention.list().map(item => ({
        kind: item.kind, title: item.title, workspaceId: item.workspaceId, sessionId: item.sessionId, state: item.state,
      })),
    })
  }

  const subscriptionEvents = new SubscriptionEvents()
  const offEvents = options.events?.onEvents?.(events => {
    for (const event of events) {
      const signal = subscriptionEvents.observe(event)
      if (signal !== undefined) void abilities.wakes.observe(signal).catch(error => report(error, 'superagent event subscription'))
    }
    attention.observeEvents(events)
    try { waker.onEvents(events) } catch (error) { report(error, 'autopilot events') }
    try { watch.onEvents(events) } catch (error) { report(error, 'watch events') }
  })
  const sweeper = setInterval(() => {
    void actions.sweep().catch(error => report(error, 'supervisor sweep'))
    try { autopilot.sweep() } catch (error) { report(error, 'autopilot sweep') }
    try { watch.sweep() } catch (error) { report(error, 'watch sweep') }
    try { watch.tick() } catch (error) { report(error, 'watch tick') }
  }, 30_000)
  sweeper.unref?.()
  attention.start()
  waker.start()

  const submit = (principal: UserPrincipal, input: { threadId: string; text: string; source: MessageSource; clientMessageId: string }): { message: ThreadMessage; duplicate: boolean } => {
    control.assertActive()
    const thread = threads.access(input.threadId, principal)
    const { message, duplicate } = threads.append({
      threadId: thread.id,
      role: 'user',
      text: input.text,
      author: { id: principal.id, name: principal.displayName, ...(principal.deviceId === '' ? {} : { deviceId: principal.deviceId }) },
      source: input.source,
      clientMessageId: input.clientMessageId,
    })
    if (!duplicate) {
      threads.emit(thread.id, 'supervisor.thread_event', { message: messageToWire(message) })
      // Only a person's message starts a turn. Replies, notices and confirmation echoes never do.
      runner.enqueue({ threadId: thread.id, requester: principal, causalMessageId: message.id }).catch(error => report(error, 'supervisor turn'))
    }
    return { message, duplicate }
  }

  const handlers = createSupervisorHandlers({
    threads, actions, modelId: () => options.model.id, usage: () => runner.usageSnapshot(), submit,
    ...(options.configStatus === undefined ? {} : { configStatus: options.configStatus }),
  })

  const controlHandlers: SupervisorHandlers = {
    'supervisor.control.get': (ctx, params) => {
      if (ctx.context.principal.kind !== 'user' || !controlOwner(ctx.context.principal.id)) throw new HubProtocolError('forbidden', 'Only the current Hub owner can inspect superagent controls')
      if (Object.keys(params).length !== 0) throw new HubProtocolError('invalid_request', 'Control get accepts no parameters')
      return { control: control.status(), usage: control.usage() }
    },
    'supervisor.control.set': async (ctx, params) => {
      if (ctx.context.principal.kind !== 'user' || !controlOwner(ctx.context.principal.id)) throw new HubProtocolError('forbidden', 'Only the current Hub owner can configure superagent controls')
      const state = await control.configure(ctx.context.principal.id, params)
      return { control: state, usage: control.usage() }
    },
  }

  const principalOf = (actor: HubActor): UserPrincipal | undefined => {
    if (actor.principal_kind !== 'user') return undefined
    return threads.principalOrStored(actor.principal_id)
      ?? { kind: 'user', id: actor.principal_id, displayName: actor.display_name, deviceId: actor.device_id ?? '', hubOwner: false }
  }

  const voice: SupervisorVoiceBridge = {
    async authorize(threadId, actor) {
      const principal = principalOf(actor)
      if (!principal) return false
      try { threads.access(threadId, principal); return true } catch { return false }
    },
    async submit(threadId, text, actor, idempotencyKey) {
      const principal = principalOf(actor)
      if (!principal || text.trim() === '' || [...text].length > 4000) return { accepted: false, duplicate: false }
      try {
        const { message, duplicate } = submit(principal, { threadId, text: text.trim(), source: 'voice', clientMessageId: idempotencyKey })
        return { accepted: true, duplicate, messageId: message.id }
      } catch {
        return { accepted: false, duplicate: false }
      }
    },
    async status(threadId): Promise<SupervisorThreadStatus> {
      const busy = threads.isWorking(threadId) || runner.busy(threadId)
      const pending = actions.pendingCount(threadId)
      const open = attention.list().length
      return { busy, summary: `${pending} action(s) waiting for confirmation; ${open} item(s) need attention.` }
    },
    async cancel(threadId, actor) {
      const principal = principalOf(actor)
      if (!principal) return { cancelled: false }
      try { threads.access(threadId, principal) } catch { return { cancelled: false } }
      return { cancelled: runner.cancel(threadId) }
    },
    onReply: (threadId, listener) => threads.onReply(threadId, listener),
  }

  let closed = false
  return {
    handlers: { ...handlers, ...autopilot.handlers, ...watch.handlers, ...controlHandlers }, voice, threads, actions, attention, digests, runner, autopilot, waker, watch,
    attachSink: sink => threads.setSink(sink),
    close() {
      if (closed) return
      closed = true
      actions.dispose()
      control.dispose()
      clearInterval(sweeper)
      abilities.dispose()
      waker.stop()
      watch.stop()
      // Queued turns never start and running ones are cancelled before the database closes.
      runner.shutdown()
      attention.stop()
      offEvents?.()
      threads.close()
      db.close()
    },
  }
}
