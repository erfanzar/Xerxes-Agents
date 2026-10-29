// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash } from 'node:crypto'
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
import { ActionService, SupervisorDenied, SupervisorGrantOverlay, type ActionExecutor, type ActionTargets, type AutopilotGrant } from './actions.js'
import { AttentionQueue, attentionToWire, type AttentionItem } from './attention.js'
import { SUPERVISOR_DISPLAY_NAME, SUPERVISOR_PRINCIPAL_ID, SupervisorAudit, supervisorActor, supervisorRequestHash } from './audit.js'
import {
  AutopilotActionBudget, AutopilotSettings, AutopilotWaker, autopilotOwnerPrincipal, displayOption, scanPendingInteractions, type PendingInteractions,
} from './autopilot.js'
import { DigestService, type DigestEventSource } from './digests.js'
import { createSupervisorHandlers, type SupervisorHandlers } from './handlers.js'
import {
  DEFAULT_SUPERVISOR_MAX_TOKENS_PER_WINDOW, SupervisorRunner, createModelDigestSummarizer, type SupervisorModelPort, type SupervisorQuotas,
} from './runner.js'
import { createSupervisorToolRegistry } from './tools.js'
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
  readonly workspaces: Pick<HubWorkspaceView, 'list' | 'projection' | 'health'>
  readonly backend: Pick<CommandBackend, 'call'>
  /** Journal for digests; observers feed attention. */
  readonly events?: DigestEventSource & { onEvents?(observer: (events: readonly WatchEvent[]) => void): () => void }
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

  const threads = new SupervisorThreads({ db, identity, clock, eligible, onError: report })
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
      // A wake turn counts only in the owner's own Autopilot thread.
      return input.threadId === autopilotThreadId(owner.id) ? { owner, wake: true } : undefined
    }
    if (input.othersInContext) return undefined
    return { owner, wake: false }
  }
  void autopilotOwnerPrincipal

  const digests = new DigestService({
    workspaces,
    ...(options.events === undefined ? {} : { events: options.events }),
    summarizer: createModelDigestSummarizer(options.model),
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
  }

  const actions = new ActionService({
    db,
    threads,
    identity,
    targets,
    execute: options.execute,
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
    const { workspaceId, sessionId, requestId } = input
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
      throw new Error('That is a permission request, not a question; decide it with answer_permission.')
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
    const scopes = delegatedScopes(owner, identity.grantsFor(owner.id), target, clock())
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

  /** The owner's XERXES: approve or reject one waiting PERMISSION request on the owner's behalf. */
  const answerPermission = async (input: {
    readonly threadId: string; readonly workspaceId: string; readonly sessionId: string; readonly requestId: string
    readonly response: 'approve' | 'approve_for_session' | 'reject'; readonly requester: UserPrincipal
    readonly reader: ScopeReader; readonly trigger: boolean; readonly othersInContext: boolean
  }): Promise<Record<string, unknown>> => {
    const { workspaceId, sessionId, requestId, response } = input
    const grant = autopilotGrant(input)
    if (grant === undefined) {
      throw new SupervisorDenied('XERXES answers a permission request only for the Hub owner, in a conversation with nobody else in it. Ask the owner to answer it.')
    }
    const target = { workspaceId, sessionId }
    // The same shared-scope check as questions: everyone this thread speaks for must be able to answer.
    if (!input.reader.canRead(target) || !input.reader.scopes(target).has('permission.delegated')) {
      throw new SupervisorDenied('permission.delegated is required on this task for everyone in this thread, so XERXES cannot answer it here.')
    }
    if (requestId.length === 0 || requestId.length > 200) throw new Error('request_id must be 1 to 200 characters')
    const pending = pendingOf(workspaceId, sessionId)
    if (pending === undefined) throw new Error('waiting permission requests cannot be read on this Hub, so XERXES will not answer blind')
    if (!pending.permissionIds.includes(requestId)) throw new Error('no such waiting permission request in this task; read pending_interactions again')
    const owner = grant.owner
    const scopes = delegatedScopes(owner, identity.grantsFor(owner.id), target, clock())
    if (!scopes.has('permission.delegated')) throw new SupervisorDenied(`${owner.displayName} cannot answer permission requests in this task, so XERXES cannot either.`)

    const actionId = `perm-${crypto.randomUUID()}`
    const actor = supervisorActor(owner.id, owner.id)
    const requestHash = supervisorRequestHash('permission.answer', workspaceId, sessionId, { request_id: requestId, response })
    const entry = (stage: 'dispatch' | 'outcome', outcome: 'accepted' | 'rejected' | 'unknown' | 'n/a') => ({
      stage, actor, actionId, tool: 'permission.answer', workspaceId, sessionId, requestHash, decision: 'allowed' as const, outcome,
    })
    try {
      await supervisorAudit.record(entry('dispatch', 'n/a'))
    } catch (error) {
      report(error, 'supervisor audit permission')
      throw new Error('the audit log is unavailable; nothing was answered')
    }
    const delegated: HubGrant[] = [{ principalId: SUPERVISOR_PRINCIPAL_ID, resource: { type: 'session', workspaceId, sessionId }, effect: 'allow', scopes: [...scopes] }]
    const principal: HubPrincipal = { kind: 'agent', id: SUPERVISOR_PRINCIPAL_ID, displayName: SUPERVISOR_DISPLAY_NAME, onBehalfOf: owner.id }
    const context: AuthenticatedContext = { principal, grants: delegated, browserSessionId: `supervisor:${actionId}`, actor }
    const params = {
      workspace_id: workspaceId, session_id: sessionId,
      command_id: `ap:${requestId}`.slice(0, 200),
      action: 'permission.answer', args: { request_id: requestId, response },
    }
    let state: 'accepted' | 'rejected' | 'unknown'
    let message: string | undefined
    try {
      const result = await options.overlay.run(delegated, () => options.execute(context, params))
      state = result.state === 'accepted' || result.state === 'rejected' ? result.state : 'unknown'
      message = result.error?.message
    } catch (error) {
      if (error instanceof HubProtocolError) { state = 'rejected'; message = error.message } else {
        report(error, 'supervisor permission dispatch')
        state = 'unknown'
        message = 'The answer may have been sent; its outcome is unknown.'
      }
    }
    await supervisorAudit.recordBestEffort(entry('outcome', state))
    const tool = pending.permissions?.find(row => row.request_id === requestId)?.tool ?? 'a tool'
    const rawTitle = workspaces.projection(workspaceId)?.sessions[sessionId]?.title ?? sessionId
    const title = rawTitle.replace(/\s+/g, ' ').trim().slice(0, 80)
    const verb = response === 'reject' ? 'rejected' : response === 'approve_for_session' ? 'approved for the session' : 'approved'
    try {
      postShared(workspaceId, sessionId, owner.id, state === 'accepted'
        ? `XERXES ${verb} ${tool} in “${title}” for ${owner.displayName}.`
        : `XERXES could not answer a permission request in “${title}”${message === undefined ? '' : ` (${message.slice(0, 200)})`}.`)
    } catch (error) {
      report(error, 'supervisor permission notice')
    }
    return {
      state, request_id: requestId, response, on_behalf_of: owner.displayName,
      ...(message === undefined ? {} : { message }),
      note: state === 'accepted' ? 'Answered.' : state === 'rejected' ? 'Not answered.' : 'The outcome is unknown. Do not retry.',
    }
  }

  const registry = createSupervisorToolRegistry({
    workspaces,
    backend: options.backend,
    digests,
    actions,
    threads,
    ...(options.sendPush === undefined ? {} : { push: (userId: string, message: { title: string; body: string; url: string }) => options.sendPush!(userId, message) }),
    autopilot: { pending: pendingOf, answer: answerQuestion, answerPermission },
    onError: report,
  })

  const runner = new SupervisorRunner({
    threads,
    registry,
    model: options.model,
    persona: options.persona ?? loadSupervisorPersona(),
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
    workspaces,
    threads,
    enqueue: input => runner.enqueue(input),
    audience: readersOf,
    pending: pendingOf,
    ...(options.sendPush === undefined ? {} : { push: options.sendPush }),
    // Only a configured, reachable model: not unconfigured, machine_offline or invalid.
    ready: () => options.configStatus === undefined || options.configStatus().status === 'configured',
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
    ready: () => options.configStatus === undefined || options.configStatus().status === 'configured',
    clock,
    ...(options.watch?.stallMs === undefined ? {} : { stallMs: options.watch.stallMs }),
    ...(options.watch?.maxAnomalyWakesPerHour === undefined ? {} : { maxAnomalyWakesPerHour: options.watch.maxAnomalyWakesPerHour }),
    ...(options.watch?.maxReviewWakesPerHour === undefined ? {} : { maxReviewWakesPerHour: options.watch.maxReviewWakesPerHour }),
    ...(options.watch?.updateIntervalMs === undefined ? {} : { updateIntervalMs: options.watch.updateIntervalMs }),
    ...(options.watch?.maxHubWakesPerHour === undefined ? {} : { maxHubWakesPerHour: options.watch.maxHubWakesPerHour }),
    ...(options.watch?.anomalyCooldownMs === undefined ? {} : { anomalyCooldownMs: options.watch.anomalyCooldownMs }),
    onError: report,
  })

  const offEvents = options.events?.onEvents?.(events => {
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
    handlers: { ...handlers, ...autopilot.handlers, ...watch.handlers }, voice, threads, actions, attention, digests, runner, autopilot, waker, watch,
    attachSink: sink => threads.setSink(sink),
    close() {
      if (closed) return
      closed = true
      clearInterval(sweeper)
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
