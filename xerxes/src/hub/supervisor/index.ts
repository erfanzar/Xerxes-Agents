// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash } from 'node:crypto'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { Database } from 'bun:sqlite'

import { canSeeWorkspace, grantedSessionIds, hasScope } from '../../security/principal.js'
import type { CommandBackend } from '../commandRouter.js'
import type { HubIdentityPort } from '../identityPort.js'
import type { HubActor } from '../protocol.js'
import type { HubWorkspaceView } from '../server.js'
import type { SupervisorThreadStatus, SupervisorVoiceBridge } from '../voice/supervisorBridge.js'
import { ActionService, SupervisorGrantOverlay, type ActionExecutor, type ActionTargets } from './actions.js'
import { AttentionQueue, attentionToWire, type AttentionItem } from './attention.js'
import { DigestService, type DigestEventSource } from './digests.js'
import { createSupervisorHandlers, type SupervisorHandlers } from './handlers.js'
import { SupervisorRunner, createModelDigestSummarizer, type SupervisorModelPort, type SupervisorQuotas } from './runner.js'
import { createSupervisorToolRegistry } from './tools.js'
import {
  SupervisorThreads, messageToWire, type MessageSource, type SupervisorEventSink, type ThreadMessage, type UserPrincipal,
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
export type { SupervisorModelPort, SupervisorQuotas } from './runner.js'

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
  readonly events?: DigestEventSource & { onEvents?(observer: (events: readonly { workspace_id: string; session_id?: string; type: string; payload: Readonly<Record<string, unknown>> }[]) => void): () => void }
  readonly model: SupervisorModelPort
  /** Whether XERXES has a model to run on; `unconfigured` makes `supervisor.message.submit` refuse. */
  readonly configStatus?: () => { readonly status: string; readonly message?: string }
  readonly persona?: string
  readonly quotas?: Partial<SupervisorQuotas>
  readonly sendPush?: PushLike
  /** Only true when the runtime guarantees `model.set` never touches defaults. Off by default. */
  readonly modelSetSessionOnly?: boolean
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
    onError: report,
  })

  const registry = createSupervisorToolRegistry({
    workspaces,
    backend: options.backend,
    digests,
    actions,
    threads,
    ...(options.sendPush === undefined ? {} : { push: (userId: string, message: { title: string; body: string; url: string }) => options.sendPush!(userId, message) }),
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

  const readerOf = (item: AttentionItem): string[] => {
    const shared = threads.get('shared')
    if (!shared) return []
    return threads.participants(shared.id).flatMap(person => {
      const principal = threads.principalOrStored(person.id)
      if (!principal) return []
      const target = { workspaceId: item.workspaceId, sessionId: item.sessionId }
      return hasScope(principal, identity.grantsFor(person.id), target, 'session.read', clock()) ? [person.id] : []
    })
  }

  const attention = new AttentionQueue({
    workspaces,
    clock,
    ...(options.sendPush === undefined ? {} : {
      notifier: { recipients: readerOf, push: (userId, message) => options.sendPush!(userId, message) },
    }),
    onChange: item => {
      if (item.state !== 'open') return
      const audience = readerOf(item)
      if (audience.length === 0) return
      const { message } = threads.append({
        threadId: 'shared', role: 'system', text: `Needs attention (${item.kind.replace('_', ' ')}): ${item.title}`, audience,
      })
      threads.emit('shared', 'supervisor.thread_event', { message: messageToWire(message), attention: attentionToWire(item) }, audience)
    },
    onError: report,
  })

  const offEvents = options.events?.onEvents?.(events => attention.observeEvents(events))
  const sweeper = setInterval(() => { void actions.sweep().catch(error => report(error, 'supervisor sweep')) }, 30_000)
  sweeper.unref?.()
  attention.start()

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
      runner.enqueue({ threadId: thread.id, requester: principal }).catch(error => report(error, 'supervisor turn'))
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

  return {
    handlers, voice, threads, actions, attention, digests, runner,
    attachSink: sink => threads.setSink(sink),
    close() {
      clearInterval(sweeper)
      attention.stop()
      offEvents?.()
      threads.close()
      db.close()
    },
  }
}
