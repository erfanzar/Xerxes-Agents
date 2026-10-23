// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { Database } from 'bun:sqlite'

import type { AuthenticatedContext } from '../src/hub/identityPort.js'
import { emptyProjection, replaceSessionSummaries, type WorkspaceProjection } from '../src/hub/projections.js'
import type { HubCommandResult } from '../src/hub/protocol.js'
import type { PoolCallResult, PoolHealth } from '../src/hub/runtimePool.js'
import { ActionService, SupervisorGrantOverlay, type ActionExecutor, type ActionRecord, type ActionTargets, type ActionTool } from '../src/hub/supervisor/actions.js'
import { DigestService } from '../src/hub/supervisor/digests.js'
import {
  SupervisorRunner, type SupervisorModelPort, type SupervisorModelRequest, type SupervisorModelResponse, type SupervisorQuotas,
} from '../src/hub/supervisor/runner.js'
import { createSupervisorToolRegistry } from '../src/hub/supervisor/tools.js'
import { SupervisorThreads, type SupervisorEventType, type UserPrincipal } from '../src/hub/supervisor/threads.js'
import type { JsonObject, ToolCall } from '../src/types/toolCalls.js'
import { FakeIdentity, registerCleanup, standardUsers } from './hubHarness.js'

export const T0 = 1_700_000_000_000

export interface SinkEvent { readonly to: readonly string[]; readonly type: SupervisorEventType; readonly payload: Readonly<Record<string, unknown>> }
export interface ExecCall { readonly context: AuthenticatedContext; readonly params: Record<string, unknown>; readonly grants: ReturnType<SupervisorGrantOverlay['current']> }

export function toolCall(name: string, args: JsonObject = {}, id?: string): ToolCall {
  return { id: id ?? `call-${Math.random().toString(36).slice(2, 10)}`, type: 'function', function: { name, arguments: args } }
}

export function scriptedModel(responses: readonly SupervisorModelResponse[], options: { delayMs?: number } = {}) {
  const requests: SupervisorModelRequest[] = []
  let index = 0
  let active = 0
  let maxActive = 0
  const model: SupervisorModelPort = {
    id: 'cheap-model',
    reportsCost: false,
    async complete(request) {
      requests.push(request)
      active += 1
      maxActive = Math.max(maxActive, active)
      try {
        if (options.delayMs) await Bun.sleep(options.delayMs)
        return responses[Math.min(index++, responses.length - 1)]!
      } finally {
        active -= 1
      }
    },
  }
  return { model, requests, maxActive: () => maxActive }
}

export function makeWorld(options: { modelSetSessionOnly?: boolean; expiryMs?: number } = {}) {
  const identity = standardUsers(new FakeIdentity())
  const db = new Database(':memory:')
  registerCleanup(() => db.close())
  const time = { now: T0 }
  const clock = (): number => time.now
  const threads = new SupervisorThreads({ db, identity, clock })
  threads.ensureShared()
  const events: SinkEvent[] = []
  threads.setSink({ send: (to, type, payload) => { events.push({ to, type, payload }) } })

  let tick = 0
  const state = { instance: 'i1', online: true, projection: emptyProjection('w1') as WorkspaceProjection }
  const setSessions = (rows: readonly Record<string, unknown>[]): void => {
    tick += 1
    state.projection = replaceSessionSummaries(state.projection, rows, new Date(T0 + tick * 1000).toISOString())
  }
  setSessions([{ id: 's1', title: 'First task', status: 'idle' }, { id: 's2', title: 'Second task', status: 'working' }])

  const workspaces = {
    list: () => [{ id: 'w1', label: 'One' }],
    projection: (id: string): WorkspaceProjection | undefined => (id === 'w1' ? state.projection : undefined),
    health: (id: string): PoolHealth | undefined =>
      id === 'w1' ? { state: state.online ? 'online' : 'offline', degraded: false, instanceId: state.instance } : undefined,
  }
  const targets: ActionTargets = {
    instanceId: id => (id === 'w1' ? state.instance : undefined),
    revision: (workspaceId, sessionId) => {
      const row = workspaceId === 'w1' ? state.projection.sessions[sessionId] : undefined
      return row === undefined ? undefined : `${row.status}|${row.goal_summary}|${row.model}`
    },
    hasGoal: (_workspaceId, sessionId) => {
      const row = state.projection.sessions[sessionId]
      return row === undefined ? undefined : row.goal_summary !== ''
    },
    online: () => state.online,
    label: id => (id === 'w1' ? 'One' : undefined),
    title: (_workspaceId, sessionId) => state.projection.sessions[sessionId]?.title,
  }

  const overlay = new SupervisorGrantOverlay()
  const calls: ExecCall[] = []
  const behavior: { run: (params: Record<string, unknown>) => HubCommandResult } = {
    run: params => ({ ok: true, command_id: String(params.command_id), state: 'accepted', result: { cancelled: true, delivered: true, session_id: 'new-1' } }),
  }
  const execute: ActionExecutor = async (context, params) => {
    calls.push({ context, params, grants: overlay.current() })
    return behavior.run(params)
  }
  const actions = new ActionService({
    db, threads, identity, targets, execute, overlay,
    access: (threadId, principal) => { threads.access(threadId, principal) },
    clock,
    ...(options.expiryMs === undefined ? {} : { expiryMs: options.expiryMs }),
    ...(options.modelSetSessionOnly === undefined ? {} : { modelSetSessionOnly: options.modelSetSessionOnly }),
  })

  const user = (name: string): UserPrincipal => {
    const principal = identity.users.get(name)?.principal
    if (principal?.kind !== 'user') throw new Error(`${name} is not a user`)
    return principal
  }
  const reader = (name: string, threadId = 'shared') => {
    const principal = user(name)
    threads.access(threadId, principal)
    return threads.scopeReader(threads.get(threadId)!, principal)
  }
  const propose = async (tool: ActionTool, args: JsonObject, who = 'alice', threadId = 'shared'): Promise<ActionRecord> =>
    actions.createIntent({ threadId, tool, args, requester: user(who), reader: reader(who, threadId) })
  const confirm = (who: string, id: string, decision: 'confirm' | 'cancel' = 'confirm') =>
    actions.respond(identity.context(who), { confirmation_id: id, decision, response_id: `r-${who}` })

  const backendCalls: { workspaceId: string; method: string; params: Record<string, unknown> }[] = []
  const backendBehavior: { run: (method: string, params: Record<string, unknown>) => PoolCallResult } = {
    run: method => (method === 'session.active_list'
      ? { state: 'accepted', result: { ok: true, sessions: [{ id: 's1', key: 'k1' }, { id: 's2', key: 'k2' }] } }
      : { state: 'accepted', result: { ok: true } }),
  }
  const backend = {
    async call(workspaceId: string, method: string, params: Readonly<Record<string, unknown>> = {}): Promise<PoolCallResult> {
      backendCalls.push({ workspaceId, method, params: { ...params } })
      return backendBehavior.run(method, { ...params })
    },
  }
  const digests = new DigestService({ workspaces, clock })
  const registry = createSupervisorToolRegistry({ workspaces, backend, digests, actions, threads })

  return {
    identity, db, time, clock, threads, events, state, setSessions, workspaces, targets, overlay, calls, behavior, actions,
    user, reader, propose, confirm, backend, backendCalls, backendBehavior, digests, registry,
  }
}
export type World = ReturnType<typeof makeWorld>

export function makeRunner(world: World, model: SupervisorModelPort, options: { quotas?: Partial<SupervisorQuotas>; onStopped?: (info: unknown) => void } = {}) {
  return new SupervisorRunner({
    threads: world.threads,
    registry: world.registry,
    model,
    persona: 'You are XERXES (test persona).',
    scopeReader: (threadId, requester) => world.threads.scopeReader(world.threads.get(threadId)!, requester),
    clock: world.clock,
    ...(options.quotas === undefined ? {} : { quotas: options.quotas }),
    ...(options.onStopped === undefined ? {} : { onStopped: options.onStopped }),
  })
}

export const steerArgs = (text = 'please focus on the tests'): JsonObject => ({ workspace_id: 'w1', task_id: 's1', text })
