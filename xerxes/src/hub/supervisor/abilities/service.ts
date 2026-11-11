// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { Database } from 'bun:sqlite'
import type { JsonObject } from '../../../types/toolCalls.js'
import type { CommandBackend } from '../../commandRouter.js'
import { supervisorRequestHash } from '../audit.js'
import type { SupervisorTurnState } from '../tools.js'
import { ABILITY_SPECS, validateAbilityInput } from './catalog.js'
import { DiskThresholdMonitor } from './disk.js'
import { SupervisorNotebook, type NotebookScope } from './memory.js'
import { SupervisorWakes, type Wake, type WakeKind } from './wakes.js'

interface AbilityOptions {
  readonly db: Database
  readonly clock: () => number
  readonly backend: Pick<CommandBackend, 'call'>
  readonly authorized: (owner: string, workspace: string) => boolean
  readonly deliver: (wake: Wake) => Promise<void>
  readonly audit: (action: string, scope: NotebookScope, requestHash: string) => Promise<void>
  readonly paused?: () => boolean
  readonly diskPolling?: { readonly intervalMs?: number }
  readonly onError?: (error: unknown) => void
}

/** Composition only. Schemas, persistence, scheduling and native workspace reads have separate owners. */
export class SuperagentAbilities {
  readonly notebook: SupervisorNotebook
  readonly wakes: SupervisorWakes
  private readonly disk: DiskThresholdMonitor | undefined
  constructor(private readonly options: AbilityOptions) {
    this.notebook = new SupervisorNotebook(options.db, options.clock)
    this.wakes = new SupervisorWakes(options.db, { clock: options.clock, paused: options.paused ?? (() => false),
      authorized: scope => options.authorized(scope.owner, scope.workspace), deliver: options.deliver,
      ...(options.onError === undefined ? {} : { onError: options.onError }) })
    this.disk = options.diskPolling === undefined ? undefined : new DiskThresholdMonitor({
      ...options.diskPolling, paused: options.paused ?? (() => false),
      workspaces: () => {
        const rows = options.db.query<NotebookScope, []>("SELECT owner,thread,workspace FROM hub_superagent_subscriptions WHERE kind='disk_threshold' LIMIT 1001").all()
        if (rows.length > 1000) throw new Error('Disk subscription inventory exceeds budget')
        return rows.filter(scope => options.authorized(scope.owner, scope.workspace)).map(scope => scope.workspace)
      },
      read: async (workspace, signal) => {
        const result = await options.backend.call(workspace, 'superagent.observe', { ability: 'machine.health', view: 'disk' }, { signal, timeoutMs: 5000 })
        if (result.state !== 'accepted') throw new Error(result.error.message)
        return result.result
      },
      observe: event => this.wakes.observe(event),
      ...(options.onError === undefined ? {} : { onError: options.onError }),
    })
  }

  async execute(name: string, input: JsonObject, turn: SupervisorTurnState, signal?: AbortSignal): Promise<unknown> {
    const checked = validateAbilityInput(name, input)
    if (!checked.ok) throw new Error(checked.error)
    const args = checked.args
    const workspace = typeof args.workspace_id === 'string' ? args.workspace_id : ''
    if (!turn.reader.canRead({ workspaceId: workspace })) throw new Error('not found, or you cannot read this workspace')
    const spec = ABILITY_SPECS.find(item => item.name === name || item.providerName === name)
    if (spec === undefined) throw new Error('Unknown ability')
    if (spec.tier === 'D') throw new Error('Risky actions require the exact owner confirmation path')
    const scope = { owner: turn.requester.id, thread: turn.threadId, workspace }
    if (this.options.paused?.()) throw new Error('Superagent is globally paused')
    if (spec.tier === 'B' && !this.options.authorized(scope.owner, workspace)) throw new Error('Only the current authorized Hub owner can use persistent abilities')
    const hash = supervisorRequestHash(spec.name, workspace, undefined, args)
    await this.options.audit(`intent:${spec.name}`, scope, hash)
    signal?.throwIfAborted()
    let result: unknown
    const text = (key: string, fallback = ''): string => typeof args[key] === 'string' ? args[key] : fallback
    const number = (key: string, fallback: number): number => typeof args[key] === 'number' ? args[key] : fallback
    switch (spec.name) {
      case 'notebook.list': result = { entries: this.notebook.list(scope), untrusted: true }; break
      case 'notebook.put': {
        const kind = text('kind')
        if (kind !== 'plan' && kind !== 'promise' && kind !== 'open_item') throw new Error('Invalid notebook kind')
        result = this.notebook.put(scope, text('key'), kind, text('text')); break
      }
      case 'notebook.remove': result = { removed: this.notebook.remove(scope, text('key')) }; break
      case 'wake.schedule': result = this.wakes.schedule(scope, {
        ...(args.at === undefined ? {} : { at: Date.parse(text('at')) }),
        ...(args.after_seconds === undefined ? {} : { after: number('after_seconds', 0) * 1000 }), reason: text('reason'),
      }); break
      case 'wake.list': result = { wakes: this.wakes.list(scope) }; break
      case 'wake.cancel': result = { cancelled: this.wakes.cancel(scope, text('id')) }; break
      case 'event.subscribe': result = this.wakes.subscribe(scope, { kind: wakeKind(text('kind')),
        ...(args.threshold === undefined ? {} : { threshold: number('threshold', 0) }) }); break
      case 'event.unsubscribe': result = { removed: this.wakes.unsubscribe(scope, wakeKind(text('kind'))) }; break
      default: {
        if (spec.name === 'ui.screenshot' && turn.captures === undefined) throw new Error('A configured vision route and capture disclosure are required')
        const deadline = AbortSignal.timeout(10_000)
        const bounded = signal === undefined ? deadline : AbortSignal.any([signal, deadline])
        // Model cannot choose project_dir or session_key: the pool pins the registered workspace directory.
        const { workspace_id: _workspaceId, ...params } = args
        const outcome = await this.options.backend.call(workspace, 'superagent.observe', { ability: spec.name, ...params }, { signal: bounded, timeoutMs: 10_000 })
        if (outcome.state !== 'accepted') throw new Error(outcome.error.message)
        if (spec.name === 'ui.screenshot') {
          const captures = turn.captures
          if (captures === undefined || typeof outcome.result.view_id !== 'string') throw new Error('Invalid capture response')
          result = await captures.sink.add(captures.source(workspace, outcome.result.view_id), outcome.result, bounded)
        } else result = { ...outcome.result, untrusted: true }
      }
    }
    // Mutation result is established already; audit failure here is observable, not an invitation to replay.
    try { await this.options.audit(`outcome:${spec.name}`, scope, hash) } catch (error) { this.options.onError?.(error) }
    return result
  }

  context(owner: string, thread: string, workspaces: readonly string[], reader: SupervisorTurnState['reader']): string {
    return workspaces.filter(workspace => reader.canRead({ workspaceId: workspace }) && this.options.authorized(owner, workspace)).slice(0, 8)
      .map(workspace => this.notebook.context({ owner, thread, workspace })).join('\n').slice(0, 32 * 1024)
  }
  dispose(): void { this.disk?.dispose(); this.wakes.dispose() }
}

function wakeKind(value: string): WakeKind {
  if (value === 'task_idle' || value === 'goal_blocked' || value === 'question_waiting' || value === 'tests_failing' || value === 'disk_threshold') return value
  throw new Error('Invalid wake kind')
}
