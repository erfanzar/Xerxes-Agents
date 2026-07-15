// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ValidationError } from '../core/errors.js'

export const MAX_AGENT_TITLE_LENGTH = 48

export type SpawnedAgentStatus = 'cancelled' | 'closed' | 'completed' | 'error' | 'idle' | 'interrupted' | 'running'

export interface SpawnedAgentDescriptor {
  readonly id: string
  readonly model?: string
  readonly name?: string
  readonly systemPrompt?: string
  readonly title?: string
}

export interface SubagentRunRequest {
  readonly agent: SpawnedAgentDescriptor
  readonly handleId: string
  readonly input: string
  readonly permissionMode: string
  readonly promptProfile: string
  readonly sourceAgentId?: string
}

export interface SubagentRunResponse {
  readonly content: string
}

/** Port through which the runtime executes one isolated spawned-agent turn. */
export type SubagentRunner = (
  request: SubagentRunRequest,
  signal: AbortSignal,
) => Promise<SubagentRunResponse | string>

export interface SpawnedAgentSnapshot {
  readonly agentId: string
  readonly apiCalls?: number
  readonly closed: boolean
  readonly completionSummary?: string
  readonly createdAt: string
  readonly creatorAgentId?: string
  readonly error?: string
  readonly filesRead?: readonly string[]
  readonly filesWritten?: readonly string[]
  readonly id: string
  readonly inputTokens?: number
  readonly lastInput?: string
  readonly lastOutput?: string
  readonly model?: string
  readonly name: string
  readonly outputTokens?: number
  readonly parentAgentId?: string
  readonly promptProfile: string
  readonly queueSize: number
  readonly queuedPreview?: string
  readonly reasoningTokens?: number
  readonly rules?: readonly string[]
  readonly sourceAgentId?: string
  readonly status: SpawnedAgentStatus
  readonly title: string
  readonly toolCalls?: number
  readonly toolsets?: readonly string[]
  readonly updatedAt: string
}

export interface SpawnedAgentManagerOptions {
  readonly defaultPermissionMode?: string
  readonly defaultPromptProfile?: string
  readonly idFactory?: () => string
  readonly now?: () => Date
  readonly runner: SubagentRunner
}

export interface SpawnAgentOptions {
  readonly agent?: SpawnedAgentDescriptor
  readonly agentId?: string
  readonly creatorAgentId?: string
  readonly message?: string
  readonly nickname?: string
  readonly parentModel?: string
  readonly parentAgentId?: string
  readonly permissionMode?: string
  readonly promptProfile?: string
  readonly rules?: readonly string[]
  readonly sourceAgentId?: string
  readonly taskDescription?: string
  readonly title?: string
  readonly toolsets?: readonly string[]
}

export interface SendAgentInputOptions {
  readonly interrupt?: boolean
  readonly message?: string
  readonly taskDescription?: string
}

interface ActiveRun {
  readonly controller: AbortController
  readonly promise: Promise<void>
  readonly token: number
}

interface SpawnedAgentHandle {
  active: ActiveRun | undefined
  readonly agent: SpawnedAgentDescriptor
  closed: boolean
  readonly creatorAgentId: string | undefined
  readonly createdAt: string
  error: string | undefined
  readonly id: string
  lastInput: string | undefined
  lastOutput: string | undefined
  readonly name: string
  readonly parentAgentId: string | undefined
  readonly permissionMode: string
  readonly promptProfile: string
  readonly queue: string[]
  readonly rules: readonly string[]
  readonly sourceAgentId: string | undefined
  status: SpawnedAgentStatus
  readonly title: string
  token: number
  readonly toolsets: readonly string[]
  updatedAt: string
}

/** Structural manager surface consumed by agent-facing compatibility tools. */
export interface SpawnedAgentManagerPort {
  close(handleId: string): SpawnedAgentSnapshot & { readonly previousStatus: SpawnedAgentStatus }
  listHandles(): SpawnedAgentSnapshot[]
  resume(handleId: string): SpawnedAgentSnapshot
  sendInput(handleId: string | undefined, options: SendAgentInputOptions): Promise<SpawnedAgentSnapshot>
  spawn(options?: SpawnAgentOptions): Promise<SpawnedAgentSnapshot>
  wait(targets: readonly string[], timeoutMs?: number): Promise<{
    readonly completed: readonly SpawnedAgentSnapshot[]
    readonly pending: readonly SpawnedAgentSnapshot[]
  }>
}

/**
 * Tracks spawned-agent work independently of the parent turn.
 *
 * The manager deliberately owns cancellation and queued input rather than
 * requiring a particular orchestrator implementation. A daemon, ACP server,
 * or test runner can supply the same `SubagentRunner` port.
 */
export class SpawnedAgentManager implements SpawnedAgentManagerPort {
  private readonly defaultPermissionMode: string
  private readonly defaultPromptProfile: string
  private readonly handles = new Map<string, SpawnedAgentHandle>()
  private readonly idFactory: () => string
  private readonly now: () => Date

  constructor(private readonly options: SpawnedAgentManagerOptions) {
    this.defaultPromptProfile = options.defaultPromptProfile ?? 'minimal'
    this.defaultPermissionMode = options.defaultPermissionMode ?? 'accept-all'
    this.idFactory = options.idFactory ?? (() => `subagent_${crypto.randomUUID().replaceAll('-', '').slice(0, 10)}`)
    this.now = options.now ?? (() => new Date())
  }

  listHandles(): SpawnedAgentSnapshot[] {
    return [...this.handles.values()]
      .sort((left, right) => left.createdAt.localeCompare(right.createdAt))
      .map(handle => this.snapshot(handle))
  }

  async spawn(options: SpawnAgentOptions = {}): Promise<SpawnedAgentSnapshot> {
    const id = options.nickname?.trim() || this.idFactory()
    if (this.handles.has(id)) throw new ValidationError('nickname', 'already identifies a spawned agent', id)
    const sourceAgentId = options.sourceAgentId ?? options.agentId
    const creatorAgentId = nonempty(options.creatorAgentId) ?? nonempty(options.agentId)
    const parentAgentId = nonempty(options.parentAgentId) ?? creatorAgentId
    const title = options.title === undefined
      ? deriveAgentTitle(options.taskDescription ?? options.message ?? options.agent?.name ?? options.nickname ?? id)
      : normalizeAgentTitle(options.title)
    const agent = Object.freeze({
      id,
      ...(options.agent?.model === undefined ? {} : { model: options.agent.model }),
      name: options.nickname?.trim() || options.agent?.name || id,
      title,
      ...(options.agent?.systemPrompt === undefined ? {} : { systemPrompt: options.agent.systemPrompt }),
    })
    const createdAt = this.now().toISOString()
    const handle: SpawnedAgentHandle = {
      id,
      agent,
      name: agent.name ?? id,
      title,
      sourceAgentId,
      creatorAgentId,
      parentAgentId,
      status: 'idle',
      createdAt,
      updatedAt: createdAt,
      promptProfile: options.promptProfile?.trim() || this.defaultPromptProfile,
      permissionMode: options.permissionMode?.trim() || this.defaultPermissionMode,
      rules: normalizeLabels(options.rules),
      toolsets: normalizeLabels(options.toolsets),
      lastInput: undefined,
      lastOutput: undefined,
      error: undefined,
      queue: [],
      active: undefined,
      closed: false,
      token: 0,
    }
    this.handles.set(id, handle)
    const message = options.message ?? options.taskDescription
    if (message?.trim()) await this.sendInput(id, { message })
    return this.snapshot(handle)
  }

  async sendInput(handleId: string | undefined, options: SendAgentInputOptions): Promise<SpawnedAgentSnapshot> {
    const id = this.resolveHandleId(handleId)
    const handle = this.requireHandle(id)
    if (handle.closed) throw new ValidationError('handle_id', 'spawned agent is closed', id)
    const input = (options.message ?? options.taskDescription)?.trim()
    if (!input) throw new ValidationError('message', 'spawned agent input is required', input)

    if (handle.active !== undefined) {
      if (!options.interrupt) {
        handle.queue.push(input)
        handle.updatedAt = this.now().toISOString()
        return this.snapshot(handle)
      }
      handle.active.controller.abort(new Error('Interrupted by parent agent'))
      handle.status = 'interrupted'
      this.start(handle, input)
      return this.snapshot(handle)
    }
    this.start(handle, input)
    return this.snapshot(handle)
  }

  async wait(targets: readonly string[], timeoutMs = 30_000): Promise<{
    readonly completed: readonly SpawnedAgentSnapshot[]
    readonly pending: readonly SpawnedAgentSnapshot[]
  }> {
    const handles = targets.map(target => this.requireHandle(target))
    const deadline = Date.now() + requireTimeout(timeoutMs)
    await Promise.all(handles.map(handle => this.waitForIdle(handle, deadline)))
    const completed: SpawnedAgentSnapshot[] = []
    const pending: SpawnedAgentSnapshot[] = []
    for (const handle of handles) {
      if (handle.active === undefined) completed.push(this.snapshot(handle))
      else pending.push(this.snapshot(handle))
    }
    return Object.freeze({ completed: Object.freeze(completed), pending: Object.freeze(pending) })
  }

  resume(handleId: string): SpawnedAgentSnapshot {
    const handle = this.requireHandle(handleId)
    handle.closed = false
    if (handle.status === 'closed') handle.status = 'idle'
    handle.updatedAt = this.now().toISOString()
    return this.snapshot(handle)
  }

  close(handleId: string): SpawnedAgentSnapshot & { readonly previousStatus: SpawnedAgentStatus } {
    const handle = this.requireHandle(handleId)
    const previousStatus = handle.status
    handle.token += 1
    handle.active?.controller.abort(new Error('Spawned agent closed'))
    handle.active = undefined
    handle.queue.length = 0
    handle.closed = true
    handle.status = 'closed'
    handle.updatedAt = this.now().toISOString()
    return Object.freeze({ ...this.snapshot(handle), previousStatus })
  }

  private start(handle: SpawnedAgentHandle, input: string): void {
    const token = handle.token + 1
    handle.token = token
    const controller = new AbortController()
    handle.status = 'running'
    handle.lastInput = input
    handle.error = undefined
    handle.updatedAt = this.now().toISOString()
    const promise = this.run(handle, input, token, controller)
    handle.active = { controller, promise, token }
  }

  private async run(handle: SpawnedAgentHandle, input: string, token: number, controller: AbortController): Promise<void> {
    try {
      const response = await this.options.runner({
        handleId: handle.id,
        agent: handle.agent,
        input,
        promptProfile: handle.promptProfile,
        permissionMode: handle.permissionMode,
        ...(handle.sourceAgentId === undefined ? {} : { sourceAgentId: handle.sourceAgentId }),
      }, controller.signal)
      if (!this.isCurrent(handle, token)) return
      handle.lastOutput = typeof response === 'string' ? response : response.content
      handle.status = 'completed'
      handle.error = undefined
    } catch (error) {
      if (!this.isCurrent(handle, token)) return
      if (controller.signal.aborted) {
        handle.status = 'cancelled'
        handle.error = 'cancelled'
      } else {
        handle.status = 'error'
        handle.error = errorMessage(error)
      }
    } finally {
      if (!this.isCurrent(handle, token)) return
      handle.active = undefined
      handle.updatedAt = this.now().toISOString()
      const next = handle.closed ? undefined : handle.queue.shift()
      if (next !== undefined) this.start(handle, next)
    }
  }

  private async waitForIdle(handle: SpawnedAgentHandle, deadline: number): Promise<void> {
    while (handle.active !== undefined && Date.now() < deadline) {
      const active = handle.active
      const remaining = Math.max(deadline - Date.now(), 0)
      await waitForCompletion(active.promise, remaining)
      if (remaining === 0) break
    }
  }

  private snapshot(handle: SpawnedAgentHandle): SpawnedAgentSnapshot {
    return Object.freeze({
      id: handle.id,
      name: handle.name,
      title: handle.title,
      agentId: handle.agent.id,
      ...(handle.sourceAgentId === undefined ? {} : { sourceAgentId: handle.sourceAgentId }),
      ...(handle.creatorAgentId === undefined ? {} : { creatorAgentId: handle.creatorAgentId }),
      ...(handle.parentAgentId === undefined ? {} : { parentAgentId: handle.parentAgentId }),
      ...(handle.agent.model === undefined ? {} : { model: handle.agent.model }),
      ...(handle.rules.length ? { rules: handle.rules } : {}),
      ...(handle.toolsets.length ? { toolsets: handle.toolsets } : {}),
      status: handle.status,
      createdAt: handle.createdAt,
      updatedAt: handle.updatedAt,
      promptProfile: handle.promptProfile,
      ...(handle.lastInput === undefined ? {} : { lastInput: handle.lastInput }),
      ...(handle.lastOutput === undefined ? {} : { lastOutput: handle.lastOutput }),
      ...(handle.status === 'completed' && handle.lastOutput !== undefined
        ? { completionSummary: handle.lastOutput.slice(0, 500) }
        : {}),
      ...(handle.error === undefined ? {} : { error: handle.error }),
      queueSize: handle.queue.length,
      ...(handle.queue[0] === undefined ? {} : { queuedPreview: handle.queue[0] }),
      closed: handle.closed,
    })
  }

  private requireHandle(handleId: string): SpawnedAgentHandle {
    const handle = this.handles.get(handleId)
    if (handle === undefined) throw new ValidationError('handle_id', 'spawned agent not found', handleId)
    return handle
  }

  private resolveHandleId(handleId: string | undefined): string {
    if (handleId?.trim()) return handleId
    const active = [...this.handles.values()]
      .filter(handle => !handle.closed)
      .sort((left, right) => right.updatedAt.localeCompare(left.updatedAt) || right.createdAt.localeCompare(left.createdAt))[0]
    if (active === undefined) throw new ValidationError('handle_id', 'is required because no open spawned agents exist')
    return active.id
  }

  private isCurrent(handle: SpawnedAgentHandle, token: number): boolean {
    return !handle.closed && handle.token === token && handle.active?.token === token
  }
}

/** Normalize a model-supplied display title into one safe terminal line. */
export function normalizeAgentTitle(value: string, field = 'title'): string {
  if (typeof value !== 'string') throw new ValidationError(field, 'must be a string', value)
  const normalized = value.replace(/[\t\r\n]+/gu, ' ').replace(/\s+/gu, ' ').trim()
  if (!normalized) throw new ValidationError(field, 'must be a non-empty human-readable title', value)
  if (/[\p{Cc}\p{Cf}]/u.test(normalized)) {
    throw new ValidationError(field, 'must not contain control or formatting characters', value)
  }
  if ([...normalized].length > MAX_AGENT_TITLE_LENGTH) {
    throw new ValidationError(field, `must be at most ${MAX_AGENT_TITLE_LENGTH} characters`, value)
  }
  return normalized
}

function deriveAgentTitle(value: string): string {
  const firstLine = value.replace(/\r\n?/gu, '\n').split('\n').find(line => line.trim()) ?? 'Subagent'
  const readable = firstLine.trim().replace(/^(?:#{1,6}|[-+*>]|\d+[.)])\s+/u, '') || 'Subagent'
  const clipped = [...readable.replace(/\s+/gu, ' ').trim()].slice(0, MAX_AGENT_TITLE_LENGTH).join('').trimEnd()
  return normalizeAgentTitle(clipped || 'Subagent')
}

function normalizeLabels(values: readonly string[] | undefined): readonly string[] {
  if (!values?.length) return Object.freeze([])
  const labels = values
    .filter(value => typeof value === 'string')
    .map(value => value.replace(/[\t\r\n]+/gu, ' ').replace(/\s+/gu, ' ').trim())
    .filter(Boolean)
  return Object.freeze([...new Set(labels)])
}

function nonempty(value: string | undefined): string | undefined {
  const normalized = value?.trim()
  return normalized || undefined
}

function requireTimeout(value: number): number {
  if (!Number.isInteger(value) || value < 0) throw new ValidationError('timeout_ms', 'must be a non-negative integer', value)
  return value
}

function waitForCompletion(promise: Promise<void>, timeoutMs: number): Promise<void> {
  return new Promise(resolve => {
    const timer = setTimeout(resolve, timeoutMs)
    void promise.finally(() => {
      clearTimeout(timer)
      resolve()
    })
  })
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}
