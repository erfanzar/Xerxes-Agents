// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { MemoryMetadata, MemorySaveOptions } from '../memory/base.js'

export type CortexTaskStatus = 'failed' | 'pending' | 'skipped' | 'succeeded'

export interface CortexTask {
  readonly agentId?: string
  /** Additional successful task outputs to supply as context without creating a dependency edge. */
  readonly contextTaskIds?: readonly string[]
  readonly dependencies?: readonly string[]
  readonly description: string
  readonly expectedOutput: string
  readonly id: string
  readonly importance?: number
  readonly metadata?: Readonly<Record<string, unknown>>
  readonly runWhen?: TaskRunCondition
}

export interface CortexAgent {
  readonly description?: string
  readonly execute?: CortexAgentExecutor
  readonly id: string
  readonly name?: string
  readonly role?: string
}

export interface TaskExecutionResult {
  readonly metadata?: Readonly<Record<string, unknown>>
  readonly output: string
}

export interface CortexTaskOutput {
  readonly agentId?: string
  readonly completedAt: Date
  readonly durationMs: number
  readonly error?: string
  readonly metadata: Readonly<Record<string, unknown>>
  readonly output: string
  readonly startedAt: Date
  readonly status: Exclude<CortexTaskStatus, 'pending'>
  readonly taskId: string
}

export interface TaskExecutionContext {
  readonly agent: CortexAgent | undefined
  readonly context: string
  readonly dependencyOutputs: ReadonlyMap<string, CortexTaskOutput>
  readonly inputs: Readonly<Record<string, unknown>>
  readonly task: CortexTask
}

export interface TaskRunConditionContext {
  readonly inputs: Readonly<Record<string, unknown>>
  readonly outputs: ReadonlyMap<string, CortexTaskOutput>
  readonly task: CortexTask
}

export type CortexAgentExecutor = (context: TaskExecutionContext) => TaskExecutionResult | string | Promise<TaskExecutionResult | string>
export type CortexTaskExecutor = CortexAgentExecutor
export type TaskRunCondition = (context: TaskRunConditionContext) => boolean | Promise<boolean>

/** Minimal writer shape satisfied by the TypeScript memory implementations. */
export interface CortexMemoryWriter {
  save(content: string, metadata?: MemoryMetadata, options?: MemorySaveOptions): unknown | Promise<unknown>
}

/** Raised before execution when a dependency graph is malformed. */
export class TaskGraphError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'TaskGraphError'
  }
}

/** Validate task IDs and dependencies, including cycles. */
export function validateTaskGraph(tasks: readonly CortexTask[]): void {
  dependencyLayers(tasks, task => task.id, task => task.dependencies ?? [], 'task')
}

/** Return task batches that may execute once all prior batches have completed. */
export function taskDependencyLayers(tasks: readonly CortexTask[]): CortexTask[][] {
  return dependencyLayers(tasks, task => task.id, task => task.dependencies ?? [], 'task')
}

/** Return a stable topological order preserving declaration order within each ready batch. */
export function topologicallyOrderTasks(tasks: readonly CortexTask[]): CortexTask[] {
  return taskDependencyLayers(tasks).flat()
}

/** Build a readable context block from dependency and explicitly requested task outputs. */
export function taskContext(task: CortexTask, outputs: ReadonlyMap<string, CortexTaskOutput>): string {
  const ids = [...new Set([...(task.dependencies ?? []), ...(task.contextTaskIds ?? [])])]
  return ids.flatMap(id => {
    const output = outputs.get(id)
    return output?.status === 'succeeded' && output.output
      ? [`Output from task ${id}:\n${output.output}`]
      : []
  }).join('\n\n')
}

/** Stable, dependency-safe ordering primitive shared by task and plan modules. */
export function dependencyLayers<T>(
  items: readonly T[],
  idFor: (item: T) => string,
  dependenciesFor: (item: T) => readonly string[],
  label: string,
): T[][] {
  const byId = new Map<string, T>()
  for (const item of items) {
    const id = idFor(item).trim()
    if (!id) throw new TaskGraphError(`${label} id cannot be empty`)
    if (byId.has(id)) throw new TaskGraphError(`Duplicate ${label} id: ${id}`)
    byId.set(id, item)
  }

  const dependencies = new Map<string, Set<string>>()
  for (const item of items) {
    const id = idFor(item)
    const required = new Set(dependenciesFor(item))
    if (required.has(id)) throw new TaskGraphError(`${label} ${id} cannot depend on itself`)
    for (const dependency of required) {
      if (!byId.has(dependency)) throw new TaskGraphError(`${label} ${id} depends on unknown ${label} ${dependency}`)
    }
    dependencies.set(id, required)
  }

  const unresolved = new Set(byId.keys())
  const completed = new Set<string>()
  const result: T[][] = []
  while (unresolved.size > 0) {
    const ready = items.filter(item => {
      const id = idFor(item)
      const required = dependencies.get(id)
      return unresolved.has(id) && required !== undefined && [...required].every(dependency => completed.has(dependency))
    })
    if (ready.length === 0) {
      throw new TaskGraphError(`Circular ${label} dependencies: ${[...unresolved].join(', ')}`)
    }
    result.push(ready)
    for (const item of ready) {
      const id = idFor(item)
      unresolved.delete(id)
      completed.add(id)
    }
  }
  return result
}

/** Deterministically derive a compact plan identifier without external state. */
export function stableIdentifier(prefix: string, source: string): string {
  let hash = 2166136261
  for (const character of source) {
    hash ^= character.codePointAt(0) ?? 0
    hash = Math.imul(hash, 16777619)
  }
  return `${prefix}-${(hash >>> 0).toString(36)}`
}
