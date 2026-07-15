// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import {
  taskContext,
  validateTaskGraph,
  type CortexAgent,
  type CortexMemoryWriter,
  type CortexTask,
  type CortexTaskExecutor,
  type CortexTaskOutput,
  type CortexTaskStatus,
  type TaskExecutionContext,
  type TaskExecutionResult,
} from './task.js'

export const CortexProcess = {
  PARALLEL: 'parallel',
  SEQUENTIAL: 'sequential',
} as const

export type CortexProcess = (typeof CortexProcess)[keyof typeof CortexProcess]

export interface CortexOrchestratorOptions {
  readonly agents?: readonly CortexAgent[]
  readonly executor?: CortexTaskExecutor
  readonly maxParallel?: number
  readonly memory?: CortexMemoryWriter
  readonly now?: () => Date
  readonly process?: CortexProcess
  readonly tasks?: readonly CortexTask[]
}

export interface CortexRunOptions {
  readonly inputs?: Readonly<Record<string, unknown>>
  readonly process?: CortexProcess
}

export interface CortexRunOutput {
  readonly executionTimeMs: number
  readonly rawOutput: string
  readonly taskOutputs: readonly CortexTaskOutput[]
}

/**
 * Dependency-aware task runner with injected execution and memory boundaries.
 *
 * It does not construct an LLM client or use a Python bridge. An application
 * can provide one global task executor or per-agent executors, making this
 * orchestration layer usable by the Bun daemon, API server, or tests alike.
 */
export class CortexOrchestrator {
  private readonly agents = new Map<string, CortexAgent>()
  private readonly executor: CortexTaskExecutor | undefined
  private readonly maxParallel: number
  private readonly memory: CortexMemoryWriter | undefined
  private readonly now: () => Date
  private readonly process: CortexProcess
  private tasks: CortexTask[]
  private latest: CortexRunOutput | undefined

  constructor(options: CortexOrchestratorOptions = {}) {
    this.executor = options.executor
    this.memory = options.memory
    this.now = options.now ?? (() => new Date())
    this.process = options.process ?? CortexProcess.SEQUENTIAL
    this.maxParallel = options.maxParallel ?? Number.POSITIVE_INFINITY
    if (!Number.isInteger(this.maxParallel) && this.maxParallel !== Number.POSITIVE_INFINITY) {
      throw new Error('maxParallel must be a positive integer')
    }
    if (this.maxParallel < 1) throw new Error('maxParallel must be a positive integer')
    for (const agent of options.agents ?? []) this.registerAgent(agent)
    this.tasks = []
    this.setTasks(options.tasks ?? [])
  }

  get lastOutput(): CortexRunOutput | undefined {
    return this.latest
  }

  get registeredAgents(): ReadonlyMap<string, CortexAgent> {
    return this.agents
  }

  get taskList(): readonly CortexTask[] {
    return this.tasks
  }

  registerAgent(agent: CortexAgent): void {
    if (!agent.id.trim()) throw new Error('Agent id cannot be empty')
    if (this.agents.has(agent.id)) throw new Error(`Agent ${agent.id} is already registered`)
    this.agents.set(agent.id, agent)
  }

  setTasks(tasks: readonly CortexTask[]): void {
    validateTaskGraph(tasks)
    this.tasks = [...tasks]
  }

  async run(options: CortexRunOptions = {}): Promise<CortexRunOutput> {
    const startedAt = this.now()
    const process = options.process ?? this.process
    const inputs = options.inputs ?? {}
    validateTaskGraph(this.tasks)
    const states = new Map<string, CortexTaskStatus>(this.tasks.map(task => [task.id, 'pending']))
    const outputs = new Map<string, CortexTaskOutput>()

    while ([...states.values()].some(status => status === 'pending')) {
      if (skipBlockedTasks(this.tasks, states, outputs, this.now)) continue
      const ready = this.tasks.filter(task => states.get(task.id) === 'pending'
        && (task.dependencies ?? []).every(dependency => states.get(dependency) === 'succeeded'))
      if (ready.length === 0) throw new Error('No executable tasks remain')

      const runnable: CortexTask[] = []
      for (const task of ready) {
        const condition = await evaluateCondition(task, outputs, inputs)
        if (condition.kind === 'skip') {
          const timestamp = this.now()
          outputs.set(task.id, skippedOutput(task, timestamp, condition.reason))
          states.set(task.id, 'skipped')
          continue
        }
        if (condition.kind === 'failure') {
          const timestamp = this.now()
          outputs.set(task.id, failedOutput(task, timestamp, timestamp, condition.error))
          states.set(task.id, 'failed')
          continue
        }
        runnable.push(task)
      }
      if (runnable.length === 0) continue

      const execute = async (task: CortexTask): Promise<void> => {
        const output = await this.executeTask(task, outputs, inputs)
        outputs.set(task.id, output)
        states.set(task.id, output.status)
      }
      if (process === CortexProcess.PARALLEL) await mapConcurrent(runnable, this.maxParallel, execute)
      else for (const task of runnable) await execute(task)
    }

    const completedAt = this.now()
    const taskOutputs = this.tasks.flatMap(task => {
      const output = outputs.get(task.id)
      return output ? [output] : []
    })
    const result: CortexRunOutput = {
      taskOutputs,
      rawOutput: taskOutputs.filter(output => output.status === 'succeeded').at(-1)?.output ?? '',
      executionTimeMs: Math.max(0, completedAt.getTime() - startedAt.getTime()),
    }
    this.latest = result
    return result
  }

  kickoff(options: CortexRunOptions = {}): Promise<CortexRunOutput> {
    return this.run(options)
  }

  private async executeTask(
    task: CortexTask,
    outputs: ReadonlyMap<string, CortexTaskOutput>,
    inputs: Readonly<Record<string, unknown>>,
  ): Promise<CortexTaskOutput> {
    const startedAt = this.now()
    const agent = task.agentId ? this.agents.get(task.agentId) : undefined
    try {
      if (task.agentId && !agent) throw new Error(`Task ${task.id} references unknown agent ${task.agentId}`)
      const executor = this.executor ?? agent?.execute
      if (!executor) throw new Error(`No executor is configured for task ${task.id}`)
      const dependencyOutputs = new Map((task.dependencies ?? []).flatMap(id => {
        const output = outputs.get(id)
        return output ? [[id, output] as const] : []
      }))
      const result = await executor({
        task,
        agent,
        inputs,
        context: taskContext(task, outputs),
        dependencyOutputs,
      })
      const normalized = normalizeResult(result)
      const completedAt = this.now()
      const output: CortexTaskOutput = {
        taskId: task.id,
        status: 'succeeded',
        output: normalized.output,
        startedAt,
        completedAt,
        durationMs: Math.max(0, completedAt.getTime() - startedAt.getTime()),
        metadata: normalized.metadata ?? {},
        ...(task.agentId ? { agentId: task.agentId } : {}),
      }
      await this.saveTaskResult(task, output)
      return output
    } catch (error) {
      return failedOutput(task, startedAt, this.now(), errorMessage(error))
    }
  }

  private async saveTaskResult(task: CortexTask, output: CortexTaskOutput): Promise<void> {
    if (!this.memory) return
    try {
      await this.memory.save(
        `Task completed: ${task.description}\n\n${output.output}`,
        { taskId: task.id, expectedOutput: task.expectedOutput, status: output.status },
        {
          ...(task.agentId ? { agentId: task.agentId } : {}),
          taskId: task.id,
          importance: task.importance ?? 0.5,
        },
      )
    } catch {
      // Memory persistence is intentionally non-critical to the execution result.
    }
  }
}

function skipBlockedTasks(
  tasks: readonly CortexTask[],
  states: Map<string, CortexTaskStatus>,
  outputs: Map<string, CortexTaskOutput>,
  now: () => Date,
): boolean {
  let changed = false
  for (const task of tasks) {
    if (states.get(task.id) !== 'pending') continue
    const blocked = (task.dependencies ?? []).some(dependency => {
      const status = states.get(dependency)
      return status === 'failed' || status === 'skipped'
    })
    if (!blocked) continue
    const timestamp = now()
    outputs.set(task.id, skippedOutput(task, timestamp, 'A dependency did not complete successfully'))
    states.set(task.id, 'skipped')
    changed = true
  }
  return changed
}

async function evaluateCondition(
  task: CortexTask,
  outputs: ReadonlyMap<string, CortexTaskOutput>,
  inputs: Readonly<Record<string, unknown>>,
): Promise<{ readonly kind: 'run' } | { readonly kind: 'skip'; readonly reason: string } | { readonly kind: 'failure'; readonly error: string }> {
  if (!task.runWhen) return { kind: 'run' }
  try {
    return await task.runWhen({ task, outputs, inputs }) ? { kind: 'run' } : { kind: 'skip', reason: 'Task condition returned false' }
  } catch (error) {
    return { kind: 'failure', error: `Task condition failed: ${errorMessage(error)}` }
  }
}

function normalizeResult(result: string | TaskExecutionResult): TaskExecutionResult {
  return typeof result === 'string' ? { output: result } : result
}

function skippedOutput(task: CortexTask, timestamp: Date, reason: string): CortexTaskOutput {
  return {
    taskId: task.id,
    status: 'skipped',
    output: '',
    error: reason,
    startedAt: timestamp,
    completedAt: timestamp,
    durationMs: 0,
    metadata: {},
    ...(task.agentId ? { agentId: task.agentId } : {}),
  }
}

function failedOutput(task: CortexTask, startedAt: Date, completedAt: Date, error: string): CortexTaskOutput {
  return {
    taskId: task.id,
    status: 'failed',
    output: '',
    error,
    startedAt,
    completedAt,
    durationMs: Math.max(0, completedAt.getTime() - startedAt.getTime()),
    metadata: {},
    ...(task.agentId ? { agentId: task.agentId } : {}),
  }
}

async function mapConcurrent<T>(items: readonly T[], maxParallel: number, run: (item: T) => Promise<void>): Promise<void> {
  const workerCount = Math.min(items.length, maxParallel)
  let next = 0
  const worker = async (): Promise<void> => {
    while (next < items.length) {
      const index = next
      next += 1
      const item = items[index]
      if (item !== undefined) await run(item)
    }
  }
  await Promise.all(Array.from({ length: workerCount }, worker))
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}
