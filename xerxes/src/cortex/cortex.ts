// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ProcessType } from './core/enums.js'
import { PromptTemplate } from './core/templates.js'
import {
  CortexOrchestrator,
  CortexProcess,
  DEFAULT_MAX_PARALLEL,
} from './orchestrator.js'
import {
  CortexPlanner,
  type ExecutionPlan,
  type PlanStep,
  type PlanStepExecutionRequest,
  type PlanStepOutput,
} from './planner.js'
import {
  validateTaskGraph,
  type CortexAgent,
  type CortexMemoryWriter,
  type CortexTask,
  type CortexTaskExecutor,
  type CortexTaskOutput,
  type TaskExecutionContext,
  type TaskExecutionResult,
} from './task.js'

export interface CortexEngineTaskExecutionRequest extends TaskExecutionContext {
  readonly signal?: AbortSignal
}

/** Host-owned task execution boundary that can receive cancellation. */
export type CortexEngineTaskRunner = (
  request: CortexEngineTaskExecutionRequest,
) => string | TaskExecutionResult | Promise<string | TaskExecutionResult>

export interface CortexManagerTaskAssignment {
  readonly agentId?: string
  readonly dependencies?: readonly string[]
  readonly taskId: string
}

export interface CortexManagerPlan {
  readonly assignments: readonly CortexManagerTaskAssignment[]
}

export interface CortexManagerPlanRequest {
  readonly agents: readonly CortexAgent[]
  readonly inputs: Readonly<Record<string, unknown>>
  readonly prompt: string
  readonly signal?: AbortSignal
  readonly tasks: readonly CortexTask[]
}

/** Produces a typed delegation plan. No JSON parser or implicit manager LLM is involved. */
export type CortexManagerPlanPort = (request: CortexManagerPlanRequest) => CortexManagerPlan | Promise<CortexManagerPlan>

export interface CortexManagerReview {
  readonly approved: boolean
  readonly feedback?: string
  readonly improvementsNeeded?: readonly string[]
}

export interface CortexManagerReviewRequest {
  readonly agent: CortexAgent | undefined
  readonly attempt: number
  readonly context: TaskExecutionContext
  readonly output: TaskExecutionResult
  readonly prompt: string
  readonly signal?: AbortSignal
  readonly task: CortexTask
}

/** Reviews a concrete agent result and may request a bounded re-execution. */
export type CortexManagerReviewPort = (
  request: CortexManagerReviewRequest,
) => CortexManagerReview | Promise<CortexManagerReview>

export interface CortexManagerSummaryRequest {
  readonly inputs: Readonly<Record<string, unknown>>
  readonly signal?: AbortSignal
  readonly taskOutputs: readonly CortexTaskOutput[]
}

/** Optional final summary boundary for a hierarchical run. */
export type CortexManagerSummaryPort = (
  request: CortexManagerSummaryRequest,
) => string | TaskExecutionResult | Promise<string | TaskExecutionResult>

export interface CortexHierarchyOptions {
  readonly maxReviewAttempts?: number
  readonly plan?: CortexManagerPlanPort
  readonly review?: CortexManagerReviewPort
  readonly summarize?: CortexManagerSummaryPort
}

export interface CortexConsensusCandidate {
  readonly agent: CortexAgent
  readonly metadata: Readonly<Record<string, unknown>>
  readonly output: string
}

export interface CortexConsensusSynthesisRequest {
  readonly candidates: readonly CortexConsensusCandidate[]
  readonly context: TaskExecutionContext
  readonly inputs: Readonly<Record<string, unknown>>
  readonly signal?: AbortSignal
  readonly task: CortexTask
}

/** An explicit semantic consensus implementation supplied by the application. */
export type CortexConsensusSynthesizer = (
  request: CortexConsensusSynthesisRequest,
) => string | TaskExecutionResult | Promise<string | TaskExecutionResult>

export interface CortexConsensusOptions {
  /**
   * Maximum concurrent candidate executions for one task. Defaults to a
   * finite cap; pass Number.POSITIVE_INFINITY to opt into unbounded fan-out.
   */
  readonly maxCandidatesParallel?: number
  readonly synthesizer?: CortexConsensusSynthesizer
}

export interface CortexDiagnostic {
  readonly code: 'consensus_candidate_failed' | 'hierarchy_plan_fallback' | 'planned_fallback' | 'planned_unmapped_task'
  readonly message: string
  readonly taskId?: string
}

export interface CortexOptions {
  readonly agents?: readonly CortexAgent[]
  readonly consensus?: CortexConsensusOptions
  readonly executor?: CortexTaskExecutor
  readonly hierarchy?: CortexHierarchyOptions
  readonly maxParallel?: number
  readonly memory?: CortexMemoryWriter
  readonly now?: () => Date
  readonly plannedParallel?: boolean
  readonly planner?: CortexPlanner
  readonly process?: ProcessType
  readonly taskRunner?: CortexEngineTaskRunner
  readonly tasks?: readonly CortexTask[]
}

export interface CortexKickoffOptions {
  readonly inputs?: Readonly<Record<string, unknown>>
  readonly signal?: AbortSignal
}

/** Aggregate native result for every Cortex topology. */
export interface CortexOutput {
  readonly diagnostics: readonly CortexDiagnostic[]
  readonly executionTimeMs: number
  readonly process: ProcessType
  readonly rawOutput: string
  readonly taskOutputs: readonly CortexTaskOutput[]
}

/** Raised to distinguish caller cancellation from an ordinary failed task. */
export class CortexCancellationError extends Error {
  constructor(reason?: unknown) {
    super(reason === undefined
      ? 'Cortex execution was cancelled'
      : `Cortex execution was cancelled: ${errorMessage(reason)}`)
    this.name = 'CortexCancellationError'
  }
}

interface TopologyResult {
  readonly rawOutput: string
  readonly taskOutputs: readonly CortexTaskOutput[]
}

interface HierarchyResolution {
  readonly tasks: readonly CortexTask[]
}

interface PlannedBinding {
  readonly missingTaskDependencies: readonly string[]
  readonly originalTaskId?: string
  readonly task: CortexTask
}

interface PlannedBindings {
  readonly byStepId: ReadonlyMap<string, PlannedBinding>
  readonly stepIdByTaskId: ReadonlyMap<string, string>
  readonly unmappedTasks: readonly CortexTask[]
}

/**
 * Native high-level Cortex engine.
 *
 * The engine is deliberately composed from the existing task graph runner and
 * XML planner. All model work remains behind injected agent/executor ports;
 * there is no Python process, subprocess adapter, or implicit provider setup.
 */
export class Cortex {
  private readonly agents: readonly CortexAgent[]
  private readonly consensus: CortexConsensusOptions
  private readonly executor: CortexTaskExecutor | undefined
  private readonly hierarchy: CortexHierarchyOptions
  private readonly maxParallel: number | undefined
  private readonly memory: CortexMemoryWriter | undefined
  private readonly now: () => Date
  private readonly plannedParallel: boolean
  private readonly planner: CortexPlanner
  private readonly process: ProcessType
  private readonly taskRunner: CortexEngineTaskRunner | undefined
  private readonly tasks: readonly CortexTask[]
  private readonly templates = new PromptTemplate()
  private latest: CortexOutput | undefined

  constructor(options: CortexOptions = {}) {
    this.agents = [...(options.agents ?? [])]
    assertUniqueAgents(this.agents)
    this.tasks = [...(options.tasks ?? [])]
    validateTaskGraph(this.tasks)
    this.process = options.process ?? ProcessType.SEQUENTIAL
    assertProcessType(this.process)
    this.executor = options.executor
    this.taskRunner = options.taskRunner
    this.memory = options.memory
    this.now = options.now ?? (() => new Date())
    this.maxParallel = validateMaxParallel(options.maxParallel)
    this.consensus = options.consensus ?? {}
    validateMaxCandidates(this.consensus.maxCandidatesParallel)
    this.hierarchy = options.hierarchy ?? {}
    validateReviewAttempts(this.hierarchy.maxReviewAttempts)
    this.planner = options.planner ?? new CortexPlanner()
    this.plannedParallel = options.plannedParallel ?? false
  }

  get lastOutput(): CortexOutput | undefined {
    return this.latest
  }

  get taskList(): readonly CortexTask[] {
    return this.tasks
  }

  get registeredAgents(): readonly CortexAgent[] {
    return this.agents
  }

  async kickoff(options: CortexKickoffOptions = {}): Promise<CortexOutput> {
    return this.run(options)
  }

  async run(options: CortexKickoffOptions = {}): Promise<CortexOutput> {
    const signal = options.signal
    throwIfCancelled(signal)
    const inputs = options.inputs ?? {}
    const diagnostics: CortexDiagnostic[] = []
    const startedAt = this.now()
    const topology = await awaitWithCancellation(this.runTopology(inputs, signal, diagnostics), signal)
    throwIfCancelled(signal)
    const completedAt = this.now()
    const output: CortexOutput = {
      process: this.process,
      rawOutput: topology.rawOutput,
      taskOutputs: topology.taskOutputs,
      diagnostics,
      executionTimeMs: Math.max(0, completedAt.getTime() - startedAt.getTime()),
    }
    this.latest = output
    return output
  }

  private async runTopology(
    inputs: Readonly<Record<string, unknown>>,
    signal: AbortSignal | undefined,
    diagnostics: CortexDiagnostic[],
  ): Promise<TopologyResult> {
    switch (this.process) {
      case ProcessType.SEQUENTIAL:
      case ProcessType.PARALLEL:
        return this.runStandard(inputs, signal)
      case ProcessType.HIERARCHICAL:
        return this.runHierarchical(inputs, signal, diagnostics)
      case ProcessType.CONSENSUS:
        return this.runConsensus(inputs, signal, diagnostics)
      case ProcessType.PLANNED:
        return this.runPlanned(inputs, signal, diagnostics)
    }
  }

  private async runStandard(
    inputs: Readonly<Record<string, unknown>>,
    signal: AbortSignal | undefined,
  ): Promise<TopologyResult> {
    const runner = this.createOrchestrator(
      this.tasks,
      context => this.executeTask(context, signal),
      this.process === ProcessType.PARALLEL ? CortexProcess.PARALLEL : CortexProcess.SEQUENTIAL,
    )
    return runner.run({ inputs, ...(signal === undefined ? {} : { signal }) })
  }

  private async runHierarchical(
    inputs: Readonly<Record<string, unknown>>,
    signal: AbortSignal | undefined,
    diagnostics: CortexDiagnostic[],
  ): Promise<TopologyResult> {
    const resolution = await this.resolveHierarchy(inputs, signal, diagnostics)
    const runner = this.createOrchestrator(
      resolution.tasks,
      context => this.executeHierarchicalTask(context, signal),
      CortexProcess.SEQUENTIAL,
    )
    const result = await runner.run({ inputs, ...(signal === undefined ? {} : { signal }) })
    if (!this.hierarchy.summarize) return result
    const summary = normalizeResult(await this.hierarchy.summarize({
      taskOutputs: result.taskOutputs,
      inputs,
      ...(signal === undefined ? {} : { signal }),
    }))
    if (!summary.output.trim()) throw new Error('Hierarchical manager summary completed without text')
    return { ...result, rawOutput: summary.output }
  }

  private async runConsensus(
    inputs: Readonly<Record<string, unknown>>,
    signal: AbortSignal | undefined,
    diagnostics: CortexDiagnostic[],
  ): Promise<TopologyResult> {
    const runner = this.createOrchestrator(
      this.tasks,
      context => this.executeConsensusTask(context, inputs, signal, diagnostics),
      CortexProcess.SEQUENTIAL,
    )
    return runner.run({ inputs, ...(signal === undefined ? {} : { signal }) })
  }

  private async runPlanned(
    inputs: Readonly<Record<string, unknown>>,
    signal: AbortSignal | undefined,
    diagnostics: CortexDiagnostic[],
  ): Promise<TopologyResult> {
    if (this.tasks.length === 0) return { rawOutput: '', taskOutputs: [] }
    const objective = plannedObjective(this.tasks)
    const created = await this.planner.createPlan({
      objective,
      agents: this.agents.map(agent => ({
        id: agent.id,
        ...(agent.role === undefined ? {} : { role: agent.role }),
        ...(agent.description === undefined ? {} : { summary: agent.description }),
      })),
    })
    throwIfCancelled(signal)
    if (created.error !== undefined) {
      diagnostics.push({ code: 'planned_fallback', message: created.error })
    }
    const bindings = buildPlannedBindings(created.plan, this.tasks, this.agents)
    for (const task of bindings.unmappedTasks) {
      diagnostics.push({
        code: 'planned_unmapped_task',
        taskId: task.id,
        message: `Planner did not produce a step for task ${task.id}`,
      })
    }
    const plan = augmentPlanDependencies(created.plan, bindings)
    const result = await this.planner.executePlan(
      plan,
      request => this.executePlannedStep(request, bindings, inputs, signal),
      { parallel: this.plannedParallel },
    )
    throwIfCancelled(signal)
    return planOutputsToTasks(result.outputs, plan, bindings, this.tasks, this.now)
  }

  private createOrchestrator(
    tasks: readonly CortexTask[],
    executor: CortexTaskExecutor,
    process: CortexProcess,
  ): CortexOrchestrator {
    return new CortexOrchestrator({
      agents: this.agents,
      executor,
      tasks,
      process,
      ...(this.maxParallel === undefined ? {} : { maxParallel: this.maxParallel }),
      ...(this.memory === undefined ? {} : { memory: this.memory }),
      now: this.now,
    })
  }

  private async resolveHierarchy(
    inputs: Readonly<Record<string, unknown>>,
    signal: AbortSignal | undefined,
    diagnostics: CortexDiagnostic[],
  ): Promise<HierarchyResolution> {
    let plan: CortexManagerPlan | undefined
    if (this.hierarchy.plan) {
      try {
        plan = await this.hierarchy.plan({
          agents: this.agents,
          tasks: this.tasks,
          inputs,
          prompt: this.templates.renderManagerDelegation(
            this.agents.map(agent => ({
              role: agent.role ?? agent.name ?? agent.id,
              goal: agent.description ?? 'Available executor',
            })),
            this.tasks,
          ),
          ...(signal === undefined ? {} : { signal }),
        })
        throwIfCancelled(signal)
        assertManagerPlan(plan)
      } catch (error) {
        if (error instanceof CortexCancellationError) throw error
        diagnostics.push({ code: 'hierarchy_plan_fallback', message: errorMessage(error) })
      }
    }

    const assignments = new Map<string, CortexManagerTaskAssignment>()
    for (const assignment of plan?.assignments ?? []) {
      if (!this.tasks.some(task => task.id === assignment.taskId)) {
        diagnostics.push({
          code: 'hierarchy_plan_fallback',
          message: `Manager assigned unknown task ${assignment.taskId}`,
          taskId: assignment.taskId,
        })
        continue
      }
      if (assignments.has(assignment.taskId)) {
        diagnostics.push({
          code: 'hierarchy_plan_fallback',
          message: `Manager assigned task ${assignment.taskId} more than once; retaining the first assignment`,
          taskId: assignment.taskId,
        })
        continue
      }
      assignments.set(assignment.taskId, assignment)
    }

    const taskIds = new Set(this.tasks.map(task => task.id))
    const resolved = this.tasks.map((task, index) => {
      const assignment = assignments.get(task.id)
      const assigned = assignment?.agentId === undefined ? undefined : findAgent(this.agents, assignment.agentId)
      if (assignment?.agentId !== undefined && assigned === undefined) {
        diagnostics.push({
          code: 'hierarchy_plan_fallback',
          taskId: task.id,
          message: `Manager assigned unknown agent ${assignment.agentId} to task ${task.id}; using deterministic fallback`,
        })
      }
      const fallback = findAgent(this.agents, task.agentId) ?? fallbackAgent(this.agents, index)
      const dependencies = uniqueDependencies(
        [...(task.dependencies ?? []), ...(assignment?.dependencies ?? [])],
        task.id,
        taskIds,
        diagnostics,
      )
      return taskWithAgent(task, assigned ?? fallback, dependencies)
    })

    try {
      validateTaskGraph(resolved)
      return { tasks: resolved }
    } catch (error) {
      diagnostics.push({
        code: 'hierarchy_plan_fallback',
        message: `Manager dependency plan was unsafe; retaining declared task dependencies: ${errorMessage(error)}`,
      })
      return {
        tasks: this.tasks.map((task, index) => taskWithAgent(
          task,
          findAgent(this.agents, task.agentId) ?? fallbackAgent(this.agents, index),
          task.dependencies ?? [],
        )),
      }
    }
  }

  private async executeHierarchicalTask(
    context: TaskExecutionContext,
    signal: AbortSignal | undefined,
  ): Promise<TaskExecutionResult> {
    let result = normalizeResult(await this.executeTask(context, signal))
    if (!this.hierarchy.review) return result
    const maxAttempts = this.hierarchy.maxReviewAttempts ?? 2
    for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
      const review = await this.hierarchy.review({
        task: context.task,
        agent: context.agent,
        context,
        attempt,
        output: result,
        prompt: this.templates.renderManagerReview({
          agentRole: context.agent?.role ?? context.agent?.name ?? context.agent?.id ?? 'Unassigned agent',
          taskDescription: context.task.description,
          output: result.output,
        }),
        ...(signal === undefined ? {} : { signal }),
      })
      throwIfCancelled(signal)
      assertManagerReview(review)
      if (review.approved) {
        return {
          output: result.output,
          metadata: {
            ...result.metadata,
            hierarchyReview: { approved: true, attempts: attempt },
          },
        }
      }
      const improvements = review.improvementsNeeded?.filter(item => item.trim()) ?? []
      const feedback = review.feedback?.trim() ?? ''
      if (attempt >= maxAttempts) {
        throw new Error(
          `Manager rejected task ${context.task.id} after ${maxAttempts} review attempts${feedback ? `: ${feedback}` : ''}`,
        )
      }
      if (!feedback && improvements.length === 0) {
        throw new Error(`Manager rejected task ${context.task.id} without actionable feedback`)
      }
      const improvementContext = [
        context.context,
        'Manager review feedback:',
        ...(feedback ? [feedback] : []),
        ...improvements.map(item => `- ${item}`),
      ].filter(Boolean).join('\n\n')
      result = normalizeResult(await this.executeTask({ ...context, context: improvementContext }, signal))
    }
    throw new Error(`Manager review loop ended unexpectedly for task ${context.task.id}`)
  }

  private async executeConsensusTask(
    context: TaskExecutionContext,
    inputs: Readonly<Record<string, unknown>>,
    signal: AbortSignal | undefined,
    diagnostics: CortexDiagnostic[],
  ): Promise<TaskExecutionResult> {
    if (this.agents.length === 0) {
      throw new Error(`Consensus task ${context.task.id} requires at least one registered agent`)
    }
    const candidates: Array<CortexConsensusCandidate | undefined> = new Array(this.agents.length)
    const errors: Array<{ readonly agentId: string; readonly message: string }> = []
    const maximum = this.consensus.maxCandidatesParallel ?? DEFAULT_MAX_PARALLEL
    await mapConcurrent(this.agents, maximum, async (agent, index) => {
      try {
        const result = normalizeResult(await this.executeTask({ ...context, agent }, signal))
        candidates[index] = {
          agent,
          output: result.output,
          metadata: result.metadata ?? {},
        }
      } catch (error) {
        if (error instanceof CortexCancellationError) throw error
        const message = errorMessage(error)
        errors.push({ agentId: agent.id, message })
        diagnostics.push({
          code: 'consensus_candidate_failed',
          taskId: context.task.id,
          message: `Consensus candidate ${agent.id} failed: ${message}`,
        })
      }
    })
    throwIfCancelled(signal)
    const completed = candidates.filter((candidate): candidate is CortexConsensusCandidate => candidate !== undefined)
    if (completed.length === 0) {
      throw new Error(
        `Every consensus candidate failed for task ${context.task.id}: ${errors.map(error => `${error.agentId}: ${error.message}`).join('; ')}`,
      )
    }
    const request: CortexConsensusSynthesisRequest = {
      task: context.task,
      context,
      inputs,
      candidates: completed,
      ...(signal === undefined ? {} : { signal }),
    }
    const synthesized = normalizeResult(this.consensus.synthesizer
      ? await this.consensus.synthesizer(request)
      : nativeConsensusSynthesis(request))
    throwIfCancelled(signal)
    return {
      output: synthesized.output,
      metadata: {
        ...synthesized.metadata,
        consensus: {
          completedCandidates: completed.map(candidate => candidate.agent.id),
          failedCandidates: errors,
          strategy: this.consensus.synthesizer ? 'injected' : 'native-aggregate',
        },
      },
    }
  }

  private async executePlannedStep(
    request: PlanStepExecutionRequest,
    bindings: PlannedBindings,
    inputs: Readonly<Record<string, unknown>>,
    signal: AbortSignal | undefined,
  ): Promise<TaskExecutionResult> {
    const binding = bindings.byStepId.get(request.step.id)
    if (!binding) throw new Error(`No task binding exists for planned step ${request.step.id}`)
    if (binding.missingTaskDependencies.length > 0) {
      throw new Error(
        `Planned step ${request.step.id} cannot satisfy task dependencies: ${binding.missingTaskDependencies.join(', ')}`,
      )
    }
    const dependencyOutputs = planDependencyOutputs(request.dependencyOutputs, bindings)
    const context = [...request.dependencyOutputs.entries()]
      .filter(([, output]) => output.status === 'succeeded' && output.output)
      .map(([stepId, output]) => `Output from planned step ${stepId}:\n${output.output}`)
      .join('\n\n')
    const task = plannedExecutionTask(binding.task, request.step, request.arguments)
    return normalizeResult(await this.executeTask({
      task,
      agent: task.agentId === undefined ? undefined : findAgent(this.agents, task.agentId),
      inputs,
      context,
      dependencyOutputs,
    }, signal))
  }

  private async executeTask(
    context: TaskExecutionContext,
    signal: AbortSignal | undefined,
  ): Promise<string | TaskExecutionResult> {
    throwIfCancelled(signal)
    const request: CortexEngineTaskExecutionRequest = {
      ...context,
      ...(signal === undefined ? {} : { signal }),
    }
    const result = this.taskRunner
      ? await this.taskRunner(request)
      : this.executor
        ? await this.executor(request)
        : context.agent?.execute
          ? await context.agent.execute(request)
          : (() => { throw new Error(`No executor is configured for task ${context.task.id}`) })()
    throwIfCancelled(signal)
    return result
  }
}

/**
 * Native, transparent consensus fallback. It returns the concrete candidate
 * answers with labels instead of claiming an unperformed LLM synthesis.
 */
export function nativeConsensusSynthesis(request: CortexConsensusSynthesisRequest): TaskExecutionResult {
  if (request.candidates.length === 0) throw new Error(`Consensus task ${request.task.id} has no candidate output`)
  return {
    output: request.candidates.map(candidate => {
      const label = candidate.agent.role ?? candidate.agent.name ?? candidate.agent.id
      return `### ${label}\n${candidate.output}`
    }).join('\n\n'),
    metadata: {
      candidateCount: request.candidates.length,
      strategy: 'native-aggregate',
    },
  }
}

function buildPlannedBindings(
  plan: ExecutionPlan,
  tasks: readonly CortexTask[],
  agents: readonly CortexAgent[],
): PlannedBindings {
  const available = new Set(tasks.map(task => task.id))
  const byStepId = new Map<string, PlannedBinding>()
  const stepIdByTaskId = new Map<string, string>()
  let nextTaskIndex = 0
  for (const step of plan.steps) {
    let source = tasks.find(task => task.id === step.id && available.has(task.id))
    while (!source && nextTaskIndex < tasks.length) {
      const candidate = tasks[nextTaskIndex]
      nextTaskIndex += 1
      if (candidate && available.has(candidate.id)) source = candidate
    }
    if (source) available.delete(source.id)
    const base = source ?? {
      id: `plan-step:${step.id}`,
      description: step.description,
      expectedOutput: `Complete planned action: ${step.action}`,
    }
    const agent = findAgent(agents, step.agentId) ?? findAgent(agents, source?.agentId) ?? agents[0]
    const task = taskWithAgent(base, agent, base.dependencies ?? [])
    byStepId.set(step.id, {
      task,
      ...(source === undefined ? {} : { originalTaskId: source.id }),
      missingTaskDependencies: [],
    })
    if (source) stepIdByTaskId.set(source.id, step.id)
  }
  const enriched = new Map<string, PlannedBinding>()
  for (const [stepId, binding] of byStepId) {
    const missingTaskDependencies = (binding.task.dependencies ?? [])
      .filter(dependency => !stepIdByTaskId.has(dependency))
    enriched.set(stepId, { ...binding, missingTaskDependencies })
  }
  const unmappedTasks = tasks.filter(task => !stepIdByTaskId.has(task.id))
  return { byStepId: enriched, stepIdByTaskId, unmappedTasks }
}

function augmentPlanDependencies(plan: ExecutionPlan, bindings: PlannedBindings): ExecutionPlan {
  const steps = plan.steps.map(step => {
    const binding = bindings.byStepId.get(step.id)
    const requiredTaskSteps = binding?.task.dependencies?.flatMap(dependency => {
      const stepId = bindings.stepIdByTaskId.get(dependency)
      return stepId === undefined ? [] : [stepId]
    }) ?? []
    return {
      ...step,
      dependencies: [...new Set([...step.dependencies, ...requiredTaskSteps])],
    }
  })
  return { ...plan, steps }
}

function planOutputsToTasks(
  outputs: readonly PlanStepOutput[],
  plan: ExecutionPlan,
  bindings: PlannedBindings,
  declaredTasks: readonly CortexTask[],
  now: () => Date,
): TopologyResult {
  const outputByStep = new Map(outputs.map(output => [output.stepId, output]))
  const stepByTaskId = new Map<string, string>()
  for (const [stepId, binding] of bindings.byStepId) {
    if (binding.originalTaskId) stepByTaskId.set(binding.originalTaskId, stepId)
  }
  const taskOutputs = declaredTasks.map(task => {
    const stepId = stepByTaskId.get(task.id)
    if (!stepId) return skippedTaskOutput(task, now(), `Planner did not produce a step for task ${task.id}`)
    const output = outputByStep.get(stepId)
    if (!output) return failedTaskOutput(task, now(), now(), `Planner did not return an output for step ${stepId}`)
    const step = plan.steps.find(candidate => candidate.id === stepId)
    const binding = bindings.byStepId.get(stepId)
    return planOutputToTaskOutput(task, output, step, binding)
  })
  const rawOutput = outputs.filter(output => output.status === 'succeeded').at(-1)?.output ?? ''
  return { rawOutput, taskOutputs }
}

function planDependencyOutputs(
  outputs: ReadonlyMap<string, PlanStepOutput>,
  bindings: PlannedBindings,
): ReadonlyMap<string, CortexTaskOutput> {
  const result = new Map<string, CortexTaskOutput>()
  for (const [stepId, output] of outputs) {
    const binding = bindings.byStepId.get(stepId)
    const taskId = binding?.originalTaskId ?? binding?.task.id ?? `plan-step:${stepId}`
    result.set(taskId, {
      taskId,
      status: output.status,
      output: output.output,
      startedAt: output.startedAt,
      completedAt: output.completedAt,
      durationMs: output.durationMs,
      metadata: output.metadata,
      ...(output.error === undefined ? {} : { error: output.error }),
      ...(binding?.task.agentId === undefined ? {} : { agentId: binding.task.agentId }),
    })
  }
  return result
}

function planOutputToTaskOutput(
  task: CortexTask,
  output: PlanStepOutput,
  step: PlanStep | undefined,
  binding: PlannedBinding | undefined,
): CortexTaskOutput {
  return {
    taskId: task.id,
    status: output.status,
    output: output.output,
    startedAt: output.startedAt,
    completedAt: output.completedAt,
    durationMs: output.durationMs,
    metadata: {
      ...output.metadata,
      planStepId: output.stepId,
      ...(step === undefined ? {} : { planAction: step.action }),
    },
    ...(output.error === undefined ? {} : { error: output.error }),
    ...(binding?.task.agentId === undefined ? {} : { agentId: binding.task.agentId }),
  }
}

function plannedExecutionTask(
  task: CortexTask,
  step: PlanStep,
  argumentsValue: Readonly<Record<string, string>>,
): CortexTask {
  const argumentsText = Object.entries(argumentsValue).map(([key, value]) => `- ${key}: ${value}`).join('\n')
  const description = [
    task.description,
    `Planned action: ${step.action}`,
    step.description,
    ...(argumentsText ? [`Planned arguments:\n${argumentsText}`] : []),
  ].join('\n\n')
  return {
    ...task,
    description,
    metadata: {
      ...task.metadata,
      plannedStep: {
        action: step.action,
        id: step.id,
        arguments: argumentsValue,
      },
    },
  }
}

function uniqueDependencies(
  candidates: readonly string[],
  taskId: string,
  knownTaskIds: ReadonlySet<string>,
  diagnostics: CortexDiagnostic[],
): readonly string[] {
  const dependencies: string[] = []
  for (const dependency of candidates) {
    if (!knownTaskIds.has(dependency) || dependency === taskId) {
      diagnostics.push({
        code: 'hierarchy_plan_fallback',
        taskId,
        message: `Manager proposed unsafe dependency ${dependency} for task ${taskId}; ignoring it`,
      })
      continue
    }
    if (!dependencies.includes(dependency)) dependencies.push(dependency)
  }
  return dependencies
}

function taskWithAgent(task: CortexTask, agent: CortexAgent | undefined, dependencies: readonly string[]): CortexTask {
  const { agentId: _oldAgentId, dependencies: _oldDependencies, ...rest } = task
  return {
    ...rest,
    ...(agent === undefined ? {} : { agentId: agent.id }),
    ...(dependencies.length === 0 ? {} : { dependencies }),
  }
}

function findAgent(agents: readonly CortexAgent[], reference: string | undefined): CortexAgent | undefined {
  if (!reference?.trim()) return undefined
  const normalized = reference.trim().toLowerCase()
  return agents.find(agent => [agent.id, agent.role, agent.name]
    .some(value => value?.trim().toLowerCase() === normalized))
}

function fallbackAgent(agents: readonly CortexAgent[], index: number): CortexAgent | undefined {
  return agents.length === 0 ? undefined : agents[index % agents.length]
}

function plannedObjective(tasks: readonly CortexTask[]): string {
  return tasks.map((task, index) => [
    `${index + 1}. ${task.description}`,
    `Expected output: ${task.expectedOutput}`,
  ].join('\n')).join('\n\n')
}

function normalizeResult(result: string | TaskExecutionResult): TaskExecutionResult {
  if (typeof result === 'string') return { output: result, metadata: {} }
  if (!result || typeof result.output !== 'string') {
    throw new TypeError('Cortex task execution must return a string or { output } result')
  }
  return { output: result.output, metadata: result.metadata ?? {} }
}

function skippedTaskOutput(task: CortexTask, timestamp: Date, reason: string): CortexTaskOutput {
  return {
    taskId: task.id,
    status: 'skipped',
    output: '',
    error: reason,
    startedAt: timestamp,
    completedAt: timestamp,
    durationMs: 0,
    metadata: {},
    ...(task.agentId === undefined ? {} : { agentId: task.agentId }),
  }
}

function failedTaskOutput(task: CortexTask, startedAt: Date, completedAt: Date, message: string): CortexTaskOutput {
  return {
    taskId: task.id,
    status: 'failed',
    output: '',
    error: message,
    startedAt,
    completedAt,
    durationMs: Math.max(0, completedAt.getTime() - startedAt.getTime()),
    metadata: {},
    ...(task.agentId === undefined ? {} : { agentId: task.agentId }),
  }
}

function assertManagerPlan(value: CortexManagerPlan): void {
  if (!value || !Array.isArray(value.assignments)) throw new TypeError('Manager plan must contain an assignments array')
  for (const assignment of value.assignments) {
    if (!assignment || typeof assignment.taskId !== 'string' || !assignment.taskId.trim()) {
      throw new TypeError('Every manager assignment requires a non-empty taskId')
    }
    if (assignment.agentId !== undefined && typeof assignment.agentId !== 'string') {
      throw new TypeError(`Manager assignment ${assignment.taskId} has an invalid agentId`)
    }
    if (assignment.dependencies !== undefined && !Array.isArray(assignment.dependencies)) {
      throw new TypeError(`Manager assignment ${assignment.taskId} has invalid dependencies`)
    }
  }
}

function assertManagerReview(value: CortexManagerReview): void {
  if (!value || typeof value.approved !== 'boolean') {
    throw new TypeError('Manager review must contain an approved boolean')
  }
  if (value.feedback !== undefined && typeof value.feedback !== 'string') {
    throw new TypeError('Manager review feedback must be text')
  }
  if (value.improvementsNeeded !== undefined && !Array.isArray(value.improvementsNeeded)) {
    throw new TypeError('Manager review improvementsNeeded must be an array')
  }
}

function assertUniqueAgents(agents: readonly CortexAgent[]): void {
  const ids = new Set<string>()
  for (const agent of agents) {
    if (!agent.id.trim()) throw new Error('Agent id cannot be empty')
    if (ids.has(agent.id)) throw new Error(`Agent ${agent.id} is already registered`)
    ids.add(agent.id)
  }
}

function assertProcessType(value: ProcessType): void {
  if (!Object.values(ProcessType).includes(value)) throw new Error(`Unknown Cortex process type: ${String(value)}`)
}

function validateMaxParallel(value: number | undefined): number | undefined {
  if (value === undefined) return undefined
  if (!Number.isInteger(value) || value < 1) throw new Error('maxParallel must be a positive integer')
  return value
}

function validateMaxCandidates(value: number | undefined): void {
  if (value === undefined || value === Number.POSITIVE_INFINITY) return
  if (!Number.isInteger(value) || value < 1) {
    throw new Error('maxCandidatesParallel must be a positive integer')
  }
}

function validateReviewAttempts(value: number | undefined): void {
  if (value !== undefined && (!Number.isInteger(value) || value < 1)) {
    throw new Error('maxReviewAttempts must be a positive integer')
  }
}

function throwIfCancelled(signal: AbortSignal | undefined): void {
  if (signal?.aborted) throw new CortexCancellationError(signal.reason)
}

async function awaitWithCancellation<T>(work: Promise<T>, signal: AbortSignal | undefined): Promise<T> {
  if (!signal) return work
  throwIfCancelled(signal)
  // The topology may settle after cancellation wins the race; observe that late
  // settlement so a detached rejection never surfaces as an unhandled rejection.
  void work.catch(() => undefined)
  return new Promise<T>((resolve, reject) => {
    const cancel = (): void => reject(new CortexCancellationError(signal.reason))
    signal.addEventListener('abort', cancel, { once: true })
    work.then(
      value => {
        signal.removeEventListener('abort', cancel)
        resolve(value)
      },
      error => {
        signal.removeEventListener('abort', cancel)
        reject(error)
      },
    )
  })
}

async function mapConcurrent<T>(
  items: readonly T[],
  maximum: number,
  run: (item: T, index: number) => Promise<void>,
): Promise<void> {
  const workers = Math.min(items.length, maximum)
  let next = 0
  const worker = async (): Promise<void> => {
    while (next < items.length) {
      const index = next
      next += 1
      const item = items[index]
      if (item !== undefined) await run(item, index)
    }
  }
  // Settle every worker so a late sibling rejection is observed instead of escaping unhandled.
  const outcomes = await Promise.allSettled(Array.from({ length: workers }, worker))
  const failure = outcomes.find((outcome): outcome is PromiseRejectedResult => outcome.status === 'rejected')
  if (failure !== undefined) throw failure.reason
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}
