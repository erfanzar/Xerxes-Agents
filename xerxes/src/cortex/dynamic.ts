// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { CortexOrchestrator, type CortexOrchestratorOptions, type CortexRunOptions, type CortexRunOutput } from './orchestrator.js'
import { materializeTaskPlan, type TaskCreationPlan, type TaskCreationRequest, type TaskCreationResult, type TaskCreator } from './taskCreator.js'
import type { CortexAgent, CortexTask } from './task.js'

export interface DynamicTaskOptions {
  readonly agentId?: string
  readonly expectedOutput?: string
  readonly id?: string
  readonly importance?: number
  readonly metadata?: Readonly<Record<string, unknown>>
}

/** Pure helpers for building one-off and chained Cortex tasks at runtime. */
export class DynamicTaskBuilder {
  static fromPrompt(prompt: string, options: DynamicTaskOptions = {}): CortexTask {
    const description = prompt.trim()
    if (!description) throw new Error('Prompt cannot be empty')
    return {
      id: options.id ?? 'dynamic-1',
      description,
      expectedOutput: options.expectedOutput ?? 'Complete the requested task',
      ...(options.agentId ? { agentId: options.agentId } : {}),
      ...(options.importance === undefined ? {} : { importance: options.importance }),
      ...(options.metadata ? { metadata: options.metadata } : {}),
    }
  }

  static chainPrompts(prompts: readonly string[], agents: readonly CortexAgent[] = []): CortexTask[] {
    return prompts.map((prompt, index) => {
      const prior = index > 0 ? `dynamic-${index}` : undefined
      const agent = agents.length ? agents[index % agents.length] : undefined
      return {
        id: `dynamic-${index + 1}`,
        description: prompt,
        expectedOutput: 'Complete the requested task and provide a detailed result',
        ...(prior ? { dependencies: [prior], contextTaskIds: [prior] } : {}),
        ...(agent ? { agentId: agent.id } : {}),
      }
    })
  }

  static fromTaskPlan(plan: TaskCreationPlan, agents: readonly CortexAgent[] = []): CortexTask[] {
    return materializeTaskPlan(plan, agents)
  }
}

export interface DynamicCortexOptions extends CortexOrchestratorOptions {
  readonly taskCreator?: TaskCreator
}

/** Small runtime facade for prompt-created tasks, backed by the same injected executor as CortexOrchestrator. */
export class DynamicCortex {
  readonly orchestrator: CortexOrchestrator
  private readonly creator: TaskCreator | undefined

  constructor(options: DynamicCortexOptions = {}) {
    this.orchestrator = new CortexOrchestrator(options)
    this.creator = options.taskCreator
  }

  get tasks(): readonly CortexTask[] {
    return this.orchestrator.taskList
  }

  async createTasks(request: TaskCreationRequest): Promise<TaskCreationResult> {
    if (!this.creator) throw new Error('DynamicCortex requires a TaskCreator for task decomposition')
    const result = await this.creator.create({ ...request, agents: request.agents ?? [...this.orchestrator.registeredAgents.values()] })
    this.orchestrator.setTasks(result.tasks)
    return result
  }

  executePrompt(prompt: string, taskOptions: DynamicTaskOptions = {}, runOptions: CortexRunOptions = {}): Promise<CortexRunOutput> {
    this.orchestrator.setTasks([DynamicTaskBuilder.fromPrompt(prompt, taskOptions)])
    return this.orchestrator.run(runOptions)
  }

  executePrompts(prompts: readonly string[], runOptions: CortexRunOptions = {}): Promise<CortexRunOutput> {
    this.orchestrator.setTasks(DynamicTaskBuilder.chainPrompts(prompts, [...this.orchestrator.registeredAgents.values()]))
    return this.orchestrator.run(runOptions)
  }
}
