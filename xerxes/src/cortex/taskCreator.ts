// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { dependencyLayers, stableIdentifier, type CortexAgent, type CortexTask } from './task.js'
import { directXmlChildren, parseXmlRoot, xmlBoolean, xmlChild, xmlIdentifiers, xmlText, type XmlElement } from './xml.js'

export type TaskPlanComplexity = 'complex' | 'medium' | 'simple'

export interface TaskDefinition {
  readonly agentRole?: string
  readonly contextNeeded: boolean
  readonly dependencies: readonly string[]
  readonly description: string
  readonly expectedOutput: string
  readonly humanFeedback: boolean
  readonly id: string
  readonly importance: number
  readonly toolsNeeded: readonly string[]
  readonly validationRequired: boolean
}

export interface TaskCreationPlan {
  readonly approach: string
  readonly complexity: TaskPlanComplexity
  readonly id: string
  readonly objective: string
  readonly sequential: boolean
  readonly tasks: readonly TaskDefinition[]
}

export interface TaskCreationRequest {
  readonly agents?: readonly CortexAgent[]
  readonly background?: string
  readonly constraints?: string
  readonly objective: string
}

export type TaskPlanGenerator = (request: TaskCreationRequest & { readonly prompt: string }) => string | Promise<string>

export interface TaskCreationResult {
  readonly error?: string
  readonly plan: TaskCreationPlan
  readonly tasks: readonly CortexTask[]
  readonly usedFallback: boolean
}

export interface TaskCreatorOptions {
  readonly generator?: TaskPlanGenerator
  readonly maxTasks?: number
}

/** Parse a constrained XML task plan without accepting JSON or unsafe XML declarations. */
export function parseTaskCreationPlan(response: string, fallbackObjective: string, maxTasks = 10): TaskCreationPlan {
  if (!Number.isInteger(maxTasks) || maxTasks < 1) throw new Error('maxTasks must be a positive integer')
  const root = parseXmlRoot(response, 'task_plan')
  const children = directXmlChildren(root.inner)
  const objective = xmlText(xmlChild(children, 'objective'), fallbackObjective) || fallbackObjective
  const taskElements = children.filter(child => child.name === 'task').slice(0, maxTasks)
  if (taskElements.length === 0) throw new Error('Task plan must contain at least one <task>')
  const plan: TaskCreationPlan = {
    id: stableIdentifier('task-plan', objective),
    objective,
    approach: xmlText(xmlChild(children, 'approach'), 'Direct execution') || 'Direct execution',
    complexity: parseComplexity(xmlText(xmlChild(children, 'complexity'), 'medium')),
    sequential: xmlBoolean(xmlChild(children, 'sequential'), true),
    tasks: taskElements.map(parseTaskDefinition),
  }
  validateTaskCreationPlan(plan)
  return plan
}

/** Build a one-task plan that can run even if an injected planning agent fails. */
export function fallbackTaskCreationPlan(objective: string, background?: string): TaskCreationPlan {
  return {
    id: stableIdentifier('fallback-task-plan', objective),
    objective,
    approach: background?.trim() || 'Direct execution',
    complexity: 'simple',
    sequential: true,
    tasks: [{
      id: '1',
      description: objective,
      expectedOutput: 'Complete the objective successfully',
      dependencies: [],
      contextNeeded: false,
      toolsNeeded: [],
      importance: 1,
      validationRequired: false,
      humanFeedback: false,
    }],
  }
}

/** Validate task-plan dependencies and return stable ready layers. */
export function taskPlanDependencyLayers(plan: TaskCreationPlan): TaskDefinition[][] {
  return dependencyLayers(plan.tasks, task => task.id, task => task.dependencies, 'task definition')
}

export function validateTaskCreationPlan(plan: TaskCreationPlan): void {
  if (!plan.objective.trim()) throw new Error('Task plan objective cannot be empty')
  taskPlanDependencyLayers(plan)
}

/** Convert declarative task definitions into executable Cortex tasks with resolved dependency IDs. */
export function materializeTaskPlan(
  plan: TaskCreationPlan,
  agents: readonly CortexAgent[] = [],
  idPrefix = plan.id,
): CortexTask[] {
  validateTaskCreationPlan(plan)
  const taskIds = new Map(plan.tasks.map(task => [task.id, `${idPrefix}:${task.id}`]))
  return plan.tasks.map(definition => {
    const dependencies = definition.dependencies.map(dependency => {
      const id = taskIds.get(dependency)
      if (!id) throw new Error(`Task ${definition.id} depends on unknown task ${dependency}`)
      return id
    })
    const agent = findAgent(definition.agentRole, agents)
    return {
      id: taskIds.get(definition.id) ?? `${idPrefix}:${definition.id}`,
      description: definition.description,
      expectedOutput: definition.expectedOutput,
      importance: definition.importance,
      metadata: {
        toolsNeeded: [...definition.toolsNeeded],
        validationRequired: definition.validationRequired,
        humanFeedback: definition.humanFeedback,
      },
      ...(dependencies.length ? { dependencies } : {}),
      ...(definition.contextNeeded && dependencies.length ? { contextTaskIds: dependencies } : {}),
      ...(agent ? { agentId: agent.id } : {}),
    }
  })
}

/** Build the model-facing prompt while leaving execution and LLM calls entirely injected. */
export function taskCreationPrompt(request: TaskCreationRequest): string {
  const agents = request.agents?.map(agent => {
    const role = agent.role ?? agent.name ?? agent.id
    return `- ${role}: ${agent.description ?? 'Available executor'}`
  }).join('\n') ?? 'No agent descriptions supplied'
  return [
    'Create a task plan as XML only, using a <task_plan> root and <task id="..."> children.',
    `Objective: ${request.objective}`,
    `Background: ${request.background ?? 'Use a direct, practical approach.'}`,
    `Constraints: ${request.constraints ?? 'None provided.'}`,
    `Available agents:\n${agents}`,
    'Each task needs description, expected_output, dependencies, and optional agent_role.',
  ].join('\n\n')
}

/** Model-agnostic task creator driven by an injected planner function. */
export class TaskCreator {
  private readonly generator: TaskPlanGenerator | undefined
  private readonly maxTasks: number

  constructor(options: TaskCreatorOptions = {}) {
    this.generator = options.generator
    this.maxTasks = options.maxTasks ?? 10
    if (!Number.isInteger(this.maxTasks) || this.maxTasks < 1) throw new Error('maxTasks must be a positive integer')
  }

  async create(request: TaskCreationRequest): Promise<TaskCreationResult> {
    if (!this.generator) {
      const plan = fallbackTaskCreationPlan(request.objective, request.background)
      return {
        plan,
        tasks: materializeTaskPlan(plan, request.agents),
        usedFallback: true,
        error: 'No task plan generator was configured',
      }
    }
    try {
      const response = await this.generator({ ...request, prompt: taskCreationPrompt(request) })
      const plan = parseTaskCreationPlan(response, request.objective, this.maxTasks)
      return { plan, tasks: materializeTaskPlan(plan, request.agents), usedFallback: false }
    } catch (error) {
      const plan = fallbackTaskCreationPlan(request.objective, request.background)
      return {
        plan,
        tasks: materializeTaskPlan(plan, request.agents),
        usedFallback: true,
        error: errorMessage(error),
      }
    }
  }
}

function parseTaskDefinition(element: XmlElement): TaskDefinition {
  const id = element.attributes.id?.trim()
  if (!id) throw new Error('Every <task> requires a non-empty id attribute')
  const children = directXmlChildren(element.inner)
  const description = xmlText(xmlChild(children, 'description'))
  if (!description) throw new Error(`Task ${id} requires a description`)
  const agentRole = xmlText(xmlChild(children, 'agent_role'))
  return {
    id,
    description,
    expectedOutput: xmlText(xmlChild(children, 'expected_output'), 'Complete the task') || 'Complete the task',
    dependencies: xmlIdentifiers(xmlChild(children, 'dependencies')),
    contextNeeded: xmlBoolean(xmlChild(children, 'context_needed'), false),
    toolsNeeded: xmlIdentifiers(xmlChild(children, 'tools_needed')),
    importance: parseImportance(xmlText(xmlChild(children, 'importance'), '0.5')),
    validationRequired: xmlBoolean(xmlChild(children, 'validation_required'), false),
    humanFeedback: xmlBoolean(xmlChild(children, 'human_feedback'), false),
    ...(agentRole ? { agentRole } : {}),
  }
}

function parseComplexity(value: string): TaskPlanComplexity {
  const normalized = value.trim().toLowerCase()
  return normalized === 'simple' || normalized === 'medium' || normalized === 'complex' ? normalized : 'medium'
}

function parseImportance(value: string): number {
  const parsed = Number(value)
  if (!Number.isFinite(parsed)) return 0.5
  return Math.min(1, Math.max(0, parsed))
}

function findAgent(role: string | undefined, agents: readonly CortexAgent[]): CortexAgent | undefined {
  if (!role) return agents[0]
  const normalized = role.trim().toLowerCase()
  return agents.find(agent => [agent.id, agent.role, agent.name].some(value => value?.trim().toLowerCase() === normalized)) ?? agents[0]
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}
