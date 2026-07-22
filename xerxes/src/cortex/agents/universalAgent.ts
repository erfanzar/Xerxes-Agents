// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { AgentError } from '../../core/errors.js'
import type { ToolExecutor } from '../../executors/toolRegistry.js'
import type { LlmClient, TokenUsage } from '../../llms/client.js'
import { createAgentState } from '../../streaming/events.js'
import { runTurn } from '../../streaming/loop.js'
import {
  DEFAULT_PERMISSION_MODE,
  type PermissionBroker,
  type PermissionMode,
  type ToolPolicy,
} from '../../streaming/permissions.js'
import type { ToolDefinition } from '../../types/toolCalls.js'
import {
  TaskCreator,
  materializeTaskPlan,
  type TaskCreationResult,
} from '../taskCreator.js'
import type {
  CortexAgent,
  CortexTask,
  TaskExecutionContext,
  TaskExecutionResult,
} from '../task.js'
import { CortexMemory } from './memoryIntegration.js'

export const UNIVERSAL_AGENT_ID = 'universal'
export const UNIVERSAL_AGENT_ROLE = 'Universal Task Executor'
export const UNIVERSAL_AGENT_GOAL = 'Execute bounded tasks through explicitly configured native LLM, tool, and '
  + 'delegation ports.'
export const UNIVERSAL_AGENT_BACKSTORY = [
  'You are a versatile Cortex agent operating only through real native capabilities supplied by the host.',
  'You can use an injected LLM, explicitly registered tools, task context, and explicit delegation ports.',
  'You do not claim filesystem, shell, web, or code-execution work unless a supplied tool actually performed it.',
].join(' ')
export const UNIVERSAL_AGENT_INSTRUCTIONS = [
  'Use only the supplied tools.',
  'When calling a tool, provide valid JSON arguments for every required field.',
  'Report concrete results or blockers; do not invent completed work.',
].join(' ')

const BASE_CAPABILITIES = [
  'General task execution through an injected LLM',
  'Structured task context and expected-output handling',
] as const

export interface UniversalAgentDelegationRequest {
  readonly context: string
  readonly inputs: Readonly<Record<string, unknown>>
  readonly sourceAgentId: string
  readonly targetAgent?: string
  readonly task: CortexTask
}

/** Explicit host boundary for a caller that wants to hand work to another agent. */
export interface UniversalAgentDelegationPort {
  delegate(request: UniversalAgentDelegationRequest, signal?: AbortSignal): Promise<TaskExecutionResult | string>
}

export interface UniversalAgentOptions {
  readonly allowDelegation?: boolean
  readonly delegation?: UniversalAgentDelegationPort
  readonly id?: string
  readonly llm: LlmClient
  readonly maxTokens?: number
  readonly maxToolTurns?: number
  readonly memory?: CortexMemory
  readonly model: string
  readonly permissionBroker?: PermissionBroker
  readonly permissionMode?: PermissionMode
  readonly policy?: ToolPolicy
  /** Explicit retry delays; the default is fail-fast rather than hidden waits. */
  readonly retryDelays?: readonly number[]
  readonly systemPrompt?: string
  readonly temperature?: number
  readonly toolExecutor?: ToolExecutor
  readonly tools?: readonly ToolDefinition[]
  readonly topP?: number
}

export interface UniversalAgentExecutionMetadata {
  readonly deniedToolCalls: number
  readonly inputTokens: number
  readonly model: string
  readonly outputTokens: number
  readonly toolCalls: number
  readonly toolFailures: number
}

export interface UniversalTaskCreationRequest {
  readonly background?: string
  readonly constraints?: string
  readonly prompt: string
  readonly specializedAgents?: readonly CortexAgent[]
}

export interface UniversalTaskCreationResult extends TaskCreationResult {
  readonly tasks: readonly CortexTask[]
}

export interface UniversalTaskCreatorOptions {
  readonly taskCreator?: TaskCreator
  readonly universalAgent: UniversalAgent
}

/**
 * General-purpose Cortex executor with no implicit process, filesystem, tool,
 * or provider access. It runs the existing native streaming loop only after
 * callers explicitly provide its LLM and any tool/delegation ports.
 */
export class UniversalAgent implements CortexAgent {
  readonly allowDelegation: boolean
  readonly backstory = UNIVERSAL_AGENT_BACKSTORY
  readonly capabilities: readonly string[]
  readonly description: string
  readonly goal = UNIVERSAL_AGENT_GOAL
  readonly id: string
  readonly instructions = UNIVERSAL_AGENT_INSTRUCTIONS
  readonly name: string
  readonly role = UNIVERSAL_AGENT_ROLE
  readonly model: string
  readonly tools: readonly ToolDefinition[]
  private readonly delegation: UniversalAgentDelegationPort | undefined
  private readonly llm: LlmClient
  private readonly maxTokens: number
  private readonly maxToolTurns: number | undefined
  private readonly memory: CortexMemory | undefined
  private readonly permissionBroker: PermissionBroker | undefined
  private readonly permissionMode: PermissionMode
  private readonly policy: ToolPolicy | undefined
  private readonly retryDelays: readonly number[]
  private readonly systemPrompt: string
  private readonly temperature: number | undefined
  private readonly toolExecutor: ToolExecutor | undefined
  private readonly topP: number | undefined

  constructor(options: UniversalAgentOptions) {
    this.id = requiredText(options.id ?? UNIVERSAL_AGENT_ID, 'id')
    this.name = UNIVERSAL_AGENT_ROLE
    this.description = 'Executes a bounded Cortex task through explicitly configured native ports.'
    this.llm = requireLlm(options.llm)
    this.model = requiredText(options.model, 'model')
    this.tools = copyTools(options.tools ?? [])
    this.toolExecutor = options.toolExecutor
    if (this.tools.length > 0 && !this.toolExecutor) {
      throw new Error('UniversalAgent requires toolExecutor when tools are configured')
    }
    this.allowDelegation = options.allowDelegation ?? true
    this.delegation = options.delegation
    this.memory = options.memory
    this.permissionBroker = options.permissionBroker
    this.permissionMode = options.permissionMode ?? DEFAULT_PERMISSION_MODE
    this.policy = options.policy
    this.retryDelays = copyRetryDelays(options.retryDelays ?? [])
    this.maxTokens = positiveInteger(options.maxTokens ?? 4_096, 'maxTokens')
    this.maxToolTurns = options.maxToolTurns === undefined
      ? undefined
      : positiveInteger(options.maxToolTurns, 'maxToolTurns')
    this.temperature = optionalFinite(options.temperature, 'temperature')
    this.topP = optionalFinite(options.topP, 'topP')
    this.systemPrompt = options.systemPrompt?.trim() || universalSystemPrompt(this.tools)
    this.capabilities = Object.freeze([
      ...BASE_CAPABILITIES,
      ...(this.tools.length ? ['Tool execution through injected handlers'] : []),
      ...(this.memory ? ['Cortex memory context and completed-result persistence'] : []),
      ...(this.allowDelegation && this.delegation ? ['Explicit delegated-task handoff'] : []),
    ])
  }

  /** Execute one Cortex task through the real Bun streaming loop. */
  async execute(context: TaskExecutionContext): Promise<TaskExecutionResult> {
    const prompt = this.taskPrompt(context)
    const state = createAgentState()
    const output: string[] = []
    let providerFailure: string | undefined
    let usage: TokenUsage = { inputTokens: 0, outputTokens: 0 }
    let toolCalls = 0
    let toolFailures = 0
    let deniedToolCalls = 0
    let unconfiguredTool: string | undefined
    const configuredToolNames = new Set(this.tools.map(tool => tool.function.name))

    for await (const event of runTurn({
      agentId: this.id,
      ...(this.maxToolTurns === undefined ? {} : { maxToolTurns: this.maxToolTurns }),
      maxTokens: this.maxTokens,
      model: this.model,
      permissionMode: this.permissionMode,
      state,
      systemPrompt: this.systemPrompt,
      ...(this.temperature === undefined ? {} : { temperature: this.temperature }),
      ...(this.topP === undefined ? {} : { topP: this.topP }),
      ...(this.tools.length ? { tools: this.tools } : {}),
      userMessage: prompt,
    }, {
      llm: this.llm,
      retryDelays: this.retryDelays,
      onUnconfiguredToolCalls: calls => {
        unconfiguredTool ??= calls[0]?.function.name
        return 'stop'
      },
      ...(this.permissionBroker === undefined ? {} : { permissionBroker: this.permissionBroker }),
      ...(this.policy === undefined ? {} : { policy: this.policy }),
      ...(this.toolExecutor === undefined ? {} : { toolExecutor: this.toolExecutor }),
    }, context.signal)) {
      if (event.type === 'text') {
        output.push(event.text)
      } else if (event.type === 'provider_retry' && event.final) {
        providerFailure = event.error
      } else if (event.type === 'tool_end') {
        if (!configuredToolNames.has(event.result.name)) unconfiguredTool = event.result.name
        if (!event.result.permitted) deniedToolCalls += 1
        if (event.result.result.startsWith('Tool execution failed:')) toolFailures += 1
      } else if (event.type === 'turn_done') {
        usage = event.usage
        toolCalls = event.toolCallsCount
      }
    }

    if (providerFailure) throw new AgentError(this.id, `LLM execution failed: ${providerFailure}`)
    if (unconfiguredTool) throw new AgentError(this.id, `LLM requested unconfigured tool: ${unconfiguredTool}`)
    const text = output.join('')
    if (!text.trim()) throw new AgentError(this.id, 'LLM completed without a text result')

    const metadata: UniversalAgentExecutionMetadata = {
      model: this.model,
      inputTokens: usage.inputTokens,
      outputTokens: usage.outputTokens,
      toolCalls,
      toolFailures,
      deniedToolCalls,
    }
    try {
      this.memory?.saveTaskResult({
        taskDescription: context.task.description,
        result: text,
        agentRole: this.role,
        ...(context.task.importance === undefined ? {} : { importance: context.task.importance }),
        metadata: {
          task_id: context.task.id,
          expected_output: context.task.expectedOutput,
          ...metadata,
        },
      })
    } catch {
      // Memory persistence is non-critical; a throwing tier must not fail an otherwise successful task.
    }
    return { output: text, metadata: { ...metadata } }
  }

  /** Explicitly hand a task to a caller-provided delegate; no automatic fallback occurs. */
  async delegateTask(
    task: CortexTask,
    context: string,
    inputs: Readonly<Record<string, unknown>>,
    options: { readonly signal?: AbortSignal; readonly targetAgent?: string } = {},
  ): Promise<TaskExecutionResult> {
    if (!this.allowDelegation) throw new AgentError(this.id, 'delegation is disabled')
    if (!this.delegation) throw new AgentError(this.id, 'no delegation port is configured')
    const result = await this.delegation.delegate({
      task,
      context,
      inputs,
      sourceAgentId: this.id,
      ...(options.targetAgent ? { targetAgent: options.targetAgent } : {}),
    }, options.signal)
    const normalized = typeof result === 'string' ? { output: result } : result
    if (!normalized.output.trim()) throw new AgentError(this.id, 'delegate completed without a text result')
    return normalized
  }

  describeCapabilities(): string {
    return [
      'Universal Agent Capabilities:',
      '',
      ...this.capabilities.map(capability => `• ${capability}`),
      '',
      `Total Tools Available: ${this.tools.length}`,
    ].join('\n')
  }

  private taskPrompt(context: TaskExecutionContext): string {
    const memoryContext = this.memory?.buildContextForTask(context.task.description, { agentRole: this.role })
    const inputs = JSON.stringify(context.inputs, null, 2)
    return [
      `Task: ${context.task.description}`,
      `Expected output: ${context.task.expectedOutput}`,
      `Task inputs:\n${inputs}`,
      ...(context.context.trim() ? [`Dependency context:\n${context.context.trim()}`] : []),
      ...(memoryContext ? [`Relevant Cortex memory:\n${memoryContext}`] : []),
      'Produce the requested result. Do not claim tool work that did not occur.',
    ].join('\n\n')
  }
}

/**
 * Task-plan adapter that assigns declared specialized roles first and routes
 * every unassigned task to the supplied UniversalAgent. It creates plans but
 * deliberately does not execute them or register agents with an orchestrator.
 */
export class UniversalTaskCreator {
  readonly taskCreator: TaskCreator
  readonly universalAgent: UniversalAgent

  constructor(options: UniversalTaskCreatorOptions) {
    this.universalAgent = options.universalAgent
    this.taskCreator = options.taskCreator ?? new TaskCreator()
  }

  async createAndAssignTasks(request: UniversalTaskCreationRequest): Promise<UniversalTaskCreationResult> {
    const prompt = requiredText(request.prompt, 'prompt')
    const specializedAgents = [...(request.specializedAgents ?? [])]
    const created = await this.taskCreator.create({
      objective: prompt,
      ...(request.background === undefined ? {} : { background: request.background }),
      ...(request.constraints === undefined ? {} : { constraints: request.constraints }),
    })
    const materialized = materializeTaskPlan(created.plan)
    const tasks = materialized.map((task, index) => {
      const definition = created.plan.tasks[index]
      const specialized = definition ? selectSpecializedAgent(definition.agentRole, specializedAgents) : undefined
      return { ...task, agentId: specialized?.id ?? this.universalAgent.id }
    })
    return { ...created, tasks }
  }
}

function copyRetryDelays(delays: readonly number[]): readonly number[] {
  for (const delay of delays) {
    if (!Number.isFinite(delay) || delay < 0) throw new Error('retryDelays must contain non-negative finite values')
  }
  return Object.freeze([...delays])
}

function copyTools(tools: readonly ToolDefinition[]): readonly ToolDefinition[] {
  const names = new Set<string>()
  const copied = tools.map(tool => ({
    type: tool.type,
    function: {
      name: requiredText(tool.function.name, 'tool.function.name'),
      description: tool.function.description,
      parameters: tool.function.parameters,
    },
  } as ToolDefinition))
  for (const tool of copied) {
    const name = tool.function.name
    if (names.has(name)) throw new Error(`Duplicate UniversalAgent tool: ${name}`)
    names.add(name)
  }
  return Object.freeze(copied)
}

function optionalFinite(value: number | undefined, name: string): number | undefined {
  if (value === undefined) return undefined
  if (!Number.isFinite(value)) throw new Error(`${name} must be finite`)
  return value
}

function positiveInteger(value: number, name: string): number {
  if (!Number.isInteger(value) || value < 1) throw new Error(`${name} must be a positive integer`)
  return value
}

function requireLlm(value: LlmClient): LlmClient {
  if (!value || typeof value.stream !== 'function') throw new Error('UniversalAgent requires an LlmClient stream port')
  return value
}

function requiredText(value: string, name: string): string {
  const normalized = value.trim()
  if (!normalized) throw new Error(`${name} cannot be empty`)
  return normalized
}

function selectSpecializedAgent(role: string | undefined, agents: readonly CortexAgent[]): CortexAgent | undefined {
  const requested = role?.trim().toLowerCase()
  if (!requested) return undefined
  return agents.find(agent => {
    const candidate = agent.role?.trim().toLowerCase()
    return candidate !== undefined && (candidate.includes(requested) || requested.includes(candidate))
  })
}

function universalSystemPrompt(tools: readonly ToolDefinition[]): string {
  const toolNames = tools.length ? tools.map(tool => tool.function.name).join(', ') : 'none'
  return [
    'You are a general-purpose Cortex task executor.',
    `Available tools: ${toolNames}.`,
    UNIVERSAL_AGENT_INSTRUCTIONS,
  ].join('\n')
}
