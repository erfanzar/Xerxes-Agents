// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { AgentError, RateLimitError, XerxesTimeoutError } from '../../core/errors.js'
import type { ToolExecutor } from '../../executors/toolRegistry.js'
import type { LlmClient, TokenUsage } from '../../llms/client.js'
import { createAgentState, type StreamEvent } from '../../streaming/events.js'
import { runTurn } from '../../streaming/loop.js'
import {
  DEFAULT_PERMISSION_MODE,
  type PermissionBroker,
  type PermissionMode,
  type ToolPolicy,
} from '../../streaming/permissions.js'
import type { AgentCapability } from '../../agents/orchestrator.js'
import type { ChatMessage, MessageContent } from '../../types/messages.js'
import type { JsonSchema, JsonValue, ToolDefinition } from '../../types/toolCalls.js'
import { AutoCompactAgent } from '../../agents/autoCompactAgent.js'
import { interpolateInputs as interpolateTemplateInputs } from '../core/stringUtils.js'
import { CortexTool } from '../core/tool.js'
import type {
  CortexAgent as CortexTaskAgent,
  CortexTask,
  TaskExecutionContext,
  TaskExecutionResult,
} from '../task.js'
import { CortexMemory } from './memoryIntegration.js'

export type CortexAgentOutputFormat = 'json' | 'xml'
export type CortexAgentRateLimitMode = 'error' | 'wait'
export type CortexAgentDelay = (milliseconds: number, signal?: AbortSignal) => Promise<void>
export type CortexAgentStreamCallback = (event: StreamEvent) => void | Promise<void>

export interface CortexAgentStep {
  readonly agent: string
  readonly error?: string
  readonly executionCount: number
  readonly executionTimeMs?: number
  readonly iteration?: number
  readonly resultLength?: number
  readonly step: 'execution_complete' | 'execution_error' | 'execution_start' | 'retry'
  readonly task: string
}

export type CortexAgentStepCallback = (step: CortexAgentStep) => void | Promise<void>

export interface CortexAgentExecutionOptions {
  readonly context?: string
  readonly expectedOutput?: string
  readonly inputs?: Readonly<Record<string, unknown>>
  /** Native JSON Schema guidance. TypeScript does not accept a Python/Pydantic class here. */
  readonly outputSchema?: JsonSchema
  readonly signal?: AbortSignal
  readonly streamCallback?: CortexAgentStreamCallback
}

export interface CortexAgentDelegationRequest {
  readonly context: string
  readonly inputs: Readonly<Record<string, unknown>>
  readonly sourceAgentId: string
  readonly sourceRole: string
  readonly targetAgentId?: string
  readonly task: CortexTask
}

/** Explicit host-owned delegation boundary; this class never discovers or invokes a peer by itself. */
export interface CortexAgentDelegationPort {
  delegate(
    request: CortexAgentDelegationRequest,
    signal?: AbortSignal,
  ): Promise<TaskExecutionResult | string> | TaskExecutionResult | string
}

export interface CortexAgentDelegateOptions {
  readonly context?: string
  readonly expectedOutput?: string
  readonly inputs?: Readonly<Record<string, unknown>>
  readonly signal?: AbortSignal
  readonly targetAgentId?: string
}

export interface CortexAgentExecutionMetadata {
  readonly attempts: number
  readonly deniedToolCalls: number
  readonly executionTimeMs: number
  readonly inputTokens: number
  readonly model: string
  readonly outputTokens: number
  readonly toolCalls: number
  readonly toolFailures: number
}

export interface CortexAgentExecutionStats {
  readonly averageExecutionTimeMs: number
  readonly maxExecutionTimeMs: number
  readonly minExecutionTimeMs: number
  readonly recentExecutionTimesMs: readonly number[]
  readonly timesExecuted: number
  readonly totalExecutionTimeMs: number
}

export interface CortexAgentRateLimitStatus {
  readonly currentRequests: number
  readonly maxRpm?: number
  readonly rateLimited: boolean
  readonly requestsRemaining?: number
}

export interface CortexAgentOptions {
  readonly allowDelegation?: boolean
  /** Optional supplied compactor. It does nothing unless the host configured a real compactor on it. */
  readonly autoCompaction?: AutoCompactAgent
  readonly autoFormatGuidance?: boolean
  readonly backstory: string
  readonly capabilities?: readonly AgentCapability[]
  readonly config?: Readonly<Record<string, unknown>>
  readonly delay?: CortexAgentDelay
  readonly delegation?: CortexAgentDelegationPort
  readonly description?: string
  /** Delays between terminal provider failures. There is never an implicit retry sleep. */
  readonly executionRetryDelays?: readonly number[]
  readonly goal: string
  readonly id?: string
  readonly instructions?: string
  readonly knowledge?: Readonly<Record<string, unknown>>
  readonly knowledgeSources?: readonly string[]
  readonly llm: LlmClient
  readonly maxExecutionTime?: number
  readonly maxIterations?: number
  readonly maxRpm?: number
  readonly maxTokens?: number
  readonly maxToolTurns?: number
  readonly maxDelegations?: number
  readonly memory?: CortexMemory
  readonly memoryEnabled?: boolean
  readonly model: string
  readonly now?: () => number
  readonly onStepCallbackError?: (error: unknown, step: CortexAgentStep) => void
  readonly outputFormatPreference?: CortexAgentOutputFormat
  readonly permissionBroker?: PermissionBroker
  readonly permissionMode?: PermissionMode
  readonly policy?: ToolPolicy
  readonly rateLimitMode?: CortexAgentRateLimitMode
  /** Passed to the existing streaming loop for retrying a single provider stream attempt. */
  readonly retryDelays?: readonly number[]
  readonly role: string
  readonly stepCallback?: CortexAgentStepCallback
  readonly temperature?: number
  readonly toolExecutor?: ToolExecutor
  readonly tools?: readonly (CortexTool | ToolDefinition)[]
  readonly topP?: number
  readonly verbose?: boolean
}

export interface CortexTaskPromptInput {
  readonly context?: string
  readonly expectedOutput?: string
  readonly formatGuidance?: string
  readonly inputs?: Readonly<Record<string, unknown>>
  readonly knowledgeContext?: string
  readonly memoryContext?: string
  readonly taskDescription: string
}

interface ExecutionRun {
  readonly cleanup: () => void
  readonly signal: AbortSignal
  readonly timedOut: () => boolean
}

interface TurnAttemptResult {
  readonly deniedToolCalls: number
  readonly inputTokens: number
  readonly messages: readonly ChatMessage[]
  readonly output: string
  readonly outputTokens: number
  readonly toolCalls: number
  readonly toolFailures: number
}

class ProviderAttemptFailure extends Error {
  constructor(readonly providerError: string) {
    super(providerError)
    this.name = 'ProviderAttemptFailure'
  }
}

/**
 * Native Cortex task agent backed only by injected LLM, tool, memory, and
 * delegation ports. It deliberately has no Xerxes/Python bridge, ambient
 * provider discovery, filesystem access, or automatic peer selection.
 */
export class CortexAgent implements CortexTaskAgent {
  readonly allowDelegation: boolean
  readonly autoFormatGuidance: boolean
  backstory: string
  readonly capabilities: readonly AgentCapability[]
  readonly config: Record<string, unknown>
  readonly description: string
  goal: string
  readonly id: string
  instructions: string
  readonly knowledge: Record<string, unknown>
  readonly knowledgeSources: string[]
  readonly maxExecutionTime: number | undefined
  readonly maxIterations: number
  readonly maxRpm: number | undefined
  readonly maxTokens: number
  readonly maxToolTurns: number | undefined
  readonly memory: CortexMemory | undefined
  readonly memoryEnabled: boolean
  readonly model: string
  readonly name: string
  readonly outputFormatPreference: CortexAgentOutputFormat
  role: string
  readonly temperature: number | undefined
  readonly tools: readonly ToolDefinition[]
  readonly topP: number | undefined
  readonly verbose: boolean

  private activeDelegations = 0
  private conversationHistory: ChatMessage[] = []
  private readonly autoCompaction: AutoCompactAgent | undefined
  private readonly delay: CortexAgentDelay
  private readonly delegation: CortexAgentDelegationPort | undefined
  private readonly executionRetryDelays: readonly number[]
  private readonly llm: LlmClient
  private readonly maxDelegations: number
  private readonly now: () => number
  private readonly onStepCallbackError: (error: unknown, step: CortexAgentStep) => void
  private readonly permissionBroker: PermissionBroker | undefined
  private readonly permissionMode: PermissionMode
  private readonly policy: ToolPolicy | undefined
  private readonly rateLimitMode: CortexAgentRateLimitMode
  private readonly retryDelays: readonly number[]
  private readonly toolExecutor: ToolExecutor | undefined
  private readonly executionTimes: number[] = []
  private readonly rpmRequests: number[] = []
  private originalBackstory: string | undefined
  private originalGoal: string | undefined
  private originalInstructions: string | undefined
  private originalRole: string | undefined
  private stepCallback: CortexAgentStepCallback | undefined
  private timesExecuted = 0

  constructor(options: CortexAgentOptions) {
    this.role = requiredText(options.role, 'role')
    this.goal = requiredText(options.goal, 'goal')
    this.backstory = requiredText(options.backstory, 'backstory')
    this.id = options.id === undefined ? agentId(this.role) : requiredText(options.id, 'id')
    this.name = this.role
    this.description = options.description?.trim() || this.goal
    this.llm = requireLlm(options.llm)
    this.model = requiredText(options.model, 'model')
    this.tools = normalizeTools(options.tools ?? [])
    this.toolExecutor = options.toolExecutor
    if (this.tools.length > 0 && this.toolExecutor === undefined) {
      throw new TypeError('CortexAgent requires toolExecutor when tools are configured')
    }
    this.allowDelegation = options.allowDelegation ?? false
    this.delegation = options.delegation
    this.maxDelegations = positiveInteger(options.maxDelegations ?? 3, 'maxDelegations')
    this.memory = options.memory
    this.memoryEnabled = options.memoryEnabled ?? true
    this.autoCompaction = options.autoCompaction
    this.autoFormatGuidance = options.autoFormatGuidance ?? true
    this.outputFormatPreference = options.outputFormatPreference ?? 'xml'
    this.maxIterations = positiveInteger(options.maxIterations ?? 10, 'maxIterations')
    this.maxTokens = positiveInteger(options.maxTokens ?? 2_048, 'maxTokens')
    this.maxToolTurns = options.maxToolTurns === undefined
      ? undefined
      : positiveInteger(options.maxToolTurns, 'maxToolTurns')
    this.maxExecutionTime = options.maxExecutionTime === undefined
      ? undefined
      : positiveFinite(options.maxExecutionTime, 'maxExecutionTime')
    this.maxRpm = options.maxRpm === undefined ? undefined : positiveInteger(options.maxRpm, 'maxRpm')
    this.rateLimitMode = options.rateLimitMode ?? 'error'
    this.delay = options.delay ?? abortableDelay
    this.retryDelays = copyDelays(options.retryDelays ?? [], 'retryDelays')
    this.executionRetryDelays = copyDelays(options.executionRetryDelays ?? [], 'executionRetryDelays')
    this.permissionBroker = options.permissionBroker
    this.permissionMode = options.permissionMode ?? DEFAULT_PERMISSION_MODE
    this.policy = options.policy
    this.temperature = optionalFinite(options.temperature, 'temperature')
    this.topP = optionalFinite(options.topP, 'topP')
    this.verbose = options.verbose ?? true
    this.now = options.now ?? Date.now
    this.stepCallback = options.stepCallback
    this.onStepCallbackError = options.onStepCallbackError ?? (() => undefined)
    this.capabilities = Object.freeze([...(options.capabilities ?? [])])
    this.config = { ...(options.config ?? {}) }
    this.knowledge = { ...(options.knowledge ?? {}) }
    this.knowledgeSources = uniqueTexts(options.knowledgeSources ?? [], 'knowledgeSources')
    this.instructions = options.instructions?.trim() || renderCortexAgentSystemPrompt({
      role: this.role,
      goal: this.goal,
      backstory: this.backstory,
      tools: this.tools,
    })
    // CortexOrchestrator deliberately stores an executor function, so retain
    // the instance receiver when this overload is read off the agent object.
    this.execute = this.execute.bind(this)
  }

  /** Execute a raw task description and resolve its real textual result. */
  execute(taskDescription: string, options?: CortexAgentExecutionOptions): Promise<string>
  /** Execute a Cortex task contract so this class can be registered directly with CortexOrchestrator. */
  execute(context: TaskExecutionContext, options?: CortexAgentExecutionOptions): Promise<TaskExecutionResult>
  async execute(
    input: string | TaskExecutionContext,
    options: CortexAgentExecutionOptions = {},
  ): Promise<string | TaskExecutionResult> {
    if (typeof input === 'string') {
      const result = await this.executeDescription(input, options)
      return result.output
    }
    return this.executeTask(input, options)
  }

  /** Explicit task-contract entrypoint, useful where overload inference is inconvenient. */
  async executeTask(
    context: TaskExecutionContext,
    options: CortexAgentExecutionOptions = {},
  ): Promise<TaskExecutionResult> {
    const inputs = { ...context.inputs, ...(options.inputs ?? {}) }
    const combinedContext = joinContexts(context.context, options.context)
    const outputSchema = options.outputSchema ?? outputSchemaFromMetadata(context.task.metadata)
    const result = await this.executeDescription(context.task.description, {
      ...options,
      expectedOutput: options.expectedOutput ?? context.task.expectedOutput,
      inputs,
      ...(combinedContext === undefined ? {} : { context: combinedContext }),
      ...(outputSchema === undefined ? {} : { outputSchema }),
    }, context.task)
    return { output: result.output, metadata: { ...result.metadata } }
  }

  /** Execute with caller-observed stream events; no background-thread facade is needed in Bun. */
  async executeStream(
    taskDescription: string,
    callback: CortexAgentStreamCallback,
    options: Omit<CortexAgentExecutionOptions, 'streamCallback'> = {},
  ): Promise<string> {
    return this.execute(taskDescription, { ...options, streamCallback: callback })
  }

  /**
   * Delegate only through a caller-supplied port. This never infers a peer,
   * falls back to a Python object, or launches an agent process.
   */
  async delegateTask(
    task: CortexTask | string,
    options: CortexAgentDelegateOptions = {},
  ): Promise<string> {
    if (!this.allowDelegation) throw new AgentError(this.id, 'delegation is disabled')
    if (this.delegation === undefined) throw new AgentError(this.id, 'no delegation port is configured')
    if (this.activeDelegations >= this.maxDelegations) {
      throw new AgentError(this.id, `delegation depth exceeds configured maximum of ${this.maxDelegations}`)
    }
    const signal = options.signal
    throwIfAborted(signal, this.id)
    const delegatedTask = typeof task === 'string'
      ? {
          id: `${this.id}-delegation-${this.activeDelegations + 1}`,
          description: requiredText(task, 'taskDescription'),
          expectedOutput: options.expectedOutput?.trim() || 'Complete the delegated task',
        }
      : task
    this.activeDelegations += 1
    try {
      const result = await this.delegation.delegate({
        task: delegatedTask,
        context: options.context?.trim() ?? '',
        inputs: { ...(options.inputs ?? {}) },
        sourceAgentId: this.id,
        sourceRole: this.role,
        ...(options.targetAgentId?.trim() ? { targetAgentId: options.targetAgentId.trim() } : {}),
      }, signal)
      throwIfAborted(signal, this.id)
      const output = typeof result === 'string' ? result : result.output
      if (!output.trim()) throw new AgentError(this.id, 'delegate completed without a text result')
      if (this.memoryEnabled && this.memory) {
        this.memory.saveAgentInteraction({
          agentRole: this.role,
          action: 'delegated_task',
          content: `Delegated task: ${delegatedTask.description.slice(0, 512)}`,
          importance: 0.6,
        })
      }
      return output
    } finally {
      this.activeDelegations -= 1
    }
  }

  /** Substitute Cortex template values in role, goal, backstory, and instructions. */
  interpolateInputs(inputs: Readonly<Record<string, unknown>>): void {
    this.originalRole ??= this.role
    this.originalGoal ??= this.goal
    this.originalBackstory ??= this.backstory
    this.originalInstructions ??= this.instructions
    this.role = interpolateTemplateInputs(this.originalRole, inputs)
    this.goal = interpolateTemplateInputs(this.originalGoal, inputs)
    this.backstory = interpolateTemplateInputs(this.originalBackstory, inputs)
    this.instructions = interpolateTemplateInputs(this.originalInstructions, inputs)
  }

  /** Return compaction statistics when a caller supplied an AutoCompactAgent. */
  getCompactionStats(): Readonly<Record<string, number | string>> | undefined {
    return this.autoCompaction?.getStatistics()
  }

  /** Return context-usage thresholds when auto compaction was configured. */
  checkContextUsage(): Readonly<Record<string, number>> | undefined {
    return this.autoCompaction?.checkUsage()
  }

  getExecutionStats(): CortexAgentExecutionStats {
    const totalExecutionTimeMs = this.executionTimes.reduce((total, value) => total + value, 0)
    return Object.freeze({
      timesExecuted: this.timesExecuted,
      totalExecutionTimeMs,
      averageExecutionTimeMs: this.executionTimes.length ? totalExecutionTimeMs / this.executionTimes.length : 0,
      minExecutionTimeMs: this.executionTimes.length ? Math.min(...this.executionTimes) : 0,
      maxExecutionTimeMs: this.executionTimes.length ? Math.max(...this.executionTimes) : 0,
      recentExecutionTimesMs: Object.freeze(this.executionTimes.slice(-5)),
    })
  }

  resetStats(): void {
    this.timesExecuted = 0
    this.executionTimes.splice(0)
    this.rpmRequests.splice(0)
    this.activeDelegations = 0
  }

  isRateLimited(): boolean {
    this.pruneRpmRequests()
    return this.maxRpm !== undefined && this.rpmRequests.length >= this.maxRpm
  }

  getRateLimitStatus(): CortexAgentRateLimitStatus {
    this.pruneRpmRequests()
    if (this.maxRpm === undefined) {
      return Object.freeze({ currentRequests: 0, rateLimited: false })
    }
    const currentRequests = this.rpmRequests.length
    return Object.freeze({
      currentRequests,
      maxRpm: this.maxRpm,
      rateLimited: currentRequests >= this.maxRpm,
      requestsRemaining: Math.max(0, this.maxRpm - currentRequests),
    })
  }

  addKnowledge(key: string, value: unknown): void {
    this.knowledge[requiredText(key, 'knowledge key')] = value
  }

  addKnowledgeSource(source: string): void {
    const normalized = requiredText(source, 'knowledge source')
    if (!this.knowledgeSources.includes(normalized)) this.knowledgeSources.push(normalized)
  }

  updateConfig(key: string, value: unknown): void {
    this.config[requiredText(key, 'config key')] = value
  }

  getConfig<T = unknown>(key: string, fallback?: T): unknown | T {
    return Object.hasOwn(this.config, key) ? this.config[key] : fallback
  }

  setStepCallback(callback: CortexAgentStepCallback | undefined): void {
    this.stepCallback = callback
  }

  private async executeDescription(
    taskDescription: string,
    options: CortexAgentExecutionOptions,
    task: CortexTask | undefined = undefined,
  ): Promise<{ readonly metadata: CortexAgentExecutionMetadata; readonly output: string }> {
    const description = requiredText(taskDescription, 'taskDescription')
    const run = executionRun(options.signal, this.maxExecutionTime, this.id)
    let startedAt: number | undefined
    try {
      await this.reserveRpmSlot(run.signal)
      throwIfExecutionAborted(run, this.id, this.maxExecutionTime)
      startedAt = this.now()
      this.timesExecuted += 1
      await this.emitStep({
        agent: this.role,
        executionCount: this.timesExecuted,
        step: 'execution_start',
        task: description,
      })

      const prompt = this.taskPrompt(description, options)
      const history = await this.historyForAttempt()
      const systemPrompt = this.instructions
      let attempt = 0
      while (attempt < this.maxIterations) {
        attempt += 1
        try {
          const result = await this.runTurnAttempt(prompt, systemPrompt, history, options.streamCallback, run.signal)
          throwIfExecutionAborted(run, this.id, this.maxExecutionTime)
          const executionTimeMs = elapsed(this.now(), startedAt)
          const metadata: CortexAgentExecutionMetadata = {
            attempts: attempt,
            model: this.model,
            executionTimeMs,
            inputTokens: result.inputTokens,
            outputTokens: result.outputTokens,
            toolCalls: result.toolCalls,
            toolFailures: result.toolFailures,
            deniedToolCalls: result.deniedToolCalls,
          }
          this.conversationHistory = result.messages.filter(message => message.role !== 'system')
          this.executionTimes.push(executionTimeMs)
          this.persistResult(description, result.output, metadata, task)
          await this.emitStep({
            agent: this.role,
            executionCount: this.timesExecuted,
            executionTimeMs,
            resultLength: result.output.length,
            step: 'execution_complete',
            task: description,
          })
          return { output: result.output, metadata }
        } catch (error) {
          throwIfExecutionAborted(run, this.id, this.maxExecutionTime)
          if (!(error instanceof ProviderAttemptFailure) || attempt >= this.maxIterations) {
            if (error instanceof ProviderAttemptFailure) {
              throw new AgentError(this.id, `LLM execution failed: ${error.providerError}`)
            }
            throw error
          }
          await this.emitStep({
            agent: this.role,
            error: error.providerError,
            executionCount: this.timesExecuted,
            iteration: attempt,
            step: 'retry',
            task: description,
          })
          const retryDelay = delayAt(this.executionRetryDelays, attempt - 1)
          if (retryDelay > 0) await this.delay(retryDelay, run.signal)
        }
      }
      throw new AgentError(this.id, 'failed to get a response after maximum iterations')
    } catch (error) {
      const executionTimeMs = startedAt === undefined ? 0 : elapsed(this.now(), startedAt)
      if (startedAt !== undefined) this.executionTimes.push(executionTimeMs)
      await this.emitStep({
        agent: this.role,
        error: errorMessage(error),
        executionCount: this.timesExecuted,
        executionTimeMs,
        step: 'execution_error',
        task: description,
      })
      throw error
    } finally {
      run.cleanup()
    }
  }

  private async runTurnAttempt(
    prompt: string,
    systemPrompt: string,
    history: readonly ChatMessage[],
    streamCallback: CortexAgentStreamCallback | undefined,
    signal: AbortSignal,
  ): Promise<TurnAttemptResult> {
    const state = createAgentState([...history])
    const text: string[] = []
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
      systemPrompt,
      ...(this.temperature === undefined ? {} : { temperature: this.temperature }),
      ...(this.topP === undefined ? {} : { topP: this.topP }),
      ...(this.tools.length ? { tools: this.tools } : {}),
      userMessage: prompt,
    }, {
      llm: this.llm,
      delay: this.delay,
      retryDelays: this.retryDelays,
      onUnconfiguredToolCalls: calls => {
        unconfiguredTool ??= calls[0]?.function.name
        return 'stop'
      },
      ...(this.permissionBroker === undefined ? {} : { permissionBroker: this.permissionBroker }),
      ...(this.policy === undefined ? {} : { policy: this.policy }),
      ...(this.toolExecutor === undefined ? {} : { toolExecutor: this.toolExecutor }),
    }, signal)) {
      throwIfAborted(signal, this.id)
      if (streamCallback) await streamCallback(event)
      if (event.type === 'text') {
        text.push(event.text)
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
    throwIfAborted(signal, this.id)
    if (providerFailure !== undefined) throw new ProviderAttemptFailure(providerFailure)
    if (unconfiguredTool !== undefined) {
      throw new AgentError(this.id, `LLM requested unconfigured tool: ${unconfiguredTool}`)
    }
    const output = text.join('')
    if (!output.trim()) throw new AgentError(this.id, 'LLM completed without a text result')
    return {
      messages: state.messages,
      output,
      inputTokens: usage.inputTokens,
      outputTokens: usage.outputTokens,
      toolCalls,
      toolFailures,
      deniedToolCalls,
    }
  }

  private async historyForAttempt(): Promise<readonly ChatMessage[]> {
    if (this.autoCompaction === undefined || this.conversationHistory.length === 0) {
      return [...this.conversationHistory]
    }
    const compacted = await this.autoCompaction.compact(this.conversationHistory as unknown as readonly Record<string, unknown>[])
    const messages = compacted.messages.map(message => compactedChatMessage(message))
    this.conversationHistory = messages
    return [...messages]
  }

  private taskPrompt(taskDescription: string, options: CortexAgentExecutionOptions): string {
    const context = options.context?.trim() ?? ''
    const knowledgeContext = this.buildKnowledgeContext()
    const memoryContext = this.memoryEnabled && this.memory
      ? this.memory.buildContextForTask(taskDescription, {
          agentRole: this.role,
          ...(context ? { additionalContext: context } : {}),
          maxItems: 10,
        })
      : ''
    const formatGuidance = options.outputSchema && this.autoFormatGuidance
      ? formatOutputGuidance(options.outputSchema, this.outputFormatPreference)
      : ''
    return renderCortexTaskPrompt({
      taskDescription,
      ...(options.expectedOutput?.trim() ? { expectedOutput: options.expectedOutput.trim() } : {}),
      ...(context ? { context } : {}),
      ...(Object.keys(options.inputs ?? {}).length ? { inputs: options.inputs } : {}),
      ...(knowledgeContext ? { knowledgeContext } : {}),
      ...(memoryContext ? { memoryContext } : {}),
      ...(formatGuidance ? { formatGuidance } : {}),
    })
  }

  private buildKnowledgeContext(): string {
    const parts: string[] = []
    const entries = Object.entries(this.knowledge)
    if (entries.length) {
      parts.push('Available knowledge:')
      for (const [key, value] of entries) parts.push(`- ${key}: ${displayValue(value)}`)
    }
    if (this.knowledgeSources.length) {
      parts.push('Knowledge sources:', ...this.knowledgeSources.map(source => `- ${source}`))
    }
    return parts.join('\n')
  }

  private persistResult(
    taskDescription: string,
    output: string,
    metadata: CortexAgentExecutionMetadata,
    task: CortexTask | undefined,
  ): void {
    if (!this.memoryEnabled || this.memory === undefined) return
    this.memory.saveTaskResult({
      taskDescription,
      result: output,
      agentRole: this.role,
      ...(task?.importance === undefined ? {} : { importance: task.importance }),
      metadata: {
        ...(task === undefined ? {} : { task_id: task.id, expected_output: task.expectedOutput }),
        ...metadata,
      },
    })
    this.memory.saveAgentInteraction({
      agentRole: this.role,
      action: 'execute_task',
      content: `Task: ${taskDescription.slice(0, 512)} - Result: ${output}`,
      importance: 0.5,
    })
  }

  private async reserveRpmSlot(signal: AbortSignal): Promise<void> {
    if (this.maxRpm === undefined) return
    for (;;) {
      throwIfAborted(signal, this.id)
      this.pruneRpmRequests()
      if (this.rpmRequests.length < this.maxRpm) {
        this.rpmRequests.push(this.now())
        return
      }
      const oldest = this.rpmRequests[0]
      if (oldest === undefined) continue
      const waitMs = Math.max(1, 60_000 - Math.max(0, this.now() - oldest))
      if (this.rateLimitMode === 'error') {
        throw new RateLimitError(`cortex-agent:${this.id}`, this.maxRpm, 'minute', Math.ceil(waitMs / 1_000))
      }
      await this.delay(waitMs, signal)
    }
  }

  private pruneRpmRequests(): void {
    const threshold = this.now() - 60_000
    while (this.rpmRequests.length > 0 && (this.rpmRequests[0] ?? Number.POSITIVE_INFINITY) <= threshold) {
      this.rpmRequests.shift()
    }
  }

  private async emitStep(step: CortexAgentStep): Promise<void> {
    if (!this.stepCallback) return
    try {
      await this.stepCallback(step)
    } catch (error) {
      // Progress callbacks are observational; a host may surface this through the explicit error hook.
      this.onStepCallbackError(error, step)
    }
  }
}

/** Construct default agent instructions from concrete native configuration. */
export function renderCortexAgentSystemPrompt(input: {
  readonly backstory: string
  readonly goal: string
  readonly role: string
  readonly tools?: readonly ToolDefinition[]
}): string {
  const role = requiredText(input.role, 'role')
  const goal = requiredText(input.goal, 'goal')
  const backstory = requiredText(input.backstory, 'backstory')
  const toolNames = input.tools?.length ? input.tools.map(tool => tool.function.name).join(', ') : 'none'
  return [
    `Role: ${role}`,
    `Goal: ${goal}`,
    `Backstory: ${backstory}`,
    `Available tools: ${toolNames}.`,
    'Use only configured tools. Report concrete results or blockers and never claim work a supplied tool did not perform.',
  ].join('\n')
}

/** Build the task prompt without inspecting the filesystem, environment, or another runtime. */
export function renderCortexTaskPrompt(input: CortexTaskPromptInput): string {
  const taskDescription = requiredText(input.taskDescription, 'taskDescription')
  const sections = [`Task:\n${taskDescription}`]
  if (input.expectedOutput?.trim()) sections.push(`Expected output:\n${input.expectedOutput.trim()}`)
  if (input.inputs && Object.keys(input.inputs).length) {
    sections.push(`Task inputs:\n${JSON.stringify(input.inputs, null, 2)}`)
  }
  if (input.knowledgeContext?.trim()) sections.push(`Knowledge context:\n${input.knowledgeContext.trim()}`)
  if (input.memoryContext?.trim()) sections.push(`Relevant Cortex memory:\n${input.memoryContext.trim()}`)
  if (input.context?.trim()) sections.push(`Additional context:\n${input.context.trim()}`)
  if (input.formatGuidance?.trim()) sections.push(input.formatGuidance.trim())
  sections.push('Produce the requested result. Do not claim tool work that did not occur.')
  return sections.join('\n\n')
}

/** Produce native format guidance from an explicit JSON Schema rather than a Python model class. */
export function formatOutputGuidance(schema: JsonSchema, preference: CortexAgentOutputFormat = 'xml'): string {
  const resolved = resolveSchemaReferences(schema)
  const title = stringValue(resolved.title) ?? 'Output'
  const example = JSON.stringify(createSchemaExample(resolved), null, 2)
  if (preference === 'json') {
    return [
      `OUTPUT FORMAT REQUIREMENT for ${title}:`,
      'Return valid JSON matching this exact structure:',
      example,
      'Do not wrap the JSON in prose or Markdown fences.',
    ].join('\n')
  }
  return [
    `OUTPUT FORMAT REQUIREMENT for ${title}:`,
    'Return the following JSON structure inside one <json> XML wrapper:',
    '<json>',
    example,
    '</json>',
    'Preserve nested object fields and array item structure exactly. Do not add prose outside the wrapper.',
  ].join('\n')
}

/** Resolve local `#/definitions/*` and `#/$defs/*` references for output-example generation. */
export function resolveSchemaReferences(schema: JsonSchema, definitions?: JsonSchema): JsonSchema {
  const available = definitions ?? schemaDefinitions(schema)
  return resolveSchema(schema, available, new Set())
}

/** Generate a deterministic JSON-compatible example from a JSON Schema. */
export function createSchemaExample(schema: JsonSchema): JsonValue {
  return schemaExample(resolveSchemaReferences(schema), 'output')
}

function resolveSchema(schema: JsonSchema, definitions: JsonSchema, resolving: Set<string>): JsonSchema {
  const reference = stringValue(schema.$ref)
  if (reference && (reference.startsWith('#/definitions/') || reference.startsWith('#/$defs/'))) {
    const referenceName = reference.split('/').at(-1)
    const candidate = referenceName === undefined ? undefined : definitions[referenceName]
    if (isRecord(candidate) && !resolving.has(reference)) {
      resolving.add(reference)
      const siblings = withoutReference(schema)
      const resolved = resolveSchema(candidate, definitions, resolving)
      resolving.delete(reference)
      return resolveSchema({ ...resolved, ...siblings }, definitions, resolving)
    }
  }

  const result: Record<string, unknown> = {}
  for (const [key, value] of Object.entries(schema)) {
    if (key === '$ref') continue
    if (key === 'properties' && isRecord(value)) {
      result[key] = Object.fromEntries(Object.entries(value).map(([name, property]) => [
        name,
        isRecord(property) ? resolveSchema(property, definitions, resolving) : property,
      ]))
      continue
    }
    if (key === 'items' && isRecord(value)) {
      result[key] = resolveSchema(value, definitions, resolving)
      continue
    }
    if ((key === 'allOf' || key === 'anyOf' || key === 'oneOf') && Array.isArray(value)) {
      result[key] = value.map(item => isRecord(item) ? resolveSchema(item, definitions, resolving) : item)
      continue
    }
    result[key] = value
  }
  return result
}

function schemaExample(schema: JsonSchema, fieldName: string): JsonValue {
  const constant = toJsonValue(schema.const)
  if (constant !== undefined) return constant
  const enumValues = schema.enum
  if (Array.isArray(enumValues)) {
    const first = toJsonValue(enumValues[0])
    if (first !== undefined) return first
  }
  const explicitExample = firstExample(schema)
  if (explicitExample !== undefined) return explicitExample

  const alternatives = schema.oneOf ?? schema.anyOf
  if (Array.isArray(alternatives)) {
    const first = alternatives.find(isRecord)
    if (first) return schemaExample(first, fieldName)
  }
  if (Array.isArray(schema.allOf)) {
    const { allOf: _allOf, ...base } = schema
    const merged = schema.allOf.filter(isRecord).reduce<Record<string, unknown>>((combined, item) => ({
      ...combined,
      ...item,
      properties: mergeProperties(combined.properties, item.properties),
    }), { ...base })
    return schemaExample(merged, fieldName)
  }

  const type = stringValue(schema.type) ?? (isRecord(schema.properties) ? 'object' : undefined)
  if (type === 'string') return stringExample(schema, fieldName)
  if (type === 'integer') return integerExample(schema)
  if (type === 'number') return numberExample(schema)
  if (type === 'boolean') return true
  if (type === 'array') {
    const count = arrayExampleCount(schema)
    const itemSchema = isRecord(schema.items) ? schema.items : {}
    return Array.from({ length: count }, (_value, index) => schemaExample(itemSchema, `${fieldName}_${index + 1}`))
  }
  if (type === 'object') {
    const properties = isRecord(schema.properties) ? schema.properties : {}
    const result: Record<string, JsonValue> = {}
    for (const [name, property] of Object.entries(properties)) {
      result[name] = isRecord(property) ? schemaExample(property, name) : `Example ${humanize(name)}`
    }
    return result
  }
  return `Example ${humanize(fieldName)}`
}

function firstExample(schema: JsonSchema): JsonValue | undefined {
  const direct = toJsonValue(schema.example ?? schema.default)
  if (direct !== undefined) return direct
  if (!Array.isArray(schema.examples)) return undefined
  return toJsonValue(schema.examples[0])
}

function stringExample(schema: JsonSchema, fieldName: string): string {
  const description = stringValue(schema.description)
  const lower = fieldName.toLowerCase()
  if (lower.includes('name') || lower.includes('title')) return `Example ${humanize(fieldName)}`
  if (schema.format === 'email') return 'example@example.com'
  if (schema.format === 'uri' || schema.format === 'url') return 'https://example.com'
  if (schema.format === 'date-time') return '2026-01-01T00:00:00Z'
  if (schema.format === 'date') return '2026-01-01'
  return `Example ${description ?? humanize(fieldName)}`
}

function integerExample(schema: JsonSchema): number {
  const minimum = numberValue(schema.minimum) ?? 1
  const maximum = numberValue(schema.maximum) ?? 10
  return Math.trunc(clamp(5, Math.ceil(minimum), Math.floor(maximum)))
}

function numberExample(schema: JsonSchema): number {
  const minimum = numberValue(schema.minimum) ?? 1
  const maximum = numberValue(schema.maximum) ?? 10
  return clamp(7.5, minimum, maximum)
}

function arrayExampleCount(schema: JsonSchema): number {
  const minimum = numberValue(schema.minItems)
  const maximum = numberValue(schema.maxItems)
  const requested = minimum === undefined ? 2 : Math.max(0, Math.ceil(minimum))
  const capped = Math.min(3, requested)
  return maximum === undefined ? capped : Math.max(0, Math.min(capped, Math.floor(maximum)))
}

function schemaDefinitions(schema: JsonSchema): JsonSchema {
  const definitions = schema.definitions ?? schema.$defs
  return isRecord(definitions) ? definitions : {}
}

function withoutReference(schema: JsonSchema): JsonSchema {
  const { $ref: _reference, ...siblings } = schema
  return siblings
}

function mergeProperties(left: unknown, right: unknown): Record<string, unknown> {
  return { ...(isRecord(left) ? left : {}), ...(isRecord(right) ? right : {}) }
}

function outputSchemaFromMetadata(metadata: Readonly<Record<string, unknown>> | undefined): JsonSchema | undefined {
  const candidate = metadata?.outputSchema
  return isRecord(candidate) ? candidate : undefined
}

function normalizeTools(tools: readonly (CortexTool | ToolDefinition)[]): readonly ToolDefinition[] {
  const seen = new Set<string>()
  const normalized = tools.map(tool => {
    const definition = tool instanceof CortexTool ? tool.toFunctionJson() : tool
    const name = requiredText(definition.function.name, 'tool.function.name')
    if (seen.has(name)) throw new TypeError(`Duplicate CortexAgent tool: ${name}`)
    seen.add(name)
    return Object.freeze({
      type: 'function' as const,
      function: Object.freeze({
        name,
        description: requiredText(definition.function.description, `tool ${name} description`),
        parameters: definition.function.parameters,
      }),
    })
  })
  return Object.freeze(normalized)
}

function compactedChatMessage(message: Record<string, unknown>): ChatMessage {
  const role = message.role
  const content = message.content
  if (!isMessageRole(role) || !isMessageContent(content)) {
    throw new TypeError('AutoCompactAgent returned a message that is not compatible with the native chat protocol')
  }
  return message as unknown as ChatMessage
}

function isMessageRole(value: unknown): value is ChatMessage['role'] {
  return value === 'assistant' || value === 'system' || value === 'tool' || value === 'user'
}

function isMessageContent(value: unknown): value is MessageContent {
  if (typeof value === 'string') return true
  if (!Array.isArray(value)) return false
  return value.every(part => isRecord(part) && (
    (part.type === 'text' && typeof part.text === 'string')
    || (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string')
  ))
}

function executionRun(parent: AbortSignal | undefined, maxExecutionTime: number | undefined, agentId: string): ExecutionRun {
  const controller = new AbortController()
  let didTimeOut = false
  const parentAbort = () => controller.abort(parent?.reason)
  if (parent?.aborted) parentAbort()
  else parent?.addEventListener('abort', parentAbort, { once: true })
  const timer = maxExecutionTime === undefined
    ? undefined
    : setTimeout(() => {
        didTimeOut = true
        controller.abort(new XerxesTimeoutError(`Cortex agent ${agentId} execution`, maxExecutionTime))
      }, maxExecutionTime * 1_000)
  return {
    signal: controller.signal,
    timedOut: () => didTimeOut,
    cleanup: () => {
      if (timer !== undefined) clearTimeout(timer)
      parent?.removeEventListener('abort', parentAbort)
    },
  }
}

function throwIfExecutionAborted(
  run: ExecutionRun,
  agentId: string,
  maxExecutionTime: number | undefined,
): void {
  if (!run.signal.aborted) return
  if (run.timedOut() && maxExecutionTime !== undefined) {
    throw new XerxesTimeoutError(`Cortex agent ${agentId} execution`, maxExecutionTime)
  }
  throwIfAborted(run.signal, agentId)
}

function throwIfAborted(signal: AbortSignal | undefined, agentId: string): void {
  if (!signal?.aborted) return
  if (signal.reason instanceof Error) throw signal.reason
  throw new AgentError(agentId, 'execution cancelled')
}

/** Delay that rejects promptly on abort and always removes its abort listener when it settles. */
export function abortableDelay(milliseconds: number, signal?: AbortSignal): Promise<void> {
  if (!Number.isFinite(milliseconds) || milliseconds < 0) {
    return Promise.reject(new TypeError('delay milliseconds must be a non-negative finite number'))
  }
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(signal.reason)
      return
    }
    const abort = () => {
      clearTimeout(timer)
      reject(signal?.reason)
    }
    const timer = setTimeout(() => {
      signal?.removeEventListener('abort', abort)
      resolve()
    }, milliseconds)
    signal?.addEventListener('abort', abort, { once: true })
  })
}

function agentId(role: string): string {
  const normalized = role.toLowerCase().trim().replace(/\s+/gu, '_').replace(/[^a-z0-9_-]/gu, '')
  return normalized.slice(0, 32) || 'cortex_agent'
}

function requiredText(value: string, name: string): string {
  const normalized = value.trim()
  if (!normalized) throw new TypeError(`${name} cannot be empty`)
  return normalized
}

function positiveInteger(value: number, name: string): number {
  if (!Number.isSafeInteger(value) || value < 1) throw new TypeError(`${name} must be a positive safe integer`)
  return value
}

function positiveFinite(value: number, name: string): number {
  if (!Number.isFinite(value) || value <= 0) throw new TypeError(`${name} must be a positive finite number`)
  return value
}

function optionalFinite(value: number | undefined, name: string): number | undefined {
  if (value === undefined) return undefined
  if (!Number.isFinite(value)) throw new TypeError(`${name} must be finite`)
  return value
}

function copyDelays(delays: readonly number[], name: string): readonly number[] {
  for (const delay of delays) {
    if (!Number.isFinite(delay) || delay < 0) throw new TypeError(`${name} must contain non-negative finite values`)
  }
  return Object.freeze([...delays])
}

function delayAt(delays: readonly number[], index: number): number {
  if (delays.length === 0) return 0
  return delays[Math.min(index, delays.length - 1)] ?? 0
}

function requireLlm(value: LlmClient): LlmClient {
  if (!value || typeof value.stream !== 'function') throw new TypeError('CortexAgent requires an injected LlmClient stream port')
  return value
}

function uniqueTexts(values: readonly string[], name: string): string[] {
  const unique: string[] = []
  for (const value of values) {
    const normalized = requiredText(value, name)
    if (!unique.includes(normalized)) unique.push(normalized)
  }
  return unique
}

function joinContexts(...contexts: readonly (string | undefined)[]): string | undefined {
  const joined = contexts.flatMap(context => context?.trim() ? [context.trim()] : []).join('\n\n')
  return joined || undefined
}

function elapsed(now: number, startedAt: number): number {
  return Math.max(0, now - startedAt)
}

function displayValue(value: unknown): string {
  if (typeof value === 'string') return value
  try {
    const serialized = JSON.stringify(value)
    return serialized === undefined ? String(value) : serialized
  } catch {
    return String(value)
  }
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function stringValue(value: unknown): string | undefined {
  return typeof value === 'string' ? value : undefined
}

function numberValue(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}

function toJsonValue(value: unknown): JsonValue | undefined {
  if (value === null || typeof value === 'boolean' || typeof value === 'string') return value
  if (typeof value === 'number') return Number.isFinite(value) ? value : undefined
  if (Array.isArray(value)) {
    const converted = value.map(toJsonValue)
    return converted.every(item => item !== undefined) ? converted as JsonValue[] : undefined
  }
  if (isRecord(value)) {
    const converted: Record<string, JsonValue> = {}
    for (const [key, child] of Object.entries(value)) {
      const json = toJsonValue(child)
      if (json === undefined) return undefined
      converted[key] = json
    }
    return converted
  }
  return undefined
}

function clamp(value: number, minimum: number, maximum: number): number {
  if (minimum > maximum) return minimum
  return Math.min(maximum, Math.max(minimum, value))
}

function humanize(value: string): string {
  const normalized = value.replace(/[_-]+/gu, ' ').replace(/\s+/gu, ' ').trim()
  return normalized.replace(/\b\p{L}/gu, character => character.toUpperCase())
}
