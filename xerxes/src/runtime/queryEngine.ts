// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ContextCompressor } from '../context/compressor.js'
import { repairToolMessageSequence } from '../context/toolPairRepair.js'
import type { ToolExecutor } from '../executors/toolRegistry.js'
import type { StreamEvent } from '../streaming/events.js'
import { createAgentState } from '../streaming/events.js'
import { runTurn, type TurnDependencies } from '../streaming/loop.js'
import { DEFAULT_PERMISSION_MODE, type PermissionMode } from '../streaming/permissions.js'
import type { ToolDefinition } from '../types/toolCalls.js'
import { CostTracker } from './costTracker.js'

export interface QueryEngineConfig {
  /** Agent identity propagated to agent-specific tools and policy decisions. */
  readonly agentId?: string
  readonly compactAfterTurns: number
  readonly maxBudgetTokens?: number
  readonly maxTokens?: number
  readonly maxTurns: number
  readonly model: string
  readonly permissionMode: PermissionMode
  readonly systemPrompt?: string
  readonly thinking: boolean
  readonly thinkingBudget?: number
}

export interface TurnResult {
  readonly inputTokens: number
  readonly output: string
  readonly outputTokens: number
  readonly prompt: string
  readonly stopReason?: 'budget_exhausted' | 'max_turns'
  readonly toolCalls: readonly string[]
}

export const DEFAULT_QUERY_ENGINE_CONFIG: QueryEngineConfig = {
  compactAfterTurns: 20,
  maxBudgetTokens: 500_000,
  maxTurns: 50,
  model: 'gpt-4o',
  permissionMode: DEFAULT_PERMISSION_MODE,
  thinking: false,
}

/** Lightweight SDK facade over the new streaming loop; the daemon remains the production source of truth. */
export class QueryEngine {
  readonly config: QueryEngineConfig
  readonly costTracker: CostTracker
  readonly sessionId: string
  private readonly contextCompressor: ContextCompressor | undefined
  private readonly dependencies: TurnDependencies
  private readonly tools: readonly ToolDefinition[]
  private turnCount = 0

  constructor(
    dependencies: TurnDependencies,
    options: {
      readonly config?: Partial<QueryEngineConfig>
      /** Optional model-backed or deterministic compactor for durable transcript pressure. */
      readonly contextCompressor?: ContextCompressor
      /** Optional externally-owned ledger; otherwise the engine creates one per session. */
      readonly costTracker?: CostTracker
      readonly sessionId?: string
      readonly tools?: readonly ToolDefinition[]
    } = {},
  ) {
    this.config = { ...DEFAULT_QUERY_ENGINE_CONFIG, ...options.config }
    this.contextCompressor = options.contextCompressor
    this.dependencies = dependencies
    this.sessionId = options.sessionId ?? crypto.randomUUID()
    this.costTracker = options.costTracker ?? new CostTracker({
      sessionId: this.sessionId,
      ...(this.config.agentId ? { agentId: this.config.agentId } : {}),
    })
    this.tools = options.tools ?? []
  }

  readonly state = createAgentState()

  get totalCost(): number {
    return this.costTracker.totalCostUsd
  }

  async submit(prompt: string, signal?: AbortSignal): Promise<TurnResult> {
    const events = this.submitStream(prompt, signal)
    while (true) {
      const next = await events.next()
      if (next.done) {
        return next.value
      }
    }
  }

  async *submitStream(prompt: string, signal?: AbortSignal): AsyncGenerator<StreamEvent, TurnResult, void> {
    const stopped = this.stopResult(prompt)
    if (stopped) {
      return stopped
    }
    this.turnCount += 1
    const output: string[] = []
    const tools: string[] = []
    let inputTokens = 0
    let outputTokens = 0
    for await (const event of runTurn({
      ...(this.config.agentId ? { agentId: this.config.agentId } : {}),
      model: this.config.model,
      permissionMode: this.config.permissionMode,
      state: this.state,
      tools: this.tools,
      userMessage: prompt,
      ...(this.config.systemPrompt ? { systemPrompt: this.config.systemPrompt } : {}),
    }, this.dependencies, signal)) {
      if (event.type === 'text') {
        output.push(event.text)
      } else if (event.type === 'tool_start') {
        tools.push(event.call.function.name)
      } else if (event.type === 'turn_done') {
        inputTokens += event.usage.inputTokens
        outputTokens += event.usage.outputTokens
        this.costTracker.recordTurn(
          this.config.model,
          event.usage.inputTokens,
          event.usage.outputTokens,
          'turn_' + this.turnCount,
          {
            sessionId: this.sessionId,
            ...(this.config.agentId ? { agentId: this.config.agentId } : {}),
            ...(event.usage.cacheReadTokens === undefined ? {} : { cacheReadTokens: event.usage.cacheReadTokens }),
            ...(event.usage.cacheCreationTokens === undefined
              ? {}
              : { cacheCreationTokens: event.usage.cacheCreationTokens }),
          },
        )
      }
      yield event
    }
    this.compactIfDue()
    return { prompt, output: output.join(''), toolCalls: tools, inputTokens, outputTokens }
  }

  private compactIfDue(): void {
    if (!this.contextCompressor || this.turnCount === 0 || this.turnCount % this.config.compactAfterTurns !== 0) {
      return
    }
    const records = this.state.messages.map(message => ({ ...message }) as Record<string, unknown>)
    const result = this.contextCompressor.compress(records)
    if (!result.compressed) return
    const repaired = repairToolMessageSequence(result.messages)
    this.state.messages.splice(0, this.state.messages.length, ...(repaired as unknown as typeof this.state.messages))
    this.state.metadata.lastCompaction = {
      compressed_count: result.compressedCount,
      pruned_tool_results: result.prunedToolResults,
      strategy: result.metadata.strategy ?? 'unknown',
      tokens_before: result.tokensBefore,
      tokens_after: result.tokensAfter,
    }
  }

  private stopResult(prompt: string): TurnResult | undefined {
    if (this.turnCount >= this.config.maxTurns) {
      return { prompt, output: 'Max turns reached.', toolCalls: [], inputTokens: 0, outputTokens: 0, stopReason: 'max_turns' }
    }
    const budget = this.config.maxBudgetTokens
    if (budget !== undefined && this.state.totalInputTokens + this.state.totalOutputTokens >= budget) {
      return {
        prompt,
        output: `Session token budget (${budget.toLocaleString()}) exhausted.`,
        toolCalls: [],
        inputTokens: 0,
        outputTokens: 0,
        stopReason: 'budget_exhausted',
      }
    }
    return undefined
  }
}

export type { ToolExecutor }
