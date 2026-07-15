// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import {
  ContextualMemory,
  EntityMemory,
  LongTermMemory,
  ShortTermMemory,
  SimpleStorage,
  UserMemory,
  type MemoryItem,
  type MemoryMetadata,
  type MemorySaveOptions,
  type MemoryStorage,
} from '../../memory/index.js'

/** Configuration for the Cortex-specific composition of the native memory tiers. */
export interface CortexMemoryOptions {
  readonly enableEntity?: boolean
  readonly enableLongTerm?: boolean
  readonly enableShortTerm?: boolean
  readonly enableUser?: boolean
  readonly longTermCapacity?: number
  /**
   * Explicit storage boundary. Omitting it intentionally selects ephemeral
   * in-process storage; this facade never reads WRITE_MEMORY or opens a path.
   */
  readonly storage?: MemoryStorage
  readonly shortTermCapacity?: number
}

export interface CortexMemoryContextOptions {
  readonly additionalContext?: string
  readonly agentRole?: string
  readonly maxItems?: number
}

export interface CortexTaskResultMemoryInput {
  readonly agentRole: string
  readonly importance?: number
  readonly metadata?: MemoryMetadata
  readonly result: string
  readonly taskDescription: string
}

export interface CortexAgentInteractionInput {
  readonly action: string
  readonly agentRole: string
  readonly content: string
  readonly importance?: number
}

export interface CortexDecisionMemoryInput {
  readonly context: string
  readonly decision: string
  readonly importance?: number
  readonly outcome?: string
}

/**
 * Unified native memory facade for Cortex agents.
 *
 * The individual tier implementations remain reusable on their own. This
 * class only composes them for Cortex task context and result persistence; it
 * does not create a database or consult process environment state by itself.
 */
export class CortexMemory {
  readonly contextual: ContextualMemory
  readonly entityMemory: EntityMemory | undefined
  readonly longTerm: LongTermMemory | undefined
  readonly shortTerm: ShortTermMemory | undefined
  readonly storage: MemoryStorage
  readonly userMemory: UserMemory | undefined

  constructor(options: CortexMemoryOptions = {}) {
    const shortTermCapacity = capacity(options.shortTermCapacity ?? 50, 'shortTermCapacity')
    const longTermCapacity = capacity(options.longTermCapacity ?? 5_000, 'longTermCapacity')
    this.storage = options.storage ?? new SimpleStorage()
    this.shortTerm = options.enableShortTerm === false
      ? undefined
      : new ShortTermMemory({ capacity: shortTermCapacity, storage: this.storage })
    this.longTerm = options.enableLongTerm === false
      ? undefined
      : new LongTermMemory({ enableEmbeddings: false, maxItems: longTermCapacity, storage: this.storage })
    this.entityMemory = options.enableEntity === false
      ? undefined
      : new EntityMemory({ storage: this.storage })
    this.userMemory = options.enableUser === true ? new UserMemory(this.storage) : undefined

    // Reuse the public long-term tier when present so Cortex searches and
    // orchestration writes observe one durable knowledge set instead of two
    // independently hydrated stores with the same storage keys.
    this.contextual = new ContextualMemory({
      shortTermCapacity,
      longTerm: this.longTerm ?? new LongTermMemory({
        enableEmbeddings: false,
        maxItems: longTermCapacity,
        storage: this.storage,
      }),
    })
  }

  /** Assemble bounded task context from active Cortex memory tiers. */
  buildContextForTask(taskDescription: string, options: CortexMemoryContextOptions = {}): string {
    const task = requiredText(taskDescription, 'taskDescription')
    const maxItems = nonNegativeInteger(options.maxItems ?? 10, 'maxItems')
    const parts: string[] = []
    const additional = options.additionalContext?.trim()
    if (additional) parts.push(`Background:\n${additional}`)

    const recent = this.shortTerm?.getRecent(5) ?? []
    if (recent.length) {
      parts.push('Recent context:')
      for (const item of recent) parts.push(`  • ${item.content.slice(0, 200)}`)
    }

    const agentRole = options.agentRole?.trim()
    const relevant = this.longTerm?.search(task, 5, agentRole ? { agentId: agentRole } : undefined) ?? []
    if (relevant.length) {
      parts.push('\nRelevant knowledge:')
      for (const item of relevant) parts.push(`  • ${item.content.slice(0, 200)}`)
    }

    const entities = this.entityMemory?.extractEntities(task) ?? []
    const known = entities.flatMap(entity => {
      const frequency = this.entityMemory?.getEntityInfo(entity).frequency ?? 0
      return frequency > 0 ? [`  • ${entity}: ${frequency} mentions`] : []
    }).slice(0, 5)
    if (known.length) parts.push('\nKnown entities:', ...known)

    const comprehensive = this.contextual.search(task, maxItems)
    if (comprehensive.length) {
      parts.push('\nRelated memories:')
      for (const item of comprehensive.slice(0, 3)) parts.push(`  • ${item.content.slice(0, 150)}`)
    }
    return parts.join('\n')
  }

  /** CortexOrchestrator-compatible writer backed by contextual memory. */
  save(content: string, metadata: MemoryMetadata = {}, options: MemorySaveOptions = {}): MemoryItem {
    return this.contextual.save(requiredText(content, 'content'), { ...metadata }, { ...options })
  }

  /** Persist an actual completed task across the relevant Cortex memory tiers. */
  saveTaskResult(input: CortexTaskResultMemoryInput): void {
    const taskDescription = requiredText(input.taskDescription, 'taskDescription')
    const result = requiredText(input.result, 'result')
    const agentRole = requiredText(input.agentRole, 'agentRole')
    const importance = importanceValue(input.importance ?? 0.5)
    const metadata: MemoryMetadata = {
      ...(input.metadata ?? {}),
      task: taskDescription.slice(0, 100),
      agent_role: agentRole,
    }

    this.shortTerm?.save(
      `Task completed: ${taskDescription.slice(0, 100)} - Result: ${result.slice(0, 200)}`,
      metadata,
      { agentId: agentRole },
    )
    this.entityMemory?.save(`${taskDescription} ${result}`, metadata, { agentId: agentRole })
    if (importance >= 0.7 && this.longTerm) {
      this.longTerm.save(result, metadata, { agentId: agentRole, importance })
      return
    }
    this.contextual.save(result, metadata, { agentId: agentRole, importance })
  }

  /** Record a concrete action and promote it only when its importance warrants it. */
  saveAgentInteraction(input: CortexAgentInteractionInput): void {
    const agentRole = requiredText(input.agentRole, 'agentRole')
    const action = requiredText(input.action, 'action')
    const content = requiredText(input.content, 'content')
    const importance = importanceValue(input.importance ?? 0.3)
    const interaction = `[${agentRole}] ${action}: ${content}`
    this.shortTerm?.save(interaction, { action }, { agentId: agentRole })
    if (importance >= 0.6) this.longTerm?.save(interaction, { action }, { agentId: agentRole, importance })
  }

  /** Record an orchestration decision in the shared contextual/long-term store. */
  saveCortexDecision(input: CortexDecisionMemoryInput): void {
    const decision = requiredText(input.decision, 'decision')
    const context = requiredText(input.context, 'context')
    const importance = importanceValue(input.importance ?? 0.7)
    const outcome = input.outcome?.trim()
    const content = [
      `Decision: ${decision}`,
      `Context: ${context}`,
      ...(outcome ? [`Outcome: ${outcome}`] : []),
    ].join('\n')
    const metadata = { type: 'cortex_decision', has_outcome: Boolean(outcome) }
    if (this.longTerm) {
      this.longTerm.save(content, metadata, { agentId: 'cortex_manager', importance })
      return
    }
    this.contextual.save(content, metadata, { agentId: 'cortex_manager', importance })
  }

  /** Return recent then durable entries belonging to one Cortex role. */
  getAgentHistory(agentRole: string, limit = 20): string[] {
    const role = requiredText(agentRole, 'agentRole')
    const count = nonNegativeInteger(limit, 'limit')
    const history = this.shortTerm?.search('', count, { agentId: role }).map(item => item.content) ?? []
    const remaining = Math.max(0, count - history.length)
    if (remaining) {
      const durable = this.longTerm?.retrieve(undefined, { agentId: role }, remaining)
      if (Array.isArray(durable)) history.push(...durable.map(item => item.content))
      else if (durable) history.push(durable.content)
    }
    return history.slice(0, count)
  }

  getUserContext(userId: string): string {
    return this.userMemory?.getUserContext(requiredText(userId, 'userId')) ?? ''
  }

  resetShortTerm(): void {
    this.shortTerm?.clear()
  }

  resetAll(): void {
    this.shortTerm?.clear()
    this.longTerm?.clear()
    this.entityMemory?.clear()
    this.contextual.clear()
  }

  /** Provide a compact cross-tier status summary, including durable-memory consolidation. */
  getSummary(): string {
    const parts: string[] = []
    if (this.shortTerm) parts.push(this.shortTerm.summarize())
    if (this.longTerm) parts.push(this.longTerm.consolidate())
    if (this.entityMemory && this.entityMemory.size > 0) {
      parts.push(`Tracking ${Object.keys(this.entityMemory.entities).length} entities`)
    }
    return parts.join('\n\n')
  }
}

function capacity(value: number, name: string): number {
  if (!Number.isInteger(value) || value < 1) throw new Error(`${name} must be a positive integer`)
  return value
}

function importanceValue(value: number): number {
  if (!Number.isFinite(value) || value < 0 || value > 1) throw new Error('importance must be between zero and one')
  return value
}

function nonNegativeInteger(value: number, name: string): number {
  if (!Number.isInteger(value) || value < 0) throw new Error(`${name} must be a non-negative integer`)
  return value
}

function requiredText(value: string, name: string): string {
  const normalized = value.trim()
  if (!normalized) throw new Error(`${name} cannot be empty`)
  return normalized
}
