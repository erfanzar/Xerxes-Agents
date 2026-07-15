// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** Input values for a completed turn that nudge rules inspect. */
export interface NudgeContextOptions {
  readonly lastAssistantMessage?: string
  readonly lastUserMessage?: string
  readonly memoryWritesSinceLastFire?: number
  readonly memoryWritesThisTurn?: number
  readonly metadata?: Readonly<Record<string, unknown>>
  readonly successfulToolCallsThisTurn?: number
  readonly toolCallsThisTurn?: number
  readonly turnIndex: number
}

/** Immutable snapshot of the session state available to nudge rules. */
export class NudgeContext {
  readonly lastAssistantMessage: string
  readonly lastUserMessage: string
  readonly memoryWritesSinceLastFire: number
  readonly memoryWritesThisTurn: number
  readonly metadata: Readonly<Record<string, unknown>>
  readonly successfulToolCallsThisTurn: number
  readonly toolCallsThisTurn: number
  readonly turnIndex: number

  constructor(options: NudgeContextOptions) {
    this.turnIndex = nonNegativeInteger(options.turnIndex, 'turnIndex')
    this.toolCallsThisTurn = nonNegativeInteger(options.toolCallsThisTurn ?? 0, 'toolCallsThisTurn')
    this.successfulToolCallsThisTurn = nonNegativeInteger(
      options.successfulToolCallsThisTurn ?? 0,
      'successfulToolCallsThisTurn',
    )
    this.memoryWritesThisTurn = nonNegativeInteger(options.memoryWritesThisTurn ?? 0, 'memoryWritesThisTurn')
    this.memoryWritesSinceLastFire = nonNegativeInteger(
      options.memoryWritesSinceLastFire ?? 0,
      'memoryWritesSinceLastFire',
    )
    this.lastUserMessage = options.lastUserMessage ?? ''
    this.lastAssistantMessage = options.lastAssistantMessage ?? ''
    this.metadata = Object.freeze({ ...(options.metadata ?? {}) })
    Object.freeze(this)
  }
}

/** A named suggestion rule evaluated after a completed agent turn. */
export interface NudgeRule {
  readonly name: string
  message(context: NudgeContext): string
  shouldFire(context: NudgeContext): boolean
}

export type NudgeResult = readonly [name: string, message: string]

/** Reminds the agent to persist durable information via save_memory. */
export class MemoryNudge implements NudgeRule {
  static readonly DURABLE_HINTS = Object.freeze([
    'decided',
    'prefers',
    'will always',
    'should never',
    'username',
    'deadline',
    'remember',
  ])

  readonly name = 'memory'
  readonly interval: number

  constructor(options: { readonly interval?: number } = {}) {
    this.interval = positiveInteger(options.interval ?? 8, 'interval')
  }

  shouldFire(context: NudgeContext): boolean {
    if (context.memoryWritesSinceLastFire > 0) return false
    if ((context.turnIndex + 1) % this.interval !== 0) return false
    const conversation = `${context.lastUserMessage} ${context.lastAssistantMessage}`.toLowerCase()
    return MemoryNudge.DURABLE_HINTS.some(hint => conversation.includes(hint))
  }

  message(_context: NudgeContext): string {
    return '[NUDGE] You haven\'t written to memory in a while and the recent turn looks like it had '
      + 'durable user info (preferences, decisions, deadlines). Consider calling '
      + '`save_memory(content=...)` so this doesn\'t get lost across sessions.'
  }
}

/** Suggests capturing a successful, tool-heavy turn as a reusable skill. */
export class SkillNudge implements NudgeRule {
  readonly name = 'skill'
  readonly threshold: number

  constructor(options: { readonly threshold?: number } = {}) {
    this.threshold = nonNegativeInteger(options.threshold ?? 6, 'threshold')
  }

  shouldFire(context: NudgeContext): boolean {
    return context.successfulToolCallsThisTurn >= this.threshold
  }

  message(_context: NudgeContext): string {
    return '[NUDGE] The just-finished turn used many tools successfully. If this pattern is likely '
      + 'to recur, consider `skill_manage(intent=\'create\', name=..., body=...)` to capture it '
      + 'as a reusable skill.'
  }
}

/** Coordinates rule enablement and per-rule fire counts for one session/runtime. */
export class NudgeManager {
  private readonly disabledRules = new Set<string>()
  private readonly firedCounts = new Map<string, number>()
  private readonly ruleList: NudgeRule[]

  constructor(rules: readonly NudgeRule[] = []) {
    this.ruleList = rules.length ? [...rules] : [new MemoryNudge(), new SkillNudge()]
  }

  /** Return a shallow copy so callers cannot reorder the manager's rule list. */
  get rules(): NudgeRule[] {
    return [...this.ruleList]
  }

  disable(ruleName: string): void {
    this.disabledRules.add(ruleName)
  }

  enable(ruleName: string): void {
    this.disabledRules.delete(ruleName)
  }

  /** Return a copy of disabled names rather than exposing mutable manager state. */
  disabled(): Set<string> {
    return new Set(this.disabledRules)
  }

  firedCount(ruleName: string): number {
    return this.firedCounts.get(ruleName) ?? 0
  }

  /** Return every active suggestion that should be injected before the next turn. */
  check(context: NudgeContext): NudgeResult[] {
    const results: NudgeResult[] = []
    for (const rule of this.ruleList) {
      if (this.disabledRules.has(rule.name) || !rule.shouldFire(context)) continue
      results.push([rule.name, rule.message(context)])
      this.firedCounts.set(rule.name, this.firedCount(rule.name) + 1)
    }
    return results
  }
}

function nonNegativeInteger(value: number, name: string): number {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new RangeError(name + ' must be a non-negative safe integer')
  }
  return value
}

function positiveInteger(value: number, name: string): number {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new RangeError(name + ' must be a positive safe integer')
  }
  return value
}
