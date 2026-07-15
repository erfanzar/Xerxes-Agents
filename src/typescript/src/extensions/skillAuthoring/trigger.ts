// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { SkillCandidate } from './model.js'

/** Minimal skill data needed for duplicate detection and semantic matching. */
export interface SkillCatalogEntry {
  readonly instructions: string
  readonly metadata: {
    readonly description: string
    readonly name: string
    readonly requiredTools: readonly string[]
    readonly tags: readonly string[]
    readonly version: string
  }
}

/** Host-owned skill inventory boundary. Existing SkillRegistry instances satisfy this shape. */
export interface SkillCatalogPort {
  all(): readonly SkillCatalogEntry[]
}

export interface SkillAuthoringConfigOptions {
  readonly enabled?: boolean
  readonly maxRetriesAllowed?: number
  readonly minToolCalls?: number
  readonly minUniqueTools?: number
  readonly requireSuccess?: boolean
  readonly skipIfSkillSignatureExists?: boolean
}

/** Immutable threshold policy for deciding whether an observed turn warrants a proposal. */
export class SkillAuthoringConfig {
  readonly enabled: boolean
  readonly maxRetriesAllowed: number
  readonly minToolCalls: number
  readonly minUniqueTools: number
  readonly requireSuccess: boolean
  readonly skipIfSkillSignatureExists: boolean

  constructor(options: SkillAuthoringConfigOptions = {}) {
    this.enabled = options.enabled ?? true
    this.minToolCalls = nonNegativeInteger(options.minToolCalls ?? 5, 'minToolCalls')
    this.maxRetriesAllowed = nonNegativeInteger(options.maxRetriesAllowed ?? 2, 'maxRetriesAllowed')
    this.minUniqueTools = nonNegativeInteger(options.minUniqueTools ?? 2, 'minUniqueTools')
    this.requireSuccess = options.requireSuccess ?? true
    this.skipIfSkillSignatureExists = options.skipIfSkillSignatureExists ?? true
  }
}

export interface SkillAuthoringDecision {
  readonly eligible: boolean
  readonly reason: string
  readonly terminalFailureIndexes: readonly number[]
}

export interface SkillAuthoringTriggerOptions {
  readonly catalog?: SkillCatalogPort
  readonly config?: SkillAuthoringConfig | SkillAuthoringConfigOptions
}

/** Deterministic policy gate; it never writes a skill or invokes an LLM. */
export class SkillAuthoringTrigger {
  readonly catalog: SkillCatalogPort | undefined
  readonly config: SkillAuthoringConfig

  constructor(options: SkillAuthoringTriggerOptions | SkillAuthoringConfig = {}) {
    const resolved = options instanceof SkillAuthoringConfig ? { config: options } : options
    this.config = resolved.config instanceof SkillAuthoringConfig
      ? resolved.config
      : new SkillAuthoringConfig(resolved.config)
    this.catalog = resolved.catalog
  }

  evaluate(candidate: SkillCandidate): SkillAuthoringDecision {
    const config = this.config
    const terminalFailures = terminalFailureIndexes(candidate)
    if (!config.enabled) {
      return decision(false, 'skill authoring disabled', terminalFailures)
    }
    if (candidate.events.length < config.minToolCalls) {
      const reason = 'only ' + candidate.events.length + ' tool calls (< ' + config.minToolCalls + ')'
      return decision(false, reason, terminalFailures)
    }
    if (config.requireSuccess && !candidate.successfulEvents.length) {
      return decision(false, 'candidate has no successful tool calls', terminalFailures)
    }
    if (config.requireSuccess && terminalFailures.length) {
      return decision(false, 'candidate has unrecovered failures', terminalFailures)
    }
    if (candidate.retries > config.maxRetriesAllowed) {
      return decision(false, candidate.retries + ' retries (> ' + config.maxRetriesAllowed + ')', terminalFailures)
    }
    if (candidate.uniqueTools.length < config.minUniqueTools) {
      const reason = candidate.uniqueTools.length + ' unique tools (< ' + config.minUniqueTools + ')'
      return decision(false, reason, terminalFailures)
    }
    if (config.skipIfSkillSignatureExists) {
      const duplicate = hasExistingCoverage(candidate, this.catalog)
      if (duplicate === 'unavailable') {
        return decision(false, 'skill catalog unavailable for duplicate detection', terminalFailures)
      }
      if (duplicate) {
        return decision(false, 'an existing skill already covers this tool combination', terminalFailures)
      }
    }
    return decision(true, 'skill-worthy', terminalFailures)
  }

  reason(candidate: SkillCandidate): string {
    return this.evaluate(candidate).reason
  }

  shouldAuthor(candidate: SkillCandidate): boolean {
    return this.evaluate(candidate).eligible
  }
}

/** Find failed events that are not eventually recovered by a successful retry chain. */
export function terminalFailureIndexes(candidate: SkillCandidate): number[] {
  const recovered = new Set<number>()
  for (let index = 0; index < candidate.events.length; index += 1) {
    const event = candidate.events[index]
    if (!event || event.status !== 'success' || event.retryOf === undefined) {
      continue
    }
    let retryOf: number | undefined = event.retryOf
    while (retryOf !== undefined && retryOf >= 0 && retryOf < candidate.events.length) {
      if (recovered.has(retryOf)) {
        break
      }
      recovered.add(retryOf)
      retryOf = candidate.events[retryOf]?.retryOf
    }
  }

  const failures: number[] = []
  for (let index = 0; index < candidate.events.length; index += 1) {
    const event = candidate.events[index]
    if (event && event.status !== 'success' && !recovered.has(index)) {
      failures.push(index)
    }
  }
  return failures
}

function decision(eligible: boolean, reason: string, terminalFailureIndexes: readonly number[]): SkillAuthoringDecision {
  return { eligible, reason, terminalFailureIndexes }
}

function hasExistingCoverage(candidate: SkillCandidate, catalog: SkillCatalogPort | undefined): boolean | 'unavailable' {
  if (!catalog) {
    return false
  }
  let skills: readonly SkillCatalogEntry[]
  try {
    skills = catalog.all()
  } catch {
    return 'unavailable'
  }
  const candidateTools = new Set(candidate.uniqueTools)
  for (const skill of skills) {
    const covered = new Set([...skill.metadata.tags, ...skill.metadata.requiredTools])
    if (covered.size && [...candidateTools].every(tool => covered.has(tool))) {
      return true
    }
  }
  return false
}

function nonNegativeInteger(value: number, name: string): number {
  if (!Number.isInteger(value) || value < 0) {
    throw new RangeError(name + ' must be a non-negative integer')
  }
  return value
}
