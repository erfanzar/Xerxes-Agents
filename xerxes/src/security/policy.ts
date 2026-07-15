// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ToolPolicyViolation } from '../core/errors.js'

export { ToolPolicyViolation } from '../core/errors.js'

/** Outcome of one tool-policy evaluation. */
export const PolicyAction = {
  ALLOW: 'allow',
  DENY: 'deny',
} as const

export type PolicyAction = (typeof PolicyAction)[keyof typeof PolicyAction]

export interface ToolPolicyOptions {
  /** Explicit allow-list. A non-empty list denies every omitted tool. */
  readonly allow?: Iterable<string>
  /** Tools denied when no explicit allow-list is active. */
  readonly deny?: Iterable<string>
  /** Existing tools which remain unavailable until explicitly allow-listed. */
  readonly optionalTools?: Iterable<string>
}

/** Per-scope, case-insensitive tool admission rules. */
export class ToolPolicy {
  readonly allow: ReadonlySet<string>
  readonly deny: ReadonlySet<string>
  readonly optionalTools: ReadonlySet<string>

  constructor(options: ToolPolicyOptions = {}) {
    this.allow = normalizeNames(options.allow)
    this.deny = normalizeNames(options.deny)
    this.optionalTools = normalizeNames(options.optionalTools)
  }

  /** Evaluate a tool name against this policy. */
  evaluate(toolName: string): PolicyAction {
    const normalized = normalizeName(toolName)
    if (this.allow.size > 0) {
      return this.allow.has(normalized) ? PolicyAction.ALLOW : PolicyAction.DENY
    }
    if (this.deny.has(normalized) || this.optionalTools.has(normalized)) {
      return PolicyAction.DENY
    }
    return PolicyAction.ALLOW
  }
}

export type PolicyListener = (toolName: string, agentId: string | undefined, action: PolicyAction) => void

export interface PolicyEngineOptions {
  readonly agentPolicies?: ReadonlyMap<string, ToolPolicy> | Readonly<Record<string, ToolPolicy>>
  readonly globalPolicy?: ToolPolicy
}

/**
 * Evaluates tool invocations against a global policy and full per-agent overrides.
 *
 * An agent policy shadows the global policy; policies are intentionally not merged.
 */
export class PolicyEngine {
  private readonly listeners: PolicyListener[] = []

  globalPolicy: ToolPolicy
  readonly agentPolicies: Map<string, ToolPolicy>

  constructor(options: PolicyEngineOptions = {}) {
    this.globalPolicy = options.globalPolicy ?? new ToolPolicy()
    this.agentPolicies = mapAgentPolicies(options.agentPolicies)
  }

  setGlobalPolicy(policy: ToolPolicy): void {
    this.globalPolicy = policy
  }

  setAgentPolicy(agentId: string, policy: ToolPolicy): void {
    this.agentPolicies.set(agentId, policy)
  }

  removeAgentPolicy(agentId: string): void {
    this.agentPolicies.delete(agentId)
  }

  addListener(listener: PolicyListener): void {
    this.listeners.push(listener)
  }

  /** Return the policy decision and notify observers without exposing enforcement gaps. */
  check(toolName: string, agentId?: string): PolicyAction {
    const policy = agentId === undefined ? undefined : this.agentPolicies.get(agentId)
    const action = (policy ?? this.globalPolicy).evaluate(toolName)

    for (const listener of this.listeners) {
      try {
        listener(toolName, agentId, action)
      } catch {
        // Audit and UI observers must never prevent the policy gate from running.
      }
    }
    return action
  }

  /** Raise when the selected policy denies the requested tool. */
  enforce(toolName: string, agentId?: string): void {
    if (this.check(toolName, agentId) === PolicyAction.DENY) {
      const agentPart = agentId === undefined ? '' : ` for agent '${agentId}'`
      throw new ToolPolicyViolation(toolName, `is denied by policy${agentPart}`, {
        ...(agentId === undefined ? {} : { agentId }),
      })
    }
  }
}

function normalizeNames(names: Iterable<string> | undefined): ReadonlySet<string> {
  return new Set(names === undefined ? [] : Array.from(names, normalizeName))
}

function normalizeName(toolName: string): string {
  return toolName.toLowerCase()
}

function mapAgentPolicies(
  policies: ReadonlyMap<string, ToolPolicy> | Readonly<Record<string, ToolPolicy>> | undefined,
): Map<string, ToolPolicy> {
  if (policies === undefined) {
    return new Map()
  }
  if (isReadonlyMap(policies)) {
    return new Map(policies)
  }
  return new Map(Object.entries(policies))
}

function isReadonlyMap(
  value: ReadonlyMap<string, ToolPolicy> | Readonly<Record<string, ToolPolicy>>,
): value is ReadonlyMap<string, ToolPolicy> {
  return typeof (value as ReadonlyMap<string, ToolPolicy>).entries === 'function'
}
