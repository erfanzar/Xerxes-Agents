// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

export const AgentSwitchTrigger = {
  EXPLICIT: 'explicit',
  CAPABILITY_BASED: 'capability_based',
  CAPABILITY_REQUIRED: 'capability_required',
  LOAD_BALANCING: 'load',
  CONTEXT_BASED: 'context',
  ERROR_RECOVERY: 'error',
  CUSTOM: 'custom',
} as const

export type AgentSwitchTrigger = (typeof AgentSwitchTrigger)[keyof typeof AgentSwitchTrigger]

export interface AgentCapability {
  readonly contextRequirements?: Readonly<Record<string, unknown>>
  readonly description: string
  readonly functionNames?: readonly string[]
  readonly name: string
  readonly performanceScore?: number
}

export const FUNCTION_CALLING_CAPABILITY: AgentCapability = Object.freeze({
  name: 'function_calling',
  description: 'Can use tools and function calls',
})

/** Minimal runtime shape needed for routing, independent of any LLM provider. */
export interface OrchestratedAgent {
  readonly capabilities?: readonly AgentCapability[]
  readonly fallbackAgentId?: string
  id?: string
  readonly name?: string
  readonly switchTriggers?: readonly AgentSwitchTrigger[]
}

export interface AgentSwitchRecord {
  readonly action: 'agent_switch'
  readonly from: string | undefined
  readonly reason: string | undefined
  readonly timestamp: string
  readonly to: string
  readonly type: 'agent_switch'
}

export type AgentSwitchHandler<Context extends Record<string, unknown> = Record<string, unknown>> = (
  context: Readonly<Context>,
  agents: ReadonlyMap<string, OrchestratedAgent>,
  currentAgentId: string | undefined,
) => string | undefined

export interface AgentOrchestratorOptions {
  readonly maxAgents?: number
  readonly now?: () => Date
  readonly onTriggerError?: (trigger: AgentSwitchTrigger, error: unknown) => void
}

/** Registry for active agents and their ordered handoff triggers. */
export class AgentOrchestrator {
  private readonly agentMap = new Map<string, OrchestratedAgent>()
  private currentId: string | undefined
  private readonly history: AgentSwitchRecord[] = []
  private readonly maxAgents: number
  private nextAutoId = 0
  private readonly now: () => Date
  private readonly triggerError: (trigger: AgentSwitchTrigger, error: unknown) => void
  private readonly triggers = new Map<AgentSwitchTrigger, AgentSwitchHandler>()

  constructor(options: AgentOrchestratorOptions = {}) {
    this.maxAgents = options.maxAgents ?? Number.POSITIVE_INFINITY
    this.now = options.now ?? (() => new Date())
    this.triggerError = options.onTriggerError ?? (() => undefined)
  }

  get agents(): ReadonlyMap<string, OrchestratedAgent> {
    return this.agentMap
  }

  get currentAgentId(): string | undefined {
    return this.currentId
  }

  get executionHistory(): readonly AgentSwitchRecord[] {
    return Object.freeze(this.history.slice())
  }

  get switchTriggers(): ReadonlyMap<AgentSwitchTrigger, AgentSwitchHandler> {
    return this.triggers
  }

  registerAgent(agent: OrchestratedAgent): string {
    let id = agent.id
    if (!id) {
      // Monotonic auto-ids never collide with explicit ids registered later.
      do {
        id = `agent_${this.nextAutoId++}`
      } while (this.agentMap.has(id))
    }
    if (this.agentMap.has(id)) {
      throw new Error(`Agent ${id} is already registered`)
    }
    if (this.agentMap.size >= this.maxAgents) {
      throw new Error(`Maximum number of agents (${this.maxAgents}) reached`)
    }
    // Store a frozen copy instead of mutating the caller's object.
    this.agentMap.set(id, Object.freeze({ ...agent, id }))
    this.currentId ??= id
    return id
  }

  registerSwitchTrigger(trigger: AgentSwitchTrigger, handler: AgentSwitchHandler): void {
    this.triggers.set(trigger, handler)
  }

  shouldSwitchAgent(context: Record<string, unknown>): string | undefined {
    const currentAgent = this.currentId ? this.agentMap.get(this.currentId) : undefined
    if (currentAgent) {
      for (const trigger of currentAgent.switchTriggers ?? []) {
        const target = this.invokeTrigger(trigger, context)
        if (target && target !== this.currentId) {
          return target
        }
      }
    }
    for (const trigger of this.triggers.keys()) {
      const target = this.invokeTrigger(trigger, context)
      if (target && target !== this.currentId) {
        return target
      }
    }
    return undefined
  }

  switchAgent(targetAgentId: string, reason?: string): void {
    if (!this.agentMap.has(targetAgentId)) {
      throw new Error(`Agent ${targetAgentId} not found`)
    }
    const from = this.currentId
    this.currentId = targetAgentId
    this.history.push(Object.freeze({
      action: 'agent_switch',
      type: 'agent_switch',
      from,
      to: targetAgentId,
      reason,
      timestamp: this.now().toISOString(),
    }))
  }

  getCurrentAgent(): OrchestratedAgent {
    if (!this.currentId) {
      throw new Error('No active agent')
    }
    const agent = this.agentMap.get(this.currentId)
    if (!agent) {
      throw new Error('No active agent')
    }
    return agent
  }

  private invokeTrigger(trigger: AgentSwitchTrigger, context: Record<string, unknown>): string | undefined {
    const handler = this.triggers.get(trigger)
    if (!handler) {
      return undefined
    }
    try {
      return handler(context, this.agentMap, this.currentId)
    } catch (error) {
      this.triggerError(trigger, error)
      return undefined
    }
  }
}

/** Pick the highest-scoring agent that advertises `context.required_capability`. */
export const capabilityBasedSwitch: AgentSwitchHandler = (context, agents) => {
  const requiredCapability = context.required_capability
  if (typeof requiredCapability !== 'string' || !requiredCapability) {
    return undefined
  }
  let selected: string | undefined
  let highestScore = 0
  for (const [agentId, agent] of agents) {
    for (const capability of agent.capabilities ?? []) {
      if (capability.name === requiredCapability && (capability.performanceScore ?? 1) > highestScore) {
        selected = agentId
        highestScore = capability.performanceScore ?? 1
      }
    }
  }
  return selected
}

/** Hand off after an execution error when the current agent names a fallback. */
export const errorRecoverySwitch: AgentSwitchHandler = (context, agents, currentAgentId) => {
  if (!context.execution_error || !currentAgentId) {
    return undefined
  }
  return agents.get(currentAgentId)?.fallbackAgentId
}

/** Register the standard Xerxes capability and error-recovery switch policies. */
export function registerDefaultSwitchTriggers(orchestrator: AgentOrchestrator): void {
  orchestrator.registerSwitchTrigger(AgentSwitchTrigger.CAPABILITY_BASED, capabilityBasedSwitch)
  orchestrator.registerSwitchTrigger(AgentSwitchTrigger.ERROR_RECOVERY, errorRecoverySwitch)
}
