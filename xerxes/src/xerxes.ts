// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import {
  AgentOrchestrator,
  registerDefaultSwitchTriggers,
  type AgentCapability,
  type AgentSwitchTrigger,
} from './agents/orchestrator.js'
import type { AgentDefinition } from './agents/definitions.js'
import type { AuditEmitter } from './audit/emitter.js'
import { ConfigurationError } from './core/errors.js'
import { ToolRegistry } from './executors/toolRegistry.js'
import { createLlmClient, requireConfiguredModel, type LlmClient } from './llms/client.js'
import { ShortTermMemory, type Memory } from './memory/index.js'
import {
  QueryEngine,
  type QueryEngineConfig,
  type TurnResult,
} from './runtime/queryEngine.js'
import {
  DEFAULT_PERMISSION_MODE,
  type PermissionBroker,
  type PermissionMode,
  type ToolPolicy,
} from './streaming/permissions.js'
import type { StreamEvent } from './streaming/events.js'
import { SandboxRouter, SandboxedToolExecutor } from './security/sandbox.js'
import { registerCoreTools, type CoreToolsOptions } from './tools/index.js'
import type { ToolDefinition } from './types/toolCalls.js'

export interface XerxesAgentOptions {
  readonly capabilities?: readonly AgentCapability[]
  readonly fallbackAgentId?: string
  readonly switchTriggers?: readonly AgentSwitchTrigger[]
}

export interface XerxesQueryOptions {
  /** Select a registered agent; defaults to the orchestrator's active agent. */
  readonly agentId?: string
  /** Isolate this turn from the agent's default in-memory conversation. */
  readonly freshSession?: boolean
  /** Additional QueryEngine configuration. Agent/model/prompt options take precedence. */
  readonly config?: Partial<QueryEngineConfig>
  readonly model?: string
  readonly permissionMode?: PermissionMode
  readonly sessionId?: string
  readonly systemPrompt?: string
}

export interface XerxesOptions {
  readonly agents?: Iterable<AgentDefinition>
  readonly auditEmitter?: AuditEmitter
  /**
   * Register the safe Bun-native baseline tools. Pass false only when providing
   * a deliberately isolated registry.
   */
  readonly coreTools?: CoreToolsOptions | false
  readonly enableMemory?: boolean
  readonly llm?: LlmClient
  readonly memory?: Memory
  readonly memoryMinChars?: number
  readonly model?: string
  readonly permissionBroker?: PermissionBroker
  readonly permissionMode?: PermissionMode
  readonly policy?: ToolPolicy
  /** Optional strict/warn/off sandbox routing around all tool calls. */
  readonly sandboxRouter?: SandboxRouter
  readonly systemPrompt?: string
  readonly toolRegistry?: ToolRegistry
  readonly workspaceRoot?: string
}

interface RegisteredAgent {
  readonly definition: AgentDefinition
  readonly options: XerxesAgentOptions
}

const DEFAULT_MEMORY_MIN_CHARS = 32

/**
 * Bun-native embedded facade for the Xerxes runtime.
 *
 * It owns the agent router, tool registry, optional memory tier, and reusable
 * QueryEngine sessions. The daemon and API server use the same streaming loop,
 * so embedding this class does not take a separate execution path.
 */
export class Xerxes {
  readonly agentOrchestrator = new AgentOrchestrator()
  readonly auditEmitter: AuditEmitter | undefined
  readonly llm: LlmClient
  readonly memory: Memory | undefined
  readonly sandboxRouter: SandboxRouter | undefined
  readonly toolRegistry: ToolRegistry

  private readonly agents = new Map<string, RegisteredAgent>()
  private readonly defaultModel: string
  private readonly defaultPermissionMode: PermissionMode
  private readonly defaultSystemPrompt: string
  private readonly engines = new Map<string, QueryEngine>()
  private readonly memoryMinChars: number
  private readonly permissionBroker: PermissionBroker | undefined
  private readonly policy: ToolPolicy | undefined

  constructor(options: XerxesOptions = {}) {
    const suppliedAgents = options.agents ? [...options.agents] : []
    this.defaultModel = optionalModel(options.model)
    const clientModel = firstConfiguredModel(this.defaultModel, ...suppliedAgents.map(agent => agent.model))
    this.llm = options.llm ?? createLlmClient(requireConfiguredModel(clientModel))
    this.toolRegistry = options.toolRegistry ?? new ToolRegistry()
    this.memory = options.memory ?? (options.enableMemory ? new ShortTermMemory({ capacity: 100 }) : undefined)
    this.memoryMinChars = options.memoryMinChars ?? DEFAULT_MEMORY_MIN_CHARS
    this.defaultPermissionMode = options.permissionMode ?? DEFAULT_PERMISSION_MODE
    this.defaultSystemPrompt = options.systemPrompt ?? ''
    this.permissionBroker = options.permissionBroker
    this.policy = options.policy
    this.sandboxRouter = options.sandboxRouter
    this.auditEmitter = options.auditEmitter

    if (options.coreTools !== false) {
      const coreOptions = options.coreTools ?? {}
      registerCoreTools(this.toolRegistry, {
        ...coreOptions,
        ...(this.memory && coreOptions.memoryTools === undefined
          ? { memoryTools: { context: { memory: this.memory } } }
          : {}),
        ...(options.workspaceRoot ?? coreOptions.workspaceRoot
          ? { workspaceRoot: options.workspaceRoot ?? coreOptions.workspaceRoot }
          : {}),
      })
    }

    registerDefaultSwitchTriggers(this.agentOrchestrator)
    if (suppliedAgents.length) {
      for (const agent of suppliedAgents) this.registerAgent(agent)
    } else {
      this.registerAgent(defaultAgentDefinition(this.defaultModel, this.defaultSystemPrompt))
    }
  }

  get agentDefinitions(): ReadonlyMap<string, AgentDefinition> {
    return new Map([...this.agents.entries()].map(([id, agent]) => [id, agent.definition]))
  }

  get currentAgentId(): string {
    return this.agentOrchestrator.currentAgentId ?? failNoAgent()
  }

  /** Register a new routable agent definition. Names are stable public IDs. */
  registerAgent(definition: AgentDefinition, options: XerxesAgentOptions = {}): string {
    const id = definition.name
    if (!id) {
      throw new ConfigurationError('agent.name', 'must not be empty')
    }
    if (this.agents.has(id)) {
      throw new ConfigurationError('agent.name', 'is already registered: ' + id)
    }
    this.agentOrchestrator.registerAgent({
      id,
      name: definition.name,
      ...(options.capabilities ? { capabilities: options.capabilities } : {}),
      ...(options.fallbackAgentId ? { fallbackAgentId: options.fallbackAgentId } : {}),
      ...(options.switchTriggers ? { switchTriggers: options.switchTriggers } : {}),
    })
    this.agents.set(id, { definition, options })
    return id
  }

  /** Select a registered agent immediately. */
  selectAgent(agentId: string, reason = 'explicit selection'): void {
    this.requireAgent(agentId)
    this.agentOrchestrator.switchAgent(agentId, reason)
  }

  /**
   * Apply registered switch triggers and switch if one produced a valid target.
   *
   * Returns the selected ID, or undefined when no handoff was warranted.
   */
  evaluateAgentSwitch(context: Record<string, unknown>): string | undefined {
    const target = this.agentOrchestrator.shouldSwitchAgent(context)
    if (!target) return undefined
    this.selectAgent(target, 'triggered switch')
    return target
  }

  /** Construct an independently usable query engine for one agent/session. */
  createQueryEngine(options: XerxesQueryOptions = {}): QueryEngine {
    const agentId = options.agentId ?? this.currentAgentId
    const agent = this.requireAgent(agentId).definition
    const model = requireConfiguredModel(firstConfiguredModel(
      options.model,
      agent.model,
      options.config?.model,
      this.defaultModel,
    ))
    const systemPrompt = options.systemPrompt ?? (agent.systemPrompt || this.defaultSystemPrompt)
    const permissionMode = options.permissionMode ?? options.config?.permissionMode ?? this.defaultPermissionMode
    const queryConfig: Partial<QueryEngineConfig> = {
      ...options.config,
      agentId,
      model,
      permissionMode,
      ...(systemPrompt ? { systemPrompt } : {}),
    }
    return new QueryEngine({
      llm: this.llm,
      toolExecutor: this.sandboxRouter
        ? new SandboxedToolExecutor(this.toolRegistry, this.sandboxRouter)
        : this.toolRegistry,
      ...(this.permissionBroker ? { permissionBroker: this.permissionBroker } : {}),
      ...(this.policy ? { policy: this.policy } : {}),
    }, {
      config: queryConfig,
      ...(options.sessionId ? { sessionId: options.sessionId } : {}),
      tools: toolsForAgent(this.toolRegistry.definitions(agentId), agent),
    })
  }

  /** Run one prompt, preserving the selected agent's default conversation by default. */
  async run(prompt: string, options: XerxesQueryOptions = {}, signal?: AbortSignal): Promise<TurnResult> {
    const stream = this.runStream(prompt, options, signal)
    while (true) {
      const next = await stream.next()
      if (next.done) return next.value
    }
  }

  /**
   * Stream one prompt through the shared loop and expose the final structured
   * turn result when iteration completes.
   */
  async *runStream(
    prompt: string,
    options: XerxesQueryOptions = {},
    signal?: AbortSignal,
  ): AsyncGenerator<StreamEvent, TurnResult, void> {
    const agentId = options.agentId ?? this.currentAgentId
    const engine = this.engineFor(options)
    this.injectRelevantMemory(engine, prompt, agentId)
    const turnId = this.auditEmitter?.emitTurnStart({ agentId, sessionId: engine.sessionId, prompt })
    try {
      const events = engine.submitStream(prompt, signal)
      while (true) {
        const next = await events.next()
        if (next.done) {
          this.indexCompletedTurn(next.value, agentId, engine.sessionId)
          this.auditEmitter?.emitTurnEnd({
            agentId,
            sessionId: engine.sessionId,
            ...(turnId ? { turnId } : {}),
            content: next.value.output,
            functionCallsCount: next.value.toolCalls.length,
          })
          return next.value
        }
        this.auditEvent(next.value, agentId, engine.sessionId, turnId)
        yield next.value
      }
    } catch (error) {
      this.auditEmitter?.emitError({
        agentId,
        sessionId: engine.sessionId,
        ...(turnId ? { turnId } : {}),
        errorType: error instanceof Error ? error.name : 'Error',
        errorMessage: error instanceof Error ? error.message : String(error),
        context: 'xerxes_facade',
      })
      this.auditEmitter?.emitTurnEnd({
        agentId,
        sessionId: engine.sessionId,
        ...(turnId ? { turnId } : {}),
        content: '',
      })
      throw error
    }
  }

  /** Discard a reusable embedded session without affecting any daemon session. */
  closeSession(sessionId?: string, agentId = this.currentAgentId): boolean {
    const key = engineKey(agentId, sessionId)
    return this.engines.delete(key)
  }

  private engineFor(options: XerxesQueryOptions): QueryEngine {
    if (options.freshSession) {
      return this.createQueryEngine(options)
    }
    const agentId = options.agentId ?? this.currentAgentId
    const key = engineKey(agentId, options.sessionId)
    const existing = this.engines.get(key)
    if (existing) return existing
    const engine = this.createQueryEngine(options)
    this.engines.set(key, engine)
    return engine
  }

  private requireAgent(agentId: string): RegisteredAgent {
    const agent = this.agents.get(agentId)
    if (!agent) {
      throw new ConfigurationError('agent', 'is not registered: ' + agentId)
    }
    return agent
  }

  private injectRelevantMemory(engine: QueryEngine, prompt: string, agentId: string): void {
    if (!this.memory) return
    const memories = this.memory.search(prompt, 3, undefined, { useSemantic: true })
      .filter(item => item.content.trim())
    if (!memories.length) return
    if (engine.config.systemPrompt && !engine.state.messages.some(message => message.role === 'system')) {
      engine.state.messages.push({ role: 'system', content: engine.config.systemPrompt })
    }
    const context = memories.map(item => '- ' + item.content).join('\n')
    engine.state.messages.push({
      role: 'system',
      content: 'Relevant retained memory for this turn:\n' + context,
    })
    engine.state.metadata.memory_context_agent = agentId
  }

  private indexCompletedTurn(result: TurnResult, agentId: string, sessionId: string): void {
    const content = result.output.trim()
    if (!this.memory || content.length < this.memoryMinChars) return
    this.memory.save(content, { source: 'xerxes_facade' }, {
      agentId,
      conversationId: sessionId,
      importance: 0.5,
      memoryType: 'turn',
    })
  }

  private auditEvent(event: StreamEvent, agentId: string, sessionId: string, turnId: string | undefined): void {
    if (!this.auditEmitter) return
    const context = {
      agentId,
      sessionId,
      ...(turnId ? { turnId } : {}),
    }
    if (event.type === 'tool_start') {
      this.auditEmitter.emitToolCallAttempt({
        ...context,
        toolName: event.call.function.name,
        args: event.call.function.arguments,
      })
    } else if (event.type === 'tool_end') {
      if (!event.result.permitted) {
        this.auditEmitter.emitToolPolicyDecision({
          ...context,
          toolName: event.result.name,
          action: 'deny',
          source: 'permission',
        })
      } else if (event.result.result.startsWith('Tool execution failed:')) {
        this.auditEmitter.emitToolCallFailure({
          ...context,
          toolName: event.result.name,
          errorType: 'ToolExecutionError',
          errorMessage: event.result.result,
        })
      } else {
        this.auditEmitter.emitToolCallComplete({
          ...context,
          toolName: event.result.name,
          durationMs: event.result.durationMs,
          result: event.result.result,
        })
      }
    } else if (event.type === 'provider_retry' && event.final) {
      this.auditEmitter.emitError({
        ...context,
        errorType: 'ProviderError',
        errorMessage: event.error,
        context: 'provider_stream',
      })
    }
  }
}

function defaultAgentDefinition(model: string, systemPrompt: string): AgentDefinition {
  return {
    name: 'default',
    description: 'Default Bun-native Xerxes agent',
    systemPrompt,
    model,
    tools: [],
    allowedTools: null,
    excludeTools: [],
    source: 'runtime',
    maxDepth: 5,
    isolation: 'shared',
  }
}

function toolsForAgent(available: readonly ToolDefinition[], agent: AgentDefinition): ToolDefinition[] {
  const declared = new Set(agent.tools)
  const allowed = agent.allowedTools === null ? undefined : new Set(agent.allowedTools)
  const excluded = new Set(agent.excludeTools)
  return available.filter(tool => {
    const name = tool.function.name
    if (excluded.has(name)) return false
    if (allowed && !allowed.has(name)) return false
    return declared.size === 0 || declared.has(name)
  })
}

function engineKey(agentId: string, sessionId: string | undefined): string {
  return agentId + ':' + (sessionId ?? 'default')
}

function failNoAgent(): never {
  throw new ConfigurationError('agent', 'no agent is registered')
}

function firstConfiguredModel(...values: readonly (string | undefined)[]): string | undefined {
  for (const value of values) {
    const configured = optionalModel(value)
    if (configured) return configured
  }
  return undefined
}

function optionalModel(value: string | undefined): string {
  return value?.trim() ?? ''
}
