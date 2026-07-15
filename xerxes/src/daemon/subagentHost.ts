// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { AgentDefinition } from '../agents/definitions.js'
import {
  SUBAGENT_BLOCKED_TOOLS,
  SubAgentManager,
  type SubAgentEvent,
  type SubAgentTask,
  type SubagentTaskRunRequest,
} from '../agents/subagentManager.js'
import { ValidationError } from '../core/errors.js'
import type { ToolExecutor } from '../executors/toolRegistry.js'
import type { LlmClient } from '../llms/client.js'
import type {
  SendAgentInputOptions,
  SpawnAgentOptions,
  SpawnedAgentManagerPort,
  SpawnedAgentSnapshot,
  SpawnedAgentStatus,
} from '../operators/subagents.js'
import { bootstrap } from '../runtime/bootstrap.js'
import { createAgentState } from '../streaming/events.js'
import { runTurn } from '../streaming/loop.js'
import type { PermissionBroker, PermissionMode } from '../streaming/permissions.js'
import type { ToolDefinition } from '../types/toolCalls.js'
import type { DaemonEvent } from './runtime.js'
import { DaemonSubagentEventBus } from './subagentEvents.js'

export interface NativeSubagentHostOptions {
  readonly agentDefinitions: ReadonlyMap<string, AgentDefinition>
  readonly cwd: string
  readonly eventBus: DaemonSubagentEventBus
  readonly llm: LlmClient
  readonly maxTokens?: number
  readonly model: string
  readonly permissionMode: PermissionMode
  readonly temperature?: number
  readonly toolExecutor: ToolExecutor
  readonly tools: readonly ToolDefinition[]
  readonly topP?: number
}

export interface NativeSubagentHost {
  readonly manager: SubAgentManager
  readonly managerPort: SpawnedAgentManagerPort
  /**
   * Apply the latest daemon/provider generation without discarding delegated
   * task handles. Existing tasks keep the execution generation they were
   * created with; subsequently spawned tasks use these options.
   */
  reconfigure(options: NativeSubagentHostOptions): void
}

/** Build the real Bun delegated-turn host used by daemon and OpenTUI sessions. */
export function createNativeSubagentHost(options: NativeSubagentHostOptions): NativeSubagentHost {
  let activeGeneration = 0
  let activeOptions = options
  const generationOptions = new Map<number, NativeSubagentHostOptions>([[activeGeneration, options]])
  const manager = new SubAgentManager({
    maxConcurrent: 8,
    maxDepth: 5,
    onEvent: event => publishSubagentEvent(options.eventBus, event),
    pathResolver: rawPath => rawPath,
    runner: request => {
      const generation = nativeHostGeneration(request.config)
      return runNativeSubagent(
        request,
        generation === undefined ? activeOptions : generationOptions.get(generation) ?? activeOptions,
      )
    },
  })
  const managerPort = new RichSubagentManagerPort(manager, options, activeGeneration)
  return {
    manager,
    managerPort,
    reconfigure(nextOptions) {
      if (nextOptions.eventBus !== options.eventBus) {
        throw new Error('A native subagent host cannot be moved to a different event bus')
      }
      activeGeneration += 1
      activeOptions = nextOptions
      generationOptions.set(activeGeneration, nextOptions)
      managerPort.reconfigure(nextOptions, activeGeneration)
    },
  }
}

interface HandleMetadata {
  readonly agentId: string
  closed: boolean
  readonly createdAt: string
  readonly creatorAgentId: string | undefined
  lastInput: string | undefined
  readonly parentAgentId: string | undefined
  readonly promptProfile: string
  readonly sourceAgentId: string | undefined
}

/** Adapt the richer native manager to the Claude-compatible tool contract. */
class RichSubagentManagerPort implements SpawnedAgentManagerPort {
  private availableTools: readonly ToolDefinition[]
  private definitions: ReadonlyMap<string, AgentDefinition>
  private fallbackModel: string
  private fallbackPermissionMode: PermissionMode
  private generation: number
  private readonly handles = new Map<string, HandleMetadata>()
  private readonly pendingResume = new Set<string>()

  constructor(
    private readonly manager: SubAgentManager,
    options: NativeSubagentHostOptions,
    generation: number,
  ) {
    this.availableTools = options.tools
    this.definitions = options.agentDefinitions
    this.fallbackModel = options.model
    this.fallbackPermissionMode = options.permissionMode
    this.generation = generation
  }

  reconfigure(options: NativeSubagentHostOptions, generation: number): void {
    this.availableTools = options.tools
    this.definitions = options.agentDefinitions
    this.fallbackModel = options.model
    this.fallbackPermissionMode = options.permissionMode
    this.generation = generation
  }

  listHandles(): SpawnedAgentSnapshot[] {
    return this.manager.listTasks().map(task => this.snapshot(task))
  }

  async spawn(options: SpawnAgentOptions = {}): Promise<SpawnedAgentSnapshot> {
    const prompt = (options.message ?? options.taskDescription)?.trim()
    if (!prompt) throw new ValidationError('message', 'spawned agent input is required', prompt)
    const name = options.nickname?.trim()
    if (name && this.manager.listTasks().some(task => task.name === name && !this.handles.get(task.id)?.closed)) {
      throw new ValidationError('nickname', 'already identifies a spawned agent', name)
    }

    const requestedType = options.promptProfile?.trim() || options.agent?.name?.trim() || 'coder'
    const definition = resolveDefinition(this.definitions, requestedType)
    const model = options.agent?.model || definition?.model || this.fallbackModel
    const permissionMode = permissionModeConfig(options.permissionMode, this.fallbackPermissionMode)
    const config = {
      model,
      permissionMode,
      _nativeSubagentHostGeneration: this.generation,
      ...(definition?.allowedTools === null || definition?.allowedTools === undefined
        ? {}
        : { _toolsAllowed: [...definition.allowedTools] }),
      ...(definition?.excludeTools.length ? { _toolsExcluded: [...definition.excludeTools] } : {}),
      ...(definition?.tools.length ? { _toolsWhitelist: [...definition.tools] } : {}),
    }
    const toolsets = subagentTools(this.availableTools, config).map(tool => tool.function.name)
    const rules = nativeRuleLabels(permissionMode, definition?.isolation || '')
    const task = await this.manager.spawn({
      prompt,
      ...(options.title ? { title: options.title } : {}),
      ...(name ? { name } : {}),
      ...(definition ? { agentDefinition: definition } : {}),
      ...(definition ? {} : options.agent?.systemPrompt ? { systemPrompt: options.agent.systemPrompt } : {}),
      ...(options.sourceAgentId ? { sourceId: options.sourceAgentId } : {}),
      ...(options.creatorAgentId ? { creatorId: options.creatorAgentId } : {}),
      ...(options.parentAgentId ? { parentId: options.parentAgentId } : {}),
      model,
      rules,
      toolsets,
      config,
    })
    this.handles.set(task.id, {
      agentId: definition?.name || requestedType,
      closed: false,
      createdAt: new Date().toISOString(),
      creatorAgentId: options.creatorAgentId,
      lastInput: prompt,
      parentAgentId: options.parentAgentId ?? options.creatorAgentId,
      promptProfile: definition?.name || requestedType,
      sourceAgentId: options.sourceAgentId,
    })
    return this.snapshot(task)
  }

  async sendInput(handleId: string | undefined, options: SendAgentInputOptions): Promise<SpawnedAgentSnapshot> {
    const task = this.requireTask(handleId)
    const input = (options.message ?? options.taskDescription)?.trim()
    if (!input) throw new ValidationError('message', 'spawned agent input is required', input)
    if (this.pendingResume.delete(task.id)) {
      const replacement = await this.manager.reset(task.id, input)
      if (!replacement) throw new ValidationError('handle_id', 'could not restart spawned agent', task.id)
      const previous = this.handles.get(task.id)
      if (previous) {
        previous.closed = true
        this.handles.set(replacement.id, { ...previous, closed: false, lastInput: input })
      }
      return this.snapshot(replacement)
    }
    if (!(await this.manager.sendMessage(task.id, input))) {
      throw new ValidationError('handle_id', 'spawned agent is not accepting input', task.id)
    }
    const metadata = this.handles.get(task.id)
    if (metadata) metadata.lastInput = input
    return this.snapshot(task)
  }

  async wait(targets: readonly string[], timeoutMs = 30_000): Promise<{
    readonly completed: readonly SpawnedAgentSnapshot[]
    readonly pending: readonly SpawnedAgentSnapshot[]
  }> {
    const ids = targets.map(target => this.requireTask(target).id)
    const result = await this.manager.waitAll(ids, timeoutMs)
    return {
      completed: result.completed.map(snapshot => this.snapshot(this.requireTask(snapshot.id))),
      pending: result.pending.map(snapshot => this.snapshot(this.requireTask(snapshot.id))),
    }
  }

  resume(handleId: string): SpawnedAgentSnapshot {
    const task = this.requireTask(handleId)
    this.pendingResume.add(task.id)
    const metadata = this.handles.get(task.id)
    if (metadata) metadata.closed = false
    return this.snapshot(task, 'idle')
  }

  close(handleId: string): SpawnedAgentSnapshot & { readonly previousStatus: SpawnedAgentStatus } {
    const task = this.requireTask(handleId)
    const previousStatus = spawnedStatus(task)
    this.manager.cancel(task.id)
    const metadata = this.handles.get(task.id)
    if (metadata) metadata.closed = true
    return { ...this.snapshot(task, 'closed'), previousStatus }
  }

  private requireTask(idOrName: string | undefined): SubAgentTask {
    const value = idOrName?.trim()
    const task = value
      ? this.manager.listTasks().find(candidate => candidate.id === value || candidate.name === value)
      : this.manager.listTasks().filter(candidate => !this.handles.get(candidate.id)?.closed).at(-1)
    if (!task) throw new ValidationError('handle_id', 'spawned agent not found', value)
    return task
  }

  private snapshot(task: SubAgentTask, statusOverride?: SpawnedAgentStatus): SpawnedAgentSnapshot {
    const metadata = this.handles.get(task.id) ?? {
      agentId: task.agentDefName || task.name,
      closed: false,
      createdAt: new Date().toISOString(),
      creatorAgentId: task.creatorId || undefined,
      lastInput: task.prompt,
      parentAgentId: task.parentId || undefined,
      promptProfile: task.agentDefName || 'coder',
      sourceAgentId: task.sourceId || undefined,
    }
    this.handles.set(task.id, metadata)
    const status = statusOverride ?? spawnedStatus(task)
    const updatedAt = new Date(task.lastActivityAt ?? Date.now()).toISOString()
    return Object.freeze({
      agentId: metadata.agentId,
      closed: metadata.closed || status === 'closed',
      createdAt: metadata.createdAt,
      ...(task.error ? { error: task.error } : {}),
      id: task.id,
      ...(metadata.lastInput ? { lastInput: metadata.lastInput } : {}),
      ...(task.result === undefined ? {} : { lastOutput: task.result }),
      name: task.name,
      title: task.title,
      ...(task.creatorId ? { creatorAgentId: task.creatorId } : {}),
      ...(task.parentId ? { parentAgentId: task.parentId } : {}),
      ...(task.model ? { model: task.model } : {}),
      rules: task.rules,
      toolsets: task.toolsets,
      ...(task.apiCalls === undefined ? {} : { apiCalls: task.apiCalls }),
      toolCalls: task.toolCallsCount,
      ...(task.inputTokens === undefined ? {} : { inputTokens: task.inputTokens }),
      ...(task.outputTokens === undefined ? {} : { outputTokens: task.outputTokens }),
      ...(task.reasoningTokens === undefined ? {} : { reasoningTokens: task.reasoningTokens }),
      filesRead: Object.freeze([...task.readFiles].sort()),
      filesWritten: Object.freeze([...task.writtenFiles].sort()),
      ...(task.result === undefined ? {} : { completionSummary: task.result.slice(0, 500) }),
      promptProfile: metadata.promptProfile,
      queueSize: task.inboxSize,
      ...(metadata.sourceAgentId ? { sourceAgentId: metadata.sourceAgentId } : {}),
      status,
      updatedAt,
    })
  }
}

async function runNativeSubagent(
  request: SubagentTaskRunRequest,
  options: NativeSubagentHostOptions,
): Promise<{ readonly content: string }> {
  const model = stringConfig(request.config.model) || options.model
  const permissionMode = permissionModeConfig(request.config.permissionMode, options.permissionMode)
  const permissionBroker = delegatedPermissionBroker(permissionMode)
  const tools = subagentTools(options.tools, request.config)
  const cwd = request.worktree?.path || options.cwd
  const boot = await bootstrap({ cwd, model, tools })
  const state = createAgentState()
  state.metadata.project_root = cwd
  let output = ''

  for await (const event of runTurn({
    agentId: request.task.agentDefName || request.task.id,
    ...(options.maxTokens === undefined ? {} : { maxTokens: options.maxTokens }),
    model,
    permissionMode,
    state,
    systemPrompt: [boot.systemPrompt, request.systemPrompt].filter(Boolean).join('\n\n'),
    ...(options.temperature === undefined ? {} : { temperature: options.temperature }),
    tools,
    ...(options.topP === undefined ? {} : { topP: options.topP }),
    userMessage: request.prompt,
  }, {
    llm: options.llm,
    ...(permissionBroker === undefined ? {} : { permissionBroker }),
    toolExecutor: options.toolExecutor,
  }, request.cancelSignal)) {
    switch (event.type) {
      case 'text':
        output += event.text
        request.report.text(event.text)
        break
      case 'thinking':
        request.report.thinking(event.text)
        break
      case 'tool_start':
        request.report.toolStart({
          inputs: event.call.function.arguments,
          name: event.call.function.name,
          toolCallId: event.call.id,
        })
        break
      case 'tool_end':
        request.report.toolEnd({
          durationMs: event.result.durationMs,
          name: event.result.name,
          permitted: event.result.permitted,
          result: event.result.result,
          toolCallId: event.result.toolCallId,
        })
        break
      case 'turn_done':
        request.report.usage({
          ...(event.apiCallsCount === undefined ? {} : { apiCalls: event.apiCallsCount }),
          model: event.model,
          toolCalls: event.toolCallsCount,
          ...(event.usageComplete ? {
            inputTokens: event.usage.inputTokens,
            outputTokens: event.usage.outputTokens,
            ...(event.usage.reasoningTokens === undefined ? {} : { reasoningTokens: event.usage.reasoningTokens }),
          } : {}),
        })
        break
      default:
        break
    }
  }
  return { content: latestAssistantText(state.messages) || output }
}

const DELEGATED_PROJECT_MEMORY_WRITES = new Set([
  'agent_memory_append',
  'agent_memory_journal',
  'agent_memory_write',
])

/**
 * A parent-approved auto-mode delegation may persist only project-scoped
 * memory through tools already admitted by the child agent definition. This
 * keeps DeepScan useful without granting children global-memory or workspace
 * write access and without opening eight concurrent approval prompts.
 */
function delegatedPermissionBroker(mode: PermissionMode): PermissionBroker | undefined {
  if (mode !== 'auto') return undefined
  return {
    request: async request =>
      DELEGATED_PROJECT_MEMORY_WRITES.has(request.toolCall.function.name)
        && request.inputs.scope === 'project'
        ? 'approve'
        : 'reject',
  }
}

function subagentTools(
  definitions: readonly ToolDefinition[],
  config: Readonly<Record<string, unknown>>,
): ToolDefinition[] {
  const whitelist = stringList(config._toolsWhitelist)
  const allowed = stringList(config._toolsAllowed)
  const excluded = new Set(stringList(config._toolsExcluded))
  return definitions.filter(definition => {
    const name = definition.function.name
    if (SUBAGENT_BLOCKED_TOOLS.has(name) || excluded.has(name)) return false
    if (whitelist.length && !whitelist.includes(name)) return false
    return !allowed.length || allowed.includes(name)
  })
}

function publishSubagentEvent(bus: DaemonSubagentEventBus, event: SubAgentEvent): void {
  if (!event.sourceId) return
  const daemonEvent = daemonEventFromSubagent(event)
  if (daemonEvent) bus.publish(event.sourceId, daemonEvent)
}

function daemonEventFromSubagent(event: SubAgentEvent): DaemonEvent | undefined {
  const base = {
    agent_id: event.taskId,
    agent_name: event.agent,
    title: event.title,
    creator_id: event.creatorId || null,
    depth: event.depth,
    files_read: event.filesRead,
    files_written: event.filesWritten,
    goal: event.goal,
    parent_id: event.parentId || null,
    model: event.model || undefined,
    rules: event.rules,
    toolsets: event.toolsets,
    tool_count: event.toolCalls,
    ...(event.apiCalls === undefined ? {} : { api_calls: event.apiCalls }),
    ...(event.inputTokens === undefined ? {} : { input_tokens: event.inputTokens }),
    ...(event.outputTokens === undefined ? {} : { output_tokens: event.outputTokens }),
    ...(event.reasoningTokens === undefined ? {} : { reasoning_tokens: event.reasoningTokens }),
    ...(event.completionSummary === undefined ? {} : { summary: event.completionSummary }),
    subagent_type: event.agentType || event.agent,
    task_index: event.sequence,
  }
  const data = event.data
  switch (event.type) {
    case 'spawn':
      return { type: 'subagent_event', payload: { ...base, event: { type: 'turn_begin', payload: { status: 'running' } } } }
    case 'thinking':
      return { type: 'subagent_event', payload: { ...base, event: { type: 'think_part', payload: { think: textValue(data.preview) } } } }
    case 'text_burst':
      return { type: 'subagent_event', payload: { ...base, event: { type: 'text_part', payload: { text: textValue(data.preview) } } } }
    case 'tool_start':
      return {
        type: 'subagent_event',
        payload: {
          ...base,
          event: {
            type: 'tool_call',
            payload: {
              arguments: textValue(data.inputPreview),
              id: textValue(data.toolCallId),
              name: textValue(data.tool),
            },
          },
        },
      }
    case 'tool_end':
      return {
        type: 'subagent_event',
        payload: {
          ...base,
          event: {
            type: 'tool_result',
            payload: {
              duration_ms: numberValue(data.durationMs),
              name: textValue(data.tool),
              permitted: data.permitted !== false,
              return_value: textValue(data.resultPreview),
              tool_call_id: textValue(data.toolCallId),
            },
          },
        },
      }
    case 'done':
      return {
        type: 'subagent_event',
        payload: {
          ...base,
          event: {
            type: 'turn_end',
            payload: {
              status: textValue(data.status) || 'completed',
              summary: textValue(data.resultPreview),
              tool_count: numberValue(data.toolCalls),
            },
          },
        },
      }
    case 'coordination':
      return {
        type: 'subagent_event',
        payload: { ...base, event: { type: 'text_part', payload: { text: `re-reading ${textValue(data.path)}` } } },
      }
    default:
      return undefined
  }
}

function nativeRuleLabels(permissionMode: PermissionMode, isolation: string): readonly string[] {
  return Object.freeze([
    `permission:${permissionMode}`,
    'delegation:blocked',
    ...(isolation ? [`isolation:${isolation}`] : []),
  ])
}

function resolveDefinition(
  definitions: ReadonlyMap<string, AgentDefinition>,
  requested: string,
): AgentDefinition | undefined {
  return definitions.get(requested)
    ?? (requested === 'general-purpose' || requested === 'general' || requested === 'explore'
      ? definitions.get(requested === 'explore' ? 'researcher' : 'coder')
      : undefined)
}

function spawnedStatus(task: SubAgentTask): SpawnedAgentStatus {
  switch (task.status) {
    case 'pending': return 'idle'
    case 'running': return 'running'
    case 'completed': return 'completed'
    case 'cancelled': return 'cancelled'
    case 'failed': return 'error'
  }
}

function permissionModeConfig(value: unknown, fallback: PermissionMode): PermissionMode {
  return value === 'accept-all' || value === 'auto' || value === 'manual' || value === 'plan' ? value : fallback
}

function nativeHostGeneration(config: Readonly<Record<string, unknown>>): number | undefined {
  const value = config._nativeSubagentHostGeneration
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0 ? value : undefined
}

function stringConfig(value: unknown): string {
  return typeof value === 'string' ? value.trim() : ''
}

function stringList(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string') : []
}

function textValue(value: unknown): string {
  return typeof value === 'string' ? value : ''
}

function numberValue(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0
}

function latestAssistantText(messages: readonly { readonly content: unknown; readonly role: string }[]): string {
  const message = messages.slice().reverse().find(candidate => candidate.role === 'assistant')
  return typeof message?.content === 'string' ? message.content : ''
}
