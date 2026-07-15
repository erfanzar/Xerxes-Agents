// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { daemonEvent, type DaemonEventFrame, type JsonRpcPayload } from '../protocol/jsonRpc.js'

export type WireEventNameStyle = 'daemon' | 'kimi'

export interface WireRequestFrame {
  readonly id: string
  readonly jsonrpc: '2.0'
  readonly method: 'request'
  readonly params: {
    readonly payload: JsonRpcPayload
    readonly type: string
  }
}

export type BridgeWireFrame = DaemonEventFrame | WireRequestFrame

/** Explicit output boundary; transports decide whether to queue, serialize, or send each frame. */
export interface WireEventSink {
  emit(frame: BridgeWireFrame): void
}

/** Explicit source of request, notification, and synthetic tool ids. */
export type WireIdFactory = () => string

export interface WireEventEmitterOptions {
  /** The daemon's snake_case protocol is the default; Kimi names are opt-in for legacy bridge clients. */
  readonly eventNameStyle?: WireEventNameStyle
  readonly idFactory: WireIdFactory
  readonly sink: WireEventSink
  readonly subagentPreviewChars?: number
}

export interface WireNotification {
  readonly body: string
  readonly category: string
  readonly id: string
  readonly payload?: JsonRpcPayload
  readonly severity: string
  readonly title: string
  readonly type: string
}

export interface WireToolResult {
  readonly durationMs?: number
  readonly permitted?: boolean
  readonly returnValue: string
  readonly toolCallId?: string
}

export type SubagentSummaryEvent =
  | 'agent_spawn'
  | 'agent_text'
  | 'agent_thinking'
  | 'agent_tool_start'
  | 'agent_tool_end'
  | 'agent_done'

/** Python-shaped agent event data, deliberately kept source-agnostic. */
export type SubagentSummaryData = Readonly<Record<string, unknown>>

const DEFAULT_SUBAGENT_PREVIEW_CHARS = 100

const KIMI_EVENT_NAME_BY_INTERNAL: Readonly<Record<string, string>> = Object.freeze({
  init_done: 'InitDone',
  turn_begin: 'TurnBegin',
  turn_end: 'TurnEnd',
  step_begin: 'StepBegin',
  step_end: 'StepEnd',
  step_interrupted: 'StepInterrupted',
  steer_input: 'SteerInput',
  compaction_begin: 'CompactionBegin',
  compaction_end: 'CompactionEnd',
  hook_triggered: 'HookTriggered',
  hook_resolved: 'HookResolved',
  mcp_loading_begin: 'MCPLoadingBegin',
  mcp_loading_end: 'MCPLoadingEnd',
  btw_begin: 'BtwBegin',
  btw_end: 'BtwEnd',
  text_part: 'TextPart',
  think_part: 'ThinkPart',
  image_url_part: 'ImageURLPart',
  audio_url_part: 'AudioURLPart',
  video_url_part: 'VideoURLPart',
  tool_call: 'ToolCall',
  tool_call_part: 'ToolCallPart',
  tool_result: 'ToolResult',
  tool_call_request: 'ToolCallRequest',
  approval_request: 'ApprovalRequest',
  approval_response: 'ApprovalResponse',
  question_request: 'QuestionRequest',
  question_response: 'QuestionResponse',
  status_update: 'StatusUpdate',
  notification: 'Notification',
  plan_display: 'PlanDisplay',
  subagent_event: 'SubagentEvent',
})

const INTERNAL_EVENT_NAME_BY_KIMI: Readonly<Record<string, string>> = Object.freeze(
  Object.fromEntries(Object.entries(KIMI_EVENT_NAME_BY_INTERNAL).map(([internal, kimi]) => [kimi, internal])),
)

/** Convert an internal snake_case wire event name to the legacy Kimi PascalCase spelling. */
export function toKimiWireEventName(eventType: string): string {
  return KIMI_EVENT_NAME_BY_INTERNAL[eventType] ?? eventType
}

/** Normalize a Kimi PascalCase wire event name to the Bun daemon's snake_case spelling. */
export function toDaemonWireEventName(eventType: string): string {
  return INTERNAL_EVENT_NAME_BY_KIMI[eventType] ?? eventType
}

/** Build an `event` JSON-RPC frame through the shared daemon protocol helper. */
export function wireEventFrame(
  eventType: string,
  payload: JsonRpcPayload,
  style: WireEventNameStyle = 'daemon',
): DaemonEventFrame {
  return daemonEvent(wireEventName(eventType, style), payload)
}

/** Build a client-response-bearing `request` JSON-RPC frame. */
export function wireRequestFrame(
  requestId: string,
  requestType: string,
  payload: JsonRpcPayload,
  style: WireEventNameStyle = 'daemon',
): WireRequestFrame {
  return {
    id: requiredText(requestId, 'requestId'),
    jsonrpc: '2.0',
    method: 'request',
    params: {
      payload,
      type: wireEventName(requestType, style),
    },
  }
}

/** Serialize a framed event for a newline-delimited transport without owning stdout or a lock. */
export function serializeWireFrame(frame: BridgeWireFrame): string {
  try {
    const serialized = JSON.stringify(frame)
    if (serialized === undefined) {
      throw new TypeError('frame is not JSON serializable')
    }
    return serialized
  } catch (error) {
    throw new WireEventSerializationError(errorMessage(error))
  }
}

/** Explicit serialization failure rather than Python's process-global stdout fallback. */
export class WireEventSerializationError extends Error {
  constructor(message: string) {
    super(`Wire frame serialization failed: ${message}`)
    this.name = 'WireEventSerializationError'
  }
}

/**
 * Connection-scoped formatter for bridge wire events.
 *
 * This class owns only formatting state (current tool id, function-tag
 * suppression, and subagent preview tails). It does not open a transport,
 * inspect process state, calculate status, or source subagent events.
 */
export class WireEventEmitter {
  private currentToolCallId = ''
  private readonly eventNameStyle: WireEventNameStyle
  private readonly idFactory: WireIdFactory
  private readonly sink: WireEventSink
  private readonly subagentParentTool = new Map<string, string>()
  private readonly subagentPreviewChars: number
  private readonly subagentTextBuffers = new Map<string, string>()
  private readonly subagentThinkingBuffers = new Map<string, string>()
  private readonly subagentToolIdFifo = new Map<string, string[]>()
  private stepCount = 0
  private suppressBuffer = ''
  private suppressingFunctionTag = false

  constructor(options: WireEventEmitterOptions) {
    this.sink = options.sink
    this.idFactory = options.idFactory
    this.eventNameStyle = options.eventNameStyle ?? 'daemon'
    this.subagentPreviewChars = normalizedPreviewChars(options.subagentPreviewChars)
  }

  get activeToolCallId(): string {
    return this.currentToolCallId
  }

  get currentStep(): number {
    return this.stepCount
  }

  /** Emit a generic protocol event through the configured sink. */
  emitEvent(eventType: string, payload: JsonRpcPayload): void {
    this.sink.emit(wireEventFrame(requiredText(eventType, 'eventType'), payload, this.eventNameStyle))
  }

  /** Emit a request frame that a client must answer with the returned id. */
  emitRequest(requestId: string, requestType: string, payload: JsonRpcPayload): void {
    this.sink.emit(wireRequestFrame(requestId, requestType, payload, this.eventNameStyle))
  }

  /** Emit a tool start, generating a stable local id when the upstream omitted it. */
  emitToolStart(toolCallId: string, name: string, arguments_: JsonRpcPayload): string {
    const id = toolCallId.trim() || this.generatedId('call_')
    this.currentToolCallId = id
    this.emitEvent('tool_call', {
      id,
      name,
      arguments: serializeArguments(arguments_),
    })
    return id
  }

  /** Emit the Python-compatible tool-result payload, resolving a missing id from the active tool. */
  emitToolResult(result: WireToolResult): void {
    const toolCallId = result.toolCallId?.trim() || this.currentToolCallId
    this.emitEvent('tool_result', {
      tool_call_id: toolCallId,
      return_value: result.returnValue,
      duration_ms: finiteNumber(result.durationMs),
      display_blocks: [],
    })
  }

  emitToolArgsPart(argumentsPart: string): void {
    this.emitEvent('tool_call_part', { arguments_part: argumentsPart })
  }

  emitThink(thinking: string): void {
    this.emitEvent('think_part', { think: thinking })
  }

  emitTurnBegin(userInput: string): void {
    this.stepCount = 0
    this.emitEvent('turn_begin', { user_input: [{ type: 'text', text: userInput }] })
  }

  emitTurnEnd(): void {
    this.emitEvent('turn_end', {})
  }

  emitStepBegin(step: number): void {
    this.stepCount = step
    this.emitEvent('step_begin', { n: step })
  }

  emitCompactionBegin(): void {
    this.emitEvent('compaction_begin', {})
  }

  emitCompactionEnd(): void {
    this.emitEvent('compaction_end', {})
  }

  /** Host-derived init data is forwarded as-is; this module does not inspect cwd, git, or skills. */
  emitInitDone(payload: JsonRpcPayload): void {
    this.emitEvent('init_done', payload)
  }

  /** Host-derived status data is forwarded as-is; token accounting remains outside this formatter. */
  emitStatus(payload: JsonRpcPayload): void {
    this.emitEvent('status_update', payload)
  }

  emitNotification(notification: WireNotification): void {
    this.emitEvent('notification', {
      id: requiredText(notification.id, 'notification.id'),
      category: notification.category,
      type: notification.type,
      severity: notification.severity,
      title: notification.title,
      body: notification.body,
      payload: notification.payload ?? {},
    })
  }

  /** Emit a permission request and return its client-facing correlation id. */
  emitPermissionRequest(toolCallId: string, name: string, description: string): string {
    const requestId = this.nextId()
    this.emitRequest(requestId, 'approval_request', {
      id: requestId,
      tool_call_id: toolCallId,
      action: name,
      description,
    })
    return requestId
  }

  /** Emit a question request and return its client-facing correlation id. */
  emitQuestionRequest(questions: readonly JsonRpcPayload[]): string {
    const requestId = this.nextId()
    this.emitRequest(requestId, 'question_request', {
      id: requestId,
      questions: [...questions],
    })
    return requestId
  }

  /**
   * Emit visible text while eliding inline `<function=...></function>` content.
   *
   * The suppression buffer is instance-local so partial chunks cannot leak
   * across clients or turns owned by other emitters.
   */
  emitText(text: string): void {
    if (this.suppressingFunctionTag) {
      this.suppressBuffer += text
      this.finishFunctionSuppressionIfClosed()
      return
    }

    const start = text.indexOf('<function=')
    if (start >= 0) {
      const before = text.slice(0, start)
      if (before.trim()) {
        this.emitEvent('text_part', { text: before })
      }
      this.suppressingFunctionTag = true
      this.suppressBuffer = text.slice(start)
      this.finishFunctionSuppressionIfClosed()
      return
    }

    const stripped = text.trim()
    if (!stripped || (stripped.startsWith('{"name":') && stripped.includes('"arguments"'))) {
      return
    }
    this.emitEvent('text_part', { text })
  }

  /**
   * Format one source-agnostic subagent lifecycle event.
   *
   * `false` means the source event is outside this bounded formatter; callers
   * must decide whether to render a separate host-specific fallback.
   */
  emitSubagentSummary(eventType: string, data: SubagentSummaryData): boolean {
    if (!isSubagentSummaryEvent(eventType)) {
      return false
    }
    const agentName = textField(data, 'agent_name', 'agentName') || textField(data, 'agent_type', 'agentType') || 'subagent'
    const agentType = textField(data, 'agent_type', 'agentType')
    const taskId = textField(data, 'task_id', 'taskId')
    const shortId = taskId.length > 8 ? `${taskId.slice(0, 8)}…` : taskId
    const prefix = shortId ? `${agentName}#${shortId}` : agentName

    switch (eventType) {
      case 'agent_spawn': {
        if (taskId && this.currentToolCallId) {
          this.subagentParentTool.set(taskId, this.currentToolCallId)
        }
        const depth = textField(data, 'depth') || '?'
        const prompt = textField(data, 'prompt').slice(0, 140)
        this.emitNotification({
          id: this.nextId(),
          category: 'subagent',
          type: eventType,
          severity: 'info',
          title: '',
          body: `${prefix} spawned (depth=${depth}): ${prompt}`,
        })
        this.emitSubagentStream(taskId, prefix, 'starting…')
        return true
      }
      case 'agent_text':
        this.streamSubagentChunk(taskId, prefix, textField(data, 'text'), 'text')
        return true
      case 'agent_thinking':
        this.streamSubagentChunk(taskId, prefix, textField(data, 'text'), 'thinking')
        return true
      case 'agent_tool_start': {
        this.emitSubagentToolEvent(taskId, agentType, data, 'start')
        const inputs = recordField(data, 'inputs')
        const firstInput = inputs ? Object.values(inputs)[0] : undefined
        this.emitSubagentStream(
          taskId,
          prefix,
          `◐ ${textField(data, 'tool_name', 'toolName') || 'tool'}(${previewValue(firstInput).slice(0, 80)})`,
        )
        return true
      }
      case 'agent_tool_end': {
        this.emitSubagentToolEvent(taskId, agentType, data, 'end')
        const permitted = booleanField(data, 'permitted', true)
        const mark = permitted ? '✓' : '✗'
        const duration = Math.round(numberField(data, 'duration_ms', 'durationMs'))
        this.emitSubagentStream(
          taskId,
          prefix,
          `${mark} ${textField(data, 'tool_name', 'toolName') || 'tool'} — ${duration}ms`,
        )
        return true
      }
      case 'agent_done':
        this.subagentParentTool.delete(taskId)
        this.subagentToolIdFifo.delete(taskId)
        this.subagentTextBuffers.delete(taskId)
        this.subagentThinkingBuffers.delete(taskId)
        this.emitSubagentStream(taskId, prefix, '')
        return true
    }
  }

  /**
   * Nest a subagent tool transition below the parent tool call when both ids
   * are available. No flat fallback is invented when correlation is missing.
   */
  emitSubagentToolEvent(
    taskId: string,
    agentType: string,
    data: SubagentSummaryData,
    kind: 'end' | 'start',
  ): boolean {
    const parentToolCallId = this.subagentParentTool.get(taskId) || this.currentToolCallId
    const rawInnerId = textField(data, 'tool_call_id', 'toolCallId')
    let innerId = rawInnerId
    if (kind === 'start') {
      innerId ||= this.generatedId('sub_')
      const ids = this.subagentToolIdFifo.get(taskId) ?? []
      ids.push(innerId)
      this.subagentToolIdFifo.set(taskId, ids)
    } else if (rawInnerId) {
      const ids = this.subagentToolIdFifo.get(taskId) ?? []
      const index = ids.indexOf(rawInnerId)
      if (index >= 0) {
        ids.splice(index, 1)
      }
    } else {
      const ids = this.subagentToolIdFifo.get(taskId) ?? []
      innerId = ids.shift() ?? ''
    }

    if (!parentToolCallId || !innerId) {
      return false
    }
    const event = kind === 'start'
      ? {
          type: 'ToolCall',
          payload: {
            id: innerId,
            name: textField(data, 'tool_name', 'toolName'),
            arguments: trySerializeArguments(recordField(data, 'inputs') ?? {}),
          },
        }
      : {
          type: 'ToolResult',
          payload: {
            tool_call_id: innerId,
            return_value: truthyText(data.result),
            display_blocks: [],
          },
        }
    this.emitEvent('subagent_event', {
      id: this.nextId(),
      parent_tool_call_id: parentToolCallId,
      agent_id: taskId,
      subagent_type: agentType,
      event,
    })
    return true
  }

  private emitSubagentStream(taskId: string, label: string, body: string): void {
    this.emitNotification({
      id: this.nextId(),
      category: 'subagent_stream',
      type: 'subagent_stream',
      severity: 'info',
      title: '',
      body,
      payload: { task_id: taskId, label },
    })
  }

  private finishFunctionSuppressionIfClosed(): void {
    const closingIndex = this.suppressBuffer.indexOf('</function>')
    if (closingIndex < 0) {
      return
    }
    const after = this.suppressBuffer.slice(closingIndex + '</function>'.length)
    this.suppressingFunctionTag = false
    this.suppressBuffer = ''
    if (after.trim()) {
      this.emitEvent('text_part', { text: after })
    }
  }

  private generatedId(prefix: string): string {
    return prefix + compactId(this.nextId()).slice(0, 12)
  }

  private nextId(): string {
    return requiredText(this.idFactory(), 'idFactory result')
  }

  private streamSubagentChunk(taskId: string, prefix: string, text: string, kind: 'text' | 'thinking'): void {
    if (!taskId || !text) {
      return
    }
    const buffers = kind === 'text' ? this.subagentTextBuffers : this.subagentThinkingBuffers
    let merged = (buffers.get(taskId) ?? '') + text
    const bufferCap = this.subagentPreviewChars * 2
    if (merged.length > bufferCap) {
      merged = merged.slice(-bufferCap)
    }
    buffers.set(taskId, merged)
    let tail = merged.replaceAll(/\s+/g, ' ').trim()
    if (!tail) {
      return
    }
    if (tail.length > this.subagentPreviewChars) {
      tail = `…${tail.slice(-this.subagentPreviewChars)}`
    }
    this.emitSubagentStream(taskId, kind === 'thinking' ? `${prefix} (thinking)` : prefix, tail)
  }
}

function booleanField(data: SubagentSummaryData, name: string, fallback: boolean): boolean {
  return typeof data[name] === 'boolean' ? data[name] : fallback
}

function compactId(value: string): string {
  const compact = value.replaceAll('-', '')
  return compact || value
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

function finiteNumber(value: number | undefined): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0
}

function isSubagentSummaryEvent(value: string): value is SubagentSummaryEvent {
  return value === 'agent_spawn'
    || value === 'agent_text'
    || value === 'agent_thinking'
    || value === 'agent_tool_start'
    || value === 'agent_tool_end'
    || value === 'agent_done'
}

function normalizedPreviewChars(value: number | undefined): number {
  if (value === undefined) {
    return DEFAULT_SUBAGENT_PREVIEW_CHARS
  }
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new TypeError('subagentPreviewChars must be a positive safe integer')
  }
  return value
}

function numberField(data: SubagentSummaryData, snakeCase: string, camelCase: string): number {
  const value = data[snakeCase] ?? data[camelCase]
  return typeof value === 'number' && Number.isFinite(value) ? value : 0
}

function previewValue(value: unknown): string {
  if (typeof value === 'string') {
    return value
  }
  if (value === undefined || value === null) {
    return ''
  }
  if (typeof value === 'number' || typeof value === 'boolean' || typeof value === 'bigint') {
    return String(value)
  }
  return trySerializeArguments(value)
}

function recordField(data: SubagentSummaryData, field: string): JsonRpcPayload | undefined {
  const value = data[field]
  return isRecord(value) ? value : undefined
}

function requiredText(value: string, field: string): string {
  if (typeof value !== 'string' || !value.trim()) {
    throw new TypeError(`${field} must be a non-empty string`)
  }
  return value.trim()
}

function serializeArguments(arguments_: unknown): string {
  try {
    const serialized = JSON.stringify(arguments_)
    if (serialized === undefined) {
      throw new TypeError('arguments are not JSON serializable')
    }
    return serialized
  } catch (error) {
    throw new WireEventSerializationError(errorMessage(error))
  }
}

function textField(data: SubagentSummaryData, snakeCase: string, camelCase?: string): string {
  const value = data[snakeCase] ?? (camelCase === undefined ? undefined : data[camelCase])
  return typeof value === 'string' ? value : value === undefined || value === null ? '' : String(value)
}

function truthyText(value: unknown): string {
  return value ? (typeof value === 'string' ? value : String(value)) : ''
}

function trySerializeArguments(value: unknown): string {
  try {
    return JSON.stringify(value) ?? ''
  } catch {
    return ''
  }
}

function wireEventName(eventType: string, style: WireEventNameStyle): string {
  const internal = toDaemonWireEventName(requiredText(eventType, 'eventType'))
  return style === 'kimi' ? toKimiWireEventName(internal) : internal
}

function isRecord(value: unknown): value is JsonRpcPayload {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
