// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import type { AnyEvent, GatewayEvent, GatewayTranscriptMessage, SubagentEventPayload } from './gatewayTypes.js'
import type { SessionInfo, Usage } from './types.js'

const TOOL_RESULT_PREVIEW_CHARS = 600

const str = (v: unknown, fallback = ''): string => (typeof v === 'string' ? v : fallback)
const optionalStr = (v: unknown): string | undefined => (typeof v === 'string' && v ? v : undefined)
const firstNonEmptyStr = (...values: unknown[]): string => {
  for (const value of values) {
    if (typeof value !== 'string') {
      continue
    }
    const trimmed = value.trim()
    if (trimmed) {
      return trimmed
    }
  }
  return ''
}
const num = (v: unknown, fallback = 0): number => (typeof v === 'number' && Number.isFinite(v) ? v : fallback)
const optionalNum = (v: unknown): number | undefined => (typeof v === 'number' && Number.isFinite(v) ? v : undefined)
const bool = (v: unknown, fallback = false): boolean => (typeof v === 'boolean' ? v : fallback)

const asRecord = (v: unknown): Record<string, unknown> =>
  v && typeof v === 'object' ? (v as Record<string, unknown>) : {}
const asStringRecord = (v: unknown): Record<string, string> =>
  Object.fromEntries(Object.entries(asRecord(v)).map(([key, value]) => [key, String(value ?? '')]))

const firstDefined = (sources: readonly Record<string, unknown>[], key: string): unknown => {
  for (const source of sources) {
    if (source[key] !== undefined && source[key] !== null) return source[key]
  }

  return undefined
}

const stringList = (value: unknown): string[] | undefined => {
  if (!Array.isArray(value)) return undefined
  const values = value.map(item => String(item).trim()).filter(Boolean)

  return values.length ? values : undefined
}

/** Preserve optional observability metadata instead of replacing missing values with zero. */
const subagentMetadata = (...sources: Record<string, unknown>[]): Partial<SubagentEventPayload> => {
  const stringField = (key: string) => optionalStr(firstDefined(sources, key))
  const numberField = (key: string) => optionalNum(firstDefined(sources, key))
  const listField = (key: string) => stringList(firstDefined(sources, key))
  const output = firstDefined(sources, 'output_tail')
  const outputTail = Array.isArray(output)
    ? output.map(value => {
        const row = asRecord(value)

        return { is_error: bool(row.is_error), preview: str(row.preview), tool: str(row.tool, 'tool') }
      })
    : undefined

  return {
    ...(stringField('agent_name') ? { agent_name: stringField('agent_name') } : {}),
    ...(stringField('agent_type') ? { agent_type: stringField('agent_type') } : {}),
    ...(numberField('api_calls') !== undefined ? { api_calls: numberField('api_calls') } : {}),
    ...(numberField('cost_usd') !== undefined ? { cost_usd: numberField('cost_usd') } : {}),
    ...(stringField('creator_id') ? { creator_id: stringField('creator_id') } : {}),
    ...(numberField('depth') !== undefined ? { depth: numberField('depth') } : {}),
    ...(numberField('duration_seconds') !== undefined ? { duration_seconds: numberField('duration_seconds') } : {}),
    ...(listField('files_read') ? { files_read: listField('files_read') } : {}),
    ...(listField('files_written') ? { files_written: listField('files_written') } : {}),
    ...(numberField('input_tokens') !== undefined ? { input_tokens: numberField('input_tokens') } : {}),
    ...(numberField('iteration') !== undefined ? { iteration: numberField('iteration') } : {}),
    ...(stringField('model') ? { model: stringField('model') } : {}),
    ...(outputTail ? { output_tail: outputTail } : {}),
    ...(numberField('output_tokens') !== undefined ? { output_tokens: numberField('output_tokens') } : {}),
    ...(numberField('reasoning_tokens') !== undefined ? { reasoning_tokens: numberField('reasoning_tokens') } : {}),
    ...(listField('rules') ? { rules: listField('rules') } : {}),
    ...(stringField('summary') ? { summary: stringField('summary') } : {}),
    ...(numberField('task_count') !== undefined ? { task_count: numberField('task_count') } : {}),
    ...(stringField('title') ? { title: stringField('title') } : {}),
    ...(numberField('tool_count') !== undefined ? { tool_count: numberField('tool_count') } : {}),
    ...(listField('toolsets') ? { toolsets: listField('toolsets') } : {})
  }
}

export function sessionInfoFromInit(payload: Record<string, unknown>): SessionInfo {
  const skills = payload.skills
  const skillList = Array.isArray(skills) ? skills.map(s => String(s)).filter(Boolean) : []
  return {
    cwd: str(payload.cwd),
    head_hash: str(payload.head_hash),
    model: str(payload.model),
    mode: str(payload.mode, 'code'),
    permission_mode: permissionMode(payload.permission_mode),
    profile_name: optionalStr(payload.agent_name),
    reasoning_effort: str(payload.reasoning_effort, 'off'),
    session_id: optionalStr(payload.session_id),
    skillDescriptions: asStringRecord(payload.skill_descriptions),
    skills: skillList.length ? { skills: skillList } : {},
    tools: {},
    usage: usageFromStatus(payload),
    version: str(payload.version)
  }
}

export function transcriptFromStoredMessages(messages: unknown): GatewayTranscriptMessage[] {
  if (!Array.isArray(messages)) {
    return []
  }

  const out: GatewayTranscriptMessage[] = []
  for (const raw of messages) {
    const msg = asRecord(raw)
    const role = str(msg.role)
    // Persisted system/tool rows are runtime state, not visible chat history.
    // Restoring them bypasses the daemon's user/assistant-only replay contract;
    // a full system prompt can also mount tens of thousands of hidden chars.
    if (role !== 'assistant' && role !== 'user') {
      continue
    }
    const text = firstNonEmptyStr(msg.text, textFromContent(msg.content))
    if (!text.trim()) {
      continue
    }
    out.push({ role, text })
  }
  return out
}

export function usageFromStatus(payload: Record<string, unknown>): Usage {
  const input = num(payload.total_input_tokens ?? payload.input_tokens)
  const output = num(payload.total_output_tokens ?? payload.output_tokens)
  const total = num(payload.total_tokens, input + output)
  return {
    calls: num(payload.calls),
    context_max: num(payload.context_limit ?? payload.max_context),
    context_used: num(payload.context_tokens),
    input,
    output,
    total
  }
}

export function adaptDaemonEvent(type: string, payload: Record<string, unknown>): AnyEvent[] {
  switch (type) {
    case 'init_done':
      return [
        { type: 'session.info', payload: sessionInfoFromInit(payload) },
        {
          type: 'status.update',
          payload: {
            kind: 'status',
            mode: str(payload.mode),
            reasoning_effort: str(payload.reasoning_effort),
            text: statusText(payload),
            usage: usageFromStatus(payload)
          }
        }
      ]

    case 'status_update': {
      const mode = optionalStr(payload.mode)
      const activePermissionMode = permissionMode(payload.permission_mode)
      const reasoningEffort = optionalStr(payload.reasoning_effort)

      return [
        {
          type: 'status.update',
          payload: {
            kind: 'status',
            ...(mode ? { mode } : {}),
            ...(activePermissionMode ? { permission_mode: activePermissionMode } : {}),
            ...(reasoningEffort ? { reasoning_effort: reasoningEffort } : {}),
            text: statusText(payload),
            usage: usageFromStatus(payload)
          }
        },
        {
          type: 'session.info',
          payload: {
            model: str(payload.model),
            ...(mode ? { mode } : {}),
            ...(activePermissionMode ? { permission_mode: activePermissionMode } : {}),
            ...(reasoningEffort ? { reasoning_effort: reasoningEffort } : {}),
            skills: {},
            tools: {},
            usage: usageFromStatus(payload)
          }
        }
      ]
    }

    case 'turn_begin':
      return [{ type: 'message.start', payload: undefined }]

    case 'text_part':
      return [{ type: 'message.delta', payload: { text: str(payload.text) } }]

    case 'think_part':
      return [{ type: 'thinking.delta', payload: { text: str(payload.think) } }]

    case 'turn_end':
      return [{ type: 'message.complete', payload: {} }]

    case 'step_interrupted':
      return [{ type: 'message.complete', payload: { text: '[interrupted]' } }]

    case 'tool_call':
      return [
        {
          type: 'tool.start',
          payload: {
            args_text: str(payload.arguments),
            name: str(payload.name, 'tool'),
            tool_id: str(payload.id, str(payload.tool_call_id, 'tool'))
          }
        }
      ]

    case 'tool_call_part':
      return [
        {
          type: 'tool.progress',
          payload: {
            name: str(payload.name, 'tool'),
            preview: str(payload.arguments_part)
          }
        }
      ]

    case 'tool_result':
      return [toolComplete(payload)]

    case 'approval_request':
      return [
        {
          type: 'approval.request',
          payload: {
            allow_permanent: bool(payload.allow_permanent, true),
            command: str(payload.action, str(payload.name, 'tool')),
            description: str(payload.description, str(payload.action, 'approval required')),
            request_id: str(payload.id, str(payload.request_id))
          }
        }
      ]

    case 'question_request':
      return [clarifyRequest(payload)]

    case 'notification':
      return notificationEvents(payload)

    case 'plan_display':
      return [
        {
          type: 'status.update',
          payload: {
            kind: 'info',
            text: str(payload.content, str(payload.file_path, 'plan updated'))
          }
        }
      ]

    case 'subagent_event':
      return subagentEvents(payload)

    default:
      return [{ type, payload } as GatewayEvent]
  }
}

function permissionMode(value: unknown): SessionInfo['permission_mode'] {
  return value === 'accept-all' || value === 'auto' || value === 'manual' || value === 'plan' ? value : undefined
}

function statusText(payload: Record<string, unknown>): string {
  const context = num(payload.context_tokens)
  const max = num(payload.max_context ?? payload.context_limit)
  const mode = str(payload.mode)
  const parts = [mode && `mode: ${mode}`, max > 0 && `context: ${context}/${max}`].filter(Boolean)
  return parts.length ? parts.join(' · ') : 'ready'
}

function toolComplete(payload: Record<string, unknown>): AnyEvent {
  const blocks = Array.isArray(payload.display_blocks) ? payload.display_blocks.map(asRecord) : []
  const diff = blocks.find(b => b.type === 'diff')
  const todo = blocks.find(b => b.type === 'todo')
  const brief = blocks.find(b => b.type === 'brief')
  const generic = blocks.find(b => b.type === 'generic')
  const text = str(payload.return_value)
  const preview = compactResultPreview(text)
  const first = firstLine(preview || text)
  const failure = firstNonEmptyStr(
    payload.error,
    payload.permitted === false ? first || 'Tool execution denied.' : '',
    /^(?:tool execution failed|error|exception|failed|failure|denied|fatal)(?:\b|:)/i.test(first) ? first : ''
  )
  return {
    type: 'tool.complete',
    payload: {
      duration_s: num(payload.duration_ms) / 1000,
      ...(failure ? { error: failure } : {}),
      inline_diff: str(diff?.diff),
      name: str(payload.name),
      result_text: preview,
      summary: str(brief?.body, str(generic?.content, first)),
      todos: Array.isArray(todo?.items) ? todo.items : undefined,
      tool_id: str(payload.tool_call_id, 'tool')
    }
  }
}

function clarifyRequest(payload: Record<string, unknown>): AnyEvent {
  const questions = Array.isArray(payload.questions) ? payload.questions.map(asRecord) : []
  const first = questions[0] ?? {}
  const choices = Array.isArray(first.options) ? first.options.map(String) : null
  const daemonRequest = str(payload.id, 'question')
  const questionId = str(first.id, 'q')
  const source = str(payload.flow) === 'provider' ? 'provider' : 'agent'
  const toolId = optionalStr(payload.tool_call_id)
  return {
    type: 'clarify.request',
    payload: {
      allow_free_form: bool(first.allow_free_form, true),
      choices,
      placeholder: optionalStr(first.placeholder),
      question: str(first.question, 'Input required'),
      question_id: questionId,
      request_id: `${daemonRequest}:${questionId}`,
      source,
      ...(toolId ? { tool_id: toolId } : {})
    }
  }
}

function notificationEvents(payload: Record<string, unknown>): AnyEvent[] {
  const category = str(payload.category)
  const kind = str(payload.type)
  const body = str(payload.body, str(payload.title))

  if (category === 'subagent_stream') {
    return subagentStreamEvents(payload, body)
  }

  if (category === 'history') {
    if (kind === 'replay_assistant') {
      return [{ type: 'transcript.append', payload: { role: 'assistant', text: body } }]
    }
    if (kind === 'replay_user') {
      // The daemon prefixes replay-only history notifications with a sparkle
      // so terminal log consumers can distinguish them. It is transport
      // metadata, not part of the user's authored message.
      return [{ type: 'transcript.append', payload: { role: 'user', text: body.replace(/^✨\s?/, '') } }]
    }
    return [
      {
        type: 'status.update',
        payload: { kind: 'info', text: body }
      }
    ]
  }

  return [
    {
      type: 'notification.show',
      payload: {
        id: str(payload.id),
        key: str(payload.category),
        kind: 'ttl',
        level: severityToLevel(str(payload.severity)),
        text: body,
        ttl_ms: 8000
      }
    }
  ]
}

function subagentStreamEvents(payload: Record<string, unknown>, body: string): AnyEvent[] {
  const streamPayload = asRecord(payload.payload)
  const taskId = str(streamPayload.task_id, str(payload.id, 'subagent'))
  const status = subagentStreamStatus(str(streamPayload.status, body ? 'running' : 'completed'))
  const count = num(streamPayload.count)
  const action = str(streamPayload.action, body)
  const result = str(streamPayload.result)
  const text = action || body || result
  const agentName = firstNonEmptyStr(streamPayload.agent_name, streamPayload.name, streamPayload.label)
  const agentType = firstNonEmptyStr(streamPayload.agent_type)
  const goal = firstNonEmptyStr(
    streamPayload.agent_type,
    streamPayload.agent_name,
    streamPayload.name,
    streamPayload.label,
    taskId,
    'subagent'
  )
  const base: SubagentEventPayload = {
    ...subagentMetadata(streamPayload, payload),
    ...(agentName ? { agent_name: agentName } : {}),
    ...(agentType ? { agent_type: agentType } : {}),
    goal,
    parent_id: optionalStr(streamPayload.parent) ?? null,
    status,
    subagent_id: taskId,
    task_index: num(streamPayload.task_index),
    text,
    ...(count > 0 ? { tool_count: count } : {})
  }

  if (status !== 'running' && status !== 'queued') {
    return [{ type: 'subagent.complete', payload: { ...base, summary: result || text } }]
  }

  if (!body && !action) {
    return [{ type: 'subagent.complete', payload: { ...base, status: 'completed' } }]
  }

  return [
    { type: 'subagent.start', payload: base },
    { type: 'subagent.progress', payload: base }
  ]
}

function subagentStreamStatus(status: string): NonNullable<SubagentEventPayload['status']> {
  if (
    status === 'completed' ||
    status === 'error' ||
    status === 'failed' ||
    status === 'interrupted' ||
    status === 'queued' ||
    status === 'running' ||
    status === 'timeout'
  ) {
    return status
  }

  if (status === 'cancelled' || status === 'canceled') {
    return 'interrupted'
  }

  if (status === 'done' || status === 'success') {
    return 'completed'
  }

  return 'running'
}

function subagentEvents(payload: Record<string, unknown>): AnyEvent[] {
  const nested = asRecord(payload.event)
  const nestedType = normalizedNestedEventType(str(nested.type))
  const nestedPayload = asRecord(nested.payload)
  const subagentId = str(payload.agent_id, str(payload.parent_tool_call_id, 'subagent'))
  const agentName = firstNonEmptyStr(payload.agent_name)
  const base: SubagentEventPayload = {
    ...subagentMetadata(nestedPayload, payload),
    ...(agentName ? { agent_name: agentName } : {}),
    agent_type: firstNonEmptyStr(payload.subagent_type, payload.agent_name),
    depth: num(payload.depth),
    goal: firstNonEmptyStr(payload.goal, payload.subagent_type, payload.agent_name, subagentId, 'subagent'),
    parent_id: optionalStr(payload.parent_id) ?? null,
    subagent_id: subagentId,
    task_index: num(payload.task_index)
  }

  if (nestedType === 'turn_begin') {
    return [{ type: 'subagent.start', payload: { ...base, status: 'running' } }]
  }
  if (nestedType === 'think_part') {
    return [{ type: 'subagent.thinking', payload: { ...base, text: str(nestedPayload.think), status: 'running' } }]
  }
  if (nestedType === 'text_part') {
    return [{ type: 'subagent.progress', payload: { ...base, text: str(nestedPayload.text), status: 'running' } }]
  }
  if (nestedType === 'tool_call') {
    return [
      {
        type: 'subagent.tool',
        payload: {
          ...base,
          status: 'running',
          tool_name: str(nestedPayload.name),
          tool_preview: str(nestedPayload.arguments)
        }
      }
    ]
  }
  if (nestedType === 'tool_result') {
    const tool = str(nestedPayload.name, 'tool')
    const result = firstNonEmptyStr(nestedPayload.return_value, nestedPayload.result)
    return [
      {
        type: 'subagent.progress',
        payload: {
          ...base,
          status: nestedPayload.permitted === false ? 'failed' : 'running',
          text: `${nestedPayload.permitted === false ? '✗' : '✓'} ${tool}${result ? ` — ${firstLine(result)}` : ''}`
        }
      }
    ]
  }
  if (nestedType === 'turn_end') {
    const summary = firstNonEmptyStr(nestedPayload.summary, nestedPayload.result)
    const toolCount = optionalNum(nestedPayload.tool_count)

    return [
      {
        type: 'subagent.complete',
        payload: {
          ...base,
          status: subagentStreamStatus(str(nestedPayload.status, 'completed')),
          ...(summary ? { summary } : {}),
          ...(toolCount !== undefined ? { tool_count: toolCount } : {})
        }
      }
    ]
  }
  return [{ type: 'subagent.progress', payload: { ...base, text: nestedType, status: 'running' } }]
}

function normalizedNestedEventType(value: string): string {
  return value
    .replace(/([a-z0-9])([A-Z])/g, '$1_$2')
    .replaceAll('-', '_')
    .toLowerCase()
}

function severityToLevel(severity: string): 'error' | 'info' | 'success' | 'warn' {
  if (severity === 'error') {
    return 'error'
  }
  if (severity === 'warning' || severity === 'warn') {
    return 'warn'
  }
  if (severity === 'success') {
    return 'success'
  }
  return 'info'
}

function textFromContent(content: unknown): string {
  if (typeof content === 'string') {
    return content
  }
  if (Array.isArray(content)) {
    return content
      .map(item => textFromContent(item))
      .filter(Boolean)
      .join('\n')
  }
  const record = asRecord(content)
  return str(record.text, str(record.content))
}

function firstLine(text: string): string {
  return text.split('\n', 1)[0]?.trim() ?? ''
}

function compactResultPreview(text: string): string {
  const compact = text.replace(/\s+/g, ' ').trim()

  return compact.length > TOOL_RESULT_PREVIEW_CHARS ? `${compact.slice(0, TOOL_RESULT_PREVIEW_CHARS - 1)}…` : compact
}
