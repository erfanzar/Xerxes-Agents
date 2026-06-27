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
const bool = (v: unknown, fallback = false): boolean => (typeof v === 'boolean' ? v : fallback)

const asRecord = (v: unknown): Record<string, unknown> =>
  v && typeof v === 'object' ? (v as Record<string, unknown>) : {}
const asStringRecord = (v: unknown): Record<string, string> =>
  Object.fromEntries(Object.entries(asRecord(v)).map(([key, value]) => [key, String(value ?? '')]))

export function sessionInfoFromInit(payload: Record<string, unknown>): SessionInfo {
  const skills = payload.skills
  const skillList = Array.isArray(skills) ? skills.map(s => String(s)).filter(Boolean) : []
  return {
    cwd: str(payload.cwd),
    head_hash: str(payload.head_hash),
    model: str(payload.model),
    mode: str(payload.mode, 'code'),
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
    if (role !== 'assistant' && role !== 'system' && role !== 'tool' && role !== 'user') {
      continue
    }
    const text = textFromContent(msg.content)
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

    case 'status_update':
      return [
        {
          type: 'status.update',
          payload: {
            kind: 'status',
            mode: str(payload.mode),
            reasoning_effort: str(payload.reasoning_effort),
            text: statusText(payload),
            usage: usageFromStatus(payload)
          }
        },
        {
          type: 'session.info',
          payload: {
            model: str(payload.model),
            mode: str(payload.mode),
            reasoning_effort: str(payload.reasoning_effort),
            skills: {},
            tools: {},
            usage: usageFromStatus(payload)
          }
        }
      ]

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
            allow_permanent: true,
            command: str(payload.action, str(payload.name, 'tool')),
            description: str(payload.description, str(payload.action, 'approval required'))
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
  return {
    type: 'tool.complete',
    payload: {
      duration_s: num(payload.duration_ms) / 1000,
      inline_diff: str(diff?.diff),
      name: str(payload.name),
      result_text: preview,
      summary: str(brief?.body, str(generic?.content, firstLine(preview || text))),
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
  return {
    type: 'clarify.request',
    payload: {
      choices,
      question: str(first.question, 'Input required'),
      request_id: `${daemonRequest}:${questionId}`
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
      return [{ type: 'transcript.append', payload: { role: 'user', text: body } }]
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
  const goal = firstNonEmptyStr(
    streamPayload.agent_type,
    streamPayload.agent_name,
    streamPayload.name,
    streamPayload.label,
    taskId,
    'subagent'
  )
  const base: SubagentEventPayload = {
    goal,
    parent_id: optionalStr(streamPayload.parent) ?? null,
    status,
    subagent_id: taskId,
    task_index: num(streamPayload.task_index),
    text,
    tool_count: count
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
  const nestedType = str(nested.type)
  const nestedPayload = asRecord(nested.payload)
  const subagentId = str(payload.agent_id, str(payload.parent_tool_call_id, 'subagent'))
  const base: SubagentEventPayload = {
    goal: firstNonEmptyStr(payload.subagent_type, payload.agent_name, subagentId, 'subagent'),
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
  if (nestedType === 'turn_end') {
    return [{ type: 'subagent.complete', payload: { ...base, status: 'completed' } }]
  }
  return [{ type: 'subagent.progress', payload: { ...base, text: nestedType, status: 'running' } }]
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
