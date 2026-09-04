// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import {
  normalizeEventType,
  type AnyEvent,
  type GatewayEvent,
  type GatewayTranscriptMessage,
  type SubagentEventPayload
} from './gatewayTypes.js'
import { compact } from './lib/compact.js'
import { summarizeToolStartDisplay } from './lib/toolStartDisplay.js'
import type { SessionInfo, Usage } from './types.js'

const TOOL_RESULT_PREVIEW_CHARS = 600

// Bounds mirroring the daemon's replay previews (REPLAY_ARGUMENTS_PREVIEW_CHARS
// / REPLAY_RESULT_PREVIEW_CHARS in xerxes/src/daemon/server.ts): enough context
// to recognize a call, never a raw dump.
const STORED_ARGUMENTS_PREVIEW_CHARS = 200
const STORED_RESULT_PREVIEW_CHARS = 160

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

  // Each accessor is called once and held: calling it twice — inside the
  // guard and again inside the object — gave TypeScript nothing to narrow,
  // so every field stayed `T | undefined` against an optional property.
  const agentName = stringField('agent_name')
  const agentType = stringField('agent_type')
  const apiCalls = numberField('api_calls')
  const costUsd = numberField('cost_usd')
  const creatorId = stringField('creator_id')
  const depth = numberField('depth')
  const durationSeconds = numberField('duration_seconds')
  const filesRead = listField('files_read')
  const filesWritten = listField('files_written')
  const inputTokens = numberField('input_tokens')
  const iteration = numberField('iteration')
  const model = stringField('model')
  const outputTokens = numberField('output_tokens')
  const cacheReadTokens = numberField('cache_read_tokens')
  const cacheCreationTokens = numberField('cache_creation_tokens')
  const reasoningTokens = numberField('reasoning_tokens')
  const rules = listField('rules')
  const summary = stringField('summary')
  const taskCount = numberField('task_count')
  const title = stringField('title')
  const toolCount = numberField('tool_count')
  const toolsets = listField('toolsets')

  return {
    ...(agentName ? { agent_name: agentName } : {}),
    ...(agentType ? { agent_type: agentType } : {}),
    ...(apiCalls !== undefined ? { api_calls: apiCalls } : {}),
    ...(costUsd !== undefined ? { cost_usd: costUsd } : {}),
    ...(creatorId ? { creator_id: creatorId } : {}),
    ...(depth !== undefined ? { depth: depth } : {}),
    ...(durationSeconds !== undefined ? { duration_seconds: durationSeconds } : {}),
    ...(filesRead ? { files_read: filesRead } : {}),
    ...(filesWritten ? { files_written: filesWritten } : {}),
    ...(inputTokens !== undefined ? { input_tokens: inputTokens } : {}),
    ...(iteration !== undefined ? { iteration: iteration } : {}),
    ...(model ? { model: model } : {}),
    ...(outputTokens !== undefined ? { output_tokens: outputTokens } : {}),
    ...(cacheReadTokens !== undefined ? { cache_read_tokens: cacheReadTokens } : {}),
    ...(cacheCreationTokens !== undefined ? { cache_creation_tokens: cacheCreationTokens } : {}),
    ...(reasoningTokens !== undefined ? { reasoning_tokens: reasoningTokens } : {}),
    ...(rules ? { rules: rules } : {}),
    ...(summary ? { summary: summary } : {}),
    ...(taskCount !== undefined ? { task_count: taskCount } : {}),
    ...(title ? { title: title } : {}),
    ...(toolCount !== undefined ? { tool_count: toolCount } : {}),
    ...(toolsets ? { toolsets: toolsets } : {}),
    ...(outputTail ? { output_tail: outputTail } : {})
  }
}

export function sessionInfoFromInit(payload: Record<string, unknown>): SessionInfo {
  const skills = payload.skills
  const skillList = Array.isArray(skills) ? skills.map(s => String(s)).filter(Boolean) : []
  return compact<SessionInfo>({
    cwd: str(payload.cwd),
    goal: optionalStr(payload.goal),
    goal_phase: optionalStr(payload.goal_phase),
    head_hash: str(payload.head_hash),
    model: str(payload.model),
    mode: str(payload.mode, 'code'),
    permission_mode: permissionMode(payload.permission_mode),
    profile_name: optionalStr(payload.profile_name) ?? optionalStr(payload.agent_name),
    reasoning_effort: str(payload.reasoning_effort, 'off'),
    session_id: optionalStr(payload.session_id),
    skillDescriptions: asStringRecord(payload.skill_descriptions),
    skills: skillList.length ? { skills: skillList } : {},
    tools: {},
    usage: usageFromStatus(payload),
    version: str(payload.version)
  })
}

/**
 * Internal user prompts the transcript must never show. This list mirrors
 * looksLikeInternalReplayMessage in xerxes/src/daemon/server.ts — keep the two
 * in sync. The daemon filters these on the replay path; the live-reattach path
 * (session.open transcripts, session previews) filters them here.
 */
export function looksLikeInternalUserPrompt(text: string): boolean {
  const head = text.trimStart().slice(0, 64)
  if (head.startsWith('[Skill') && head.includes('activated')) {
    return true
  }
  if (
    [
      '[sub-agent events]',
      '[mid-turn steer from user]',
      '[steer from user]',
      '[steer from user saved for next turn]',
      '[Workspace guard]',
      '[Objective gate]',
      '[Previous conversation summary'
    ].some(prefix => head.startsWith(prefix))
  ) {
    return true
  }
  return [
    'Please compact this conversation:',
    'Write a reusable agent skill called',
    'Generate an image matching this brief'
  ].some(prefix => text.trimStart().startsWith(prefix))
}

/** Compact an arguments/result payload into one bounded single-line preview (server.ts replayPreviewText semantics). */
function storedPreviewText(value: unknown, limit: number): string {
  const raw =
    typeof value === 'string'
      ? value
      : value === undefined || value === null
        ? ''
        : safeJsonStringify(value)
  const compact = raw.replace(/\s+/g, ' ').trim()

  return compact.length > limit ? `${compact.slice(0, limit - 1)}…` : compact
}

function safeJsonStringify(value: unknown): string {
  try {
    return JSON.stringify(value) ?? ''
  } catch {
    return ''
  }
}

/** One compact diagnostic line for a failed stored tool result, '' when it succeeded. */
function storedToolFailure(msg: Record<string, unknown>, content: string): string {
  const first = firstLine(content)
  if (msg.is_error === true) {
    return storedPreviewText(content, STORED_RESULT_PREVIEW_CHARS) || 'Tool execution failed.'
  }
  if (msg.permitted === false) {
    return first || 'Tool execution denied.'
  }
  return /^(?:tool execution failed|error|exception|failed|failure|denied|fatal)(?:\b|:)/i.test(first) ? first : ''
}

/**
 * Map persisted raw session messages to the same rich rows the replay path
 * renders: assistant rows keep their thinking, assistant tool_calls become
 * tool rows carrying a compact arguments preview, and following role:"tool"
 * results reconcile ok/error (and duration, when persisted) onto their call.
 * Tool result CONTENT never reaches the transcript — replay parity is one
 * semantic row per call. Internal user prompts are filtered exactly like the
 * daemon's replay path. System rows stay hidden runtime state.
 */
export function transcriptFromStoredMessages(messages: unknown): GatewayTranscriptMessage[] {
  if (!Array.isArray(messages)) {
    return []
  }

  const out: GatewayTranscriptMessage[] = []
  const toolRowByCallId = new Map<string, GatewayTranscriptMessage>()
  for (const raw of messages) {
    const msg = asRecord(raw)
    const role = str(msg.role).toLowerCase()
    if (role === 'user') {
      const text = firstNonEmptyStr(msg.text, textFromContent(msg.content))
      if (!text.trim() || looksLikeInternalUserPrompt(text)) {
        continue
      }
      out.push({ role: 'user', text })
      continue
    }
    if (role === 'assistant') {
      const text = firstNonEmptyStr(msg.text, textFromContent(msg.content))
      const thinking = optionalStr(msg.thinking)
      if (text.trim()) {
        out.push({ role: 'assistant', text, ...(thinking?.trim() ? { thinking } : {}) })
      }
      const calls = Array.isArray(msg.tool_calls) ? msg.tool_calls : []
      for (const call of calls) {
        const record = asRecord(call)
        const fn = asRecord(record.function)
        const name = firstNonEmptyStr(fn.name, record.name) || 'tool'
        // OpenAI wire shape nests {name, arguments} under 'function'; legacy
        // rows carry top-level name/input instead. Run the raw arguments through
        // the same summarizer live tool.start events use, so a reattached row
        // reads "directory_path=…" instead of a raw JSON blob. Summarize the
        // FULL arguments and truncate the rendered summary afterwards: cutting
        // the JSON first leaves an unparseable fragment, which the summarizer
        // then echoes back as the raw blob this comment promises to avoid.
        const rawValue = fn.arguments ?? record.input
        const argumentsText = typeof rawValue === 'string' ? rawValue : safeJsonStringify(rawValue)
        const summary = argumentsText.trim()
          ? summarizeToolStartDisplay(name, '', argumentsText.replace(/\s+/g, ' ').trim()).context
          : ''
        const context =
          summary.length > STORED_ARGUMENTS_PREVIEW_CHARS
            ? `${summary.slice(0, STORED_ARGUMENTS_PREVIEW_CHARS - 1)}…`
            : summary
        const row: GatewayTranscriptMessage = { role: 'tool', name, ...(context ? { context } : {}) }
        const callId = optionalStr(record.id)
        if (callId) {
          toolRowByCallId.set(callId, row)
        }
        out.push(row)
      }
      continue
    }
    if (role === 'tool') {
      // Results reconcile onto their call; an orphan result (its assistant
      // call was trimmed from retained history) is dropped, exactly like the
      // daemon replay, which only re-emits orphans from tool_executions.
      const callId = optionalStr(msg.tool_call_id)
      const existing = callId ? toolRowByCallId.get(callId) : undefined
      if (!existing) {
        continue
      }
      const content = firstNonEmptyStr(msg.text, textFromContent(msg.content))
      const failure = storedToolFailure(msg, content)
      const durationMs = optionalNum(msg.duration_ms ?? msg.durationMs)
      if (failure) {
        existing.error = failure
      }
      if (durationMs !== undefined) {
        existing.duration_s = durationMs / 1000
      }
      continue
    }
    // Persisted system rows are runtime state, not visible chat history; a
    // full system prompt can also mount tens of thousands of hidden chars.
  }
  return out
}

/**
 * Read the usage a status payload actually carries.
 *
 * Absent fields are omitted rather than reported as zero. Consumers merge this
 * over the previous usage with a spread, so a zero for a field the payload
 * never mentioned is not a no-op — it erases the running value. That is how a
 * mid-turn update carrying only token counts blanked the context meter, and how
 * any status event without cache figures reset the cached total to nothing.
 */
export function usageFromStatus(payload: Record<string, unknown>): Usage {
  const present = (...keys: string[]) => keys.some(key => payload[key] !== undefined)
  const hasInput = present('total_input_tokens', 'input_tokens')
  const hasOutput = present('total_output_tokens', 'output_tokens')
  const input = num(payload.total_input_tokens ?? payload.input_tokens)
  const output = num(payload.total_output_tokens ?? payload.output_tokens)
  const total = num(payload.total_tokens, input + output)

  return {
    ...(present('cache_read_tokens') ? { cache_read: num(payload.cache_read_tokens) } : {}),
    ...(present('cache_creation_tokens') ? { cache_write: num(payload.cache_creation_tokens) } : {}),
    ...(present('calls') ? { calls: num(payload.calls) } : {}),
    ...(present('context_limit', 'max_context')
      ? { context_max: num(payload.context_limit ?? payload.max_context) }
      : {}),
    ...(present('context_tokens') ? { context_used: num(payload.context_tokens) } : {}),
    ...(hasInput ? { input } : {}),
    ...(hasOutput ? { output } : {}),
    ...(present('total_tokens') || (hasInput && hasOutput) ? { total } : {}),
    // Cumulative telemetry (additive v35): absent keys stay absent so the
    // stats row renders nothing instead of fabricated zeros on old daemons.
    ...(present('turn_count') ? { turns: num(payload.turn_count) } : {}),
    ...(present('llm_steps') ? { llm_steps: num(payload.llm_steps) } : {}),
    ...(present('tool_steps') ? { tool_steps: num(payload.tool_steps) } : {}),
    ...(present('llm_duration_ms') ? { llm_ms: num(payload.llm_duration_ms) } : {}),
    ...(present('tool_duration_ms') ? { tool_ms: num(payload.tool_duration_ms) } : {}),
    ...(present('ttft_avg_ms') ? { ttft_avg_ms: num(payload.ttft_avg_ms) } : {}),
    ...(present('ttft_samples') ? { ttft_samples: num(payload.ttft_samples) } : {}),
    ...(present('ttft_total_ms') ? { ttft_total_ms: num(payload.ttft_total_ms) } : {}),
    ...(present('tokens_per_second') ? { tok_per_sec: num(payload.tokens_per_second) } : {}),
    ...(present('cache_hit_rate') ? { cache_hit_rate: num(payload.cache_hit_rate) } : {})
  } as Usage
}

export function adaptDaemonEvent(type: string, payload: Record<string, unknown>): AnyEvent[] {
  const normalizedType = normalizeEventType(type)

  switch (normalizedType) {
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
      const usage = usageFromStatus(payload)
      // Streamed usage_update frames carry one round's duration, while session
      // status payloads carry cumulative telemetry under the same v35 field.
      // Mark the former explicitly and keep it out of the ordinary spread merge.
      const llmDurationMs = optionalNum(payload.llm_duration_ms)
      const ttftMs = optionalNum(payload.ttft_ms)
      const timingIsDelta = llmDurationMs !== undefined
        && payload.llm_steps === undefined
        && payload.turn_count === undefined
      const stableUsage: Usage = { ...usage }
      if (timingIsDelta) delete stableUsage.llm_ms
      const telemetryDelta = timingIsDelta
        ? {
            llm_ms: llmDurationMs,
            ...(ttftMs === undefined ? {} : { ttft_ms: ttftMs })
          }
        : undefined

      return [
        {
          type: 'status.update',
          payload: {
            kind: 'status',
            ...(mode ? { mode } : {}),
            ...(activePermissionMode ? { permission_mode: activePermissionMode } : {}),
            ...(reasoningEffort ? { reasoning_effort: reasoningEffort } : {}),
            ...(telemetryDelta ? { telemetry_delta: telemetryDelta } : {}),
            text: statusText(payload),
            usage: stableUsage
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
            usage: stableUsage
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
      // The native daemon flags a cancelled turn's final edge. Forward it so
      // the UI can tell a daemon-confirmed interruption from a natural
      // completion that merely raced the user's Esc keystroke. `unstarted`
      // additionally marks a cancel that fired before any turn_begin or
      // assistant content existed (setup abort / suppressed launch), so the
      // handler settles the turn without synthesizing an assistant row.
      return [{
        type: 'message.complete',
        payload: {
          ...(bool(payload.cancelled) ? { interrupted: true } : {}),
          ...(bool(payload.unstarted) ? { unstarted: true } : {})
        }
      }]

    case 'step_interrupted':
      return [{ type: 'message.complete', payload: { text: '[interrupted]' } }]

    case 'tool_call':
      return [
        {
          type: 'tool.start',
          payload: {
            args_text: str(payload.arguments),
            name: str(payload.name, 'tool'),
            tool_id: str(payload.id, str(payload.tool_call_id, 'tool')),
            ...(optionalStr(payload.reasoning) ? { reasoning: optionalStr(payload.reasoning) } : {})
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

    case 'background.complete':
      return [{
        type: 'background.complete',
        payload: {
          task_id: str(payload.task_id),
          text: str(payload.text)
        }
      }]

    case 'session_title':
      return [{
        type: 'session_title',
        payload: {
          session_id: optionalStr(payload.session_id),
          title: optionalStr(payload.title)
        }
      }]

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
      return [{ type: normalizedType, payload } as GatewayEvent]
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
      tool_id: str(payload.tool_call_id, 'tool'),
      ...(optionalStr(payload.reasoning) ? { reasoning: optionalStr(payload.reasoning) } : {})
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
  // The bridge notification shape is `{category, title, body, severity}`, but
  // the Bun runtime emits its own turn-level notices as `{level, message}`.
  // Without the second spelling those notices — turn errors, saved steers,
  // stopped delegated agents — reached the TUI as a blank info toast.
  const body = firstNonEmptyStr(payload.body, payload.title, payload.message)

  if (category === 'subagent_stream') {
    return subagentStreamEvents(payload, body)
  }

  if (category === 'history') {
    if (kind === 'replay_assistant') {
      // The daemon nests persisted thinking under the notification's payload
      // sub-object; keep the top-level read as a fallback for older emitters.
      const nested = asRecord(payload.payload)
      const thinking = str(nested.thinking) || str(payload.thinking)
      return [{
        type: 'transcript.append',
        payload: { role: 'assistant', text: body, ...(thinking ? { thinking } : {}) }
      }]
    }
    if (kind === 'replay_tool') {
      const toolPayload = asRecord(payload.payload)
      const ok = bool(toolPayload.ok, true)
      const durationMs = optionalNum(toolPayload.duration_ms)
      const replayName = str(toolPayload.name, 'tool')
      const rawContext = str(toolPayload.context)
      // The daemon ships a bounded arguments preview; summarize JSON argument
      // blobs into the same friendly context live rows carry.
      const context = rawContext ? summarizeToolStartDisplay(replayName, '', rawContext).context : ''
      const error = str(toolPayload.preview) || 'Tool execution failed.'
      return [{
        type: 'transcript.append',
        payload: {
          role: 'tool',
          name: replayName,
          ...(context ? { context } : {}),
          ...(ok ? {} : { error }),
          ...(durationMs === undefined ? {} : { duration_s: durationMs / 1000 })
        }
      }]
    }
    if (kind === 'replay_user') {
      // The daemon prefixes replay-only history notifications with a sparkle
      // so terminal log consumers can distinguish them. It is transport
      // metadata, not part of the user's authored message.
      return [{ type: 'transcript.append', payload: { role: 'user', text: body.replace(/^✨\s?/, '') } }]
    }
    if (kind === 'compaction') {
      // Compaction changes the conversation itself, so keep a durable visible
      // transcript row instead of an eight-second toast/status replacement.
      return [{ type: 'transcript.append', payload: { role: 'system', text: body } }]
    }
    return [
      {
        type: 'status.update',
        payload: { kind: 'info', text: body }
      }
    ]
  }

  // A toast with nothing to say is noise the user cannot act on; drop it
  // rather than flashing an empty box for eight seconds.
  if (!body) {
    return []
  }

  return [
    {
      type: 'notification.show',
      payload: {
        id: str(payload.id),
        key: str(payload.category),
        kind: 'ttl',
        level: severityToLevel(firstNonEmptyStr(payload.severity, payload.level)),
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
          ...(optionalStr(nestedPayload.id) ? { tool_call_id: str(nestedPayload.id) } : {}),
          tool_name: str(nestedPayload.name),
          tool_preview: str(nestedPayload.arguments)
        }
      }
    ]
  }
  if (nestedType === 'tool_result') {
    const tool = str(nestedPayload.name, 'tool')
    const result = firstNonEmptyStr(nestedPayload.return_value, nestedPayload.result)
    const permitted = nestedPayload.permitted !== false
    const callId = optionalStr(nestedPayload.tool_call_id)
    const durationMs = optionalNum(nestedPayload.duration_ms)
    // Two events from one: the note keeps the live rail's running commentary,
    // and the structured result is what the inspector pairs with its tool_call
    // to show a duration. Dropping either one loses a surface.
    return [
      {
        type: 'subagent.tool_result',
        payload: {
          ...base,
          status: permitted ? 'running' : 'failed',
          ...(callId ? { tool_call_id: callId } : {}),
          ...(durationMs === undefined ? {} : { tool_duration_ms: durationMs }),
          tool_name: tool,
          tool_ok: permitted,
          ...(result ? { tool_preview: result } : {})
        }
      },
      {
        type: 'subagent.progress',
        payload: {
          ...base,
          status: permitted ? 'running' : 'failed',
          text: `${permitted ? '✓' : '✗'} ${tool}${result ? ` — ${firstLine(result)}` : ''}`
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
