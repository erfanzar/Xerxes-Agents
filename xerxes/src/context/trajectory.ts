// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** Provider-shaped input; callers need not depend on a provider or session implementation. */
export interface TrajectoryMessage {
  readonly role?: string
  readonly content?: unknown
  readonly tool_calls?: unknown
  readonly tool_call_id?: unknown
  readonly is_error?: unknown
  readonly isError?: unknown
  readonly status?: unknown
  readonly ok?: unknown
}

export type TrajectoryStep =
  | { readonly type: 'user'; readonly text: string }
  | { readonly type: 'assistant'; readonly text: string }
  | { readonly type: 'action'; readonly tool: string; readonly callId: string; readonly args: string }
  | { readonly type: 'observation'; readonly callId: string; readonly text: string; readonly originalChars: number; readonly clipped: boolean }
  | { readonly type: 'error'; readonly callId?: string; readonly text: string }

export interface TrajectoryRenderOptions {
  /** Head/tail ceiling, including the disclosure. A too-small ceiling yields to a complete disclosure. */
  readonly maxChars?: number
  /** Tool output preview ceiling; defaults to 2,000 UTF-16 characters. */
  readonly maxObservationChars?: number
  /** Tool argument preview ceiling; defaults to 200 UTF-16 characters. */
  readonly maxArgumentChars?: number
}

/**
 * Normalize a transcript without clipping, reordering, deduplicating or mutating it.
 * Results remain at their chronological position and link to actions by callId (empty for
 * identifier-less orphans). Duplicate IDs do not discard any results. Unknown/system roles
 * are user steps with an explicit source-role annotation, not falsely attributed user text.
 * `originalChars` measures rendered textual content, before preview clipping; `clipped` is
 * always false here because only renderTrajectory applies preview budgets.
 */
export function trajectorySteps(messages: readonly TrajectoryMessage[]): TrajectoryStep[] {
  if (!Array.isArray(messages)) throw new TypeError('trajectory messages must be an array')
  const steps: TrajectoryStep[] = []
  for (const [index, message] of messages.entries()) {
    if (!isRecord(message)) throw new TypeError(`trajectory message ${index} must be an object`)
    if (message.role !== undefined && typeof message.role !== 'string') {
      throw new TypeError(`trajectory message ${index} role must be a string`)
    }
    if (message.role === 'tool' || message.role === 'error') {
      steps.push(resultStep(message, firstText(message.tool_call_id), message.role === 'error'))
      continue
    }
    const start = steps.length
    const parts = Array.isArray(message.content) ? message.content : [message.content]
    let text: string[] = []
    const flush = () => {
      if (!text.length) return
      steps.push(textStep(message.role, text.join('\n')))
      text = []
    }
    for (const part of parts) {
      if (isRecord(part) && part.type === 'tool_use') {
        flush()
        steps.push(actionStep(part))
      } else if (isRecord(part) && part.type === 'tool_result') {
        flush()
        steps.push(resultStep(part, firstText(part.tool_use_id, part.tool_call_id)))
      } else if (part !== undefined) {
        text.push(contentText(part))
      }
    }
    flush()
    if (message.tool_calls !== undefined && message.tool_calls !== null) {
      if (Array.isArray(message.tool_calls)) {
        steps.push(...message.tool_calls.map(actionStep))
      } else {
        // Malformed calls must not disappear merely because they cannot be interpreted.
        steps.push(textStep(message.role, `tool_calls=${stableJson(message.tool_calls)}`))
      }
    }
    if (steps.length === start) steps.push(textStep(message.role, ''))
  }
  return steps
}

/**
 * Shared model-readable grammar: [USER]/[ASSISTANT] text, [ACTION] tool(args) call_id=JSON,
 * and [OBSERVATION]/[ERROR] call_id=JSON text. IDs are omitted only when unavailable.
 * Every preview states its source size and that the renderer, not the tool, clipped it.
 * Non-positive or non-finite limits disable the respective clipping operation.
 */
export function renderTrajectory(
  messages: readonly TrajectoryMessage[],
  options: TrajectoryRenderOptions = {},
): string {
  const steps = trajectorySteps(messages)
  const resultIds = new Set(steps.flatMap(step => (
    (step.type === 'observation' || step.type === 'error') && step.callId ? [step.callId] : []
  )))
  const rendered = steps.map(step => {
    switch (step.type) {
      case 'user': return `[USER] ${step.text}`
      case 'assistant': return `[ASSISTANT] ${step.text}`
      case 'action': {
        const preview = previewText(step.args, options.maxArgumentChars ?? 200, 'argument')
        const missing = !step.callId || !resultIds.has(step.callId) ? ' (no result in this slice)' : ''
        return `[ACTION] ${step.tool}(${preview.text})${renderCallId(step.callId)}${missing}${preview.note}`
      }
      case 'observation':
      case 'error': {
        const preview = previewText(step.text, options.maxObservationChars ?? 2_000, 'observation')
        return `[${step.type === 'error' ? 'ERROR' : 'OBSERVATION'}]${renderCallId(step.callId)} ${preview.text}${preview.note}`
      }
    }
  })
  return clipTranscript(rendered, options.maxChars)
}

function textStep(role: unknown, text: string): TrajectoryStep {
  if (role === 'assistant') return { type: 'assistant', text }
  return {
    type: 'user',
    text: role === undefined || role === 'user' ? text : `[source role=${stableJson(role)}] ${text}`,
  }
}

function actionStep(call: unknown): TrajectoryStep {
  if (!isRecord(call)) return { type: 'action', tool: 'unknown_tool', callId: '', args: stableJson(call) }
  const fn = isRecord(call.function) ? call.function : undefined
  const tool = firstText(fn?.name, call.name, call.tool_name)
  const args = [fn?.arguments, call.arguments, call.input, call.args].find(value => value !== undefined)
  return {
    type: 'action',
    tool: tool || 'unknown_tool',
    callId: firstText(call.id, call.tool_call_id),
    args: tool ? argumentText(args) : stableJson(call),
  }
}

function resultStep(result: Record<string, unknown>, callId: string, forceError = false): TrajectoryStep {
  const text = contentText(result.content)
  if (forceError || failedResult(result, text)) {
    return { type: 'error', ...(callId ? { callId } : {}), text }
  }
  return { type: 'observation', callId, text, originalChars: text.length, clipped: false }
}

/**
 * Conservative failure rule: only explicit top-level is_error/isError === true, ok === false,
 * status === 'error'/'failed', the exact leading runtime 'Tool execution failed:' sentinel,
 * or those same flags in a complete JSON-object output. No substring/error-word heuristics,
 * nested flags, truthy strings, nonzero counts, or mere `error` fields imply tool failure.
 */
function failedResult(result: Record<string, unknown>, text: string): boolean {
  if (failureFlag(result) || (isRecord(result.content) && failureFlag(result.content))) return true
  if (text.trimStart().startsWith('Tool execution failed:')) return true
  try {
    const parsed: unknown = JSON.parse(text)
    return isRecord(parsed) && failureFlag(parsed)
  } catch {
    return false
  }
}

function failureFlag(value: Record<string, unknown>): boolean {
  return value.is_error === true || value.isError === true || value.ok === false
    || value.status === 'error' || value.status === 'failed'
}

function contentText(content: unknown): string {
  if (typeof content === 'string') return content
  if (content === undefined) return ''
  if (Array.isArray(content)) return content.map(contentText).join('\n')
  if (isRecord(content)) {
    if (content.type === 'image' || content.type === 'image_url' || content.type === 'input_image') {
      return '[image omitted]'
    }
    if (typeof content.text === 'string'
      && (content.type === 'text' || content.type === 'input_text' || Object.keys(content).length === 1)) {
      return content.text
    }
  }
  return stableJson(content)
}

function argumentText(value: unknown): string {
  if (value === undefined) return ''
  if (typeof value !== 'string') return stableJson(value)
  try {
    const parsed: unknown = JSON.parse(value)
    return stableJson(parsed)
  } catch {
    return value
  }
}

function previewText(text: string, maximumChars: number, kind: 'argument' | 'observation'): {
  readonly text: string
  readonly note: string
} {
  if (!Number.isFinite(maximumChars) || maximumChars <= 0 || text.length <= maximumChars) {
    return { text, note: '' }
  }
  const end = prefixEnd(text, Math.floor(maximumChars))
  const note = kind === 'argument'
    ? `[argument preview clipped by the renderer: showing ${end} of ${text.length} chars; original arguments unchanged]`
    : `[preview clipped by the renderer: showing ${end} of ${text.length} chars; the tool's real output was not clipped by the renderer]`
  return { text: text.slice(0, end), note: `\n${note}` }
}

function renderCallId(callId: string | undefined): string {
  return callId ? ` call_id=${stableJson(callId)}` : ''
}

function clipTranscript(steps: readonly string[], maximumChars: number | undefined): string {
  const text = steps.join('\n')
  if (maximumChars === undefined || !Number.isFinite(maximumChars) || maximumChars <= 0 || text.length <= maximumChars) {
    return text
  }
  const spans: Array<{ start: number; end: number }> = []
  let offset = 0
  for (const step of steps) {
    spans.push({ start: offset, end: offset + step.length })
    offset += step.length + 1
  }
  // Prefer the requested ceiling, but never silently shorten the disclosure to meet a tiny
  // one. At least a character from each end survives; surrogate boundaries may shorten it.
  let available = Math.max(2, Math.floor(maximumChars))
  for (;;) {
    const headEnd = prefixEnd(text, Math.ceil(available / 2))
    const tailStart = suffixStart(text, text.length - Math.floor(available / 2))
    const omittedChars = tailStart - headEnd
    let whole = 0
    let partial = 0
    for (const span of spans) {
      if (span.start >= headEnd && span.end <= tailStart) whole += 1
      else if (span.start < tailStart && span.end > headEnd) partial += 1
    }
    const marker = `\n[transcript truncated by the renderer: omitted ${omittedChars} of ${text.length} chars; ${whole} whole steps omitted, ${partial} steps partially clipped of ${steps.length} total; partial steps are renderer previews, not incomplete tool output]\n`
    const clipped = text.slice(0, headEnd) + marker + text.slice(tailStart)
    if (clipped.length <= maximumChars || available <= 2) return clipped
    available = Math.max(2, available - Math.ceil(clipped.length - maximumChars))
  }
}

function prefixEnd(text: string, end: number): number {
  return end > 0 && /[\uD800-\uDBFF]/u.test(text[end - 1]!) ? end - 1 : end
}

function suffixStart(text: string, start: number): number {
  return /[\uDC00-\uDFFF]/u.test(text[start] ?? '') ? start + 1 : start
}

function firstText(...values: readonly unknown[]): string {
  return values.find((value): value is string => typeof value === 'string' && value.length > 0) ?? ''
}

function stableJson(value: unknown): string {
  try {
    return JSON.stringify(value, (_key, item: unknown) => {
      if (typeof item === 'bigint') return item.toString()
      if (!isRecord(item)) return item
      return Object.fromEntries(Object.keys(item).sort().map(key => [key, item[key]]))
    }) ?? String(value)
  } catch {
    return `[unserializable value: ${String(value)}]`
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
