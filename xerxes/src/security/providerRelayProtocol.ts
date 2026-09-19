// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { COMPACTION_SUMMARY_MARKER } from '../context/compressor.js'
import { parseChatCompletionRequest } from '../api-server/protocol.js'
import { isQuerySource, type CompletionRequest, type LlmDelta } from '../llms/client.js'
import type { ToolDefinition } from '../types/toolCalls.js'
import { LocalProviderRelayError, type RelayFailureCode } from './localProviderRelay.js'

export type ProviderRelayRequest = { op: 'next'; id: string; request?: CompletionRequest } | { op: 'cancel'; id: string }
export type ProviderRelayReply = { done: boolean; deltas?: LlmDelta[] } | { error: RelayFailureCode }
export const MAX_RELAY_REQUEST_BYTES = 16 * 1024 * 1024
export const MAX_RELAY_DELTA_BYTES = 1024 * 1024
const bad = (): never => { throw new LocalProviderRelayError('invalid_request') }
const object = (value: unknown): Record<string, unknown> => value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : bad()
const string = (value: unknown): string => typeof value === 'string' ? value : bad()
const boolean = (value: unknown): boolean => typeof value === 'boolean' ? value : bad()
const number = (value: unknown): number => typeof value === 'number' && Number.isFinite(value) ? value : bad()
function keys(value: Record<string, unknown>, allowed: readonly string[]): void { if (Object.keys(value).some(key => !allowed.includes(key))) bad() }
function bounded(value: unknown, limit: number): void { try { if (Buffer.byteLength(JSON.stringify(value)) > limit) bad() } catch { bad() } }

/** Native completion codec: the public API parser validates shared message/tool
 * shapes, while native reasoning, replay and cache fields are retained explicitly.
 * Endpoint/credential overrides are never admitted. Failures never echo input. */
export function decodeRelayCompletion(value: unknown): CompletionRequest {
  try {
    bounded(value, MAX_RELAY_REQUEST_BYTES)
    const raw = object(value)
    keys(raw, ['model','messages','frequencyPenalty','maxTokens','minP','presencePenalty','querySource','systemSegments','thinking','repetitionPenalty','serviceTier','sessionId','stop','temperature','toolChoice','toolChoiceFunctionName','tools','topK','topP'])
    const parsed = parseChatCompletionRequest({ model: raw.model, messages: raw.messages, tools: raw.tools,
      max_tokens: raw.maxTokens, frequency_penalty: raw.frequencyPenalty, presence_penalty: raw.presencePenalty,
      stop: raw.stop, temperature: raw.temperature, top_p: raw.topP,
      tool_choice: raw.toolChoiceFunctionName === undefined ? raw.toolChoice : { type: 'function', function: { name: string(raw.toolChoiceFunctionName) } },
    }).completion
    const messages = parsed.messages.map((message, index) => {
      const source = object((raw.messages as unknown[])[index])
      const common = ['role', 'content']
      keys(source, [...common, ...(message.role === 'assistant' ? ['thinking','thinking_signature','tool_calls'] : message.role === 'tool' ? ['name','tool_call_id','is_error','added_tool_names'] : message.role === 'user' ? ['displayText', COMPACTION_SUMMARY_MARKER] : [])])
      if (message.role === 'assistant') return { ...message,
        ...(source.thinking === undefined ? {} : { thinking: string(source.thinking) }),
        ...(source.thinking_signature === undefined ? {} : { thinking_signature: string(source.thinking_signature) }) }
      if (message.role === 'tool') return { ...message,
        ...(source.is_error === undefined ? {} : { is_error: boolean(source.is_error) }),
        ...(source.added_tool_names === undefined ? {} : { added_tool_names: strings(source.added_tool_names) }) }
      // Internal summary provenance stays in the remote transcript, not the model request.
      if (source[COMPACTION_SUMMARY_MARKER] !== undefined) boolean(source[COMPACTION_SUMMARY_MARKER])
      return message
    })
    const tools = parsed.tools?.map((tool, index): ToolDefinition => {
      const source = object((raw.tools as unknown[])[index])
      keys(source, ['type','function','constrainedSampling'])
      if (source.constrainedSampling === undefined) return tool
      if (source.constrainedSampling === false) return { ...tool, constrainedSampling: false }
      const constraint = object(source.constrainedSampling); keys(constraint, ['type','variants'])
      if (constraint.type !== 'grammar') bad()
      const variants = object(constraint.variants); keys(variants, ['openai_lark','openai_regex'])
      return { ...tool, constrainedSampling: { type: 'grammar', variants: {
        ...(variants.openai_lark === undefined ? {} : { openai_lark: string(variants.openai_lark) }),
        ...(variants.openai_regex === undefined ? {} : { openai_regex: string(variants.openai_regex) }),
      } } }
    })
    const extra: { minP?: number; repetitionPenalty?: number; topK?: number } = {}
    for (const name of ['minP','repetitionPenalty','topK'] as const) if (raw[name] !== undefined) {
      const n = number(raw[name]); if (n < 0 || (name === 'minP' && n > 1) || (name === 'topK' && !Number.isSafeInteger(n))) bad(); extra[name] = n
    }
    let thinking: CompletionRequest['thinking']
    if (raw.thinking !== undefined) {
      const t = object(raw.thinking); keys(t, ['effort','budgetTokens'])
      if (t.budgetTokens !== undefined && (!Number.isSafeInteger(t.budgetTokens) || number(t.budgetTokens) < 0)) bad()
      thinking = { ...(t.effort === undefined ? {} : { effort: string(t.effort) }), ...(t.budgetTokens === undefined ? {} : { budgetTokens: number(t.budgetTokens) }) }
    }
    const segments = raw.systemSegments === undefined ? undefined : array(raw.systemSegments).map(value => {
      const segment = object(value); keys(segment, ['name','text','volatile'])
      return { name: string(segment.name), text: string(segment.text), ...(segment.volatile === undefined ? {} : { volatile: boolean(segment.volatile) }) }
    })
    if (raw.querySource !== undefined && !isQuerySource(raw.querySource)) bad()
    return { ...parsed, messages, ...extra, ...(tools ? { tools } : {}), ...(thinking ? { thinking } : {}),
      ...(segments ? { systemSegments: segments } : {}),
      ...(isQuerySource(raw.querySource) ? { querySource: raw.querySource } : {}),
      ...(raw.serviceTier === undefined ? {} : { serviceTier: string(raw.serviceTier) }),
      ...(raw.sessionId === undefined ? {} : { sessionId: string(raw.sessionId) }),
    }
  } catch { return bad() }
}
function array(value: unknown): unknown[] { return Array.isArray(value) ? value : bad() }
function strings(value: unknown): string[] { return array(value).map(string) }

export function decodeRelayRequest(value: unknown): ProviderRelayRequest {
  bounded(value, MAX_RELAY_REQUEST_BYTES)
  const raw = object(value); keys(raw, ['op','id','request'])
  const id = string(raw.id)
  if (!/^[a-zA-Z0-9_-]{1,64}$/.test(id)) bad()
  if (raw.op === 'cancel' && raw.request === undefined) return { op: 'cancel', id }
  if (raw.op !== 'next') return bad()
  return { op: 'next', id, ...(raw.request === undefined ? {} : { request: decodeRelayCompletion(raw.request) }) }
}

export function decodeRelayDelta(value: unknown): LlmDelta {
  bounded(value, MAX_RELAY_DELTA_BYTES)
  const raw = object(value); keys(raw, ['content','finishReason','thinking','thinkingSignature','toolCalls','usage'])
  const text: { content?: string; finishReason?: string; thinking?: string; thinkingSignature?: string } = {}
  for (const key of ['content','finishReason','thinking','thinkingSignature'] as const) if (raw[key] !== undefined) text[key] = string(raw[key])
  const toolCalls = raw.toolCalls === undefined ? undefined : parseChatCompletionRequest({ model: 'relay-validation', messages: [{ role: 'assistant', content: '', tool_calls: raw.toolCalls }] }).completion.messages[0]
  let usage: LlmDelta['usage']
  if (raw.usage !== undefined) {
    const u = object(raw.usage); keys(u, ['inputTokens','outputTokens','cacheCreationTokens','cacheReadTokens','reasoningTokens','serviceTier'])
    const tokens: { cacheCreationTokens?: number; cacheReadTokens?: number; reasoningTokens?: number } = {}
    for (const name of ['inputTokens','outputTokens','cacheCreationTokens','cacheReadTokens','reasoningTokens'] as const) {
      if (u[name] === undefined && name !== 'inputTokens' && name !== 'outputTokens') continue
      if (!Number.isSafeInteger(u[name]) || number(u[name]) < 0) bad()
      if (name !== 'inputTokens' && name !== 'outputTokens') tokens[name] = number(u[name])
    }
    usage = { inputTokens: number(u.inputTokens), outputTokens: number(u.outputTokens), ...tokens, ...(u.serviceTier === undefined ? {} : { serviceTier: string(u.serviceTier) }) }
  }
  return { ...text, ...(toolCalls?.role === 'assistant' ? { toolCalls: toolCalls.tool_calls ?? [] } : {}), ...(usage ? { usage } : {}) }
}
