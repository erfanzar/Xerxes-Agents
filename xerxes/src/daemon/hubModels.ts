// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { hostname as machineHostname } from 'node:os'

import { profileLabel, signsInPerMachine, type ProfileStore, type ProviderProfile } from '../bridge/profiles.js'
import {
  CompletionDeadlineError,
  closeLlmClient,
  completeLlm,
  type CompletionRequest,
  type LlmClient,
  type ThinkingRequest,
} from '../llms/client.js'
import { REASONING_OFF, resolveEffort, selectableEfforts, type ReasoningLevelSet } from '../llms/reasoningLevels.js'
import type { JsonRpcPayload } from '../protocol/jsonRpc.js'
import { resolveTurnThinking } from '../runtime/thinkingLevels.js'
import type { ChatMessage, MessageContent } from '../types/messages.js'
import { isJsonObject, parseToolArguments, type JsonObject, type ToolCall, type ToolDefinition } from '../types/toolCalls.js'
import { profileDiscoveryApiKey, sanitizeModelDiscoveryError } from './modelDiscovery.js'
import { profileAcceptsModel } from './sessionProvider.js'

/**
 * Model access for the Hub's XERXES supervisor (`hub_models_v1`).
 *
 * The Hub never holds a provider credential. It asks this daemon, which owns
 * the provider profiles, to list them (`hub.models.catalog`), describe one
 * (`hub.models.options`) and run a single non-streaming completion
 * (`hub.model.complete`). No result here ever carries an API key, a token or a
 * base URL, and an error message is scrubbed of both before it leaves.
 */

/** Largest accepted `hub.model.complete` request, as JSON. */
export const HUB_MODEL_MAX_REQUEST_BYTES = 1024 * 1024
export const HUB_MODEL_DEFAULT_MAX_TOKENS = 2048
export const HUB_MODEL_MAX_TOKENS = 8192
export const HUB_MODEL_MAX_CONCURRENT = 2
export const HUB_MODEL_TIMEOUT_MS = 120_000

const MAX_MESSAGES = 500
const MAX_TOOLS = 64
const MAX_LISTED_MODELS = 500
const MAX_ERROR_CHARS = 240
const TOOL_NAME = /^[A-Za-z0-9_.-]{1,128}$/

export type HubModelErrorCode = 'unknown_profile' | 'invalid_request' | 'busy' | 'provider_failed' | 'timeout'

export interface HubModelsHost {
  readonly profiles: Pick<ProfileStore, 'get' | 'list'>
  /** The daemon's model discovery for one stored profile (never an explicit endpoint). */
  fetchModels(profileName: string): Promise<JsonRpcPayload>
  /** Reasoning levels of `model` as served by `profile`, resolved the way `reasoning_levels` does. */
  reasoningLevels(model: string, profile: ProviderProfile, signal?: AbortSignal): Promise<ReasoningLevelSet>
  createClient(model: string, profile: ProviderProfile): LlmClient
}

export interface HubModelsOptions {
  /** Test seam: replaces the provider client built from the stored profile. */
  readonly clientFactory?: (model: string, profile: ProviderProfile) => LlmClient
  readonly timeoutMs?: number
}

class InvalidRequest extends Error {}

function failure(code: HubModelErrorCode, error: string): JsonRpcPayload {
  return { ok: false, code, error }
}

function text(value: unknown): string | undefined {
  return typeof value === 'string' && value.trim() ? value.trim() : undefined
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function parseContent(raw: unknown, where: string, required: boolean): MessageContent {
  if (raw === undefined && !required) return ''
  if (typeof raw === 'string') return raw
  if (Array.isArray(raw)) {
    return raw.map((part, index) => {
      if (!isRecord(part) || part.type !== 'text' || typeof part.text !== 'string') {
        throw new InvalidRequest(`${where}.content[${index}] must be a text part`)
      }
      return { type: 'text' as const, text: part.text }
    })
  }
  throw new InvalidRequest(`${where}.content must be a string or a list of text parts`)
}

function parseToolCalls(raw: unknown, where: string): ToolCall[] {
  if (!Array.isArray(raw)) throw new InvalidRequest(`${where}.tool_calls must be a list`)
  return raw.map((call, index): ToolCall => {
    const at = `${where}.tool_calls[${index}]`
    if (!isRecord(call) || typeof call.id !== 'string' || !call.id || !isRecord(call.function)) {
      throw new InvalidRequest(`${at} needs an id and a function`)
    }
    const name = call.function.name
    if (typeof name !== 'string' || !TOOL_NAME.test(name)) throw new InvalidRequest(`${at}.function.name is not a valid tool name`)
    const args = call.function.arguments
    let parsed: JsonObject
    try {
      parsed = typeof args === 'string' || args === undefined ? parseToolArguments(args) : isJsonObject(args) ? args : (() => { throw new Error('not an object') })()
    } catch {
      throw new InvalidRequest(`${at}.function.arguments must be a JSON object`)
    }
    return { id: call.id, type: 'function', function: { name, arguments: parsed } }
  })
}

function parseMessage(raw: unknown, index: number): ChatMessage {
  const where = `messages[${index}]`
  if (!isRecord(raw)) throw new InvalidRequest(`${where} must be an object`)
  switch (raw.role) {
    case 'system':
      return { role: 'system', content: parseContent(raw.content, where, true) }
    case 'user':
      return { role: 'user', content: parseContent(raw.content, where, true) }
    case 'assistant': {
      const calls = raw.tool_calls === undefined ? undefined : parseToolCalls(raw.tool_calls, where)
      return { role: 'assistant', content: parseContent(raw.content, where, calls === undefined), ...(calls?.length ? { tool_calls: calls } : {}) }
    }
    case 'tool': {
      if (typeof raw.content !== 'string') throw new InvalidRequest(`${where}.content must be a string`)
      if (typeof raw.tool_call_id !== 'string' || !raw.tool_call_id) throw new InvalidRequest(`${where}.tool_call_id is required`)
      return {
        role: 'tool',
        content: raw.content,
        tool_call_id: raw.tool_call_id,
        ...(typeof raw.name === 'string' && raw.name ? { name: raw.name } : {}),
        ...(raw.is_error === true ? { is_error: true } : {}),
      }
    }
    default:
      throw new InvalidRequest(`${where}.role must be system, user, assistant or tool`)
  }
}

function parseTools(raw: unknown): ToolDefinition[] {
  if (raw === undefined || raw === null) return []
  if (!Array.isArray(raw) || raw.length > MAX_TOOLS) throw new InvalidRequest(`tools must be a list of at most ${MAX_TOOLS} definitions`)
  return raw.map((tool, index): ToolDefinition => {
    const where = `tools[${index}]`
    if (!isRecord(tool) || tool.type !== 'function' || !isRecord(tool.function)) throw new InvalidRequest(`${where} must be a function tool`)
    const { name, description, parameters } = tool.function
    if (typeof name !== 'string' || !TOOL_NAME.test(name)) throw new InvalidRequest(`${where}.function.name is not a valid tool name`)
    if (typeof description !== 'string') throw new InvalidRequest(`${where}.function.description must be a string`)
    if (!isRecord(parameters)) throw new InvalidRequest(`${where}.function.parameters must be a schema object`)
    return { type: 'function', function: { name, description, parameters } }
  })
}

interface ParsedCompletion {
  readonly profile: string
  readonly model: string
  readonly effort: string | undefined
  readonly system: string
  readonly messages: readonly ChatMessage[]
  readonly tools: readonly ToolDefinition[]
  readonly maxTokens: number
}

function parseCompletion(params: JsonRpcPayload): ParsedCompletion {
  const profile = text(params.profile)
  if (!profile || profile.length > 512) throw new InvalidRequest('profile is required')
  const model = text(params.model)
  if (!model || model.length > 512) throw new InvalidRequest('model is required')
  let effort: string | undefined
  if (params.effort !== undefined && params.effort !== null) {
    effort = text(params.effort)
    if (!effort || effort.length > 64) throw new InvalidRequest('effort must be a short non-empty string')
  }
  if (typeof params.system !== 'string') throw new InvalidRequest('system must be a string')
  if (!Array.isArray(params.messages) || params.messages.length === 0 || params.messages.length > MAX_MESSAGES) {
    throw new InvalidRequest(`messages must be a list of 1 to ${MAX_MESSAGES} messages`)
  }
  const raw = params.max_tokens
  let maxTokens = HUB_MODEL_DEFAULT_MAX_TOKENS
  if (raw !== undefined && raw !== null) {
    if (typeof raw !== 'number' || !Number.isInteger(raw) || raw < 1 || raw > HUB_MODEL_MAX_TOKENS) {
      throw new InvalidRequest(`max_tokens must be an integer from 1 to ${HUB_MODEL_MAX_TOKENS}`)
    }
    maxTokens = raw
  }
  return {
    profile, model, effort, system: params.system,
    messages: params.messages.map(parseMessage),
    tools: parseTools(params.tools),
    maxTokens,
  }
}

export class HubModelService {
  private active = 0
  private readonly controllers = new Map<object, Set<AbortController>>()
  private readonly timeoutMs: number

  constructor(private readonly host: HubModelsHost, private readonly settings: HubModelsOptions = {}) {
    this.timeoutMs = settings.timeoutMs ?? HUB_MODEL_TIMEOUT_MS
  }

  get inFlight(): number { return this.active }

  catalog(): JsonRpcPayload {
    return {
      ok: true,
      hostname: machineHostname(),
      profiles: this.host.profiles.list().map(profile => ({
        name: profile.name,
        label: profileLabel(profile.name),
        provider: profile.provider,
        default_model: profile.model,
        signs_in: signsInPerMachine(profile.provider),
      })),
    }
  }

  async options(params: JsonRpcPayload): Promise<JsonRpcPayload> {
    const name = text(params.profile)
    if (!name || name.length > 512) return failure('invalid_request', 'profile is required')
    const profile = this.host.profiles.get(name)
    if (!profile) return failure('unknown_profile', `No provider profile named ${name}`)
    let model = profile.model.trim()
    if (params.model !== undefined && params.model !== null) {
      const requested = text(params.model)
      if (!requested || requested.length > 512) return failure('invalid_request', 'model must be a short non-empty string')
      model = requested
    }

    let found: JsonRpcPayload
    try {
      found = await this.host.fetchModels(profile.name)
    } catch (error) {
      found = { ok: false, error: this.safe(error, profile) }
    }
    const listed = found.ok === true && Array.isArray(found.models) ? found.models.filter((id): id is string => typeof id === 'string' && id !== '') : []
    const labels = new Map<string, string>()
    if (Array.isArray(found.catalog)) {
      for (const entry of found.catalog) {
        if (isRecord(entry) && typeof entry.id === 'string' && typeof entry.display_name === 'string' && entry.display_name) labels.set(entry.id, entry.display_name)
      }
    }
    const configured = profile.model.trim()
    const ids = listed.length ? listed : configured ? [configured] : []
    const note = found.ok === true ? text(found.warning) : text(found.error)
    const reasons = note ?? (listed.length ? undefined : 'the provider listed no models')

    const levels = await this.host.reasoningLevels(model, profile)
    const efforts = selectableEfforts(levels).map(effort => {
      const description = effort === REASONING_OFF
        ? 'No extended reasoning; fastest replies'
        : levels.levels.find(level => level.effort === effort)?.description
      return { effort, ...(description === undefined ? {} : { description }) }
    })
    return {
      ok: true,
      profile: profile.name,
      provider: profile.provider,
      models: ids.slice(0, MAX_LISTED_MODELS).map(id => ({ id, ...(labels.has(id) ? { label: labels.get(id) } : {}) })),
      ...(reasons === undefined ? {} : { models_note: this.safe(reasons, profile) }),
      model,
      efforts,
      default_effort: levels.defaultEffort ?? null,
    }
  }

  async complete(owner: object, params: JsonRpcPayload): Promise<JsonRpcPayload> {
    if (Buffer.byteLength(JSON.stringify(params), 'utf8') > HUB_MODEL_MAX_REQUEST_BYTES) {
      return failure('invalid_request', `the request exceeds ${HUB_MODEL_MAX_REQUEST_BYTES} bytes`)
    }
    let request: ParsedCompletion
    try {
      request = parseCompletion(params)
    } catch (error) {
      if (error instanceof InvalidRequest) return failure('invalid_request', error.message)
      throw error
    }
    const profile = this.host.profiles.get(request.profile)
    if (!profile) return failure('unknown_profile', `No provider profile named ${request.profile}`)
    if (!profileAcceptsModel(profile, request.model)) {
      return failure('invalid_request', `Provider ${profile.name} cannot serve ${request.model}`)
    }
    if (this.active >= HUB_MODEL_MAX_CONCURRENT) return failure('busy', 'Too many model completions are running on this machine; retry shortly')

    const controller = new AbortController()
    this.active += 1
    let set = this.controllers.get(owner)
    if (!set) this.controllers.set(owner, set = new Set())
    set.add(controller)
    let client: LlmClient | undefined
    try {
      let thinking: ThinkingRequest | undefined
      if (request.effort !== undefined) {
        const levels = await this.host.reasoningLevels(request.model, profile, controller.signal)
        const resolved = resolveEffort(levels, request.effort)
        if (resolved === undefined) {
          const offered = selectableEfforts(levels)
          return failure('invalid_request', `Unsupported thinking effort for ${request.model}: ${request.effort}${offered.length ? ` (offered: ${offered.join(', ')})` : ''}`)
        }
        if (resolved === REASONING_OFF) {
          // Claude Code thinks by default, so "off" has to be said explicitly (as the turn loop does).
          if (/^claude[-_]code\//i.test(request.model)) thinking = { effort: 'none' }
        } else {
          const directive = resolveTurnThinking({ defaults: { effort: resolved }, prompt: '', ultraMode: false })
          if (directive) thinking = { budgetTokens: directive.budgetTokens, effort: directive.effort }
        }
      }
      client = (this.settings.clientFactory ?? ((model, stored) => this.host.createClient(model, stored)))(request.model, profile)
      const completionRequest: CompletionRequest = {
        model: request.model,
        messages: [{ role: 'system', content: request.system }, ...request.messages],
        maxTokens: request.maxTokens,
        querySource: 'classification',
        ...(request.tools.length ? { tools: request.tools } : {}),
        ...(thinking === undefined ? {} : { thinking }),
      }
      const completion = await completeLlm(client, completionRequest, controller.signal, { timeoutMs: this.timeoutMs })
      return {
        ok: true,
        text: completion.content ?? '',
        tool_calls: completion.toolCalls.map(call => ({ id: call.id, name: call.function.name, arguments: JSON.stringify(call.function.arguments ?? {}) })),
        usage: { input_tokens: completion.usage?.inputTokens ?? 0, output_tokens: completion.usage?.outputTokens ?? 0 },
      }
    } catch (error) {
      if (error instanceof CompletionDeadlineError) return failure('timeout', `the model did not answer within ${Math.round(this.timeoutMs / 1000)} seconds`)
      if (controller.signal.aborted) return failure('provider_failed', 'the request was cancelled')
      return failure('provider_failed', this.safe(error, profile))
    } finally {
      this.active -= 1
      set.delete(controller)
      if (set.size === 0 && this.controllers.get(owner) === set) this.controllers.delete(owner)
      if (client) {
        try { await closeLlmClient(client) } catch { /* a failed close must not replace the answer */ }
      }
    }
  }

  /** Abort every completion a closed connection started. */
  abortConnection(owner: object): void {
    const set = this.controllers.get(owner)
    if (!set) return
    for (const controller of set) controller.abort(new Error('connection closed'))
  }

  abortAll(): void {
    for (const set of this.controllers.values()) for (const controller of set) controller.abort(new Error('daemon stopping'))
  }

  /** A short single-line message with this profile's key, endpoint and any token-looking text removed. */
  private safe(error: unknown, profile: ProviderProfile): string {
    let apiKey = ''
    try { apiKey = profileDiscoveryApiKey(profile) } catch { /* no resolvable key: nothing more to scrub */ }
    let message = sanitizeModelDiscoveryError(error, { apiKey, baseUrl: profile.base_url })
    for (const secret of [profile.api_key, profile.base_url]) {
      if (secret.trim()) message = message.replaceAll(secret.trim(), '[redacted]')
    }
    message = message.replace(/\s+/g, ' ').trim()
    return message.length > MAX_ERROR_CHARS ? `${message.slice(0, MAX_ERROR_CHARS - 1)}…` : message
  }
}
