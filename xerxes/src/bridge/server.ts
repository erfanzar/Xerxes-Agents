// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { isIP } from 'node:net'

import {
  BridgeSlashRouter,
  type BridgeProviderProfile,
  type BridgeSlashRouterHost,
} from './slashRouter.js'
import { BridgeSession, type BridgeSessionSnapshot } from './session.js'
import {
  WireEventEmitter,
  serializeWireFrame,
  type BridgeWireFrame,
  type WireEventNameStyle,
} from './wireEvents.js'
import type { ProviderProfile, SaveProfileInput } from './profiles.js'
import { estimateContextTokens } from '../context/windowUsage.js'
import { calcCost, getContextLimit, resolveProvider } from '../llms/providerRegistry.js'
import { CostTracker } from '../runtime/costTracker.js'
import { normalizeInteractionMode } from '../runtime/interactionModes.js'
import { createAgentState, type AgentState, type StreamEvent } from '../streaming/events.js'
import type { ChatMessage, ContentPart, MessageContent } from '../types/messages.js'
import type { JsonObject, ToolCall } from '../types/toolCalls.js'

type Awaitable<Value> = Value | Promise<Value>

/** Legacy line emitted by the original non-wire bridge protocol. */
export interface LegacyBridgeEvent {
  readonly data: Record<string, unknown>
  readonly event: string
}

/** Output is intentionally injected so the bridge never writes process stdout by itself. */
export interface BridgeServerOutput {
  writeLegacy(event: LegacyBridgeEvent): void
  writeWire(frame: BridgeWireFrame): void
}

/** Small line writer used by {@link NdjsonBridgeOutput}. */
export interface BridgeLineWriter {
  write(line: string): void
}

/** Incoming old-bridge request. JSON-RPC ids are optional because legacy clients used notifications. */
export interface BridgeServerRequest {
  readonly id?: number | string
  readonly method: string
  readonly params: Record<string, unknown>
}

/** Result returned to an embedding host; the public bridge protocol still communicates with events. */
export interface BridgeDispatchResult {
  readonly accepted: boolean
  readonly error?: string
}

/** Real provider-profile persistence boundary. No profile data is synthesized by the bridge. */
export interface BridgeProfileStore {
  active(): Awaitable<ProviderProfile | undefined>
  delete(name: string): Awaitable<boolean>
  list(): Awaitable<readonly (ProviderProfile & { readonly active?: boolean })[]>
  save(input: SaveProfileInput): Awaitable<ProviderProfile>
  setActive(name: string): Awaitable<boolean>
  updateActiveModel?(model: string): Awaitable<ProviderProfile | undefined>
  updateSampling?(name: string, updates: Record<string, unknown>): Awaitable<ProviderProfile | undefined>
}

/** Model lookup belongs to a caller-owned transport with its own DNS and credential policy. */
export interface BridgeModelDiscoveryPort {
  fetchModels(input: {
    readonly apiKey: string
    readonly baseUrl: string
    readonly provider: string
  }): Awaitable<readonly string[]>
}

export interface BridgeRuntimeTool {
  readonly description: string
  readonly name: string
  readonly safe?: boolean
}

export interface BridgeRuntimeAgent {
  readonly description: string
  readonly name: string
  readonly source?: string
}

/** Facts discovered by the real runtime at initialization time. */
export interface BridgeRuntimeInfo {
  readonly agents?: readonly BridgeRuntimeAgent[]
  readonly skills?: readonly string[]
  readonly tools?: readonly BridgeRuntimeTool[]
}

export interface BridgeRuntimeInitializeInput {
  readonly config: Readonly<Record<string, unknown>>
  readonly session: BridgeSessionSnapshot
}

export interface BridgeRuntimeTurnInput {
  readonly config: Readonly<Record<string, unknown>>
  readonly session: BridgeSessionSnapshot
  /** Mutable state is the portable native agent-loop state, not a Python-shaped proxy. */
  readonly state: AgentState
  readonly text: string
}

/**
 * Runtime boundary for the in-process bridge.
 *
 * The bridge owns protocol translation and persistence. Providers, tools,
 * interactions, and process shutdown are supplied by the embedding runtime.
 */
export interface BridgeRuntimePort {
  cancel?(): Awaitable<void>
  cancelAll?(): Awaitable<number>
  configure?(config: Readonly<Record<string, unknown>>): Awaitable<void>
  initialize?(input: BridgeRuntimeInitializeInput): Awaitable<BridgeRuntimeInfo>
  respondPermission?(input: { readonly requestId: string; readonly response: string }): Awaitable<boolean>
  respondQuestion?(input: {
    readonly answers: Readonly<Record<string, string>>
    readonly requestId: string
  }): Awaitable<boolean>
  shutdown?(): Awaitable<void>
  steer?(content: string): Awaitable<boolean>
  streamTurn(input: BridgeRuntimeTurnInput, signal: AbortSignal): AsyncIterable<StreamEvent>
}

export interface BridgeServerOptions {
  readonly contextLimit?: (model: string) => number
  readonly idFactory?: () => string
  readonly modelDiscovery?: BridgeModelDiscoveryPort
  readonly output: BridgeServerOutput
  readonly profileStore: BridgeProfileStore
  readonly runtime: BridgeRuntimePort
  readonly session: BridgeSession
  /** Extra slash capabilities are supplied by an embedding host, never invented locally. */
  readonly slashHost?: BridgeSlashRouterHost
  /** Kimi-style event names are the native wire bridge default. */
  readonly wireEventNameStyle?: WireEventNameStyle
  readonly wireMode?: boolean
}

/**
 * Native in-process bridge server.
 *
 * It supports the Python bridge's public method family (`init`, `query`,
 * cancellation, provider CRUD, model discovery, modes, and slash dispatch)
 * without owning stdio, global configuration, subprocesses, provider HTTP,
 * or a Python fallback. `NdjsonBridgeOutput` and `serveBridgeNdjson` supply
 * the optional stdio framing layer separately.
 */
export class BridgeServer {
  private activeAbortController: AbortController | undefined
  private activeTurn: Promise<void> | undefined
  private readonly contextLimit: (model: string) => number
  private config: Record<string, unknown> = {}
  private initialized = false
  private runtimeInfo: BridgeRuntimeInfo = {}
  private running = true
  private state: AgentState
  private readonly wire: WireEventEmitter

  constructor(private readonly options: BridgeServerOptions) {
    this.contextLimit = options.contextLimit ?? getContextLimit
    this.state = stateFromSnapshot(options.session.snapshot)
    this.wire = new WireEventEmitter({
      eventNameStyle: options.wireEventNameStyle ?? 'kimi',
      idFactory: options.idFactory ?? (() => crypto.randomUUID()),
      sink: { emit: frame => options.output.writeWire(frame) },
    })
  }

  get configuration(): Readonly<Record<string, unknown>> {
    return Object.freeze({ ...this.config })
  }

  get isInitialized(): boolean {
    return this.initialized
  }

  get isQueryRunning(): boolean {
    return this.activeTurn !== undefined
  }

  get session(): BridgeSessionSnapshot {
    return this.options.session.snapshot
  }

  /** Parse and dispatch one NDJSON request, surfacing malformed input as a bridge error event. */
  async handleLine(line: string): Promise<BridgeDispatchResult> {
    try {
      return await this.dispatch(parseBridgeRequestLine(line))
    } catch (error) {
      const message = errorMessage(error)
      this.emitError(message)
      return { accepted: false, error: message }
    }
  }

  /** Dispatch one already-decoded bridge request. Query work starts asynchronously so cancel messages can arrive. */
  async dispatch(request: BridgeServerRequest): Promise<BridgeDispatchResult> {
    if (!this.running && request.method !== 'shutdown') {
      return this.reject('Bridge is shut down.')
    }

    switch (request.method) {
      case 'init':
      case 'initialize':
        return this.initialize(request.params)
      case 'query':
      case 'prompt':
        return this.startQuery(request.params)
      case 'permission_response':
        return this.permissionResponse(request.params)
      case 'question_response':
        return this.questionResponse(request.params)
      case 'steer':
        return this.steer(request.params)
      case 'set_plan_mode':
        return this.setPlanMode(request.params)
      case 'set_mode':
        return this.setMode(request.params)
      case 'cancel':
        return this.cancel()
      case 'cancel_all':
        return this.cancelAll()
      case 'slash':
        return this.slash(request.params)
      case 'provider_list':
        return this.providerList()
      case 'fetch_models':
        return this.fetchModels(request.params)
      case 'provider_save':
        return this.providerSave(request.params)
      case 'provider_select':
        return this.providerSelect(request.params)
      case 'provider_delete':
        return this.providerDelete(request.params)
      case 'replay':
        if (this.options.wireMode) {
          this.wire.emitNotification({
            id: this.nextId(),
            category: 'session',
            type: 'replay_unavailable',
            severity: 'warning',
            title: 'Replay unavailable',
            body: 'This Bun bridge does not persist provider-specific wire replay records.',
          })
        }
        return { accepted: true }
      case 'shutdown':
        return this.shutdown()
      default:
        return this.reject(`Unknown method: ${request.method}`)
    }
  }

  /** Wait until the current asynchronous query has fully emitted its terminal events. */
  async waitForIdle(): Promise<void> {
    while (this.activeTurn !== undefined) {
      const active = this.activeTurn
      await active
      if (this.activeTurn === active) return
    }
  }

  private async initialize(params: Record<string, unknown>): Promise<BridgeDispatchResult> {
    if (this.activeTurn) return this.reject('A query is already running. Wait or send cancel.')

    const resumeId = textValue(params.resume_session_id)
    let resumed = false
    if (resumeId) {
      const result = await this.options.session.resume(resumeId)
      resumed = result.status === 'resumed'
      this.state = stateFromSnapshot(this.options.session.snapshot)
    }

    const suppliedModel = textValue(params.model)
    const suppliedBaseUrl = textValue(params.base_url)
    const suppliedApiKey = textValue(params.api_key)
    const profile = !suppliedModel && !suppliedBaseUrl ? await this.options.profileStore.active() : undefined
    const mode = normalizeInteractionMode(params.mode, booleanValue(params.plan_mode, false))
    const projectDirectory = textValue(params.project_dir)

    const nextConfig: Record<string, unknown> = {
      permission_mode: textValue(params.permission_mode) || 'accept-all',
      verbose: booleanValue(params.verbose, false),
      thinking: booleanValue(params.thinking, true),
      debug: booleanValue(params.debug, false),
      mode,
      plan_mode: mode === 'plan',
      model: suppliedModel || profile?.model || '',
    }
    const baseUrl = suppliedBaseUrl || profile?.base_url || ''
    const apiKey = suppliedApiKey || profile?.api_key || ''
    const provider = profile?.provider || textValue(params.provider)
    if (baseUrl) nextConfig.base_url = baseUrl
    if (apiKey) nextConfig.api_key = apiKey
    if (provider) nextConfig.provider = provider
    for (const [name, value] of Object.entries(profile?.sampling ?? {})) nextConfig[name] = value
    await this.autoSelectSoleModel(nextConfig, suppliedBaseUrl, suppliedApiKey, suppliedModel)

    replaceRecord(this.config, nextConfig)
    if (projectDirectory) this.options.session.update({ cwd: projectDirectory })
    this.options.session.update({
      interactionMode: mode,
      model: textValue(this.config.model),
      planMode: mode === 'plan',
    })
    this.state = stateFromSnapshot(this.options.session.snapshot)
    this.state.metadata = { ...this.options.session.snapshot.metadata }
    await this.options.runtime.configure?.(this.configuration)
    this.runtimeInfo = await this.options.runtime.initialize?.({
      config: this.configuration,
      session: this.options.session.snapshot,
    }) ?? {}
    this.initialized = true

    if (this.options.wireMode) {
      this.emitWireInitDone()
      if (resumed && this.state.messages.length) this.emitResumedHistory()
    }
    this.emitLegacy('ready', {
      model: textValue(this.config.model),
      provider: resolveProvider(textValue(this.config.model), this.config),
      tools: this.runtimeInfo.tools?.length ?? 0,
      permission_mode: textValue(this.config.permission_mode),
      has_profile: Boolean(textValue(this.config.model)),
      skills: [...(this.runtimeInfo.skills ?? [])].sort(),
      agents: (this.runtimeInfo.agents ?? []).map(agent => agent.name),
    })
    return { accepted: true }
  }

  private startQuery(params: Record<string, unknown>): BridgeDispatchResult {
    if (!this.initialized) return this.reject("Not initialized. Send 'init' first.")
    if (this.activeTurn) return this.reject('A query is already running. Wait or send cancel.')
    const text = queryText(params.text ?? params.user_input)
    if (!text) return this.reject('Empty query.')

    const controller = new AbortController()
    let task: Promise<void>
    task = this.runQuery(text, params, controller)
      .catch(error => this.emitError(`${errorName(error)}: ${errorMessage(error)}`))
      .finally(() => {
        if (this.activeTurn === task) {
          this.activeTurn = undefined
          this.activeAbortController = undefined
        }
      })
    this.activeAbortController = controller
    this.activeTurn = task
    return { accepted: true }
  }

  private async runQuery(text: string, params: Record<string, unknown>, controller: AbortController): Promise<void> {
    const planMode = params.plan_mode === undefined
      ? booleanValue(this.config.plan_mode, false)
      : booleanValue(params.plan_mode, false)
    const mode = normalizeInteractionMode(params.mode ?? this.config.mode, planMode)
    this.config.mode = mode
    this.config.plan_mode = mode === 'plan'
    await this.options.runtime.configure?.(this.configuration)

    if (this.options.wireMode) this.wire.emitTurnBegin(text)
    try {
      for await (const event of this.options.runtime.streamTurn({
        config: this.configuration,
        session: this.options.session.snapshot,
        state: this.state,
        text,
      }, controller.signal)) {
        this.emitStreamEvent(event)
      }
    } catch (error) {
      if (!controller.signal.aborted) this.emitError(`${errorName(error)}: ${errorMessage(error)}`)
    } finally {
      this.synchronizeSession()
      try {
        await this.options.session.save()
      } catch (error) {
        this.emitError(`Could not save session: ${errorMessage(error)}`)
      }
      if (this.options.wireMode) {
        this.wire.emitTurnEnd()
        this.emitWireStatus()
      }
      this.emitLegacy('query_done', {})
      this.emitLegacy('state', this.statePayload())
    }
  }

  private emitStreamEvent(event: StreamEvent): void {
    switch (event.type) {
      case 'text':
        if (this.options.wireMode) this.wire.emitText(event.text)
        this.emitLegacy('text_chunk', { text: event.text })
        return
      case 'thinking':
        if (this.options.wireMode) this.wire.emitThink(event.text)
        this.emitLegacy('thinking_chunk', { text: event.text })
        return
      case 'provider_retry':
        if (this.options.wireMode) {
          this.wire.emitNotification({
            id: 'provider-connection',
            category: 'provider_connection',
            type: event.final ? 'failed' : 'retrying',
            severity: event.final ? 'error' : 'warning',
            title: 'Provider connection',
            body: event.final
              ? `${event.error}\nUse /retry-connection to retry the last prompt.`
              : `${event.error}\nRetrying provider connection in ${event.delay}ms `
                + `(${event.attempt}/${event.maxAttempts}).`,
          })
        }
        this.emitLegacy('provider_retry', {
          error: event.error,
          attempt: event.attempt,
          max_attempts: event.maxAttempts,
          delay: event.delay,
          final: event.final,
        })
        return
      case 'tool_start': {
        const toolCallId = this.options.wireMode
          ? this.wire.emitToolStart(event.call.id, event.call.function.name, event.call.function.arguments)
          : event.call.id
        this.emitLegacy('tool_start', {
          name: event.call.function.name,
          inputs: event.call.function.arguments,
          tool_call_id: toolCallId,
        })
        return
      }
      case 'permission_request':
        if (this.options.wireMode) {
          this.wire.emitRequest(event.request.requestId, 'approval_request', {
            id: event.request.requestId,
            tool_call_id: event.request.toolCall.id || this.wire.activeToolCallId,
            action: event.request.toolCall.function.name,
            description: event.request.description,
          })
        }
        this.emitLegacy('permission_request', {
          id: event.request.requestId,
          tool_name: event.request.toolCall.function.name,
          description: event.request.description,
          inputs: event.request.inputs,
        })
        return
      case 'tool_end':
        if (this.options.wireMode) {
          this.wire.emitToolResult({
            toolCallId: event.result.toolCallId,
            returnValue: event.result.result,
            permitted: event.result.permitted,
            durationMs: event.result.durationMs,
          })
        }
        this.emitLegacy('tool_end', {
          name: event.result.name,
          result: event.result.result,
          permitted: event.result.permitted,
          tool_call_id: event.result.toolCallId,
          duration_ms: event.result.durationMs,
        })
        return
      case 'turn_done':
        this.costTracker.recordTurn(event.model, event.usage.inputTokens, event.usage.outputTokens, '', {
          ...(event.usage.cacheReadTokens === undefined ? {} : { cacheReadTokens: event.usage.cacheReadTokens }),
          ...(event.usage.cacheCreationTokens === undefined
            ? {}
            : { cacheCreationTokens: event.usage.cacheCreationTokens }),
          sessionId: this.options.session.snapshot.sessionId,
        })
        this.emitLegacy('turn_done', {
          input_tokens: event.usage.inputTokens,
          output_tokens: event.usage.outputTokens,
          tool_calls_count: event.toolCallsCount,
          model: event.model,
        })
        return
      case 'skill_suggestion':
        this.emitLegacy('skill_suggested', {
          skill_name: event.skillName,
          version: event.version,
          source_path: event.sourcePath,
          tool_count: event.toolCount,
          unique_tools: [...event.uniqueTools],
        })
        if (this.options.wireMode) {
          this.wire.emitNotification({
            id: this.nextId(),
            category: 'skills',
            type: 'skill_suggested',
            severity: 'info',
            title: 'Skill suggested',
            body: event.skillName,
            payload: { version: event.version, source_path: event.sourcePath },
          })
        }
    }
  }

  private async permissionResponse(params: Record<string, unknown>): Promise<BridgeDispatchResult> {
    if (!this.options.runtime.respondPermission) {
      return this.reject('Permission responses require a configured runtime interaction port.')
    }
    const requestId = textValue(params.request_id) || textValue(params.id)
    if (!requestId) return this.reject('request_id is required')
    const response = textValue(params.response) || (params.granted === true ? 'approve' : 'reject')
    const accepted = await this.options.runtime.respondPermission({ requestId, response })
    return accepted ? { accepted: true } : this.reject('Permission response did not match an active request.')
  }

  private async questionResponse(params: Record<string, unknown>): Promise<BridgeDispatchResult> {
    if (!this.options.runtime.respondQuestion) {
      return this.reject('Question responses require a configured runtime interaction port.')
    }
    const requestId = textValue(params.request_id) || textValue(params.id)
    if (!requestId) return this.reject('request_id is required')
    const answers = stringAnswers(params.answers, textValue(params.answer))
    const accepted = await this.options.runtime.respondQuestion({ requestId, answers })
    return accepted ? { accepted: true } : this.reject('Question response did not match an active request.')
  }

  private async steer(params: Record<string, unknown>): Promise<BridgeDispatchResult> {
    const content = textValue(params.user_input) || textValue(params.content) || textValue(params.text)
    if (!content) return this.reject('Steering text is required.')
    const steer = this.options.runtime.steer
    if (!steer) return this.reject('Steering requires a configured runtime port.')
    const accepted = await steer(content)
    if (!accepted) return this.reject('No active query accepted the steering input.')
    if (this.options.wireMode) this.wire.emitEvent('steer_input', { content })
    return { accepted: true }
  }

  private async setPlanMode(params: Record<string, unknown>): Promise<BridgeDispatchResult> {
    const enabled = booleanValue(params.enabled, booleanValue(params.plan_mode, false))
    const mode = normalizeInteractionMode(params.mode ?? this.config.mode, enabled)
    this.config.mode = mode
    this.config.plan_mode = mode === 'plan'
    await this.options.runtime.configure?.(this.configuration)
    this.synchronizeSession()
    if (this.options.wireMode) this.emitWireStatus()
    return { accepted: true }
  }

  private async setMode(params: Record<string, unknown>): Promise<BridgeDispatchResult> {
    const mode = normalizeInteractionMode(params.mode, false)
    this.config.mode = mode
    this.config.plan_mode = mode === 'plan'
    await this.options.runtime.configure?.(this.configuration)
    this.synchronizeSession()
    if (this.options.wireMode) this.emitWireStatus()
    return { accepted: true }
  }

  private async cancel(): Promise<BridgeDispatchResult> {
    const task = this.activeTurn
    if (!task) return { accepted: false, error: 'No query is running.' }
    this.activeAbortController?.abort(new Error('Turn cancelled'))
    await this.options.runtime.cancel?.()
    await this.options.runtime.cancelAll?.()
    return { accepted: true }
  }

  private async cancelAll(): Promise<BridgeDispatchResult> {
    this.activeAbortController?.abort(new Error('Turn cancelled'))
    await this.options.runtime.cancel?.()
    const cancelled = await this.options.runtime.cancelAll?.() ?? (this.activeTurn ? 1 : 0)
    if (cancelled) this.emitLegacy('slash_result', { output: `Cancelled ${cancelled} running sub-agent(s).` })
    return { accepted: true }
  }

  private async slash(params: Record<string, unknown>): Promise<BridgeDispatchResult> {
    if (this.activeTurn) return this.reject('A query is already running. Wait or send cancel.')
    const command = textValue(params.command)
    if (!command.startsWith('/')) return this.reject(`Not a slash command: ${command}`)
    const router = new BridgeSlashRouter({
      config: this.config,
      cwd: this.options.session.snapshot.cwd,
      host: this.slashHost(),
      state: this.state,
    })
    const result = await router.dispatch(command)
    if (result.output) this.emitLegacy('slash_result', { output: result.output })
    this.synchronizeSession()
    const statusChangingCommand = result.command === 'thinking'
      || result.command === 'reasoning'
      || result.command === 'plan'
    if (this.options.wireMode && statusChangingCommand) {
      this.emitWireStatus()
    }
    return result.status === 'handled' ? { accepted: true } : { accepted: false, error: result.output }
  }

  private async providerList(): Promise<BridgeDispatchResult> {
    const profiles = await this.options.profileStore.list()
    this.emitLegacy('provider_list', { profiles: profiles.map(profilePayload) })
    return { accepted: true }
  }

  private async fetchModels(params: Record<string, unknown>): Promise<BridgeDispatchResult> {
    const baseUrl = textValue(params.base_url)
    if (!baseUrl) return this.reject('base_url is required for fetch_models')
    const validationError = validateBridgeModelFetchUrl(baseUrl)
    if (validationError) return this.reject(`fetch_models blocked: ${validationError}`)
    const discovery = this.options.modelDiscovery
    if (!discovery) return this.reject('Model discovery is not configured for this bridge.')
    try {
      const models = await discovery.fetchModels({
        baseUrl,
        apiKey: textValue(params.api_key),
        provider: textValue(params.provider) || resolveProvider(textValue(this.config.model), this.config),
      })
      this.emitLegacy('models_list', { models: [...models], base_url: baseUrl })
      return { accepted: true }
    } catch (error) {
      return this.reject(`Failed to fetch models: ${errorMessage(error)}`)
    }
  }

  private async providerSave(params: Record<string, unknown>): Promise<BridgeDispatchResult> {
    const name = textValue(params.name)
    const baseUrl = textValue(params.base_url)
    const apiKey = textValue(params.api_key)
    const model = textValue(params.model)
    if (!name || !baseUrl || !model) return this.reject('name, base_url, and model are required')
    try {
      const profile = await this.options.profileStore.save({
        name,
        baseUrl,
        apiKey,
        model,
        ...(textValue(params.provider) ? { provider: textValue(params.provider) } : {}),
        setActive: true,
      })
      await this.applyProfile(profile)
      this.emitLegacy('provider_saved', {
        profile: profilePayload(profile),
        message: `Profile '${name}' saved and activated. Model: ${model}`,
      })
      return { accepted: true }
    } catch (error) {
      return this.reject(`Could not save provider profile: ${errorMessage(error)}`)
    }
  }

  private async providerSelect(params: Record<string, unknown>): Promise<BridgeDispatchResult> {
    const name = textValue(params.name)
    if (!name) return this.reject('Profile name is required')
    try {
      if (!await this.options.profileStore.setActive(name)) return this.reject(`Profile '${name}' not found`)
      const profile = await this.options.profileStore.active()
      if (!profile) return this.reject(`Profile '${name}' could not be loaded`)
      await this.applyProfile(profile)
      this.emitLegacy('provider_saved', {
        profile: profilePayload(profile),
        message: `Switched to profile '${name}'. Model: ${profile.model}`,
      })
      return { accepted: true }
    } catch (error) {
      return this.reject(`Could not select provider profile: ${errorMessage(error)}`)
    }
  }

  private async providerDelete(params: Record<string, unknown>): Promise<BridgeDispatchResult> {
    const name = textValue(params.name)
    if (!name) return this.reject('Profile name is required')
    if (!await this.options.profileStore.delete(name)) return this.reject(`Profile '${name}' not found`)
    this.emitLegacy('slash_result', { output: `Profile '${name}' deleted.` })
    return { accepted: true }
  }

  private async shutdown(): Promise<BridgeDispatchResult> {
    this.running = false
    await this.options.runtime.cancel?.()
    await this.options.runtime.shutdown?.()
    return { accepted: true }
  }

  private async applyProfile(profile: ProviderProfile): Promise<void> {
    this.config.model = profile.model
    this.config.base_url = profile.base_url
    if (profile.provider) this.config.provider = profile.provider
    else delete this.config.provider
    if (profile.api_key) this.config.api_key = profile.api_key
    else delete this.config.api_key
    for (const [name, value] of Object.entries(profile.sampling)) this.config[name] = value
    await this.options.runtime.configure?.(this.configuration)
    this.options.session.update({ model: profile.model })
    this.emitLegacy('model_changed', {
      model: profile.model,
      provider: resolveProvider(profile.model, this.config),
    })
    if (this.options.wireMode && this.initialized) {
      this.emitWireInitDone()
      this.emitWireStatus()
    }
  }

  private slashHost(): BridgeSlashRouterHost {
    const supplied = this.options.slashHost ?? {}
    const host: BridgeSlashRouterHost = {
      ...supplied,
      configChanged: async config => {
        await this.options.runtime.configure?.(config)
        await supplied.configChanged?.(config)
      },
      costTracker: supplied.costTracker ?? this.costTracker,
      models: supplied.models ?? {
        list: async input => {
          const discovery = this.options.modelDiscovery
          if (!discovery || !input.baseUrl) return []
          const validationError = validateBridgeModelFetchUrl(input.baseUrl)
          if (validationError) return []
          return discovery.fetchModels({
            baseUrl: input.baseUrl,
            apiKey: input.apiKey,
            provider: resolveProvider(input.currentModel, this.config),
          })
        },
        switchModel: async model => this.switchModel(model),
      },
      providers: supplied.providers ?? {
        active: () => this.options.profileStore.active(),
        list: () => this.options.profileStore.list(),
        saveSampling: async (name, sampling) => {
          const updateSampling = this.options.profileStore.updateSampling
          return updateSampling ? updateSampling(name, { ...sampling }) : undefined
        },
        select: async name => {
          if (!await this.options.profileStore.setActive(name)) return undefined
          const profile = await this.options.profileStore.active()
          if (!profile) return undefined
          await this.applyProfile(profile)
          return profile
        },
      },
      tools: supplied.tools ?? (() => [...(this.runtimeInfo.tools ?? [])]),
      agents: supplied.agents ?? (() => [...(this.runtimeInfo.agents ?? [])]),
    }
    return host
  }

  private async switchModel(model: string): Promise<void> {
    const value = model.trim()
    if (!value) throw new TypeError('model must not be empty')
    this.config.model = value
    await this.options.profileStore.updateActiveModel?.(value)
    await this.options.runtime.configure?.(this.configuration)
    this.options.session.update({ model: value })
    this.emitLegacy('model_changed', { model: value, provider: resolveProvider(value, this.config) })
    if (this.options.wireMode && this.initialized) {
      this.emitWireInitDone()
      this.emitWireStatus()
    }
  }

  private synchronizeSession(): void {
    this.options.session.update({
      interactionMode: textValue(this.config.mode) || 'code',
      messages: this.state.messages.map(rawMessage),
      metadata: { ...this.state.metadata },
      model: textValue(this.config.model),
      planMode: booleanValue(this.config.plan_mode, false),
      thinkingContent: [...this.state.thinkingContent],
      toolExecutions: this.state.toolExecutions.map(record => ({ ...record })),
      totalInputTokens: this.state.totalInputTokens,
      totalOutputTokens: this.state.totalOutputTokens,
      turnCount: this.state.turnCount,
    })
  }

  private emitWireInitDone(): void {
    const snapshot = this.options.session.snapshot
    this.wire.emitInitDone({
      model: textValue(this.config.model),
      session_id: snapshot.sessionId,
      cwd: snapshot.cwd,
      git_branch: '',
      head_hash: '',
      context_limit: this.contextLimit(textValue(this.config.model)),
      agent_name: snapshot.agentId,
      skills: [...(this.runtimeInfo.skills ?? [])].sort(),
      skill_descriptions: {},
      mode: textValue(this.config.mode) || 'code',
      version: '0.3.0',
    })
  }

  private emitWireStatus(): void {
    const model = textValue(this.config.model)
    this.wire.emitStatus({
      context_tokens: this.contextTokens(),
      max_context: this.contextLimit(model),
      mcp_status: {},
      plan_mode: booleanValue(this.config.plan_mode, false),
      mode: textValue(this.config.mode) || 'code',
      reasoning_effort: textValue(this.config.reasoning_effort) || 'off',
    })
  }

  private emitResumedHistory(): void {
    for (const record of this.options.session.historyReplayRecords()) {
      this.wire.emitNotification({
        id: this.nextId(),
        category: record.category,
        type: record.type,
        severity: record.severity,
        title: '',
        body: record.body,
      })
    }
  }

  private statePayload(): Record<string, unknown> {
    const model = textValue(this.config.model)
    const contextLimit = this.contextLimit(model)
    const usedContext = this.contextTokens()
    return {
      turn_count: this.state.turnCount,
      total_input_tokens: this.state.totalInputTokens,
      total_output_tokens: this.state.totalOutputTokens,
      message_count: this.state.messages.length,
      tool_execution_count: this.state.toolExecutions.length,
      context_limit: contextLimit,
      remaining_context: Math.max(0, contextLimit - usedContext),
      used_context: usedContext,
      cost_usd: calcCost(model, this.state.totalInputTokens, this.state.totalOutputTokens),
      reasoning_effort: textValue(this.config.reasoning_effort) || 'off',
    }
  }

  private contextTokens(): number {
    const model = textValue(this.config.model)
    return estimateContextTokens(this.state.messages.map(rawMessage), { model })
  }

  private get costTracker(): CostTracker {
    return this.serverCostTracker
  }

  private readonly serverCostTracker = new CostTracker()

  private async autoSelectSoleModel(
    config: Record<string, unknown>,
    baseUrl: string,
    apiKey: string,
    suppliedModel: string,
  ): Promise<void> {
    if (!baseUrl || suppliedModel || !this.options.modelDiscovery) return
    if (validateBridgeModelFetchUrl(baseUrl) !== undefined) return
    try {
      const models = await this.options.modelDiscovery.fetchModels({
        baseUrl,
        apiKey,
        provider: textValue(config.provider) || 'custom',
      })
      if (models.length !== 1 || !models[0]?.trim()) return
      const model = models[0].trim()
      config.model = model
      await this.options.profileStore.updateActiveModel?.(model)
    } catch {
      // Discovery is a best-effort startup convenience. Explicit fetch_models
      // remains available to surface the host transport error to a client.
    }
  }

  private emitLegacy(event: string, data: Record<string, unknown>): void {
    if (this.options.wireMode) {
      if (event === 'slash_result') {
        this.wire.emitNotification({
          id: this.nextId(),
          category: 'slash',
          type: 'slash_result',
          severity: 'info',
          title: '',
          body: textValue(data.output),
        })
      }
      return
    }
    this.options.output.writeLegacy({ event, data })
  }

  private emitError(message: string): void {
    if (this.options.wireMode) {
      this.wire.emitNotification({
        id: this.nextId(),
        category: 'bridge',
        type: 'error',
        severity: 'error',
        title: 'Bridge error',
        body: message,
      })
      return
    }
    this.options.output.writeLegacy({ event: 'error', data: { message } })
  }

  private reject(message: string): BridgeDispatchResult {
    this.emitError(message)
    return { accepted: false, error: message }
  }

  private nextId(): string {
    const factory = this.options.idFactory ?? (() => crypto.randomUUID())
    const value = factory().trim()
    if (!value) throw new TypeError('idFactory must return a non-empty string')
    return value
  }
}

/** Serialize bridge output exactly once at a caller-owned NDJSON transport boundary. */
export class NdjsonBridgeOutput implements BridgeServerOutput {
  constructor(private readonly writer: BridgeLineWriter) {}

  writeLegacy(event: LegacyBridgeEvent): void {
    this.writer.write(`${JSON.stringify({ event: event.event, data: event.data })}\n`)
  }

  writeWire(frame: BridgeWireFrame): void {
    this.writer.write(`${serializeWireFrame(frame)}\n`)
  }
}

/** Consume newline-delimited input without opening stdin or stdout itself. */
export async function serveBridgeNdjson(
  input: AsyncIterable<string | Uint8Array>,
  server: BridgeServer,
): Promise<void> {
  const decoder = new TextDecoder()
  let buffer = ''
  for await (const chunk of input) {
    buffer += typeof chunk === 'string' ? chunk : decoder.decode(chunk, { stream: true })
    const lines = buffer.split(/\r?\n/u)
    buffer = lines.pop() ?? ''
    for (const line of lines) {
      if (line.trim()) await server.handleLine(line)
    }
  }
  buffer += decoder.decode()
  if (buffer.trim()) await server.handleLine(buffer)
}

/** Parse legacy bridge notifications and JSON-RPC 2.0 request envelopes without accepting ambiguous values. */
export function parseBridgeRequestLine(line: string): BridgeServerRequest {
  let parsed: unknown
  try {
    parsed = JSON.parse(line) as unknown
  } catch {
    throw new TypeError('Invalid JSON')
  }
  if (!isRecord(parsed) || typeof parsed.method !== 'string' || !parsed.method.trim()) {
    throw new TypeError('Invalid bridge request')
  }
  if (parsed.jsonrpc !== undefined && parsed.jsonrpc !== '2.0') {
    throw new TypeError('Invalid JSON-RPC version')
  }
  if (parsed.id !== undefined && typeof parsed.id !== 'string' && typeof parsed.id !== 'number') {
    throw new TypeError('JSON-RPC request id must be a string or number')
  }
  return {
    method: parsed.method,
    params: isRecord(parsed.params) ? { ...parsed.params } : {},
    ...(typeof parsed.id === 'string' || typeof parsed.id === 'number' ? { id: parsed.id } : {}),
  }
}

/** Validate the model-discovery URL before it reaches a caller-owned HTTP port. */
export function validateBridgeModelFetchUrl(baseUrl: string): string | undefined {
  let parsed: URL
  try {
    parsed = new URL(baseUrl)
  } catch {
    return 'Invalid URL format'
  }
  if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') return 'URL scheme must be http or https'
  const hostname = parsed.hostname.toLowerCase().replace(/^\[/u, '').replace(/\]$/u, '')
  if (!hostname) return 'URL must have a host'
  if (hostname === 'localhost' || hostname === '0.0.0.0') return 'Private/localhost addresses are not allowed'
  if (isPrivateAddress(hostname)) return 'Private IP addresses are not allowed'
  return undefined
}

function isPrivateAddress(hostname: string): boolean {
  if (!isIP(hostname)) return false
  if (hostname === '::1' || hostname === '::') return true
  if (hostname.includes(':')) {
    const normalized = hostname.toLowerCase()
    return normalized.startsWith('fc') || normalized.startsWith('fd') || normalized.startsWith('fe8')
      || normalized.startsWith('fe9') || normalized.startsWith('fea') || normalized.startsWith('feb')
  }
  const parts = hostname.split('.').map(part => Number(part))
  const first = parts[0] ?? -1
  const second = parts[1] ?? -1
  return first === 0 || first === 10 || first === 127 || first >= 224
    || (first === 100 && second >= 64 && second <= 127)
    || (first === 169 && second === 254)
    || (first === 172 && second >= 16 && second <= 31)
    || (first === 192 && second === 168)
}

function stateFromSnapshot(snapshot: BridgeSessionSnapshot): AgentState {
  const state = createAgentState(snapshot.messages.flatMap(chatMessageFromRaw))
  state.metadata = { ...snapshot.metadata }
  state.thinkingContent = snapshot.thinkingContent.filter((entry): entry is string => typeof entry === 'string')
  state.toolExecutions = snapshot.toolExecutions.flatMap(toolExecutionFromRaw)
  state.totalInputTokens = snapshot.totalInputTokens
  state.totalOutputTokens = snapshot.totalOutputTokens
  state.turnCount = snapshot.turnCount
  return state
}

function chatMessageFromRaw(message: Readonly<Record<string, unknown>>): ChatMessage[] {
  const role = textValue(message.role)
  const content = messageContent(message.content)
  if (content === undefined) return []
  if (role === 'system' || role === 'user') return [{ role, content }]
  if (role === 'assistant') {
    const toolCalls = Array.isArray(message.tool_calls) ? message.tool_calls.flatMap(toolCallFromRaw) : []
    return [{
      role,
      content,
      ...(typeof message.thinking === 'string' ? { thinking: message.thinking } : {}),
      ...(typeof message.thinking_signature === 'string' ? { thinking_signature: message.thinking_signature } : {}),
      ...(toolCalls.length ? { tool_calls: toolCalls } : {}),
    }]
  }
  if (role === 'tool' && typeof content === 'string' && textValue(message.tool_call_id)) {
    return [{
      role,
      content,
      tool_call_id: textValue(message.tool_call_id),
      ...(typeof message.name === 'string' ? { name: message.name } : {}),
      ...(message.is_error === true ? { is_error: true } : {}),
    }]
  }
  return []
}

function messageContent(value: unknown): MessageContent | undefined {
  if (typeof value === 'string') return value
  if (isRecord(value) && typeof value.text === 'string') return value.text
  if (!Array.isArray(value)) return undefined
  const parts: ContentPart[] = []
  for (const part of value) {
    if (!isRecord(part) || typeof part.type !== 'string') return []
    if (part.type === 'text' && typeof part.text === 'string') {
      parts.push({ type: 'text', text: part.text })
      continue
    }
    if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
      const detail = part.image_url.detail
      const imageUrl: { detail?: 'auto' | 'high' | 'low'; url: string } = { url: part.image_url.url }
      if (detail === 'auto' || detail === 'high' || detail === 'low') imageUrl.detail = detail
      parts.push({ type: 'image_url', image_url: imageUrl })
      continue
    }
  }
  return parts
}

function toolCallFromRaw(value: unknown): ToolCall[] {
  if (!isRecord(value) || textValue(value.id) === '') return []
  const functionValue = isRecord(value.function) ? value.function : undefined
  const name = textValue(functionValue?.name) || textValue(value.name)
  const arguments_ = jsonObjectValue(functionValue?.arguments ?? value.input)
  if (!name || !isJsonObject(arguments_)) return []
  return [{ id: textValue(value.id), type: 'function', function: { name, arguments: arguments_ } }]
}

function toolExecutionFromRaw(value: unknown): AgentState['toolExecutions'] {
  if (!isRecord(value)) return []
  const durationMs = numberValue(value.durationMs ?? value.duration_ms)
  const name = textValue(value.name ?? value.tool_name)
  const permitted = typeof value.permitted === 'boolean' ? value.permitted : true
  const result = value.result ?? value.return_value
  const toolCallId = textValue(value.toolCallId ?? value.tool_call_id)
  const inputs = value.inputs ?? value.arguments
  if (durationMs === undefined || !name || typeof result !== 'string' || !toolCallId || !isJsonObject(inputs)) return []
  return [{
    durationMs,
    inputs,
    name,
    permitted,
    result,
    toolCallId,
  }]
}

function rawMessage(message: ChatMessage): Record<string, unknown> {
  switch (message.role) {
    case 'assistant':
      return {
        role: message.role,
        content: message.content,
        ...(message.thinking ? { thinking: message.thinking } : {}),
        ...(message.thinking_signature ? { thinking_signature: message.thinking_signature } : {}),
        ...(message.tool_calls?.length ? { tool_calls: [...message.tool_calls] } : {}),
      }
    case 'tool':
      return {
        role: message.role,
        content: message.content,
        tool_call_id: message.tool_call_id,
        ...(message.name ? { name: message.name } : {}),
        ...(message.is_error ? { is_error: true } : {}),
      }
    case 'system':
    case 'user':
      return { role: message.role, content: message.content }
  }
}

/**
 * Serialized profile payloads are served to bridge clients, so the raw API
 * key never leaves this process. Selection resolves the real key server-side
 * through the profile store.
 */
function profilePayload(profile: ProviderProfile & { readonly active?: boolean }): BridgeProviderProfile {
  return {
    name: profile.name,
    api_key: profile.api_key ? '********' : '',
    base_url: profile.base_url,
    model: profile.model,
    provider: profile.provider,
    sampling: { ...profile.sampling },
    ...(profile.active === undefined ? {} : { active: profile.active }),
  }
}

function queryText(value: unknown): string {
  if (typeof value === 'string') return value.trim()
  if (!Array.isArray(value)) return ''
  return value.map(part => isRecord(part) ? String(part.text ?? '') : String(part)).join('\n').trim()
}

function stringAnswers(value: unknown, directAnswer: string): Record<string, string> {
  if (isRecord(value)) {
    const answers = Object.entries(value)
      .flatMap(([key, item]) => typeof item === 'string' ? [[key, item] as const] : [])
    return Object.fromEntries(answers)
  }
  return directAnswer ? { answer: directAnswer } : {}
}

function replaceRecord(target: Record<string, unknown>, source: Readonly<Record<string, unknown>>): void {
  for (const key of Object.keys(target)) delete target[key]
  Object.assign(target, source)
}

function textValue(value: unknown): string {
  return typeof value === 'string' ? value.trim() : ''
}

function booleanValue(value: unknown, fallback: boolean): boolean {
  return typeof value === 'boolean' ? value : fallback
}

function numberValue(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}

function isJsonObject(value: unknown): value is JsonObject {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function jsonObjectValue(value: unknown): JsonObject | undefined {
  if (isJsonObject(value)) return value
  if (typeof value !== 'string' || !value.trim()) return undefined
  try {
    const parsed: unknown = JSON.parse(value)
    return isJsonObject(parsed) ? parsed : undefined
  } catch {
    return undefined
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

function errorName(error: unknown): string {
  return error instanceof Error ? error.name : 'Error'
}
