// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash } from 'node:crypto'
import { Database } from 'bun:sqlite'

import { canonicalJson, paramsHash, type HubAuthClaims } from '../../security/hubAuthEnvelope.js'
import { parseToolArguments, type ToolCall } from '../../types/toolCalls.js'
import { HUB_AUTH_LIFETIME_MS } from '../commandRouter.js'
import type { AuthenticatedContext, HubAuditInput } from '../identityPort.js'
import { HubProtocolError, isRecord, readId, type HubMethod } from '../protocol.js'
import type { PoolCallResult, PoolMachine } from '../runtimePool.js'
import type { HubMethodHandler } from '../server.js'
import type { HubSigner } from '../signer.js'
import { SUPERVISOR_DISPLAY_NAME, SUPERVISOR_PRINCIPAL_ID } from './audit.js'
import type { SupervisorModelPort, SupervisorModelRequest, SupervisorModelResponse } from './runner.js'

/**
 * Which model XERXES runs on is chosen at runtime, not by environment variables.
 *
 * One row in the Hub database names a machine, one of its provider profiles, a
 * model and a thinking effort. Every supervisor model call is a signed
 * `hub.model.complete` to that machine's daemon: the Hub signs as the supervisor
 * agent principal with the scope `supervisor.model`, and the provider key never
 * leaves the machine. Changing the row swaps the model port at once; nothing
 * restarts.
 */

export const UNCONFIGURED_MESSAGE = "Choose XERXES's model in XERXES settings"
const MAX_REQUEST_BYTES = 1024 * 1024
const MAX_LIST = 500
const FAILOVER_SKIP_CODES = new Set(['unknown_profile', 'not_found', 'invalid_request', 'unsupported', 'unavailable'])

export type SupervisorConfigStatus = 'configured' | 'unconfigured' | 'machine_offline' | 'invalid'

export interface SupervisorModelChoice {
  readonly machineId: string
  readonly profile: string
  readonly model: string
  /** null: the provider default applies. */
  readonly effort: string | null
}
export interface StoredSupervisorModel extends SupervisorModelChoice {
  readonly updatedBy: string
  readonly updatedAt: number
}

interface Row {
  machine_id: string
  profile: string
  model: string
  effort: string | null
  updated_by: string
  updated_at: number
}

/** The single stored choice, in `hub.sqlite`. */
export class SupervisorModelStore {
  private readonly db: Database
  constructor(databasePath: string) {
    this.db = new Database(databasePath, { create: true, strict: true })
    this.db.exec(`
      PRAGMA journal_mode = WAL;
      PRAGMA synchronous = FULL;
      PRAGMA busy_timeout = 5000;
      CREATE TABLE IF NOT EXISTS hub_supervisor_model (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        machine_id TEXT NOT NULL, profile TEXT NOT NULL, model TEXT NOT NULL, effort TEXT,
        updated_by TEXT NOT NULL, updated_at INTEGER NOT NULL
      );
    `)
  }

  get(): StoredSupervisorModel | undefined {
    const row = this.db.query<Row, []>('SELECT machine_id, profile, model, effort, updated_by, updated_at FROM hub_supervisor_model WHERE id = 1').get()
    return row ? { machineId: row.machine_id, profile: row.profile, model: row.model, effort: row.effort, updatedBy: row.updated_by, updatedAt: row.updated_at } : undefined
  }

  set(choice: SupervisorModelChoice, updatedBy: string, updatedAt: number): StoredSupervisorModel {
    this.db.query(`INSERT INTO hub_supervisor_model (id, machine_id, profile, model, effort, updated_by, updated_at) VALUES (1, ?, ?, ?, ?, ?, ?)
      ON CONFLICT(id) DO UPDATE SET machine_id = excluded.machine_id, profile = excluded.profile, model = excluded.model,
        effort = excluded.effort, updated_by = excluded.updated_by, updated_at = excluded.updated_at`)
      .run(choice.machineId, choice.profile, choice.model, choice.effort, updatedBy, updatedAt)
    return { ...choice, updatedBy, updatedAt }
  }

  close(): void { this.db.close() }
}

/** What the control needs from the runtime pool. */
export interface SupervisorMachines {
  machines(): readonly PoolMachine[]
  callModel(machineId: string, method: string, params: Readonly<Record<string, unknown>>, options?: { readonly signal?: AbortSignal }): Promise<PoolCallResult>
}

export interface SupervisorModelControlOptions {
  readonly databasePath: string
  readonly signer: Pick<HubSigner, 'hubId' | 'sign'>
  readonly machines: SupervisorMachines
  /** Appends to the Hub audit log (identity.audit). A failure refuses the change. */
  readonly audit: (record: HubAuditInput) => Promise<void>
  /** Only used while nothing is stored: the `XERXES_HUB_SUPERVISOR_*` environment model. */
  readonly fallback?: SupervisorModelPort
  /** Pushes an event to every connected person; the builder runs once per recipient. */
  readonly broadcast?: (build: (context: AuthenticatedContext) => Readonly<Record<string, unknown>>) => void
  /** Announces a serving-machine change; `to` is undefined on return to the configured machine. */
  readonly onFailover?: (change: { readonly from: string; readonly to: string | undefined }) => void
  readonly clock?: () => number
  readonly onError?: (error: unknown, context: string) => void
}

export interface SupervisorConfigDescription {
  readonly status: SupervisorConfigStatus
  readonly config?: { readonly machineId: string; readonly machineLabel: string; readonly profile: string; readonly model: string; readonly effort: string | null }
  readonly fallback?: { readonly machineId: string; readonly label: string }
  readonly message?: string
}

function isOwner(context: AuthenticatedContext): boolean {
  return context.principal.kind === 'user' && context.principal.hubOwner
}

function exactKeys(params: Readonly<Record<string, unknown>>, allowed: readonly string[]): void {
  for (const key of Object.keys(params)) {
    if (!allowed.includes(key)) throw new HubProtocolError('invalid_request', `unknown parameter ${JSON.stringify(key.slice(0, 40))}`)
  }
}

function field(value: unknown, name: string, max: number): string {
  if (typeof value !== 'string' || value.trim() === '' || value.length > max || /[\x00-\x1f\x7f]/.test(value)) {
    throw new HubProtocolError('invalid_request', `${name} must be a non-empty single-line string of at most ${max} characters`)
  }
  return value
}

function shortText(value: unknown, max = 200): string | undefined {
  return typeof value === 'string' && value.trim() !== '' && value.length <= max && !/[\x00-\x1f\x7f]/.test(value) ? value : undefined
}

function clip(text: string, max: number): string {
  return text.length <= max ? text : `${text.slice(0, Math.max(0, max - 1))}…`
}

type Refusal = Extract<PoolCallResult, { readonly state: 'rejected' | 'unknown' }>

/** A readable message for a refused daemon call, without leaking anything the daemon should not have said. */
function refusalMessage(outcome: Refusal): string {
  const message = outcome.error.message
  if (/hub_auth rejected \((?:unknown_hub|hub_not_enrolled[a-z_]*)\)/.test(message)) {
    return 'This machine has not enrolled this Hub. Run `xerxes hub link` on it again.'
  }
  if (/hub_auth/.test(message)) return 'This machine refused the Hub\'s signed model request. Update Xerxes on it and link it again.'
  return clip(message, 300)
}

function refusal(outcome: Refusal): HubProtocolError {
  switch (outcome.error.code) {
    case 'unsupported': return new HubProtocolError('unsupported', refusalMessage(outcome))
    case 'unknown_profile': return new HubProtocolError('not_found', refusalMessage(outcome))
    case 'invalid_request': return new HubProtocolError('invalid_request', refusalMessage(outcome))
    default: return new HubProtocolError('unavailable', refusalMessage(outcome))
  }
}

interface Options {
  readonly profiles: readonly { readonly name: string; readonly label: string; readonly provider: string; readonly default_model: string }[]
  readonly models: readonly { readonly id: string; readonly label?: string }[]
  readonly model?: string
  readonly efforts: readonly { readonly effort: string; readonly description?: string }[]
  readonly default_effort: string | null
}

export class SupervisorModelControl {
  /** The model port the supervisor runs on; it follows the stored choice live. */
  readonly port: SupervisorModelPort
  readonly handlers: Readonly<Partial<Record<HubMethod, HubMethodHandler>>>
  private readonly store: SupervisorModelStore
  private readonly clock: () => number
  private current: StoredSupervisorModel | undefined
  private failover: { readonly machineId: string; readonly label: string } | undefined

  constructor(private readonly options: SupervisorModelControlOptions) {
    this.clock = options.clock ?? Date.now
    this.store = new SupervisorModelStore(options.databasePath)
    this.current = this.store.get()
    const self = this
    this.port = {
      get id() { return self.modelId() },
      reportsCost: false,
      complete: request => self.complete(request),
    }
    this.handlers = {
      'supervisor.config.get': (ctx, params) => this.getHandler(ctx.context, params),
      'supervisor.config.options': (ctx, params) => this.optionsHandler(ctx.context, params),
      'supervisor.config.set': (ctx, params) => this.setHandler(ctx.context, params),
    }
  }

  close(): void { this.store.close() }

  private modelId(): string {
    return this.current?.model ?? this.options.fallback?.id ?? 'unconfigured'
  }

  /** The stored choice, or `undefined` while XERXES is not configured on a machine. */
  stored(): StoredSupervisorModel | undefined { return this.current }

  describe(): SupervisorConfigDescription {
    const stored = this.current
    if (stored === undefined) {
      return this.options.fallback === undefined
        ? { status: 'unconfigured', message: UNCONFIGURED_MESSAGE }
        : { status: 'configured', message: 'XERXES runs on the model named in this Hub\'s environment (XERXES_HUB_SUPERVISOR_*). Choose one in XERXES settings to replace it.' }
    }
    const machines = this.options.machines.machines()
    const machine = machines.find(row => row.id === stored.machineId)
    const config = { machineId: stored.machineId, machineLabel: machine?.label ?? stored.machineId, profile: stored.profile, model: stored.model, effort: stored.effort }
    if (machine === undefined || machine.health !== 'online') {
      const fallback = this.failover
      if (fallback !== undefined && machines.some(row => row.id === fallback.machineId && row.health === 'online')) {
        return { status: 'configured', config, fallback, message: `${config.machineLabel} is offline; XERXES is answering on ${fallback.label} with the same model.` }
      }
      if (machines.some(row => row.id !== stored.machineId && row.health === 'online' && row.modelsSupported)) {
        return { status: 'machine_offline', config, message: `${config.machineLabel} is offline; XERXES will try the same model on another online machine.` }
      }
      if (machine === undefined) return { status: 'invalid', config, message: 'The machine XERXES was set to run on is no longer linked to this Hub. Choose another.' }
      return { status: 'machine_offline', config, message: `${machine.label} is offline; XERXES cannot answer until it is back.` }
    }
    if (!machine.modelsSupported) return { status: 'invalid', config, message: `${machine.label} runs a Xerxes version that cannot run XERXES's model calls. Update Xerxes there or choose another machine.` }
    return { status: 'configured', config }
  }

  /** The `supervisor.config.get` result for one reader. */
  view(canEdit: boolean): Record<string, unknown> {
    const description = this.describe()
    return {
      status: description.status,
      ...(description.config === undefined ? {} : {
        config: {
          machine_id: description.config.machineId, machine_label: description.config.machineLabel,
          profile: description.config.profile, model: description.config.model, effort: description.config.effort,
        },
      }),
      can_edit: canEdit,
      ...(description.fallback === undefined ? {} : { fallback: { machine_id: description.fallback.machineId, machine_label: description.fallback.label } }),
      ...(description.message === undefined ? {} : { message: description.message }),
    }
  }

  // ── signed calls to the chosen machine ──────────────────────────────────

  private async call(machineId: string, method: string, params: Readonly<Record<string, unknown>>, requestedBy: string | undefined, signal?: AbortSignal): Promise<PoolCallResult> {
    const now = this.clock()
    const claims: HubAuthClaims = {
      hub_id: this.options.signer.hubId,
      principal_id: SUPERVISOR_PRINCIPAL_ID,
      principal_kind: 'agent',
      display_name: SUPERVISOR_DISPLAY_NAME,
      ...(requestedBy === undefined ? {} : { requested_by: requestedBy }),
      // Any workspace of the machine identifies its lane; the machine entry itself is the stable one.
      workspace_id: machineId,
      method,
      params_hash: paramsHash(params),
      request_id: crypto.randomUUID(),
      nonce: crypto.randomUUID(),
      issued_at: now,
      expires_at: now + HUB_AUTH_LIFETIME_MS,
      scopes: ['supervisor.model'],
    }
    const envelope = this.options.signer.sign(claims)
    return this.options.machines.callModel(machineId, method, { ...params, hub_auth: envelope }, signal === undefined ? {} : { signal })
  }

  private async complete(request: SupervisorModelRequest): Promise<SupervisorModelResponse> {
    request.signal?.throwIfAborted()
    const stored = this.current
    if (stored === undefined) {
      if (this.options.fallback !== undefined) return this.options.fallback.complete(request)
      throw new Error(UNCONFIGURED_MESSAGE)
    }
    const params = {
      profile: stored.profile, model: stored.model, ...(stored.effort === null ? {} : { effort: stored.effort }),
      system: request.system, messages: request.messages, tools: request.tools, max_tokens: request.maxTokens,
    }
    if (Buffer.byteLength(JSON.stringify(params)) > MAX_REQUEST_BYTES) throw new Error('the request to the model is too large')
    const machines = this.options.machines.machines()
    const machine = machines.find(row => row.id === stored.machineId)
    const configuredLabel = machine?.label ?? stored.machineId
    if (machine?.health === 'online' && machine.modelsSupported) {
      const outcome = await this.call(stored.machineId, 'hub.model.complete', params, undefined, request.signal)
      request.signal?.throwIfAborted()
      if (outcome.state === 'accepted') {
        const response = parseCompletion(outcome.result)
        if (this.current === stored) this.setFailover(configuredLabel, undefined)
        return response
      }
      if (outcome.error.code !== 'unavailable') throw new Error(refusalMessage(outcome))
    }
    const preferred = this.failover?.machineId
    const candidates = machines.filter(row => row.id !== stored.machineId && row.health === 'online' && row.modelsSupported)
      .sort((a, b) => Number(b.id === preferred) - Number(a.id === preferred) || a.label.localeCompare(b.label))
    for (const candidate of candidates) {
      request.signal?.throwIfAborted()
      const outcome = await this.call(candidate.id, 'hub.model.complete', params, undefined, request.signal)
      request.signal?.throwIfAborted()
      if (outcome.state === 'accepted') {
        const response = parseCompletion(outcome.result)
        if (this.current === stored) this.setFailover(configuredLabel, { machineId: candidate.id, label: candidate.label })
        return response
      }
      if (!FAILOVER_SKIP_CODES.has(outcome.error.code)) throw new Error(refusalMessage(outcome))
    }
    throw new Error(`${configuredLabel} is offline and no other online machine offers ${stored.profile}/${stored.model}.`)
  }

  private setFailover(from: string, next: { readonly machineId: string; readonly label: string } | undefined): void {
    if (this.failover?.machineId === next?.machineId && this.failover?.label === next?.label) return
    this.failover = next
    this.options.broadcast?.(recipient => this.view(isOwner(recipient)))
    try {
      this.options.onFailover?.({ from, to: next?.label })
    } catch (error) {
      this.options.onError?.(error, 'supervisor model failover')
    }
  }

  private async fetchOptions(machineId: string, profile: string | undefined, model: string | undefined, requestedBy: string): Promise<Options> {
    const catalog = await this.call(machineId, 'hub.models.catalog', {}, requestedBy)
    if (catalog.state !== 'accepted') throw refusal(catalog)
    const profiles = (Array.isArray(catalog.result.profiles) ? catalog.result.profiles : []).slice(0, MAX_LIST).flatMap(row => {
      if (!isRecord(row)) return []
      const name = shortText(row.name), provider = shortText(row.provider)
      if (name === undefined || provider === undefined) return []
      return [{ name, label: shortText(row.label) ?? name, provider, default_model: typeof row.default_model === 'string' ? clip(row.default_model, 200) : '' }]
    })
    if (profile === undefined) return { profiles, models: [], efforts: [], default_effort: null }
    const detail = await this.call(machineId, 'hub.models.options', { profile, ...(model === undefined ? {} : { model }) }, requestedBy)
    if (detail.state !== 'accepted') throw refusal(detail)
    const models = (Array.isArray(detail.result.models) ? detail.result.models : []).slice(0, MAX_LIST).flatMap(row => {
      if (!isRecord(row)) return []
      const id = shortText(row.id)
      if (id === undefined) return []
      const label = shortText(row.label)
      return [{ id, ...(label === undefined ? {} : { label }) }]
    })
    const efforts = (Array.isArray(detail.result.efforts) ? detail.result.efforts : []).slice(0, 50).flatMap(row => {
      if (!isRecord(row)) return []
      const effort = shortText(row.effort, 64)
      if (effort === undefined) return []
      const description = shortText(row.description, 300)
      return [{ effort, ...(description === undefined ? {} : { description }) }]
    })
    const chosen = shortText(detail.result.model)
    return {
      profiles, models, ...(chosen === undefined ? {} : { model: chosen }), efforts,
      default_effort: shortText(detail.result.default_effort, 64) ?? null,
    }
  }

  // ── Hub methods ─────────────────────────────────────────────────────────

  private requireUser(context: AuthenticatedContext): void {
    if (context.principal.kind !== 'user') throw new HubProtocolError('forbidden', 'only a signed-in person can use the supervisor settings')
  }

  private getHandler(context: AuthenticatedContext, params: Readonly<Record<string, unknown>>): Record<string, unknown> {
    this.requireUser(context)
    exactKeys(params, [])
    return this.view(isOwner(context))
  }

  private async optionsHandler(context: AuthenticatedContext, params: Readonly<Record<string, unknown>>): Promise<Record<string, unknown>> {
    this.requireUser(context)
    exactKeys(params, ['machine_id', 'profile', 'model'])
    if (!isOwner(context)) throw new HubProtocolError('forbidden', 'only the Hub owner can choose XERXES\'s model')
    const machineId = readId(params.machine_id, 'machine_id')
    const profile = params.profile === undefined ? undefined : field(params.profile, 'profile', 200)
    const model = params.model === undefined ? undefined : field(params.model, 'model', 200)
    if (model !== undefined && profile === undefined) throw new HubProtocolError('invalid_request', 'model needs a profile')
    this.machine(machineId)
    const found = await this.fetchOptions(machineId, profile, model, context.principal.id)
    return {
      machine_id: machineId,
      profiles: found.profiles,
      ...(profile === undefined ? {} : { profile }),
      models: found.models,
      ...(found.model === undefined ? {} : { model: found.model }),
      efforts: found.efforts,
      default_effort: found.default_effort,
    }
  }

  private async setHandler(context: AuthenticatedContext, params: Readonly<Record<string, unknown>>): Promise<Record<string, unknown>> {
    this.requireUser(context)
    exactKeys(params, ['machine_id', 'profile', 'model', 'effort'])
    const hash = createHash('sha256').update(canonicalJson({ machine_id: params.machine_id ?? null, profile: params.profile ?? null, model: params.model ?? null, effort: params.effort ?? null })).digest('hex')
    const record = (decision: 'allowed' | 'denied', outcome: HubAuditInput['outcome']): HubAuditInput =>
      ({ actor: context.actor, action: 'supervisor.config.set', requestHash: hash, decision, outcome })
    if (!isOwner(context)) {
      await this.options.audit(record('denied', 'rejected')).catch(error => this.options.onError?.(error, 'audit denial'))
      throw new HubProtocolError('forbidden', 'only the Hub owner can choose XERXES\'s model')
    }
    const choice: SupervisorModelChoice = {
      machineId: readId(params.machine_id, 'machine_id'),
      profile: field(params.profile, 'profile', 200),
      model: field(params.model, 'model', 200),
      effort: params.effort === undefined || params.effort === null ? null : field(params.effort, 'effort', 64),
    }
    this.machine(choice.machineId)
    const offered = await this.fetchOptions(choice.machineId, choice.profile, choice.model, context.principal.id)
    if (!offered.models.some(row => row.id === choice.model)) {
      throw new HubProtocolError('invalid_request', `${choice.profile} does not offer the model ${JSON.stringify(clip(choice.model, 80))}`)
    }
    if (choice.effort !== null && !offered.efforts.some(row => row.effort === choice.effort)) {
      throw new HubProtocolError('invalid_request', `${JSON.stringify(choice.effort)} is not a thinking effort ${choice.model} offers`)
    }
    try {
      await this.options.audit(record('allowed', 'n/a'))
    } catch (error) {
      this.options.onError?.(error, 'audit supervisor model')
      throw new HubProtocolError('unavailable', 'the audit log is unavailable; the model was not changed')
    }
    this.current = this.store.set(choice, context.principal.id, this.clock())
    await this.options.audit(record('allowed', 'accepted')).catch(error => this.options.onError?.(error, 'audit supervisor model outcome'))
    this.options.broadcast?.(recipient => this.view(isOwner(recipient)))
    return this.view(true)
  }

  private machine(id: string): PoolMachine {
    const machine = this.options.machines.machines().find(row => row.id === id)
    if (machine === undefined) throw new HubProtocolError('not_found', 'unknown machine')
    return machine
  }
}

function parseCompletion(result: Readonly<Record<string, unknown>>): SupervisorModelResponse {
  const text = typeof result.text === 'string' ? result.text : ''
  const toolCalls: ToolCall[] = []
  if (Array.isArray(result.tool_calls)) {
    for (const row of result.tool_calls) {
      if (!isRecord(row) || typeof row.id !== 'string' || !row.id || typeof row.name !== 'string' || !row.name) {
        throw new Error('the machine returned a malformed tool call')
      }
      let args
      try { args = parseToolArguments(typeof row.arguments === 'string' ? row.arguments : undefined) } catch { throw new Error('the machine returned tool arguments that are not a JSON object') }
      toolCalls.push({ id: row.id, type: 'function', function: { name: row.name, arguments: args } })
    }
  }
  const usage = isRecord(result.usage) ? result.usage : undefined
  const count = (value: unknown): number => (typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : 0)
  return {
    text, toolCalls,
    ...(usage === undefined ? {} : { usage: { inputTokens: count(usage.input_tokens), outputTokens: count(usage.output_tokens) } }),
  }
}
