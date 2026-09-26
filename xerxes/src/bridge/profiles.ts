// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { randomUUID } from 'node:crypto'
import { chmodSync, mkdirSync, readFileSync, renameSync, rmSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'

import { xerxesHome } from '../daemon/paths.js'
import { modelsDev, type LiveCost, type LiveReasoning } from '../llms/modelsDev.js'

export const CLAUDE_CODE_PROFILE_NAME = 'cc'

/**
 * What people see for a profile. The id stays as stored (`cc` is in saved
 * sessions and `/provider cc`); only the built-in Claude Code profile has a
 * friendlier name.
 */
export function profileLabel(name: string): string {
  return name === CLAUDE_CODE_PROFILE_NAME ? 'Claude Code' : name
}

export const CODEX_PROFILE_NAME = 'codex'
export const CODEX_PROFILE_BASE_URL = 'https://chatgpt.com/backend-api/codex'

export const SAMPLING_PARAMS = new Set([
  'temperature', 'top_p', 'top_k', 'max_tokens', 'frequency_penalty', 'presence_penalty', 'repetition_penalty', 'min_p',
  'thinking', 'reasoning_effort', 'thinking_budget', 'service_tier',
])

export interface ProviderModelCapabilities {
  /** Provider-reported context window. */
  readonly context_limit?: number
  /** Provider-reported maximum output tokens. */
  readonly max_output_tokens?: number
  /** The provider's own name for the model. */
  readonly display_name?: string
  /** How the provider says the model reasons (see `LiveReasoning`). */
  readonly reasoning?: {
    readonly supported: boolean
    readonly can_disable: boolean
    readonly efforts: readonly string[]
    readonly default_effort?: string
  }
  /** Provider-published prices, USD per million tokens. */
  readonly cost?: { readonly input?: number; readonly output?: number; readonly cache_read?: number; readonly cache_write?: number }
}

export interface ProviderModelOverride {
  readonly context_limit?: number
  readonly max_output_tokens?: number
}

export interface ModelCapabilityUpdates {
  /** null clears the user override and reveals provider/catalog metadata. */
  readonly contextLimit?: number | null
  /** null clears the user override and reveals provider/catalog metadata. */
  readonly maxOutputTokens?: number | null
}

export type ModelCapabilitySource = 'catalog' | 'override' | 'provider' | 'unknown'

export interface ResolvedModelCapabilities {
  readonly contextLimit?: number
  readonly contextSource: ModelCapabilitySource
  readonly maxOutputTokens?: number
  readonly outputSource: ModelCapabilitySource
}

export interface ProviderProfile {
  readonly api_key: string
  readonly base_url: string
  readonly model: string
  /** Live provider metadata cached by exact model id; never filled from static tables. */
  readonly model_capabilities?: Record<string, ProviderModelCapabilities>
  /** User-edited model capacities, kept separate from discovered metadata. */
  readonly model_overrides?: Record<string, ProviderModelOverride>
  readonly name: string
  readonly provider: string
  readonly sampling: Record<string, unknown>
}

interface ProfilesDocument {
  active: string | null
  profiles: Record<string, ProviderProfile>
}

export interface SaveProfileInput {
  readonly apiKey: string
  readonly baseUrl: string
  readonly model: string
  readonly name: string
  readonly provider?: string
  readonly sampling?: Record<string, unknown>
  readonly setActive?: boolean
}

/** Compatibility store for `$XERXES_HOME/profiles.json`. */
export class ProfileStore {
  readonly filePath: string

  constructor(filePath = join(xerxesHome(), 'profiles.json')) {
    this.filePath = filePath
  }

  list(): Array<ProviderProfile & { readonly active: boolean }> {
    const document = this.load()
    const profiles = this.merged(document)
    const active = this.activeName(document, profiles)
    return Object.entries(profiles).map(([name, profile]) => ({ ...profile, active: name === active }))
  }

  active(): ProviderProfile | undefined {
    const document = this.load()
    const profiles = this.merged(document)
    return profiles[this.activeName(document, profiles)]
  }

  /** Resolve one exact profile without changing the process-wide active selection. */
  get(name: string): ProviderProfile | undefined {
    const clean = name.trim()
    if (!clean) {
      return undefined
    }
    const profiles = this.merged(this.load())
    return Object.hasOwn(profiles, clean) ? profiles[clean] : undefined
  }

  save(input: SaveProfileInput): ProviderProfile {
    const document = this.load()
    const existing = Object.hasOwn(document.profiles, input.name) ? document.profiles[input.name] : undefined
    const baseUrl = input.baseUrl.replace(/\/+$/, '')
    const provider = input.provider?.trim().toLowerCase().replace('claude_code', 'claude-code') || guessProvider(baseUrl)
    const sameConnection = existing !== undefined
      && existing.base_url === baseUrl
      && existing.provider === provider
    const profile: ProviderProfile = {
      name: input.name,
      base_url: baseUrl,
      api_key: input.apiKey,
      model: input.model,
      model_capabilities: sameConnection ? existing.model_capabilities ?? {} : {},
      model_overrides: sameConnection ? existing.model_overrides ?? {} : {},
      provider,
      sampling: input.sampling ?? existing?.sampling ?? {},
    }
    document.profiles[input.name] = profile
    if (input.setActive ?? true) {
      document.active = input.name
    }
    this.write(document)
    return profile
  }

  updateSampling(name: string, updates: Record<string, unknown>): ProviderProfile | undefined {
    const document = this.load()
    const profile = this.ensureWritable(document, name)
    if (!profile) {
      return undefined
    }
    const sampling = { ...profile.sampling }
    for (const [key, value] of Object.entries(updates)) {
      if (!SAMPLING_PARAMS.has(key)) {
        continue
      }
      if (value === null || value === undefined) {
        delete sampling[key]
      } else {
        sampling[key] = value
      }
    }
    const updated = { ...profile, sampling }
    document.profiles[name] = updated
    this.write(document)
    return updated
  }

  updateActiveModel(model: string): ProviderProfile | undefined {
    const document = this.load()
    const active = this.activeName(document, this.merged(document))
    const profile = this.ensureWritable(document, active)
    if (!profile) {
      return undefined
    }
    const updated = { ...profile, model }
    document.profiles[active] = updated
    this.write(document)
    return updated
  }

  /** Replace provider metadata while preserving separately stored user overrides. */
  replaceModelCapabilities(
    name: string,
    capabilities: Readonly<Record<string, ProviderModelCapabilities>>,
  ): ProviderProfile | undefined {
    const document = this.load()
    const profile = this.ensureWritable(document, name)
    if (!profile) return undefined
    const modelCapabilities: Record<string, ProviderModelCapabilities> = Object.create(null)
    for (const [model, value] of Object.entries(capabilities)) {
      const id = model.trim()
      if (!id || id.length > 512) continue
      const contextLimit = positiveInteger(value.context_limit)
      const maxOutputTokens = positiveInteger(value.max_output_tokens)
      const displayName = typeof value.display_name === 'string' && value.display_name.trim() ? value.display_name.trim() : undefined
      const reasoning = storedReasoning(value.reasoning)
      const cost = storedCost(value.cost)
      const declaredCapability = value.context_limit !== undefined || value.max_output_tokens !== undefined
      if (declaredCapability && contextLimit === undefined && maxOutputTokens === undefined && !displayName && !reasoning && !cost) continue
      modelCapabilities[id] = {
        ...(contextLimit === undefined ? {} : { context_limit: contextLimit }),
        ...(maxOutputTokens === undefined ? {} : { max_output_tokens: maxOutputTokens }),
        ...(displayName ? { display_name: displayName } : {}),
        ...(reasoning ? { reasoning } : {}),
        ...(cost ? { cost } : {}),
      }
    }
    const updated = { ...profile, model_capabilities: modelCapabilities }
    document.profiles[name] = updated
    this.write(document)
    return updated
  }

  /** Set or clear user capacity overrides for one cached model. */
  updateModelCapabilities(
    name: string,
    model: string,
    updates: ModelCapabilityUpdates,
  ): ProviderProfile | undefined {
    const document = this.load()
    const profile = this.ensureWritable(document, name)
    const id = model.trim()
    if (!profile || !id || id.length > 512) return undefined
    const modelOverrides = { ...profile.model_overrides }
    const existing: MutableModelOverride = { ...(modelOverrides[id] ?? {}) }
    applyCapabilityOverride(existing, 'context_limit', updates.contextLimit)
    applyCapabilityOverride(existing, 'max_output_tokens', updates.maxOutputTokens)
    if (Object.keys(existing).length === 0) delete modelOverrides[id]
    else modelOverrides[id] = existing
    const updated = { ...profile, model_overrides: modelOverrides }
    document.profiles[name] = updated
    this.write(document)
    return updated
  }

  delete(name: string): boolean {
    const document = this.load()
    if (!Object.hasOwn(document.profiles, name)) {
      return false
    }
    delete document.profiles[name]
    if (document.active === name) {
      document.active = null
    }
    this.write(document)
    return true
  }

  setActive(name: string): boolean {
    const document = this.load()
    if (!Object.hasOwn(this.merged(document), name)) {
      return false
    }
    document.active = name
    this.write(document)
    return true
  }

  /**
   * Whether someone chose the active profile. On a fresh install nothing is
   * saved and the built-in Claude Code profile is active only as a fallback —
   * which must not quietly start spending a Claude plan.
   */
  activeIsExplicit(): boolean {
    const document = this.load()
    return Boolean(document.active && Object.hasOwn(this.merged(document), document.active))
  }

  private activeName(document: ProfilesDocument, profiles: Record<string, ProviderProfile>): string {
    return document.active && Object.hasOwn(profiles, document.active) ? document.active : CLAUDE_CODE_PROFILE_NAME
  }

  private builtinProfiles(): Record<string, ProviderProfile> {
    const profiles: Record<string, ProviderProfile> = Object.create(null)
    profiles[CLAUDE_CODE_PROFILE_NAME] = {
      name: CLAUDE_CODE_PROFILE_NAME,
      base_url: 'claude-code://local',
      api_key: '',
      // No built-in model: the first one Claude Code reports is taken when
      // the profile is chosen (see selectProvider).
      model: '',
      model_capabilities: {},
      provider: 'claude-code',
      sampling: {},
    }
    // Subscription-backed like `cc`: the credential is an OAuth session rather
    // than a stored key, so the profile carries no api_key and is listed
    // whether or not the user has signed in yet. Selecting it without a
    // session fails with the sign-in command instead of hiding the option.
    profiles[CODEX_PROFILE_NAME] = {
      name: CODEX_PROFILE_NAME,
      base_url: CODEX_PROFILE_BASE_URL,
      api_key: '',
      // No built-in model: the first one the plan's Codex catalog lists is
      // taken when the profile is chosen.
      model: '',
      model_capabilities: {},
      provider: 'openai-codex',
      sampling: {},
    }
    return profiles
  }

  private ensureWritable(document: ProfilesDocument, name: string): ProviderProfile | undefined {
    const existing = Object.hasOwn(document.profiles, name) ? document.profiles[name] : undefined
    if (existing) {
      return existing
    }
    const builtins = this.builtinProfiles()
    const builtin = Object.hasOwn(builtins, name) ? builtins[name] : undefined
    if (!builtin) {
      return undefined
    }
    const copy = { ...builtin, sampling: { ...builtin.sampling } }
    document.profiles[name] = copy
    return copy
  }

  private load(): ProfilesDocument {
    try {
      const parsed: unknown = JSON.parse(readFileSync(this.filePath, 'utf8'))
      if (isRecord(parsed)) {
        const profiles: Record<string, ProviderProfile> = Object.create(null)
        if (isRecord(parsed.profiles)) {
          for (const [name, value] of Object.entries(parsed.profiles)) {
            const profile = providerProfile(value)
            if (profile) profiles[name] = profile
          }
        }
        return {
          active: typeof parsed.active === 'string' ? parsed.active : null,
          profiles,
        }
      }
    } catch {
      // Corrupt/missing stores intentionally start empty, matching Python behavior.
    }
    return { active: null, profiles: Object.create(null) }
  }

  private merged(document: ProfilesDocument): Record<string, ProviderProfile> {
    const profiles: Record<string, ProviderProfile> = Object.create(null)
    for (const [name, profile] of Object.entries(this.builtinProfiles())) {
      profiles[name] = profile
    }
    for (const [name, profile] of Object.entries(document.profiles)) {
      profiles[name] = profile
    }
    return profiles
  }

  private write(document: ProfilesDocument): void {
    mkdirSync(dirname(this.filePath), { recursive: true, mode: 0o700 })
    // Temp file plus same-directory rename so a profile store containing API
    // keys is never left half-written.
    const temporary = `${this.filePath}.${process.pid}.${randomUUID()}.tmp`
    try {
      writeFileSync(temporary, `${JSON.stringify(document, null, 2)}\n`, { encoding: 'utf8', mode: 0o600 })
      renameSync(temporary, this.filePath)
      // `mode` only applies when writeFileSync creates the file. Repair older
      // profile stores that may have inherited a permissive process umask.
      chmodSync(this.filePath, 0o600)
    } finally {
      rmSync(temporary, { force: true })
    }
  }
}

function storedReasoning(value: unknown): ProviderModelCapabilities['reasoning'] {
  if (!isRecord(value) || typeof value.supported !== 'boolean') return undefined
  const efforts = Array.isArray(value.efforts) ? value.efforts.filter((effort): effort is string => typeof effort === 'string' && effort.trim() !== '') : []
  const defaultEffort = typeof value.default_effort === 'string' && efforts.includes(value.default_effort) ? value.default_effort : undefined
  return { supported: value.supported, can_disable: value.can_disable === true, efforts, ...(defaultEffort ? { default_effort: defaultEffort } : {}) }
}

function storedCost(value: unknown): ProviderModelCapabilities['cost'] {
  if (!isRecord(value)) return undefined
  const price = (field: unknown) => typeof field === 'number' && Number.isFinite(field) && field >= 0 ? field : undefined
  const cost = {
    ...(price(value.input) === undefined ? {} : { input: price(value.input)! }),
    ...(price(value.output) === undefined ? {} : { output: price(value.output)! }),
    ...(price(value.cache_read) === undefined ? {} : { cache_read: price(value.cache_read)! }),
    ...(price(value.cache_write) === undefined ? {} : { cache_write: price(value.cache_write)! }),
  }
  return Object.keys(cost).length ? cost : undefined
}

/** A model's provider-reported reasoning, as stored on its profile. */
/** The prices the provider reported for a model, as the request builders read them. */
export function reportedModelCost(profile: ProviderProfile | undefined, model: string): LiveCost | undefined {
  const stored = modelRecord(profile?.model_capabilities, model)?.cost
  if (!stored) return undefined
  const cost: LiveCost = {
    ...(stored.input === undefined ? {} : { input: stored.input }),
    ...(stored.output === undefined ? {} : { output: stored.output }),
    ...(stored.cache_read === undefined ? {} : { cacheRead: stored.cache_read }),
    ...(stored.cache_write === undefined ? {} : { cacheWrite: stored.cache_write }),
  }
  return Object.keys(cost).length ? cost : undefined
}

export function reportedModelReasoning(profile: ProviderProfile | undefined, model: string): LiveReasoning | undefined {
  const stored = modelRecord(profile?.model_capabilities, model)?.reasoning
  return stored ? { supported: stored.supported, canDisable: stored.can_disable, efforts: stored.efforts, ...(stored.default_effort ? { defaultEffort: stored.default_effort } : {}) } : undefined
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function providerProfile(value: unknown): ProviderProfile | undefined {
  if (!isRecord(value)
    || typeof value.name !== 'string'
    || typeof value.base_url !== 'string'
    || typeof value.api_key !== 'string'
    || typeof value.model !== 'string'
    || typeof value.provider !== 'string'
    || !isRecord(value.sampling)) return undefined
  const modelCapabilities: Record<string, ProviderModelCapabilities> = Object.create(null)
  if (isRecord(value.model_capabilities)) {
    for (const [model, raw] of Object.entries(value.model_capabilities)) {
      if (!isRecord(raw)) continue
      const id = model.trim()
      if (!id || id.length > 512) continue
      const contextLimit = positiveInteger(raw.context_limit)
      const maxOutputTokens = positiveInteger(raw.max_output_tokens)
      const displayName = typeof raw.display_name === 'string' && raw.display_name.trim() ? raw.display_name.trim() : undefined
      const reasoning = storedReasoning(raw.reasoning)
      const cost = storedCost(raw.cost)
      // `{}` means "the provider listed this id" and is kept; an entry that
      // had fields but none of them valid is dropped.
      if (Object.keys(raw).length > 0 && contextLimit === undefined && maxOutputTokens === undefined && !displayName && !reasoning && !cost) continue
      modelCapabilities[id] = {
        ...(contextLimit === undefined ? {} : { context_limit: contextLimit }),
        ...(maxOutputTokens === undefined ? {} : { max_output_tokens: maxOutputTokens }),
        ...(displayName ? { display_name: displayName } : {}),
        ...(reasoning ? { reasoning } : {}),
        ...(cost ? { cost } : {}),
      }
    }
  }
  const modelOverrides: Record<string, ProviderModelOverride> = Object.create(null)
  if (isRecord(value.model_overrides)) {
    for (const [model, raw] of Object.entries(value.model_overrides)) {
      if (!isRecord(raw)) continue
      const id = model.trim()
      if (!id || id.length > 512) continue
      const contextLimit = positiveInteger(raw.context_limit)
      const maxOutputTokens = positiveInteger(raw.max_output_tokens)
      if (contextLimit === undefined && maxOutputTokens === undefined) continue
      modelOverrides[id] = {
        ...(contextLimit === undefined ? {} : { context_limit: contextLimit }),
        ...(maxOutputTokens === undefined ? {} : { max_output_tokens: maxOutputTokens }),
      }
    }
  }
  return {
    api_key: value.api_key,
    base_url: value.base_url,
    model: value.model,
    model_capabilities: modelCapabilities,
    model_overrides: modelOverrides,
    name: value.name,
    provider: value.provider,
    sampling: value.sampling,
  }
}

function positiveInteger(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isSafeInteger(value) && value > 0 ? value : undefined
}

type MutableModelOverride = {
  -readonly [Key in keyof ProviderModelOverride]: ProviderModelOverride[Key]
}

function applyCapabilityOverride(
  capabilities: MutableModelOverride,
  key: keyof ProviderModelOverride,
  value: number | null | undefined,
): void {
  if (value === undefined) return
  if (value === null) {
    delete capabilities[key]
    return
  }
  const normalized = positiveInteger(value)
  if (normalized === undefined) throw new Error(`${key} must be a positive safe integer or null`)
  capabilities[key] = normalized
}

function modelRecord<Value>(
  records: Readonly<Record<string, Value>> | undefined,
  model: string,
): Value | undefined {
  const configured = model.trim()
  if (!configured || !records) return undefined
  const exact = records[configured]
  if (exact !== undefined) return exact
  const slash = configured.indexOf('/')
  return slash < 0 ? undefined : records[configured.slice(slash + 1)]
}

/** Resolve a user override over provider-reported context metadata. */
export function profileContextLimit(profile: ProviderProfile | undefined, model: string): number | undefined {
  const override = modelRecord(profile?.model_overrides, model)
  const capabilities = modelRecord(profile?.model_capabilities, model)
  return override?.context_limit ?? capabilities?.context_limit
}

/** Resolve a user override over provider-reported output metadata. */
export function profileMaxOutputTokens(profile: ProviderProfile | undefined, model: string): number | undefined {
  const override = modelRecord(profile?.model_overrides, model)
  const capabilities = modelRecord(profile?.model_capabilities, model)
  return override?.max_output_tokens ?? capabilities?.max_output_tokens
}

/** Resolve editable profile metadata over Pi's catalog, preserving unknown. */
export function resolvedProfileModelCapabilities(
  profile: ProviderProfile | undefined,
  model: string,
): ResolvedModelCapabilities {
  const override = modelRecord(profile?.model_overrides, model)
  const cached = modelRecord(profile?.model_capabilities, model)
  // Nothing reported by the provider: what models.dev says (Kimi Code's approach).
  const catalog = modelsDev.find({ model, ...(profile?.provider ? { provider: profile.provider } : {}), ...(profile?.base_url ? { baseUrl: profile.base_url } : {}) })
  const contextLimit = override?.context_limit ?? cached?.context_limit ?? catalog?.contextLimit
  const maxOutputTokens = override?.max_output_tokens ?? cached?.max_output_tokens ?? catalog?.maxOutputTokens
  return {
    ...(contextLimit === undefined ? {} : { contextLimit }),
    contextSource: override?.context_limit !== undefined
      ? 'override'
      : cached?.context_limit !== undefined
        ? 'provider'
        : catalog?.contextLimit !== undefined
          ? 'catalog'
          : 'unknown',
    ...(maxOutputTokens === undefined ? {} : { maxOutputTokens }),
    outputSource: override?.max_output_tokens !== undefined
      ? 'override'
      : cached?.max_output_tokens !== undefined
        ? 'provider'
        : catalog?.maxOutputTokens !== undefined
          ? 'catalog'
          : 'unknown',
  }
}

export function resolvedProfileContextLimit(profile: ProviderProfile | undefined, model: string): number | undefined {
  return resolvedProfileModelCapabilities(profile, model).contextLimit
}

export function resolvedProfileMaxOutputTokens(profile: ProviderProfile | undefined, model: string): number | undefined {
  return resolvedProfileModelCapabilities(profile, model).maxOutputTokens
}

function guessProvider(baseUrl: string): string {
  const url = baseUrl.toLowerCase()
  if (url.startsWith('claude-code://')) {
    return 'claude-code'
  }
  if (url.includes('openrouter.ai')) {
    return 'openrouter'
  }
  if (url.includes('openai')) {
    return 'openai'
  }
  if (url.includes('anthropic')) {
    return 'anthropic'
  }
  if (url.includes('localhost') || url.includes('127.0.0.1')) {
    return url.includes('11434') ? 'ollama' : 'local'
  }
  if (url.includes('deepseek')) {
    return 'deepseek'
  }
  if (url.includes('together')) {
    return 'together'
  }
  if (url.includes('groq')) {
    return 'groq'
  }
  if (url.includes('kimi.com/coding')) {
    return 'kimi-code'
  }
  if (url.includes('kimi') || url.includes('moonshot')) {
    return 'kimi'
  }
  if (url.includes('minimax') || url.includes('minimaxi')) {
    return 'minimax'
  }
  if (url.includes('z.ai') || url.includes('zhipu') || url.includes('bigmodel')) {
    return 'zhipu'
  }
  return 'custom'
}
