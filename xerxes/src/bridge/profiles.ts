// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { randomUUID } from 'node:crypto'
import { chmodSync, mkdirSync, readFileSync, renameSync, rmSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'

import { xerxesHome } from '../daemon/paths.js'
import { piCatalogModelCapabilities } from '../llms/piModelCatalog.js'

export const CLAUDE_CODE_PROFILE_NAME = 'cc'
export const CLAUDE_CODE_DEFAULT_MODEL = 'claude-code/default'

export const CODEX_PROFILE_NAME = 'codex'
// The plan's newest model. Xerxes uses the ChatGPT subscription as an
// entitlement and runs its own agent loop, so every model the catalog returns
// is selectable; this is only the starting point.
export const CODEX_DEFAULT_MODEL = 'codex/gpt-5.6-sol'
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
      const declaredCapability = value.context_limit !== undefined || value.max_output_tokens !== undefined
      if (declaredCapability && contextLimit === undefined && maxOutputTokens === undefined) continue
      modelCapabilities[id] = {
        ...(contextLimit === undefined ? {} : { context_limit: contextLimit }),
        ...(maxOutputTokens === undefined ? {} : { max_output_tokens: maxOutputTokens }),
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

  private activeName(document: ProfilesDocument, profiles: Record<string, ProviderProfile>): string {
    return document.active && Object.hasOwn(profiles, document.active) ? document.active : CLAUDE_CODE_PROFILE_NAME
  }

  private builtinProfiles(): Record<string, ProviderProfile> {
    const profiles: Record<string, ProviderProfile> = Object.create(null)
    profiles[CLAUDE_CODE_PROFILE_NAME] = {
      name: CLAUDE_CODE_PROFILE_NAME,
      base_url: 'claude-code://local',
      api_key: '',
      model: CLAUDE_CODE_DEFAULT_MODEL,
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
      model: CODEX_DEFAULT_MODEL,
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
      const declaredCapability = raw.context_limit !== undefined || raw.max_output_tokens !== undefined
      if ((declaredCapability || Object.keys(raw).length > 0)
        && contextLimit === undefined
        && maxOutputTokens === undefined) continue
      modelCapabilities[id] = {
        ...(contextLimit === undefined ? {} : { context_limit: contextLimit }),
        ...(maxOutputTokens === undefined ? {} : { max_output_tokens: maxOutputTokens }),
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
  const catalog = piCatalogModelCapabilities(model, profile?.provider ?? '')
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
