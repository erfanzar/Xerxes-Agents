// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { chmodSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'

import { xerxesHome } from '../daemon/paths.js'

export const CLAUDE_CODE_PROFILE_NAME = 'cc'
export const CLAUDE_CODE_DEFAULT_MODEL = 'claude-code/default'

export const SAMPLING_PARAMS = new Set([
  'temperature', 'top_p', 'top_k', 'max_tokens', 'frequency_penalty', 'presence_penalty', 'repetition_penalty', 'min_p',
  'thinking', 'reasoning_effort', 'thinking_budget',
])

export interface ProviderProfile {
  readonly api_key: string
  readonly base_url: string
  readonly model: string
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
    const profile: ProviderProfile = {
      name: input.name,
      base_url: baseUrl,
      api_key: input.apiKey,
      model: input.model,
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
      provider: 'claude-code',
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
            if (isProfile(value)) {
              profiles[name] = value
            }
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
    writeFileSync(this.filePath, `${JSON.stringify(document, null, 2)}\n`, { encoding: 'utf8', mode: 0o600 })
    // `mode` only applies when writeFileSync creates the file. Repair older
    // profile stores that may have inherited a permissive process umask.
    chmodSync(this.filePath, 0o600)
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function isProfile(value: unknown): value is ProviderProfile {
  return isRecord(value)
    && typeof value.name === 'string'
    && typeof value.base_url === 'string'
    && typeof value.api_key === 'string'
    && typeof value.model === 'string'
    && typeof value.provider === 'string'
    && isRecord(value.sampling)
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
