// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { unknownProfileQuota, type ProfileQuota } from '../auth/profileUsage.js'
export interface InventoryProfile { name: string; provider: string; model: string; active: boolean }
/** Suggest exact host-owned names without remapping a requested credential source. */
export function unavailableAgentProfile(name: string, model: string, profiles: readonly Pick<InventoryProfile, 'name' | 'provider' | 'model'>[]): Error {
  const matches = profiles.filter(profile => profile.provider !== 'claude-code' && profile.model === model).slice(0, 8)
  const hint = matches.length ? ` Configured profiles for this model on the execution host: ${matches.map(profile => JSON.stringify(profile.name)).join(', ')}.` : ''
  return new Error(`Agent provider profile unavailable: ${JSON.stringify(name)}.${hint} Use list_available_models with provider_profile omitted to discover this session's choices, then retry with the exact provider_profile and model. No fallback provider was selected.`)
}
export interface InventoryModel { id: string; context_limit?: number; max_output_tokens?: number; context_source?: string; output_source?: string }
export interface InventoryReasoning {
  efforts: readonly string[]
  source: 'provider_reported' | 'bundled_catalog' | 'provider_fallback' | 'unknown'
  shape: 'effort' | 'toggle' | 'inherent'
  defaultEffort?: string
}
export interface ModelInventoryPort {
  quota?(profile: string, signal?: AbortSignal): Promise<ProfileQuota>
  routingNotes?(): readonly { provider_profile: string; model: string; note: string; revision: number }[]
  profiles(): InventoryProfile[]
  discover(profile: string): Promise<{ models: InventoryModel[]; source: string; warning?: string }>
  reasoning(profile: string, model: string): Promise<InventoryReasoning>
}
/** Bounded, credential-free projection. Configuration is not proof of provider access. */
export async function modelInventory(port: ModelInventoryPort, params: Record<string, unknown>, signal?: AbortSignal) {
  signal?.throwIfAborted()
  const offset = params.offset ?? 0, limit = params.limit ?? 20
  if (typeof offset !== 'number' || !Number.isSafeInteger(offset) || offset < 0 || typeof limit !== 'number' || !Number.isSafeInteger(limit) || limit < 1 || limit > 50) throw new Error('Inventory offset must be nonnegative; limit must be 1–50')
  for (const key of ['provider_profile', 'query', 'revision']) if (params[key] !== undefined && (typeof params[key] !== 'string' || (params[key] as string).length > 512)) throw new Error(`Invalid inventory ${key}`)
  if (params.include_usage !== undefined && typeof params.include_usage !== 'boolean') throw new Error('Invalid include_usage')
  const profileName = (params.provider_profile as string | undefined)?.trim() || undefined
  if (params.include_usage === true && profileName === undefined) throw new Error('Usage lookup requires provider_profile')
  const profiles = [...port.profiles()].sort((a, b) => a.name.localeCompare(b.name))
  const query = (params.query as string | undefined)?.toLowerCase() ?? ''
  const notes = port.routingNotes?.() ?? []
  const guidance = (profile: string, model?: string) => {
    const selected = notes.filter(note => note.provider_profile === profile && (note.model === '' || note.model === model) && note.note)
    return selected.length ? { routing_notes: selected.map(note => ({ scope: note.model ? 'model' : 'provider', note: note.note, source: 'user_preference', revision: note.revision })) } : {}
  }
  let entries: Array<Record<string, unknown>>
  let source = 'configured_profiles', warning: string | undefined
  if (profileName !== undefined) {
    const profile = profiles.find(value => value.name === profileName)
    if (!profile) throw new Error('Unknown provider profile')
    const catalog = await port.discover(profile.name)
    signal?.throwIfAborted()
    source = catalog.source; warning = catalog.warning
    entries = [...new Map(catalog.models.map(model => [model.id, model])).values()].filter(model => model.id.toLowerCase().includes(query)).sort((a,b) => a.id.localeCompare(b.id)).map(model => ({
      provider_profile: profile.name, provider: profile.provider, model: model.id,
      ...guidance(profile.name, model.id),
      spawn_supported: profile.provider !== 'claude-code',
      context_window: model.context_limit ?? null, max_output_tokens: model.max_output_tokens ?? null,
      context_source: model.context_source ?? 'unknown', output_source: model.output_source ?? 'unknown',
    }))
  } else entries = profiles.filter(profile => `${profile.name} ${profile.provider} ${profile.model}`.toLowerCase().includes(query)).map(profile => ({
    provider_profile: profile.name, provider: profile.provider, configured_model: profile.model, active: profile.active,
    ...guidance(profile.name),
    spawn_supported: profile.provider !== 'claude-code',
  }))
  // Resolve capability metadata before hashing the catalog. Otherwise a model's
  // effort ladder can change while a caller continues with an old page token.
  if (profileName !== undefined) for (const entry of entries) {
    signal?.throwIfAborted()
    const reasoning = entry.spawn_supported ? await port.reasoning(profileName, entry.model as string) : undefined
    entry.reasoning_efforts = reasoning?.efforts ?? []
    entry.reasoning_source = reasoning?.source ?? 'unavailable'
    entry.reasoning_shape = reasoning?.shape ?? null
    entry.default_reasoning_effort = reasoning?.defaultEffort ?? null
  }
  signal?.throwIfAborted()
  const revision = Bun.hash(JSON.stringify({ profileName, query, source, warning, entries })).toString(16)
  // Page zero starts a fresh snapshot, including when a caller carries the
  // previous query/profile's token or fills an optional revision with "".
  // Only continuation pages must match the catalog being paginated.
  if (offset > 0 && params.revision !== revision) throw new Error('Inventory changed or revision missing. Retry with offset: 0 and omit revision; then use the returned revision and next_offset for subsequent pages.')
  const page = entries.slice(offset, offset + limit)
  const quota = params.include_usage === true && profileName !== undefined && port.quota
    ? await port.quota(profileName, signal) : unknownProfileQuota('Subscription usage was not requested or no adapter is available.')
  signal?.throwIfAborted()
  const observedAt = new Date().toISOString()
  const response = () => ({ ok: true, mode: profileName === undefined ? 'providers' : 'models', configured_profiles: profiles.length,
    configured_providers: new Set(profiles.map(profile => profile.provider)).size,
    observed_at: observedAt, source, ...(warning ? { warning } : {}), revision, total: entries.length,
    next_offset: offset + page.length < entries.length ? offset + page.length : null, entries: page,
    quota,
    guidance: 'Use provider_profile, model and reasoning_effort when delegating. Configuration and catalog discovery do not guarantee model access. Context capacity is not subscription allowance.' })
  const encoder = new TextEncoder()
  while (encoder.encode(JSON.stringify(response())).byteLength > 60000) {
    if (page.length <= 1) throw new Error('Inventory entry or metadata exceeds the response size limit')
    page.pop()
  }
  return response()
}
