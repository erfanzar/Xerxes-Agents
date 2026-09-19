// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import type { ProfileStore, ProviderProfile } from '../bridge/profiles.js'
import { bareModel, resolveProvider } from '../llms/providerRegistry.js'

/** Native subscription endpoints do not proxy other vendors' model names. */
export function profileAcceptsModel(profile: ProviderProfile, model: string): boolean {
  model = bareModel(model)
  if (profile.provider === 'claude-code') return false
  if (profile.provider === 'kimi-code') return /^(kimi|k[0-9])(?:[-./]|$)/i.test(model)
  if (profile.provider === 'openai-codex') return /^(gpt-|o[1-9]|codex)/i.test(model)
  return true
}
export function sessionProvider(profiles: Pick<ProfileStore, 'get' | 'list' | 'active'>, session: { metadata: Record<string, unknown> }, model: string): ProviderProfile | undefined {
  const pinned = session.metadata.provider_profile
  if (typeof pinned === 'string' && pinned) {
    const profile = profiles.get(pinned)
    if (!profile || !profileAcceptsModel(profile, model)) throw new Error(`Provider profile ${pinned} cannot serve ${model}. Use /model to select its provider and model together.`)
    return profile
  }
  if (!profiles.list().some(profile => profile.provider !== 'claude-code')) return undefined
  let inferred = ''
  try { inferred = resolveProvider(model) } catch { /* Custom model names require a configured profile below. */ }
  const compatible = profiles.list().filter(p => profileAcceptsModel(p, model))
  const native = compatible.filter(p => p.provider === inferred || (inferred === 'openai' && p.provider === 'openai-codex'))
  const known = native.length ? native : compatible.filter(p => p.model === model || Object.hasOwn(p.model_capabilities ?? {}, model))
  const active = profiles.active()
  const profile = known.find(p => p.name === active?.name) ?? (known.length === 1 ? known[0] : undefined)
    ?? (!known.length && active && profileAcceptsModel(active, model) && (!inferred || active.provider === 'openai') ? active : undefined)
  if (!profile && profiles.list().length) throw new Error(`No unambiguous provider profile for ${model}. Use /model to select its provider and model together.`)
  if (profile) session.metadata.provider_profile = profile.name
  return profile
}
