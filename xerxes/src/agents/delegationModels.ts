// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * The models a delegated agent can run on, as a prompt section. Nothing here
 * is a curated list: each provider's entries are the models it reported (or
 * the one saved on the profile), with the prices it or models.dev published.
 * The model decides which is cheap enough for a mechanical job from that.
 */

import type { ProviderProfile } from '../bridge/profiles.js'

/** Catalogs larger than this are summarized; the model searches them with list_available_models. */
const INLINE_CATALOG_LIMIT = 12

export function renderDelegationModels(
  profiles: readonly (ProviderProfile & { readonly active?: boolean })[],
  currentProfile?: string,
): string {
  const lines: string[] = []
  for (const profile of profiles) {
    const reported = Object.keys(profile.model_capabilities ?? {})
    const models = reported.length ? reported : profile.model ? [profile.model] : []
    if (!models.length) continue
    const marker = profile.name === currentProfile ? ' (this conversation)' : ''
    if (models.length > INLINE_CATALOG_LIMIT) {
      const saved = profile.model ? `; saved model ${describe(profile, profile.model)}` : ''
      lines.push(`- ${profile.name}${marker}: ${models.length} models${saved}. Search them with list_available_models.`)
      continue
    }
    lines.push(`- ${profile.name}${marker}: ${models.map(model => describe(profile, model)).join(', ')}`)
  }
  if (!lines.length) return ''
  return [
    '# Models for delegated agents',
    'Pick a model per agent (with profile when it lives on another provider than this conversation). Use a fast, cheap one for mechanical reading, search, extraction and classification, and a strong one for judgement, subtle verification and synthesis. Prices are USD per million input/output tokens where published.',
    ...lines,
  ].join('\n')
}

function describe(profile: ProviderProfile, model: string): string {
  const cost = profile.model_capabilities?.[model]?.cost
  const input = typeof cost?.input === 'number' ? cost.input : undefined
  const output = typeof cost?.output === 'number' ? cost.output : undefined
  if (input === undefined && output === undefined) return model
  return `${model} ($${price(input)}/$${price(output)})`
}

function price(value: number | undefined): string {
  if (value === undefined) return '?'
  return value >= 1 ? String(Math.round(value * 100) / 100) : String(Math.round(value * 1000) / 1000)
}
