// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import type { ProfileStore, ProviderProfile } from '../bridge/profiles.js'
import { claudeCodeCatalog } from '../llms/claudeCodeCatalog.js'
import { bareModel, resolveProvider } from '../llms/providerRegistry.js'

/** Subscription endpoints serve only their own plan's models; they never proxy another vendor's. */
const SUBSCRIPTION_PROVIDERS = new Set(['claude-code', 'kimi-code', 'openai-codex'])

/**
 * Whether a profile can serve a model — decided by what the provider
 * reported, never by name patterns. `claim` is Xerxes inferring a route;
 * `explicit` is a profile the user chose. A subscription endpoint serves the models
 * it listed (Claude Code's own list; the /models or Codex catalog cached on
 * the profile). Claude Code models also carry Xerxes' `claude-code/` id
 * namespace. Other providers take any model id.
 */
export function profileAcceptsModel(profile: ProviderProfile, model: string, mode: 'explicit' | 'claim' = 'explicit'): boolean {
  if (!SUBSCRIPTION_PROVIDERS.has(profile.provider)) return true
  const bare = bareModel(model)
  if (profile.provider === 'claude-code') {
    return /^claude[-_]code\//i.test(model) || Boolean(claudeCodeCatalog.find(bare))
  }
  // Once the provider has listed its models that list is authoritative — a
  // saved model it does not list (a polluted config) is refused too. Before
  // it has listed anything: an explicit choice has nothing to be refused on,
  // but when Xerxes is only inferring which profile serves a model, an
  // unlisted subscription profile claims just its own saved model.
  const listed = Object.keys(profile.model_capabilities ?? {})
  if (listed.length) return listed.includes(model) || listed.includes(bare)
  return mode === 'explicit' || model === profile.model || bare === bareModel(profile.model)
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
  const compatible = profiles.list().filter(p => profileAcceptsModel(p, model, 'claim'))
  const native = compatible.filter(p => p.provider === inferred || (inferred === 'openai' && p.provider === 'openai-codex'))
  const known = native.length ? native : compatible.filter(p => p.model === model || Object.hasOwn(p.model_capabilities ?? {}, model))
  const active = profiles.active()
  const profile = known.find(p => p.name === active?.name) ?? (known.length === 1 ? known[0] : undefined)
    ?? (!known.length && active && profileAcceptsModel(active, model, 'claim') && (!inferred || active.provider === 'openai') ? active : undefined)
  if (!profile && profiles.list().length) throw new Error(`No unambiguous provider profile for ${model}. Use /model to select its provider and model together.`)
  if (profile) session.metadata.provider_profile = profile.name
  return profile
}

/**
 * The one profile a model id plainly belongs to, if any: its `<prefix>/`
 * names a profile or a profile's provider (`claude-code/opus`, `cc/opus`),
 * or it is among the models a profile saved or discovered. Undefined when
 * no profile or more than one does.
 */
export function profileOwningModel(profiles: readonly ProviderProfile[], model: string): ProviderProfile | undefined {
  const slash = model.indexOf('/')
  const prefix = slash > 0 ? model.slice(0, slash).toLowerCase() : ''
  const owners = profiles.filter(profile =>
    (prefix && (profile.name.toLowerCase() === prefix || profile.provider.toLowerCase() === prefix))
    || profile.model === model
    || Object.hasOwn(profile.model_capabilities ?? {}, model))
  return owners.length === 1 ? owners[0] : undefined
}

/**
 * Which profile a delegated agent runs on. It inherits the parent's, unless
 * the parent's profile does not list the model and another profile plainly
 * owns it: asking for `claude-code/opus` from a z.ai conversation means
 * Claude Code, and an OpenAI-compatible profile, which accepts any id, must
 * not claim it on the parent's behalf.
 */
export function agentProvider(profiles: Pick<ProfileStore, 'get' | 'list' | 'active'>, session: { metadata: Record<string, unknown> }, model: string): ProviderProfile | undefined {
  const pinned = typeof session.metadata.provider_profile === 'string' ? profiles.get(session.metadata.provider_profile) : undefined
  // Only a parent profile that has listed its models can be said not to
  // serve one; before discovery (or for a gateway whose ids carry a vendor
  // prefix, `anthropic/…` on OpenRouter) it keeps the child as before.
  const listed = pinned ? Object.keys(pinned.model_capabilities ?? {}) : []
  if (pinned && (pinned.model === model || listed.includes(model) || !listed.length)) return sessionProvider(profiles, session, model)
  const owner = profileOwningModel(profiles.list(), model)
  if (owner && profileAcceptsModel(owner, model)) return owner
  return sessionProvider(profiles, session, model)
}
