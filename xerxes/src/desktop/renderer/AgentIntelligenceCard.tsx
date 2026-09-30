// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Settings → Agent intelligence. The light, balanced and smart tiers an agent
 * or workflow asks for by name, each mapped to any provider profile, any model
 * that provider offers (or one typed in), and any reasoning effort that model
 * supports. Backed by the daemon's agent.settings.* RPCs; nothing here guesses
 * a provider's models or efforts — both come from the provider.
 */

import { useEffect, useState, type ReactElement } from 'react'

import { desktopCall, desktopError } from './desktopRpc.js'
import type { Snapshot } from './store.js'

export const INTELLIGENCE_TIERS = [
  { id: 'light', label: 'Light', hint: 'Quick, cheap work: searches, lookups, mechanical edits.' },
  { id: 'balanced', label: 'Balanced', hint: 'Most agent work: reading, reviewing, implementing.' },
  { id: 'smart', label: 'Smart', hint: 'The hardest calls: design, subtle bugs, final judgement.' },
] as const
export type TierId = typeof INTELLIGENCE_TIERS[number]['id']

export interface TierDraft {
  readonly provider_profile: string
  readonly model: string
  readonly reasoning_effort: string
}
export interface IntelligenceDraft {
  readonly default: 'inherit' | TierId
  readonly tiers: Readonly<Record<TierId, TierDraft>>
}
interface ProfileOption { readonly name: string; readonly label: string; readonly provider: string; readonly model: string }

const EMPTY_TIER: TierDraft = { provider_profile: '', model: '', reasoning_effort: '' }

/** The daemon's stored settings as an editable draft; a bare string tier is a model on the active provider. */
export function draftOf(settings: unknown): IntelligenceDraft {
  const record = settings && typeof settings === 'object' && !Array.isArray(settings) ? settings as Record<string, unknown> : {}
  const tier = (value: unknown): TierDraft => {
    if (typeof value === 'string') return { ...EMPTY_TIER, model: value }
    if (!value || typeof value !== 'object') return EMPTY_TIER
    const fields = value as Record<string, unknown>
    const text = (field: unknown) => typeof field === 'string' ? field : ''
    return { provider_profile: text(fields.provider_profile), model: text(fields.model), reasoning_effort: text(fields.reasoning_effort) }
  }
  const level = record.default
  return {
    default: level === 'light' || level === 'balanced' || level === 'smart' ? level : 'inherit',
    tiers: { light: tier(record.light), balanced: tier(record.balanced), smart: tier(record.smart) },
  }
}

/** The draft in the daemon's wire shape: an unset tier is omitted, empty fields are dropped. */
export function settingsOf(draft: IntelligenceDraft): Record<string, unknown> {
  const settings: Record<string, unknown> = { default: draft.default }
  for (const { id } of INTELLIGENCE_TIERS) {
    const tier = draft.tiers[id]
    if (!tier.model.trim()) continue
    settings[id] = {
      model: tier.model.trim(),
      ...(tier.provider_profile ? { provider_profile: tier.provider_profile } : {}),
      ...(tier.reasoning_effort ? { reasoning_effort: tier.reasoning_effort } : {}),
    }
  }
  return settings
}

/** Why the draft cannot be saved yet, or undefined. */
export function draftProblem(draft: IntelligenceDraft): string | undefined {
  if (draft.default !== 'inherit' && !draft.tiers[draft.default].model.trim()) {
    return `Set a model for ${draft.default}, or choose another default.`
  }
  return undefined
}

export function AgentIntelligenceCard({ snap }: { snap: Snapshot }): ReactElement {
  const key = snap.sessionKey
  const [draft, setDraft] = useState<IntelligenceDraft | null>(null)
  const [saved, setSaved] = useState<IntelligenceDraft | null>(null)
  const [revision, setRevision] = useState(0)
  const [profiles, setProfiles] = useState<readonly ProfileOption[]>([])
  const [models, setModels] = useState<Readonly<Record<string, readonly string[]>>>({})
  const [efforts, setEfforts] = useState<Readonly<Record<string, readonly string[]>>>({})
  const [status, setStatus] = useState<{ kind: 'idle' | 'saving' | 'saved' } | { kind: 'error'; message: string }>({ kind: 'idle' })
  const [loadError, setLoadError] = useState('')
  const online = snap.connection === 'online'

  useEffect(() => {
    if (!online) return
    let live = true
    void desktopCall(window.xerxes, key, 'agent.settings.get').then(result => {
      if (!live) return
      const next = draftOf(result.settings)
      setDraft(next)
      setSaved(next)
      setRevision(typeof result.revision === 'number' ? result.revision : 0)
      setProfiles(Array.isArray(result.profiles) ? result.profiles.flatMap(raw => {
        const row = raw && typeof raw === 'object' ? raw as Record<string, unknown> : {}
        return typeof row.name === 'string' ? [{ name: row.name, label: String(row.label ?? row.name), provider: String(row.provider ?? ''), model: String(row.model ?? '') }] : []
      }) : [])
      setLoadError('')
    }).catch(error => { if (live) setLoadError(desktopError(error)) })
    return () => { live = false }
  }, [key, online])

  // Each provider's model list, fetched once per profile the tiers use.
  const usedProfiles = draft ? [...new Set(INTELLIGENCE_TIERS.map(({ id }) => draft.tiers[id].provider_profile))] : []
  useEffect(() => {
    for (const profile of usedProfiles) {
      if (models[profile] !== undefined || !online) continue
      setModels(current => ({ ...current, [profile]: [] }))
      void desktopCall(window.xerxes, key, 'fetch_models', profile ? { profile_name: profile } : {})
        .then(result => setModels(current => ({ ...current, [profile]: Array.isArray(result.models) ? result.models.filter((model): model is string => typeof model === 'string') : [] })))
        .catch(() => { /* Typing a model name still works without the list. */ })
    }
  }, [usedProfiles.join('\n'), online])

  // The efforts a tier's model supports, from its provider.
  const effortKey = (tier: TierDraft) => `${tier.provider_profile}\n${tier.model.trim()}`
  useEffect(() => {
    if (!draft || !online) return
    for (const { id } of INTELLIGENCE_TIERS) {
      const tier = draft.tiers[id]
      const cacheKey = effortKey(tier)
      if (!tier.model.trim() || efforts[cacheKey] !== undefined) continue
      setEfforts(current => ({ ...current, [cacheKey]: [] }))
      void desktopCall(window.xerxes, key, 'agent.settings.options', { model: tier.model.trim(), ...(tier.provider_profile ? { provider_profile: tier.provider_profile } : {}) })
        .then(result => setEfforts(current => ({ ...current, [cacheKey]: Array.isArray(result.reasoning_efforts) ? result.reasoning_efforts.filter((effort): effort is string => typeof effort === 'string') : [] })))
        .catch(() => { /* No effort choices; the model's default applies. */ })
    }
  }, [draft && INTELLIGENCE_TIERS.map(({ id }) => effortKey(draft.tiers[id])).join('|'), online])

  if (!online) return <><h2 className="modal__title">Agent intelligence</h2><p className="modal__sub">Connect to the runtime to edit agent intelligence.</p></>
  if (loadError) return <><h2 className="modal__title">Agent intelligence</h2><p role="alert" className="studio-error">{loadError}</p></>
  if (!draft) return <><h2 className="modal__title">Agent intelligence</h2><p className="modal__sub" role="status">Loading…</p></>

  const setTier = (id: TierId, patch: Partial<TierDraft>) => {
    setStatus({ kind: 'idle' })
    setDraft(current => current && { ...current, tiers: { ...current.tiers, [id]: { ...current.tiers[id], ...patch } } })
  }
  const problem = draftProblem(draft)
  const dirty = JSON.stringify(settingsOf(draft)) !== JSON.stringify(saved && settingsOf(saved))
  const save = async () => {
    if (problem) return
    setStatus({ kind: 'saving' })
    try {
      const result = await desktopCall(window.xerxes, key, 'agent.settings.save', { settings: settingsOf(draft), revision })
      setRevision(typeof result.revision === 'number' ? result.revision : revision + 1)
      setSaved(draft)
      setStatus({ kind: 'saved' })
    } catch (error) { setStatus({ kind: 'error', message: desktopError(error) }) }
  }
  const activeProfile = profiles.find(profile => profile.name === snap.providers.find(provider => provider.active)?.name)

  return (
    <>
      <h2 className="modal__title">Agent intelligence</h2>
      <p className="modal__sub">
        Agents and workflow steps ask for a tier by name. Point each tier at any provider, any model and any reasoning effort. An agent given an explicit model uses that instead.
      </p>
      <div className="tierlist">
        {INTELLIGENCE_TIERS.map(({ id, label, hint }) => {
          const tier = draft.tiers[id]
          const suggestions = models[tier.provider_profile] ?? []
          const choices = efforts[effortKey(tier)] ?? []
          return (
            <section key={id} className="tiercard" aria-label={`${label} tier`}>
              <header className="tiercard__head">
                <span className="tiercard__name">{label}</span>
                <span className="tiercard__hint">{hint}</span>
                {tier.model && <button className="chipbtn" onClick={() => setTier(id, EMPTY_TIER)}>Clear</button>}
              </header>
              <div className="tiercard__fields">
                <label className="field">
                  <span>Provider</span>
                  <select value={tier.provider_profile} onChange={event => setTier(id, { provider_profile: event.target.value, reasoning_effort: '' })}>
                    <option value="">Active provider{activeProfile ? ` (${activeProfile.label})` : ''}</option>
                    {profiles.map(profile => <option key={profile.name} value={profile.name}>{profile.label}{profile.provider ? ` · ${profile.provider}` : ''}</option>)}
                  </select>
                </label>
                <label className="field">
                  <span>Model</span>
                  <input list={`tier-models-${id}`} value={tier.model} spellCheck={false} placeholder="Not set"
                    onChange={event => setTier(id, { model: event.target.value, reasoning_effort: '' })} />
                  <datalist id={`tier-models-${id}`}>{suggestions.map(model => <option key={model} value={model} />)}</datalist>
                </label>
                <label className="field">
                  <span>Reasoning effort</span>
                  <select value={tier.reasoning_effort} disabled={!tier.model.trim()} onChange={event => setTier(id, { reasoning_effort: event.target.value })}>
                    <option value="">Model default</option>
                    {choices.map(effort => <option key={effort} value={effort}>{effort}</option>)}
                    {tier.reasoning_effort && !choices.includes(tier.reasoning_effort) && <option value={tier.reasoning_effort}>{tier.reasoning_effort}</option>}
                  </select>
                </label>
              </div>
            </section>
          )
        })}
      </div>
      <label className="field tierdefault">
        <span>New agents use</span>
        <select value={draft.default} onChange={event => { setStatus({ kind: 'idle' }); setDraft({ ...draft, default: event.target.value as IntelligenceDraft['default'] }) }}>
          <option value="inherit">The task's own model (no tier)</option>
          {INTELLIGENCE_TIERS.map(({ id, label }) => <option key={id} value={id} disabled={!draft.tiers[id].model.trim()}>{label}{draft.tiers[id].model.trim() ? '' : ' — set a model first'}</option>)}
        </select>
      </label>
      {problem && <p className="studio-error" role="alert">{problem}</p>}
      {status.kind === 'error' && <p className="studio-error" role="alert">{status.message}</p>}
      <div className="approval__row">
        <button className="btn btn--solid" disabled={!dirty || !!problem || status.kind === 'saving'} onClick={() => void save()}>{status.kind === 'saving' ? 'Saving…' : 'Save'}</button>
        {dirty && <button className="btn" onClick={() => { setDraft(saved); setStatus({ kind: 'idle' }) }}>Revert</button>}
        {status.kind === 'saved' && !dirty && <span className="row__s" role="status">Saved. New agents use these tiers.</span>}
      </div>
    </>
  )
}
