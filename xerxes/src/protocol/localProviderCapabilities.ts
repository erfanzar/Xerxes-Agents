// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** Public display/selection metadata only. Never provider configuration or authority. */
export interface LocalProviderCapabilities {
  readonly version: 1
  readonly model: string
  readonly reasoning: {
    readonly shape: 'effort' | 'toggle' | 'inherent'
    readonly efforts: readonly string[]
    readonly canDisable: boolean
    readonly provenance: 'provider_reported' | 'bundled_catalog' | 'provider_fallback'
  }
}
const record = (value: unknown): value is Record<string, unknown> => !!value && typeof value === 'object' && !Array.isArray(value)
const invalid = () => new Error('Invalid local provider capability snapshot. Review local setup before retrying.')

/** Reconstruct a bounded allowlist at every transport/persistence boundary. */
export function parseLocalProviderCapabilities(value: unknown, model: string): LocalProviderCapabilities | undefined {
  if (value === undefined) return undefined
  if (!record(value) || value.version !== 1 || value.model !== model || !model.trim() || model.length > 512 || /[\x00-\x1f\x7f]/.test(model) ||
    Object.keys(value).some(key => !['version','model','reasoning'].includes(key)) || !record(value.reasoning)) throw invalid()
  const r = value.reasoning
  if (!['effort','toggle','inherent'].includes(String(r.shape)) || typeof r.canDisable !== 'boolean' ||
    !['provider_reported','bundled_catalog','provider_fallback'].includes(String(r.provenance)) ||
    Object.keys(r).some(key => !['shape','efforts','canDisable','provenance'].includes(key)) ||
    !Array.isArray(r.efforts) || r.efforts.length > 16 ||
    r.efforts.some(effort => typeof effort !== 'string' || !/^[a-z][a-z0-9_-]{0,31}$/.test(effort) || effort === 'off') ||
    new Set(r.efforts).size !== r.efforts.length ||
    (r.shape === 'inherent' && (r.efforts.length > 0 || r.canDisable)) ||
    (r.shape === 'toggle' && (r.efforts.length !== 1 || r.efforts[0] !== 'on')) ||
    (r.shape === 'effort' && (!r.efforts.length || r.efforts.includes('on')))) throw invalid()
  return Object.freeze({version:1,model,reasoning:Object.freeze({
    shape:r.shape as LocalProviderCapabilities['reasoning']['shape'],
    efforts:Object.freeze([...r.efforts] as string[]),canDisable:r.canDisable,
    provenance:r.provenance as LocalProviderCapabilities['reasoning']['provenance'],
  })})
}
