// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
export interface ShareableLocalProfile {
  readonly name: string
  readonly model: string
  readonly credentialSource: string
  readonly supported: boolean
  readonly providerControlledOutput: boolean
  readonly setup: string
}
const record = (v: unknown): Record<string, unknown> => v && typeof v === "object" && !Array.isArray(v) ? v as Record<string, unknown> : {}
const label = (v: unknown, maximum = 512): string => typeof v === "string" && v.length <= maximum && !/[\x00-\x1f\x7f]/.test(v) ? v : ""
const failure = () => new Error("Remote task setup failed. Review the connection and selected provider before retrying.")

export function shareableLocalProfiles(value: unknown): readonly ShareableLocalProfile[] {
  const result = record(value)
  if (result.ok !== true || !Array.isArray(result.profiles) || result.profiles.length > 1000) throw failure()
  return Object.freeze(result.profiles.map(raw => {
    const p = record(raw)
    const name = label(p.name), model = label(p.model)
    if (!name || !model) throw failure()
    return Object.freeze({name, model, credentialSource: label(p.credential_source) || 'local provider configuration',
      supported: p.supported === true, providerControlledOutput: p.output_limit_mode === 'provider-controlled', setup: label(p.setup)})
  }))
}

