// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Account failover inside one subscription family: a Codex sign-in hands over
 * to another Codex sign-in, a Claude sign-in to another Claude sign-in. Pure
 * decision logic; the daemon supplies the profiles and their measured usage.
 * Never crosses families, never moves to an account whose usage is unknown or
 * also near its limit.
 */

export type FailoverFamily = 'claude' | 'codex'

export interface FailoverSettings {
  readonly enabled: boolean
  /** Peak window use (percent) at or above which an account is "near its limit". */
  readonly threshold: number
}

export const DEFAULT_FAILOVER_SETTINGS: FailoverSettings = { enabled: true, threshold: 95 }

export interface FailoverCandidate {
  readonly name: string
  readonly provider: string
  /** Highest used percent over the account's windows; undefined when unknown. */
  readonly peakUsed: number | undefined
}

export interface FailoverChoice {
  readonly to: string
  readonly peakUsed: number
}

export function failoverFamily(provider: string): FailoverFamily | undefined {
  const name = provider.trim().toLowerCase()
  // Sign-in transports only: an API-key `anthropic` profile cannot serve a `claude-code/...` model.
  if (name === 'claude-code') return 'claude'
  if (name === 'openai-codex') return 'codex'
  return undefined
}

/** The sibling with the most headroom, or undefined when none is clearly usable. */
export function chooseFailover(
  current: { readonly name: string; readonly provider: string },
  candidates: readonly FailoverCandidate[],
  threshold: number,
): FailoverChoice | undefined {
  const family = failoverFamily(current.provider)
  if (!family) return undefined
  let best: FailoverChoice | undefined
  for (const candidate of candidates) {
    if (candidate.name === current.name || failoverFamily(candidate.provider) !== family) continue
    if (candidate.peakUsed === undefined || !Number.isFinite(candidate.peakUsed) || candidate.peakUsed >= threshold) continue
    if (!best || candidate.peakUsed < best.peakUsed) best = { to: candidate.name, peakUsed: candidate.peakUsed }
  }
  return best
}

/** Parse `runtime.account_failover` (on|off) and `runtime.account_failover_threshold` (1–100). */
export function parseFailoverSettings(runtime: Readonly<Record<string, unknown>>): FailoverSettings {
  const mode = runtime.account_failover
  const enabled = mode === undefined ? DEFAULT_FAILOVER_SETTINGS.enabled : mode === true || mode === 'on'
  const raw = runtime.account_failover_threshold
  const threshold = typeof raw === 'number' && Number.isFinite(raw) && raw >= 1 && raw <= 100 ? raw : DEFAULT_FAILOVER_SETTINGS.threshold
  return { enabled, threshold }
}
