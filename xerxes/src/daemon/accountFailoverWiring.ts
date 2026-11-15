// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Daemon side of same-family account failover. Reads the measured usage of
 * every sibling sign-in (same family, able to serve the session's model) and
 * re-pins the session to the one with the most headroom. Usage is never
 * guessed: an account whose windows could not be read is never chosen.
 */

import { buildUsageReport, UsageReportCache } from '../auth/usageReport.js'
import type { ProfileStore, ProviderProfile } from '../bridge/profiles.js'
import { chooseFailover, failoverFamily, type FailoverSettings } from '../runtime/accountFailover.js'
import { profileAcceptsModel } from './sessionProvider.js'

export type FailoverReason = 'preflight' | 'quota'

export interface FailoverOutcome {
  readonly from: string
  readonly to: string
  readonly reason: string
}

interface UsageRow {
  readonly profile: string
  readonly status: string
  readonly windows: readonly { readonly usedPercent: number }[]
}

export interface AccountFailoverDeps {
  readonly profiles: Pick<ProfileStore, 'get' | 'list' | 'active'>
  readonly settings: () => FailoverSettings
  /** Measured usage per profile; `refresh` bypasses the one-minute cache. Injected in tests. */
  readonly usage?: (profiles: readonly ProviderProfile[], refresh: boolean) => Promise<readonly UsageRow[]>
}

export function createAccountFailover(deps: AccountFailoverDeps) {
  const cache = new UsageReportCache()
  const usage = deps.usage ?? ((profiles, refresh) => buildUsageReport(profiles, { cache, refresh }))
  return async (session: { metadata: Record<string, unknown> }, model: string, reason: FailoverReason): Promise<FailoverOutcome | undefined> => {
    const settings = deps.settings()
    if (!settings.enabled) return undefined
    const pinned = session.metadata.provider_profile
    const current = typeof pinned === 'string' && pinned ? deps.profiles.get(pinned) : deps.profiles.active()
    const family = current ? failoverFamily(current.provider) : undefined
    if (!current || !family) return undefined
    const siblings = deps.profiles.list().filter(profile => failoverFamily(profile.provider) === family && profileAcceptsModel(profile, model))
    if (siblings.length < 2 || !siblings.some(profile => profile.name === current.name)) return undefined
    const report = await usage(siblings, reason === 'quota')
    const peak = (name: string): number | undefined => {
      const row = report.find(entry => entry.profile === name)
      return row && row.status === 'ok' && row.windows.length ? Math.max(...row.windows.map(window => window.usedPercent)) : undefined
    }
    const currentPeak = peak(current.name)
    if (reason === 'preflight' && (currentPeak === undefined || currentPeak < settings.threshold)) return undefined
    const choice = chooseFailover(current, siblings.map(profile => ({ name: profile.name, provider: profile.provider, peakUsed: peak(profile.name) })), settings.threshold)
    if (!choice) return undefined
    session.metadata.provider_profile = choice.to
    return {
      from: current.name,
      to: choice.to,
      reason: reason === 'quota'
        ? `${current.name} hit its usage limit; ${choice.to} is at ${Math.round(choice.peakUsed)}%`
        : `${current.name} is at ${Math.round(currentPeak ?? 0)}% of its limit; ${choice.to} is at ${Math.round(choice.peakUsed)}%`,
    }
  }
}
