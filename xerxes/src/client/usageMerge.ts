// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { parseUsageReport, usageSourceLabel, type ModelUsageView, type UsageProfileView, type UsageReportView } from '../auth/usageView.js'
import { isRecord } from '../hub/protocol.js'
import type { UsageExtras } from './hubStore.js'
import { arr, num, rec, str } from './parse.js'

/**
 * Combine `hub.usage` answers. A plan or key shared by several workspaces is
 * one account, so its windows are shown once (newest fetch wins), not once per
 * workspace. Per-model totals are summed only across distinct runtimes: reports
 * that name the same `runtime_id` are the same runtime counted once.
 */

function accountKey(profile: UsageProfileView): string {
  return [usageSourceLabel(profile.source, profile.provider), profile.plan ?? '', profile.profile].join('\u001f')
}

export function mergeUsageReports(reports: readonly UsageReportView[], fetchedAt: number = Date.now()): UsageReportView {
  const profiles = new Map<string, UsageProfileView>()
  for (const report of reports) {
    for (const profile of report.profiles) {
      const key = accountKey(profile)
      const known = profiles.get(key)
      if (!known || (profile.fetchedAt ?? 0) > (known.fetchedAt ?? 0)) profiles.set(key, profile)
    }
  }
  const models = new Map<string, ModelUsageView>()
  let modelsSince: number | undefined
  for (const report of reports) {
    for (const model of report.models ?? []) {
      const key = `${model.profile ?? ''}\u001f${model.model}`
      const known = models.get(key)
      models.set(key, known ? sumModels(known, model) : model)
    }
    if (report.modelsSince !== undefined) modelsSince = Math.min(modelsSince ?? report.modelsSince, report.modelsSince)
  }
  return {
    fetchedAt: reports.reduce((newest, report) => Math.max(newest, report.fetchedAt), 0) || fetchedAt,
    profiles: [...profiles.values()],
    ...(models.size > 0 ? { models: [...models.values()] } : {}),
    ...(modelsSince === undefined ? {} : { modelsSince }),
  }
}

function sumModels(a: ModelUsageView, b: ModelUsageView): ModelUsageView {
  const costUsd = a.costUsd === undefined && b.costUsd === undefined ? undefined : (a.costUsd ?? 0) + (b.costUsd ?? 0)
  const costBasis = a.costBasis === 'subscription' && b.costBasis === 'subscription' ? ('subscription' as const) : undefined
  return {
    model: a.model,
    ...(a.profile === undefined ? {} : { profile: a.profile }),
    calls: a.calls + b.calls,
    input: a.input + b.input,
    output: a.output + b.output,
    cacheRead: a.cacheRead + b.cacheRead,
    cacheWrite: a.cacheWrite + b.cacheWrite,
    ...(costUsd === undefined ? {} : { costUsd }),
    ...(costBasis ? { costBasis } : {}),
  }
}

/**
 * Hub-level totals from a `hub.usage` result. The Hub sums distinct runtimes
 * once and keeps plan-paid cost apart from billable cost.
 */
export function usageExtrasFromResult(result: Readonly<Record<string, unknown>>): UsageExtras | undefined {
  if (!isRecord(result.totals)) return undefined
  const totals = result.totals
  const billable = num(totals.billable_cost_usd)
  const label = str(totals.subscription_label)
  return {
    inputTokens: num(totals.input) ?? 0,
    outputTokens: num(totals.output) ?? 0,
    ...(billable === undefined ? {} : { billableCostUsd: billable }),
    ...(label ? { subscriptionLabel: label } : {}),
    unavailable: arr(result.unavailable_workspaces).filter((id): id is string => typeof id === 'string'),
  }
}

/** Parse a `hub.usage` result: `by_model` rows, or a daemon-style `report` / `reports[]`. */
export function usageFromResult(result: Readonly<Record<string, unknown>>): UsageReportView {
  if (Array.isArray(result.by_model)) return parseUsageReport({ models: result.by_model, fetched_at: Date.now() })
  if (Array.isArray(result.reports)) {
    const seen = new Set<string>()
    const reports: UsageReportView[] = []
    for (const raw of arr(result.reports)) {
      const item = rec(raw)
      const runtime = str(item.runtime_id)
      if (runtime !== undefined) {
        if (seen.has(runtime)) continue
        seen.add(runtime)
      }
      reports.push(parseUsageReport(item.report ?? item))
    }
    return mergeUsageReports(reports)
  }
  return parseUsageReport(result.report ?? result)
}
