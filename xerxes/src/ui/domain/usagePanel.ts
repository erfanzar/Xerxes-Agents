// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
//
// `/usage` as transcript panel sections: this session's statistics, then
// every imported provider profile with its plan windows (5-hour, weekly,
// per-model), its API balance, or the reason there is nothing to show.
import {
  compactTokens,
  modelUsageSummary,
  resetIn,
  usd,
  sessionUsageRows,
  usageSourceLabel,
  windowLabel,
  windowNote,
  type UsageReportView
} from '../../auth/usageView.js'
import type { PanelSection } from '../types.js'

export function usagePanelSections(report: UsageReportView, now = Date.now()): PanelSection[] {
  const sections: PanelSection[] = []
  const session = report.session

  if (session) {
    const cost = session.costUsd !== undefined ? `$${session.costUsd.toFixed(session.costUsd < 0.01 ? 4 : 2)}` : undefined
    const rows = sessionUsageRows(session).filter(([label]) => label !== 'Cost')
    sections.push({
      heading: { label: 'This session', notes: session.model ? [session.model] : [], ...(cost ? { right: cost } : {}), state: 'active' },
      rows,
      ...(session.contextMax > 0
        ? {
            meters: [{
              label: 'Context',
              percent: (session.contextUsed / session.contextMax) * 100,
              note: `${compactTokens(session.contextUsed)} / ${compactTokens(session.contextMax)}`
            }]
          }
        : {})
    })
  }

  if (!report.profiles.length) {
    sections.push({ title: 'Plans & keys', text: 'No provider profiles imported yet. Add one with /provider.' })
    return withModelSpend(sections, report)
  }

  sections.push({ title: 'Plans & keys', count: String(report.profiles.length) })
  for (const profile of report.profiles) {
    const notes = [usageSourceLabel(profile.source, profile.provider), profile.plan].filter((note): note is string => Boolean(note))
    const right = profile.balance ?? (profile.status === 'unsupported' ? 'no limits published' : profile.status === 'error' ? 'unavailable' : undefined)
    sections.push({
      heading: {
        label: profile.label,
        notes: [notes.join(' · ')].filter(Boolean),
        ...(right ? { right } : {}),
        state: profile.status === 'error' ? 'failed' : profile.active ? 'active' : 'idle'
      },
      meters: profile.windows.map(window => {
        const reset = resetIn(window, profile.fetchedAt, now)
        const note = windowNote(window)
        return {
          label: windowLabel(window),
          percent: window.usedPercent,
          ...(note ? { note } : {}),
          ...(reset ? { right: `resets in ${reset}` } : {})
        }
      }),
      ...(profile.facts?.length ? { rows: profile.facts.map(fact => [fact.label, fact.value] as [string, string]) } : {}),
      ...(profile.status === 'error' && profile.message ? { text: profile.message } : {})
    })
  }
  return withModelSpend(sections, report)
}

/** Spend by model, from the runtime's own record; absent on runtimes that do not keep one. */
function withModelSpend(sections: PanelSection[], report: UsageReportView): PanelSection[] {
  const models = report.models
  if (!models) return sections
  if (!models.length) {
    sections.push({ title: 'By model · last 30 days', text: 'No model calls recorded yet.' })
    return sections
  }
  const priced = models.filter(model => model.costUsd !== undefined)
  const total = priced.reduce((sum, model) => sum + (model.costUsd ?? 0), 0)
  const ordered = [...models].sort((a, b) => (b.costUsd ?? -1) - (a.costUsd ?? -1) || (b.input + b.output) - (a.input + a.output))
  sections.push({
    heading: { label: 'By model', notes: ['last 30 days'], ...(priced.length ? { right: usd(total) } : {}), state: 'idle' },
    rows: ordered.map(model => [model.model, `${model.costUsd === undefined ? 'price unknown' : usd(model.costUsd)} · ${modelUsageSummary(model)}`] as [string, string])
  })
  return sections
}
