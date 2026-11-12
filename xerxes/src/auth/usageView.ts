// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * The `usage.report` answer as both surfaces read it — the desktop Usage tab
 * and the TUI's /usage panel. Pure (no node imports) so the renderer bundle
 * can share it: parsing tolerates older runtimes, and every label and
 * countdown is produced here so the two surfaces never disagree.
 */

export interface UsageWindowView {
  readonly label: string
  readonly usedPercent: number
  readonly resetsAt?: number
  readonly resetAfterSeconds?: number
  readonly detail?: string
}

export interface UsageProfileView {
  readonly profile: string
  /** Display name; the id when an older runtime sends none. */
  readonly label: string
  readonly provider: string
  readonly model: string
  readonly active: boolean
  readonly source?: string
  readonly status: 'ok' | 'unsupported' | 'error'
  readonly plan?: string
  readonly windows: readonly UsageWindowView[]
  readonly balance?: string
  /** Label/value rows beyond windows: credit left, spend today, this month… */
  readonly facts?: readonly { readonly label: string; readonly value: string }[]
  readonly message?: string
  readonly fetchedAt?: number
}

/** Tokens the runtime recorded for one model over the report window. */
export interface ModelUsageView {
  readonly model: string
  readonly profile?: string
  readonly calls: number
  readonly input: number
  readonly output: number
  readonly cacheRead: number
  readonly cacheWrite: number
  /** USD at published prices; absent when no one publishes a price for the model. */
  readonly costUsd?: number
  /** `subscription` when a plan pays for these calls; costUsd is then only what API prices would charge. */
  readonly costBasis?: 'subscription'
  /** True on this machine, false in the cloud, null when the profile is unknown; absent on older runtimes. */
  readonly local?: boolean | null
}

export type ModelRole = 'worker' | 'advisor' | 'subagent'

/** One model's share of a single task. */
export interface TaskModelUsageView extends ModelUsageView {
  readonly role?: ModelRole
  /** Bytes the calls sent, when measured (advisor calls are). */
  readonly bytesSent?: number
}

/** The cloud advisor's per-task limits, present when an advisor is configured. */
export interface AdvisorLimitsView {
  readonly profile: string
  readonly maxCalls: number
  readonly maxUsd: number
}

/**
 * The one cost label every view shows: "Included in plan" when a plan pays,
 * the published-price estimate otherwise, and "price unknown" when nobody
 * publishes a price. Never an invented $0.
 */
export function costLabel(costUsd: number | undefined, costBasis: 'subscription' | undefined | null): string {
  if (costBasis === 'subscription') return 'Included in plan'
  return costUsd === undefined ? 'price unknown' : usd(costUsd)
}

export interface SessionUsageView {
  readonly model: string
  readonly turns: number
  readonly steps: number
  readonly modelMs: number
  readonly toolMs: number
  readonly ttftMs?: number
  readonly tokensPerSecond?: number
  readonly cacheHitRate?: number
  readonly input: number
  readonly output: number
  readonly calls?: number
  readonly contextUsed: number
  readonly contextMax: number
  readonly costUsd?: number
  readonly costBasis?: 'subscription'
  /** Which models this task used; absent on runtimes that do not record it. */
  readonly models?: readonly TaskModelUsageView[]
  readonly advisor?: AdvisorLimitsView
}

export interface UsageReportView {
  readonly fetchedAt: number
  readonly profiles: readonly UsageProfileView[]
  readonly session?: SessionUsageView
  /** Per-model totals the runtime recorded since `modelsSince`; absent on older runtimes. */
  readonly models?: readonly ModelUsageView[]
  readonly modelsSince?: number
}

const SOURCE_NAMES: Readonly<Record<string, string>> = {
  claude: 'Claude',
  codex: 'ChatGPT',
  kimi: 'Kimi Code',
  zai: 'Z.ai',
  openrouter: 'OpenRouter',
  deepseek: 'DeepSeek',
  moonshot: 'Moonshot',
}

/** "ChatGPT", "Z.ai" — the account a profile's numbers come from. */
export function usageSourceLabel(source: string | undefined, provider: string): string {
  return (source && SOURCE_NAMES[source]) || provider
}

const record = (value: unknown): Record<string, unknown> =>
  value !== null && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {}
const num = (value: unknown): number | undefined =>
  typeof value === 'number' && Number.isFinite(value) ? value : undefined
const str = (value: unknown): string | undefined => typeof value === 'string' && value.trim() ? value : undefined

function windowOf(value: unknown): UsageWindowView | null {
  const item = record(value)
  const label = str(item.label)
  const usedPercent = num(item.usedPercent)
  if (!label || usedPercent === undefined) return null
  const resetsAt = num(item.resetsAt)
  const resetAfterSeconds = num(item.resetAfterSeconds)
  const detail = str(item.detail)
  return {
    label,
    usedPercent: Math.max(0, Math.min(100, usedPercent)),
    ...(resetsAt === undefined ? {} : { resetsAt }),
    ...(resetAfterSeconds === undefined ? {} : { resetAfterSeconds }),
    ...(detail ? { detail } : {}),
  }
}

function profileOf(value: unknown): UsageProfileView | null {
  const item = record(value)
  const profile = str(item.profile)
  if (!profile) return null
  const status = item.status === 'ok' || item.status === 'error' ? item.status : 'unsupported'
  const optional = (key: 'source' | 'plan' | 'balance' | 'message') => { const text = str(item[key]); return text ? { [key]: text } : {} }
  const fetchedAt = num(item.fetchedAt)
  const facts = Array.isArray(item.facts)
    ? item.facts.map(record).flatMap(fact => { const label = str(fact.label); const value = str(fact.value); return label && value ? [{ label, value }] : [] })
    : []
  return {
    profile,
    label: str(item.label) ?? profile,
    provider: str(item.provider) ?? '',
    model: str(item.model) ?? '',
    active: item.active === true,
    status,
    windows: Array.isArray(item.windows) ? item.windows.map(windowOf).filter((entry): entry is UsageWindowView => entry !== null) : [],
    ...optional('source'), ...optional('plan'), ...optional('balance'), ...optional('message'),
    ...(facts.length ? { facts } : {}),
    ...(fetchedAt === undefined ? {} : { fetchedAt }),
  }
}

function sessionOf(value: unknown): SessionUsageView | undefined {
  const item = record(value)
  if (!Object.keys(item).length) return undefined
  const optional = (key: string, name: keyof SessionUsageView) => { const n = num(item[key]); return n === undefined || n <= 0 ? {} : { [name]: n } }
  return {
    model: str(item.model) ?? '',
    turns: num(item.turn_count) ?? 0,
    steps: (num(item.llm_steps) ?? 0) + (num(item.tool_steps) ?? 0),
    modelMs: num(item.llm_duration_ms) ?? 0,
    toolMs: num(item.tool_duration_ms) ?? 0,
    input: num(item.input) ?? 0,
    output: num(item.output) ?? 0,
    contextUsed: num(item.context_used) ?? 0,
    contextMax: num(item.context_max) ?? 0,
    ...optional('ttft_avg_ms', 'ttftMs'),
    ...optional('tokens_per_second', 'tokensPerSecond'),
    ...(num(item.cache_hit_rate) === undefined ? {} : { cacheHitRate: num(item.cache_hit_rate)! }),
    ...(num(item.calls) === undefined ? {} : { calls: num(item.calls)! }),
    ...optional('cost_usd', 'costUsd'),
    ...(item.cost_basis === 'subscription' ? { costBasis: 'subscription' } : {}),
    ...taskModelsOf(item),
  } as SessionUsageView
}

/** `models` and `advisor` from a `session.usage` answer (or a report's session); empty on older runtimes. */
export function taskModelsOf(value: unknown): Pick<SessionUsageView, 'models' | 'advisor'> {
  const item = record(value)
  const advisor = record(item.advisor)
  const profile = str(advisor.profile)
  const maxCalls = num(advisor.max_calls)
  const maxUsd = num(advisor.max_usd)
  return {
    ...(Array.isArray(item.models) ? { models: item.models.map(taskModelOf).filter((entry): entry is TaskModelUsageView => entry !== null) } : {}),
    ...(profile && maxCalls !== undefined && maxUsd !== undefined ? { advisor: { profile, maxCalls, maxUsd } } : {}),
  }
}

function taskModelOf(value: unknown): TaskModelUsageView | null {
  const base = modelOf(value)
  if (!base) return null
  const item = record(value)
  const role = item.role === 'worker' || item.role === 'advisor' || item.role === 'subagent' ? item.role : undefined
  const bytes = num(item.bytes_sent)
  return { ...base, ...(role ? { role } : {}), ...(bytes !== undefined && bytes > 0 ? { bytesSent: bytes } : {}) }
}

function modelOf(value: unknown): ModelUsageView | null {
  const item = record(value)
  const model = str(item.model)
  if (!model) return null
  const profile = str(item.profile)
  const cost = num(item.cost_usd)
  return {
    model,
    ...(profile ? { profile } : {}),
    calls: num(item.calls) ?? 0,
    input: num(item.input_tokens) ?? 0,
    output: num(item.output_tokens) ?? 0,
    cacheRead: num(item.cache_read_tokens) ?? 0,
    cacheWrite: num(item.cache_write_tokens) ?? 0,
    ...(cost === undefined ? {} : { costUsd: cost }),
    ...(item.cost_basis === 'subscription' ? { costBasis: 'subscription' as const } : {}),
    ...(typeof item.local === 'boolean' || item.local === null ? { local: item.local as boolean | null } : {}),
  }
}

/** Every token a round processed: what it read (fresh and cached) and what it wrote. */
export function processedTokens(model: Pick<ModelUsageView, 'input' | 'output' | 'cacheRead' | 'cacheWrite'>): number {
  return model.input + model.output + model.cacheRead + model.cacheWrite
}

/** How a task split between this machine and the cloud. */
export interface TaskSplit {
  readonly localTokens: number
  readonly cloudTokens: number
  /** Tokens from a profile that no longer exists, so where they ran is unknown. */
  readonly unknownTokens: number
  /** Share of the known tokens that ran on this machine, 0–1; null with no known tokens. */
  readonly localShare: number | null
  readonly advisorCalls: number
  /** Cloud spend at published prices; undefined when a cloud call has no published price. */
  readonly cloudUsd: number | undefined
  /** Every cloud call is paid by a plan, so there is no spend to show. */
  readonly cloudOnPlan: boolean
  readonly bytesSent: number
}

export function taskSplit(models: readonly TaskModelUsageView[]): TaskSplit {
  let localTokens = 0, cloudTokens = 0, unknownTokens = 0, advisorCalls = 0, bytesSent = 0
  let cloudUsd: number | undefined = 0
  let cloudOnPlan = true, anyCloud = false
  for (const model of models) {
    const tokens = processedTokens(model)
    if (model.role === 'advisor') advisorCalls += model.calls
    bytesSent += model.bytesSent ?? 0
    if (model.local === true) { localTokens += tokens; continue }
    if (model.local !== false) { unknownTokens += tokens; continue }
    cloudTokens += tokens
    anyCloud = true
    if (model.costBasis === 'subscription') continue
    cloudOnPlan = false
    cloudUsd = model.costUsd === undefined || cloudUsd === undefined ? undefined : cloudUsd + model.costUsd
  }
  const known = localTokens + cloudTokens
  return {
    localTokens, cloudTokens, unknownTokens,
    localShare: known > 0 ? localTokens / known : null,
    advisorCalls,
    cloudUsd: anyCloud && cloudOnPlan ? undefined : cloudUsd,
    cloudOnPlan: anyCloud && cloudOnPlan,
    bytesSent,
  }
}

/** "4.2 KB", "1.3 MB": bytes as a person reads them. */
export function compactBytes(bytes: number): string {
  if (bytes < 1_000) return `${bytes} B`
  if (bytes < 1_000_000) return `${(bytes / 1_000).toFixed(bytes < 10_000 ? 1 : 0)} KB`
  return `${(bytes / 1_000_000).toFixed(1)} MB`
}

/** "Main" for the conversation's own model (the lead in Hybrid), "Advisor" and "Subagent" for the others. */
export function modelRoleLabel(role: ModelRole | undefined): string {
  return role === 'advisor' ? 'Advisor' : role === 'subagent' ? 'Subagent' : role === 'worker' ? 'Main' : 'Model'
}

/** Normalize a `usage.report` result; malformed entries are dropped, never guessed. */
export function parseUsageReport(value: unknown): UsageReportView {
  const item = record(value)
  const session = sessionOf(item.session)
  const modelsSince = num(item.models_since)
  return {
    fetchedAt: num(item.fetched_at) ?? Date.now(),
    profiles: Array.isArray(item.profiles) ? item.profiles.map(profileOf).filter((entry): entry is UsageProfileView => entry !== null) : [],
    ...(session ? { session } : {}),
    ...(Array.isArray(item.models) ? { models: item.models.map(modelOf).filter((entry): entry is ModelUsageView => entry !== null) } : {}),
    ...(modelsSince === undefined ? {} : { modelsSince }),
  }
}

/** "$1.24", "$0.0031" — precise below a cent, rounded above. */
export function usd(amount: number): string {
  return `$${amount.toFixed(amount > 0 && amount < 0.01 ? 4 : 2)}`
}

/** One model's line: "412 calls · 1.2M in · 88K out" plus cache reads when there were any. */
export function modelUsageSummary(model: ModelUsageView): string {
  const parts = [`${model.calls} ${model.calls === 1 ? 'call' : 'calls'}`, `${compactTokens(model.input)} in`, `${compactTokens(model.output)} out`]
  if (model.cacheRead > 0) parts.push(`${compactTokens(model.cacheRead)} cached`)
  return parts.join(' · ')
}

/** Time until a window resets, as "3d 23h", "4h 12m" or "18m"; undefined when unknown. */
export function resetIn(window: Pick<UsageWindowView, 'resetsAt' | 'resetAfterSeconds'>, fetchedAt: number | undefined, now = Date.now()): string | undefined {
  const target = window.resetsAt ?? (window.resetAfterSeconds !== undefined && fetchedAt !== undefined ? fetchedAt + window.resetAfterSeconds * 1_000 : undefined)
  if (target === undefined) return undefined
  const minutes = Math.max(0, Math.round((target - now) / 60_000))
  const days = Math.floor(minutes / 1_440)
  const hours = Math.floor((minutes % 1_440) / 60)
  const rest = minutes % 60
  if (days > 0) return `${days}d ${hours}h`
  if (hours > 0) return `${hours}h ${rest}m`
  return `${rest}m`
}

/**
 * "5-hour", "Weekly · opus" — the window's name with the model it covers.
 * Counts and money go to `windowNote`; raw provider identifiers
 * (`tokens_limit`) are dropped rather than shown.
 */
export function windowLabel(window: UsageWindowView): string {
  const name = window.label.charAt(0).toUpperCase() + window.label.slice(1)
  const detail = window.detail?.trim()
  if (!detail || windowNote(window) || /^[a-z]+(?:_[a-z]+)+$/i.test(detail) || detail.toLowerCase() === window.label.toLowerCase()) return name
  return `${name} · ${detail}`
}

/** "$2.50 of $10.00", "1.2M left" — a window's quantity, when it has one. */
export function windowNote(window: UsageWindowView): string | undefined {
  const detail = window.detail?.trim()
  if (!detail) return undefined
  if (detail.startsWith('$') || detail.startsWith('¥')) return detail
  const remaining = /^(\d+(?:\.\d+)?)\s+remaining$/i.exec(detail)
  if (remaining) return `${compactTokens(Number(remaining[1]))} left`
  return undefined
}

export function compactTokens(tokens: number): string {
  if (tokens < 1_000) return String(Math.round(tokens))
  if (tokens < 1_000_000) return `${(tokens / 1_000).toFixed(tokens < 10_000 ? 1 : 0)}K`
  return `${(tokens / 1_000_000).toFixed(1)}M`
}

function duration(milliseconds: number): string {
  const seconds = Math.max(0, Math.round(milliseconds / 1_000))
  if (seconds < 60) return `${seconds}s`
  const minutes = Math.floor(seconds / 60)
  const remainder = seconds % 60
  return `${minutes}m${remainder ? ` ${remainder}s` : ''}`
}

/**
 * Session statistics as label/value rows, in reading order. A metric the
 * runtime never measured is omitted rather than shown as "0" or "—".
 */
export function sessionUsageRows(session: SessionUsageView): [string, string][] {
  const rows: [string, string][] = [
    ['Turns', String(session.turns)],
    ['Steps', String(session.steps)],
    ['Model time', duration(session.modelMs)],
    ['Tool time', duration(session.toolMs)],
  ]
  if (session.ttftMs !== undefined) rows.push(['First response', session.ttftMs < 1_000 ? `${Math.round(session.ttftMs)}ms` : `${(session.ttftMs / 1_000).toFixed(1)}s`])
  if (session.tokensPerSecond !== undefined) rows.push(['Generation', `${session.tokensPerSecond.toFixed(1)} tok/s`])
  if (session.cacheHitRate !== undefined) rows.push(['Cache hit', `${Math.round(session.cacheHitRate * 100)}%`])
  rows.push(['Tokens', `${compactTokens(session.input)} in · ${compactTokens(session.output)} out`])
  if (session.calls !== undefined && session.calls > 0) rows.push(['API calls', String(session.calls)])
  if (session.costBasis === 'subscription') rows.push(['Cost', costLabel(session.costUsd, session.costBasis)])
  else if (session.costUsd !== undefined) rows.push(['Cost', `$${session.costUsd.toFixed(session.costUsd < 0.01 ? 4 : 2)}`])
  return rows
}

function textBar(percent: number, cells = 16): string {
  const filled = percent <= 0 ? 0 : Math.min(cells, Math.max(1, Math.round((percent / 100) * cells)))
  return '▰'.repeat(filled) + '▱'.repeat(cells - filled)
}

/** Plain-text rendering for clients without a Usage view (chat channels, logs). */
export function formatUsageText(report: UsageReportView, now = Date.now()): string {
  const lines: string[] = []
  const session = report.session
  if (session) {
    lines.push(`Session${session.model ? ` · ${session.model}` : ''}`)
    for (const [label, value] of sessionUsageRows(session)) lines.push(`  ${label.padEnd(15)}${value}`)
    if (session.contextMax > 0) {
      const percent = (session.contextUsed / session.contextMax) * 100
      lines.push(`  ${'Context'.padEnd(15)}${textBar(percent)} ${Math.round(percent)}%  ${compactTokens(session.contextUsed)} / ${compactTokens(session.contextMax)}`)
    }
    const models = session.models ?? []
    if (models.length) {
      const split = taskSplit(models)
      if (split.localShare !== null && split.cloudTokens > 0) {
        lines.push(`  ${'Where it ran'.padEnd(15)}${textBar(split.localShare * 100)} ${Math.round(split.localShare * 100)}% on this machine`)
      }
      for (const model of models) {
        const where = model.local === true ? 'this machine' : model.local === false ? 'cloud' : 'unknown'
        lines.push(`  ${modelRoleLabel(model.role).padEnd(15)}${model.model} (${where})  ${modelUsageSummary(model)}${model.local === false ? `  ${costLabel(model.costUsd, model.costBasis)}` : ''}`)
      }
      if (session.advisor) {
        const spent = split.cloudOnPlan ? 'included in plan' : split.cloudUsd === undefined ? 'spend unknown' : `${usd(split.cloudUsd)} of ${usd(session.advisor.maxUsd)}`
        lines.push(`  ${'Advisor budget'.padEnd(15)}${split.advisorCalls} of ${session.advisor.maxCalls} calls · ${spent}`)
      }
    }
  }
  if (report.profiles.length) {
    if (lines.length) lines.push('')
    lines.push('Plans & keys')
    for (const profile of report.profiles) {
      const head = [profile.label, usageSourceLabel(profile.source, profile.provider), profile.plan].filter(Boolean).join(' · ')
      lines.push(`${profile.active ? '●' : '○'} ${head}${profile.balance ? `  —  ${profile.balance}` : ''}`)
      if (profile.status !== 'ok') lines.push(`    ${profile.status === 'unsupported' ? 'No usage limits published' : profile.message ?? 'Unavailable'}`)
      for (const fact of profile.facts ?? []) lines.push(`    ${fact.label.padEnd(18)}${fact.value}`)
      for (const window of profile.windows) {
        const reset = resetIn(window, profile.fetchedAt, now)
        const note = windowNote(window)
        lines.push(`    ${windowLabel(window).padEnd(18)}${textBar(window.usedPercent)} ${String(Math.round(window.usedPercent)).padStart(3)}%${note ? `  ${note}` : ''}${reset ? `  resets in ${reset}` : ''}`)
      }
    }
  }
  if (report.models?.length) {
    if (lines.length) lines.push('')
    lines.push('By model · last 30 days')
    for (const model of report.models) lines.push(`  ${model.model.padEnd(28)}${costLabel(model.costUsd, model.costBasis).padEnd(17)}${modelUsageSummary(model)}`)
  }
  return lines.join('\n') || 'No usage recorded yet.'
}
