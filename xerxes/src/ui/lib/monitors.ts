// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import type { GatewayRpc } from '../app/interfaces.js'
function reactionTokenLabel(health: Record<string, unknown>): string {
  if (health.tokenBudget == null) return ''
  const budget = health.tokenBudget as Record<string, unknown>
  const usage = health.usage as Record<string, unknown> | undefined
  if (!budget || typeof budget.maximum !== 'number' || !Number.isSafeInteger(budget.maximum) || budget.maximum < 1 || typeof budget.blocked !== 'boolean'
    || !usage || typeof usage.inputTokens !== 'number' || typeof usage.outputTokens !== 'number'
    || !Number.isSafeInteger(usage.inputTokens) || usage.inputTokens < 0 || !Number.isSafeInteger(usage.outputTokens) || usage.outputTokens < 0
    || typeof usage.complete !== 'boolean') throw new Error('Invalid monitor token budget')
  const total = usage.inputTokens + usage.outputTokens
  return ` · tokens ${Number.isSafeInteger(total) && total >= 0 ? total : 'unknown'}/${budget.maximum}${usage.complete ? '' : ' (incomplete)'}${budget.blocked ? ' · token admission blocked' : ''}`
}
export interface MonitorView {
  policy?: { revision: string; maxReactions: number; maxDurationMs: number; maxTotalTokens: number | null }
  id: string; terminalId: string; match: string; state: string; expiresAt: number
  source?: { kind: 'terminal'; terminalId: string } | { kind: 'file'; path: string; workspace: string } | { kind: 'websocket'; url: string } | { kind: 'webhook'; name: string }
  trigger: 'output' | 'completion' | 'change'
  stopAction?: 'stop-watch' | 'cancel-reactions' | null
  sourceStatus?: string
  reaction: string; error: string; events: string[]; omittedEvents: number
}
function parse(value: unknown): MonitorView {
  if (!value || typeof value !== 'object') throw new Error('Invalid monitor response')
  const row = value as Record<string, unknown>
  if (typeof row.id !== 'string' || typeof row.terminalId !== 'string' || typeof row.match !== 'string' || typeof row.state !== 'string'
    || !['watching', 'stopped', 'expired', 'source-ended', 'limit-reached', 'failed', 'archived', 'interrupted', 'detached'].includes(row.state)) throw new Error('Invalid monitor response')
  const trigger = row.trigger === undefined ? 'output' : row.trigger
  if (trigger !== 'output' && trigger !== 'completion' && trigger !== 'change') throw new Error('Invalid monitor trigger')
  let source: MonitorView['source']
  if (row.source !== undefined) {
    if (!row.source || typeof row.source !== 'object') throw new Error('Invalid monitor source')
    const candidate = row.source as Record<string, unknown>
    if (candidate.kind === 'terminal' && typeof candidate.terminalId === 'string' && candidate.terminalId.trim()) source = { kind: 'terminal', terminalId: candidate.terminalId }
    else if (candidate.kind === 'file' && typeof candidate.path === 'string' && candidate.path.trim() && typeof candidate.workspace === 'string' && candidate.workspace.trim()) source = { kind: 'file', path: candidate.path, workspace: candidate.workspace }
    else if (candidate.kind === 'websocket' && typeof candidate.url === 'string' && candidate.url.trim()) source = { kind: 'websocket', url: candidate.url }
    else if (candidate.kind === 'webhook' && typeof candidate.name === 'string' && candidate.name.trim()) source = { kind: 'webhook', name: candidate.name }
    else throw new Error('Invalid monitor source')
    if ((source.kind === 'file' && trigger !== 'change') || ((source.kind === 'websocket' || source.kind === 'webhook') && trigger !== 'output') || (source.kind === 'terminal' && trigger === 'change')) throw new Error('Invalid monitor source trigger')
  } else {
    if (trigger === 'change') throw new Error('File monitor source is required for change events')
    source = { kind: 'terminal', terminalId: row.terminalId }
  }
  if (row.stopAction !== undefined && row.stopAction !== null && row.stopAction !== 'stop-watch' && row.stopAction !== 'cancel-reactions') throw new Error('Invalid monitor action')
  const health = row.reactionHealth && typeof row.reactionHealth === 'object' ? row.reactionHealth as Record<string, unknown> : undefined
  const policy = health?.policy as MonitorView['policy']
  if (policy && (typeof policy.revision !== 'string' || !Number.isSafeInteger(policy.maxReactions) || !Number.isSafeInteger(policy.maxDurationMs)
    || (policy.maxTotalTokens !== null && !Number.isSafeInteger(policy.maxTotalTokens)))) throw new Error('Invalid monitor policy')
  return { id: row.id, terminalId: row.terminalId, match: row.match, state: row.state, trigger, source, expiresAt: typeof row.expiresAt === 'number' ? row.expiresAt : 0,
    ...(policy ? { policy } : {}),
    stopAction: row.stopAction === undefined ? row.state === 'watching' ? 'stop-watch' : null : row.stopAction,
    sourceStatus: typeof row.sourceStatus === 'string' ? row.sourceStatus : '',
    reaction: health && typeof health.state === 'string' && typeof health.attempts === 'number' && typeof health.maxReactions === 'number'
      ? `${health.state} · ${health.attempts}/${health.maxReactions} attempts${reactionTokenLabel(health)}` : 'Notifications only',
    error: [typeof row.error === 'string' ? row.error : '', typeof health?.lastError === 'string' ? health.lastError : '',
      typeof row.deliveryError === 'string' ? `Notification delivery: ${row.deliveryError}` : ''].filter(Boolean).join('\n'),
    events: Array.isArray(row.events) ? row.events.flatMap(event => event && typeof event === 'object' && typeof event.text === 'string' ? [event.text] : []) : [],
    omittedEvents: typeof row.omittedEvents === 'number' ? row.omittedEvents : 0 }
}
interface Response { ok?: boolean; error?: string; monitors?: unknown; monitor?: unknown }
export async function listMonitors(rpc: GatewayRpc): Promise<MonitorView[]> {
  const response = await rpc<Response>('monitor.list', {})
  if (!response?.ok || !Array.isArray(response.monitors)) throw new Error(response?.error || 'Monitors unavailable')
  return response.monitors.map(parse)
}
export async function monitorAction(rpc: GatewayRpc, id: string, action: 'inspect' | 'stop'): Promise<MonitorView> {
  const response = await rpc<Response>(`monitor.${action}`, { monitor_id: id })
  if (!response?.ok) throw new Error(response?.error || 'Monitor action failed')
  const watch = parse(response.monitor)
  if (watch.id !== id) throw new Error('Monitor identity changed')
  return watch
}

export interface MonitorSettings { max_total_tokens?: number; terminal_id?: string; source_kind?: 'file' | 'websocket' | 'webhook'; file_path?: string; websocket_url?: string; webhook_name?: string; trigger?: 'output' | 'completion' | 'change'; match?: string; duration_seconds: number; react: boolean; max_reactions: number; reaction_timeout_seconds?: number }
export async function createMonitor(rpc: GatewayRpc, settings: MonitorSettings): Promise<MonitorView> {
  const response = await rpc<Response>('monitor.create', { ...settings })
  if (!response?.ok) throw new Error(response?.error || 'Could not create monitor')
  return parse(response.monitor)
}

export async function listMonitorWebhooks(rpc: GatewayRpc): Promise<string[]> {
  const response = await rpc<{ ok?: boolean; error?: string; webhooks?: unknown }>('monitor.sources', {})
  if (!response?.ok || !Array.isArray(response.webhooks)) throw new Error(response?.error || 'Webhook sources unavailable')
  return response.webhooks.map(value => {
    if (!value || typeof value !== 'object') throw new Error('Invalid webhook source response')
    const candidate = value as Record<string, unknown>
    if (typeof candidate.name !== 'string' || !candidate.name.trim()) throw new Error('Invalid webhook source response')
    return candidate.name
  })
}
