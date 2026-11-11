// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { isRecord } from '../hub/protocol.js'

export const CONTROL_METRICS = ['actions', 'wakes', 'modelCalls', 'tokens', 'usd'] as const
export type ControlMetric = (typeof CONTROL_METRICS)[number]
export type DailyLimits = Readonly<Partial<Record<ControlMetric, number>>>
export type DailyLimitsDraft = Readonly<Record<ControlMetric, string>>
export interface SuperagentControlSnapshot {
  readonly revision: number
  readonly paused: boolean
  readonly killed: boolean
  readonly limits: DailyLimits
  readonly usage: { readonly actions: number; readonly wakes: number; readonly modelCalls: number; readonly tokens: number | null; readonly usd: number | null }
}
export interface SuperagentControlPatch { readonly paused?: boolean; readonly killed?: boolean; readonly limits?: DailyLimits }
export interface SuperagentControlView {
  readonly phase: 'idle' | 'loading' | 'ready' | 'saving' | 'error'
  readonly data?: SuperagentControlSnapshot
  readonly error?: string
}
export interface SuperagentControlClient {
  getSnapshot(): SuperagentControlView
  subscribe(listener: () => void): () => void
  load(): Promise<void>
  save(patch: SuperagentControlPatch): Promise<boolean>
  dispose(): void
}
function record(value: unknown): Record<string, unknown> {
  if (!isRecord(value)) throw new Error('Invalid superagent control response')
  return value
}
function amount(value: unknown, metric: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0 || (metric !== 'usd' && !Number.isSafeInteger(value))) throw new Error(`Invalid ${metric} budget value`)
  return value
}
function metric(value: string): value is ControlMetric { return CONTROL_METRICS.some(key => key === value) }
function limits(value: unknown): DailyLimits {
  const result: Partial<Record<ControlMetric, number>> = {}
  for (const [key, item] of Object.entries(record(value))) {
    if (!metric(key)) throw new Error('Unknown daily budget metric')
    result[key] = amount(item, key)
  }
  return result
}
export function parseSuperagentControl(value: unknown): SuperagentControlSnapshot {
  const raw = record(value); const control = record(raw.control); const usage = record(raw.usage)
  if (typeof control.paused !== 'boolean' || typeof control.killed !== 'boolean') throw new Error('Invalid superagent control flags')
  return { revision: amount(control.revision, 'revision'), paused: control.paused, killed: control.killed, limits: limits(control.limits),
    usage: { actions: amount(usage.actions, 'actions'), wakes: amount(usage.wakes, 'wakes'), modelCalls: amount(usage.modelCalls, 'modelCalls'),
      tokens: usage.tokens === null ? null : amount(usage.tokens, 'tokens'), usd: usage.usd === null ? null : amount(usage.usd, 'usd') } }
}
export function parseDailyLimits(draft: DailyLimitsDraft): DailyLimits {
  const result: Partial<Record<ControlMetric, number>> = {}
  for (const key of CONTROL_METRICS) if (draft[key].trim() !== '') result[key] = amount(Number(draft[key]), key)
  return result
}

/** Human UI state: writes once, with a revision; ambiguous/refused saves require an explicit read. */
export function createSuperagentControlClient(options: {
  readonly allowed: () => boolean
  readonly request: (method: 'supervisor.control.get' | 'supervisor.control.set', params: Readonly<Record<string, unknown>>) => Promise<unknown>
}): SuperagentControlClient {
  let view: SuperagentControlView = { phase: 'idle' }
  let generation = 0; let disposed = false
  const listeners = new Set<() => void>()
  const publish = (next: SuperagentControlView): void => { view = next; for (const listener of listeners) listener() }
  const allowed = (): boolean => !disposed && options.allowed()
  const client: SuperagentControlClient = {
    getSnapshot: () => view,
    subscribe(listener) { listeners.add(listener); return () => { listeners.delete(listener) } },
    async load() {
      if (!allowed()) { if (!disposed) publish({ phase: 'error', error: 'Owner controls require a live connection as the Hub owner.' }); return }
      if (view.phase === 'saving') return
      const token = ++generation
      publish({ phase: 'loading', ...(view.data === undefined ? {} : { data: view.data }) })
      try {
        const result = await options.request('supervisor.control.get', {})
        if (disposed || token !== generation) return
        if (!allowed()) { publish({ phase: 'error', error: 'Owner access or connection changed. Reload after reconnecting.' }); return }
        publish({ phase: 'ready', data: parseSuperagentControl(result) })
      } catch (error) {
        if (!disposed && token === generation) publish({ phase: 'error', error: error instanceof Error ? error.message : 'Unable to read superagent controls' })
      }
    },
    async save(patch) {
      if (!allowed() || view.phase !== 'ready' || view.data === undefined) return false
      const token = ++generation; const data = view.data
      publish({ phase: 'saving', data })
      try {
        const result = await options.request('supervisor.control.set', { ...patch, revision: data.revision })
        if (disposed || token !== generation) return false
        if (!allowed()) { publish({ phase: 'error', error: 'Owner access or connection changed. Reload before further changes.' }); return false }
        publish({ phase: 'ready', data: parseSuperagentControl(result) })
        return true
      } catch (error) {
        if (!disposed && token === generation) publish({ phase: 'error', data,
          error: `${error instanceof Error ? error.message : 'Control update outcome is unknown'}. Reload before further changes; nothing will be retried.` })
        return false
      }
    },
    dispose() { disposed = true; generation += 1; listeners.clear() },
  }
  return client
}
