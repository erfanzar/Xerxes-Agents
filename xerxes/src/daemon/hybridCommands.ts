// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Slash commands for M2/M3: `/route`, `/plans`, `/calibration`. Every command
 * here only reads or explains; none of them turns a feature on by itself, and
 * with the settings unset each one reports that the feature is off.
 */

import type { PlanStore } from '../cortex/planStore.js'
import { parseHybridPlansConfig } from '../cortex/hybridPlan.js'
import { parseRoutingConfig } from '../routing/config.js'
import { PolicyRouter, renderRouteDecision } from '../routing/router.js'
import type { DataTrust, StepKind, TaskRequirements } from '../routing/types.js'
import { recommend, renderCalibration, type CalibrationStore } from '../runtime/calibration.js'
import { parseEscalationConfig } from '../runtime/escalationPolicy.js'
import { parseDataBoundaryConfig } from '../security/dataBoundary.js'

export interface HybridCommandResult {
  readonly ok: boolean
  readonly report: string
  readonly [key: string]: unknown
}

type Env = Record<string, string | undefined>

const STEP_KINDS: readonly StepKind[] = ['plan', 'tool_use', 'analysis', 'reasoning', 'review', 'summarize', 'research']

export const ROUTE_USAGE = [
  'Usage:',
  '  /route                                   routing mode, configured routes and related M2 settings',
  '  /route explain [kind] [--private] [--tokens N]',
  `                                           which route a step would take and why (kind: ${STEP_KINDS.join('|')})`,
  'Turn routing on with runtime.routing: rules (or XERXES_ROUTING) and list routes in runtime.routing_routes.',
].join('\n')

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

/** `/route`: show or explain the policy router. Never changes the active model. */
export function routeCommand(
  args: string,
  ports: { readonly cwd: string; readonly runtime: Record<string, unknown>; readonly env?: Env; readonly trustOf?: (profile: string) => DataTrust | undefined },
): HybridCommandResult {
  const env = ports.env ?? process.env
  const words = args.trim().split(/\s+/).filter(Boolean)
  const action = (words[0] ?? 'status').toLowerCase()
  try {
    const routing = parseRoutingConfig(ports.runtime, env)
    const boundary = parseDataBoundaryConfig(ports.runtime, env, { root: ports.cwd })
    if (action === 'help') return { ok: true, report: ROUTE_USAGE }
    if (action === 'status') {
      const escalation = parseEscalationConfig(ports.runtime, env)
      const lines = [
        `Routing: ${routing.mode.toUpperCase()}${routing.mode === 'off' ? ' (the configured model is always used)' : ''}`,
        `Data boundary: ${boundary.mode}`,
        `Escalation: ${escalation.mode}${escalation.mode === 'on' ? ` (local retries ${escalation.maxLocalRetries}, model changes ${escalation.maxModelChanges}, advisor calls ${escalation.maxAdvisorCalls}, $${escalation.maxUsd})` : ''}`,
        `Routes (${routing.routes.length}):`,
        ...routing.routes.map(route => `  ${route.id}: ${route.profile}/${route.model} trust=${route.trust ?? ports.trustOf?.(route.profile) ?? 'cloud'}${route.tier === undefined ? '' : ` tier=${route.tier}`}`),
      ]
      if (!routing.routes.length) lines.push('  none (add runtime.routing_routes)')
      return { ok: true, report: lines.join('\n'), mode: routing.mode }
    }
    if (action !== 'explain') return { ok: false, report: ROUTE_USAGE }
    let kind: StepKind = 'analysis'
    let touchesPrivate = false
    let tokens = 4000
    for (let index = 1; index < words.length; index += 1) {
      const word = words[index]!
      if (word === '--private') touchesPrivate = true
      else if (word === '--tokens' && /^\d+$/.test(words[index + 1] ?? '')) { tokens = Number(words[index + 1]); index += 1 }
      else if ((STEP_KINDS as readonly string[]).includes(word)) kind = word as StepKind
      else return { ok: false, report: ROUTE_USAGE }
    }
    const router = new PolicyRouter({
      mode: routing.mode === 'off' ? 'rules' : routing.mode,
      routes: routing.routes,
      boundaryMode: boundary.mode,
      ...(ports.trustOf ? { trustOf: ports.trustOf } : {}),
    })
    const request: TaskRequirements = {
      kind,
      touchesPrivate,
      estimatedInputTokens: tokens,
      needsTools: kind === 'tool_use',
    }
    const decision = router.decide(request)
    const prefix = routing.mode === 'off' ? 'Routing is OFF, so nothing is rerouted. With rules it would pick:\n' : ''
    return { ok: true, report: prefix + renderRouteDecision(decision), decision }
  } catch (error) {
    return { ok: false, report: `Routing settings are invalid: ${errorText(error)}` }
  }
}

export const PLANS_USAGE = 'Usage: /plans [list] | /plans show <id> | /plans remove <id>'

/** `/plans`: saved hybrid plans. Running them is a separate, explicit step. */
export async function plansCommand(
  args: string,
  ports: { readonly runtime: Record<string, unknown>; readonly env?: Env; readonly store: Pick<PlanStore, 'list' | 'load' | 'remove' | 'listInvalid'> },
): Promise<HybridCommandResult> {
  const env = ports.env ?? process.env
  const words = args.trim().split(/\s+/).filter(Boolean)
  const action = (words[0] ?? 'list').toLowerCase()
  let enabled: boolean
  try {
    enabled = parseHybridPlansConfig(ports.runtime, env)
  } catch (error) {
    return { ok: false, report: `hybrid_plans setting is invalid: ${errorText(error)}` }
  }
  const header = `Hybrid plans: ${enabled ? 'ON' : 'OFF (set runtime.hybrid_plans: on or XERXES_HYBRID_PLANS=on)'}`
  if (action === 'list' && words.length <= 1) {
    const plans = await ports.store.list()
    const invalid = await ports.store.listInvalid()
    const lines = [header, plans.length ? `Saved plans (${plans.length}):` : 'No saved plans.']
    for (const plan of plans) {
      lines.push(`  ${plan.id}${plan.name ? ` "${plan.name}"` : ''}: ${plan.steps} steps, ${plan.reviewed ? 'reviewed' : 'not reviewed'}, ${plan.savedAt} — ${plan.objective.slice(0, 80)}`)
    }
    for (const bad of invalid) lines.push(`  ${bad.id}: unreadable (${bad.error})`)
    return { ok: true, report: lines.join('\n'), count: plans.length }
  }
  const id = words[1]
  if ((action === 'show' || action === 'remove') && id && words.length === 2) {
    if (action === 'remove') {
      const removed = await ports.store.remove(id)
      return removed ? { ok: true, report: `Removed plan ${id}.` } : { ok: false, report: `No saved plan ${id}.` }
    }
    const stored = await ports.store.load(id)
    if (!stored) return { ok: false, report: `No saved plan ${id}.` }
    const lines = [
      `${stored.id}${stored.name ? ` "${stored.name}"` : ''} (${stored.reviewed ? 'reviewed' : 'not reviewed'}): ${stored.plan.objective}`,
      ...stored.plan.steps.map(step => `  ${step.id} [${step.placement}, ${step.kind}]${step.dependencies.length ? ` after ${step.dependencies.join(', ')}` : ''}: ${step.description.slice(0, 100)}`),
    ]
    return { ok: true, report: lines.join('\n') }
  }
  return { ok: false, report: PLANS_USAGE }
}

/** `/calibration [workload]`: measured outcomes per strategy on this hardware. */
export async function calibrationCommand(
  args: string,
  ports: { readonly runtime: Record<string, unknown>; readonly env?: Env; readonly store: Pick<CalibrationStore, 'stats'>; readonly hardwareKey: () => Promise<string> },
): Promise<HybridCommandResult> {
  const workload = args.trim()
  // Nothing records outcomes yet; routing measurement arrives with M2.
  const header = 'Calibration: shows outcomes stored for this machine. Xerxes does not record them yet.'
  if (!workload) return { ok: true, report: `${header}\nUsage: /calibration <workload>  (e.g. csv-reconcile)` }
  if (!/^[\w.:+-]{1,80}$/.test(workload)) return { ok: false, report: 'Workload names use letters, digits and . : + - _ (max 80).' }
  const key = await ports.hardwareKey()
  const stats = ports.store.stats(workload, key)
  return { ok: true, report: `${header}\nWorkload ${workload} on ${key}:\n${renderCalibration(stats, recommend(stats))}`, hardwareKey: key }
}
