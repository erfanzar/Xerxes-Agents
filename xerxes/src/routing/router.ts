// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Policy router (M2.1): pick a local or cloud route per call.
 *
 * Privacy decides which routes are ALLOWED; the router only optimizes among
 * them. When nothing allowed fits it says so (`no_allowed_route`) and lists
 * every exclusion; it never falls back to a disallowed route. With mode `off`
 * it never reorders or substitutes anything.
 */

import { ValidationError } from '../core/errors.js'
import type { ConfiguredRoute } from './config.js'
import type {
  BudgetState,
  CalibrationLookup,
  DataBoundaryMode,
  DataTrust,
  RouteCandidate,
  RouteDecision,
  RouteExclusion,
  RoutingMode,
  StepKind,
  StrategyStats,
  TaskRequirements,
} from './types.js'

/** Minimum recorded runs of a route before the learned mode trusts its stats. */
export const MIN_CALIBRATION_RUNS = 3

const LOCAL_FIRST_KINDS: ReadonlySet<StepKind> = new Set<StepKind>(['tool_use', 'summarize', 'analysis'])
const STRONG_FIRST_KINDS: ReadonlySet<StepKind> = new Set<StepKind>(['reasoning', 'review', 'plan'])

export interface PolicyRouterOptions {
  readonly mode: RoutingMode
  /** Routes in declared order. A missing `trust` is resolved through `trustOf`, then fails closed to `cloud`. */
  readonly routes: readonly ConfiguredRoute[]
  readonly boundaryMode: DataBoundaryMode
  readonly trustOf?: (profile: string) => DataTrust | undefined
  readonly calibration?: CalibrationLookup
  readonly hardwareKey?: string
  readonly now?: () => number
}

/**
 * Estimated USD for one call. Undefined unless BOTH prices are known: an
 * unknown price is never invented.
 */
export function estimateCostUsd(
  route: Pick<RouteCandidate, 'inputUsdPerMTok' | 'outputUsdPerMTok'>,
  inputTokens: number,
  outputTokens: number,
): number | undefined {
  if (route.inputUsdPerMTok === undefined || route.outputUsdPerMTok === undefined) return undefined
  return (inputTokens * route.inputUsdPerMTok + outputTokens * route.outputUsdPerMTok) / 1_000_000
}

/** One short line plus the excluded routes, for `/route explain`. */
export function renderRouteDecision(decision: RouteDecision): string {
  const lines: string[] = []
  if (decision.ok) {
    const { route } = decision
    const cost = decision.estimatedCostUsd === undefined ? '' : `, est. $${decision.estimatedCostUsd.toFixed(4)}`
    lines.push(`route ${route.id} (${route.profile}/${route.model}, ${route.trust}) via ${decision.source}: ${decision.reason}${cost}`)
  } else {
    lines.push(`no allowed route: ${decision.reason}`)
  }
  for (const item of decision.excluded) lines.push(`  excluded ${item.id}: ${item.reason}`)
  return lines.join('\n')
}

function compareCost(a: number | undefined, b: number | undefined): number {
  if (a !== undefined && b !== undefined) return a - b
  if (a !== undefined) return -1
  if (b !== undefined) return 1
  return 0
}

interface Scored {
  readonly route: RouteCandidate
  readonly order: number
  readonly cost: number | undefined
}

interface Choice {
  readonly route: RouteCandidate
  readonly reason: string
}

export class PolicyRouter {
  private readonly mode: RoutingMode
  private readonly routes: readonly RouteCandidate[]
  private readonly boundaryMode: DataBoundaryMode
  private readonly calibration: CalibrationLookup | undefined
  private readonly hardwareKey: string | undefined
  private readonly now: () => number

  constructor(options: PolicyRouterOptions) {
    this.mode = options.mode
    this.boundaryMode = options.boundaryMode
    this.calibration = options.calibration
    this.hardwareKey = options.hardwareKey
    this.now = options.now ?? (() => performance.now())
    const seen = new Set<string>()
    this.routes = options.routes.map((route): RouteCandidate => {
      if (seen.has(route.id)) throw new ValidationError('routes', `duplicate route id "${route.id}"`, route.id)
      seen.add(route.id)
      const trust: DataTrust = route.trust ?? options.trustOf?.(route.profile) ?? 'cloud'
      return { ...route, trust }
    })
  }

  decide(req: TaskRequirements, budget?: BudgetState, fixed?: RouteCandidate): RouteDecision {
    const start = this.now()
    const elapsed = (): number => Math.max(0, this.now() - start)
    if (this.mode === 'off') return this.decideOff(req, fixed, elapsed)

    const excluded: RouteExclusion[] = []
    const allowed: RouteCandidate[] = []
    for (const route of this.routes) {
      const reason = this.exclusionReason(route, req, budget)
      if (reason === undefined) allowed.push(route)
      else excluded.push({ id: route.id, reason })
    }
    if (allowed.length === 0) return this.noRoute(excluded, elapsed(), this.routes.length === 0 ? 'no routes are configured' : undefined)

    const rules = this.chooseByRules(allowed, req)
    if (this.mode === 'learned') {
      const learned = this.chooseByStats(allowed, req)
      if (learned.kind === 'chosen') return this.chosen(learned.choice, 'learned', req, budget, excluded, elapsed())
      return this.chosen({ route: rules.route, reason: `${learned.why}; ${rules.reason}` }, 'rules', req, budget, excluded, elapsed())
    }
    return this.chosen(rules, 'rules', req, budget, excluded, elapsed())
  }

  /**
   * Routing off: keep today's selection. Only the data boundary may still
   * refuse it (privacy is not an optimization); budget, context and tier are
   * not applied because they would change behavior that is meant to be fixed.
   */
  private decideOff(req: TaskRequirements, fixed: RouteCandidate | undefined, elapsed: () => number): RouteDecision {
    const route = fixed ?? this.routes[0]
    if (route === undefined) return this.noRoute([], elapsed(), 'routing is off and no route is configured')
    const refusal = this.boundaryReason(route, req)
    if (refusal !== undefined) return this.noRoute([{ id: route.id, reason: refusal }], elapsed())
    return this.chosen(
      { route, reason: fixed ? 'routing is off: using the configured model' : 'routing is off: using the first configured route' },
      'fixed',
      req,
      undefined,
      [],
      elapsed(),
    )
  }

  private boundaryReason(route: RouteCandidate, req: TaskRequirements): string | undefined {
    if (route.trust === 'local') return undefined
    if (this.boundaryMode === 'private') return 'data boundary is private: only local routes are allowed'
    if (this.boundaryMode !== 'off' && req.touchesPrivate) return 'step touches private data: only local routes are allowed'
    return undefined
  }

  private exclusionReason(route: RouteCandidate, req: TaskRequirements, budget: BudgetState | undefined): string | undefined {
    const boundary = this.boundaryReason(route, req)
    if (boundary !== undefined) return boundary
    if (req.placement === 'local' && route.trust !== 'local') return 'placement requires a local route'
    if (req.placement === 'cloud' && route.trust === 'local') return 'placement requires a cloud route'
    const outputTokens = req.estimatedOutputTokens ?? 0
    if (route.contextLimit !== undefined && route.contextLimit < req.estimatedInputTokens + outputTokens) {
      return `context limit ${route.contextLimit} is below the estimated ${req.estimatedInputTokens + outputTokens} tokens`
    }
    if (req.minTier !== undefined) {
      if (route.tier === undefined) return `tier is unknown but the step needs tier ${req.minTier}`
      if (route.tier < req.minTier) return `tier ${route.tier} is below the required tier ${req.minTier}`
    }
    if (budget?.remainingUsd !== undefined) {
      const cost = estimateCostUsd(route, req.estimatedInputTokens, outputTokens)
      if (cost !== undefined && cost > budget.remainingUsd) {
        return `estimated $${cost.toFixed(4)} exceeds the remaining budget $${budget.remainingUsd.toFixed(4)}`
      }
    }
    if (budget?.remainingCalls !== undefined && budget.remainingCalls <= 0 && route.trust !== 'local') {
      return 'no cloud calls remain in the budget'
    }
    return undefined
  }

  private chooseByRules(allowed: readonly RouteCandidate[], req: TaskRequirements): Choice {
    const outputTokens = req.estimatedOutputTokens ?? 0
    const scored: Scored[] = allowed.map((route, order) => ({
      route,
      order,
      cost: estimateCostUsd(route, req.estimatedInputTokens, outputTokens),
    }))
    const byCost = (a: Scored, b: Scored): number => compareCost(a.cost, b.cost) || a.order - b.order
    const first = (list: readonly Scored[]): Scored => {
      const head = list[0]
      if (head === undefined) throw new Error('chooseByRules requires at least one allowed route')
      return head
    }

    if (LOCAL_FIRST_KINDS.has(req.kind)) {
      const locals = scored.filter((item) => item.route.trust === 'local').sort(byCost)
      if (locals.length > 0) {
        const pick = first(locals)
        return { route: pick.route, reason: `${req.kind} work prefers a local route; ${pick.route.id} is the best allowed local route` }
      }
      const pick = first([...scored].sort(byCost))
      return { route: pick.route, reason: `${req.kind} work prefers a local route but none is allowed; ${pick.route.id} is the cheapest allowed route` }
    }
    if (STRONG_FIRST_KINDS.has(req.kind)) {
      const ranked = [...scored].sort((a, b) => (b.route.tier ?? 0) - (a.route.tier ?? 0) || byCost(a, b))
      const pick = first(ranked)
      const tier = pick.route.tier === undefined ? 'with no declared tier' : `at tier ${pick.route.tier}`
      return { route: pick.route, reason: `${req.kind} work prefers the highest tier; ${pick.route.id} is the strongest allowed route, ${tier}` }
    }
    const pick = first([...scored].sort(byCost))
    return { route: pick.route, reason: `no placement preference for ${req.kind} work; ${pick.route.id} is the cheapest allowed route in declared order` }
  }

  private chooseByStats(
    allowed: readonly RouteCandidate[],
    req: TaskRequirements,
  ): { readonly kind: 'chosen'; readonly choice: Choice } | { readonly kind: 'insufficient'; readonly why: string } {
    const need = `not enough calibration data (need ${MIN_CALIBRATION_RUNS} runs of an allowed route)`
    if (!req.workload || !this.calibration || !this.hardwareKey) return { kind: 'insufficient', why: need }
    const byId = new Map<string, StrategyStats>()
    for (const stat of this.calibration.stats(req.workload, this.hardwareKey)) byId.set(stat.strategy, stat)
    const candidates = allowed
      .map((route, order) => ({ route, order, stat: byId.get(route.id) }))
      .filter((item): item is { route: RouteCandidate; order: number; stat: StrategyStats } =>
        item.stat !== undefined && item.stat.runs >= MIN_CALIBRATION_RUNS)
    candidates.sort((a, b) =>
      b.stat.acceptRate - a.stat.acceptRate
      || compareCost(a.stat.meanCostUsd, b.stat.meanCostUsd)
      || a.stat.medianWallMs - b.stat.medianWallMs
      || a.order - b.order)
    const best = candidates[0]
    if (best === undefined) return { kind: 'insufficient', why: need }
    const percent = Math.round(best.stat.acceptRate * 100)
    return {
      kind: 'chosen',
      choice: {
        route: best.route,
        reason: `${best.route.id} has the best recorded outcome for ${req.workload} on this hardware (${percent}% accepted over ${best.stat.runs} runs)`,
      },
    }
  }

  private chosen(
    choice: Choice,
    source: 'fixed' | 'rules' | 'learned',
    req: TaskRequirements,
    budget: BudgetState | undefined,
    excluded: readonly RouteExclusion[],
    decisionMs: number,
  ): RouteDecision {
    const cost = estimateCostUsd(choice.route, req.estimatedInputTokens, req.estimatedOutputTokens ?? 0)
    const note = budget?.remainingUsd !== undefined && cost === undefined
      ? ` (price unknown, so the USD budget was not checked for ${choice.route.id})`
      : ''
    const base = { ok: true, route: choice.route, source, reason: `${choice.reason}${note}`, excluded, decisionMs } as const
    return cost === undefined ? base : { ...base, estimatedCostUsd: cost }
  }

  private noRoute(excluded: readonly RouteExclusion[], decisionMs: number, detail?: string): RouteDecision {
    const reason = detail ?? `all ${excluded.length} configured route${excluded.length === 1 ? ' is' : 's are'} excluded`
    return { ok: false, code: 'no_allowed_route', reason, excluded, decisionMs }
  }
}
