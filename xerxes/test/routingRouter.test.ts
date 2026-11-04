// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, test } from 'bun:test'

import { ValidationError } from '../src/core/errors.js'
import { parseRoutingConfig, type ConfiguredRoute } from '../src/routing/config.js'
import { estimateCostUsd, PolicyRouter, renderRouteDecision, type PolicyRouterOptions } from '../src/routing/router.js'
import type {
  CalibrationLookup,
  RouteCandidate,
  RouteDecision,
  StrategyStats,
  TaskRequirements,
} from '../src/routing/types.js'

const LOCAL: RouteCandidate = {
  id: 'local-small',
  profile: 'llama',
  model: 'bonsai-8b',
  trust: 'local',
  tier: 1,
  contextLimit: 8000,
}
const CLOUD_MID: RouteCandidate = {
  id: 'cloud-mid',
  profile: 'hosted',
  model: 'mid',
  trust: 'cloud',
  tier: 2,
  inputUsdPerMTok: 1,
  outputUsdPerMTok: 5,
  contextLimit: 100_000,
}
const CLOUD_BIG: RouteCandidate = {
  id: 'cloud-big',
  profile: 'hosted',
  model: 'big',
  trust: 'cloud',
  tier: 3,
  inputUsdPerMTok: 3,
  outputUsdPerMTok: 15,
  contextLimit: 200_000,
}

const REQ: TaskRequirements = { kind: 'analysis', touchesPrivate: false, estimatedInputTokens: 1000, needsTools: false }

function router(overrides: Partial<PolicyRouterOptions> = {}): PolicyRouter {
  return new PolicyRouter({
    mode: 'rules',
    routes: [LOCAL, CLOUD_MID, CLOUD_BIG],
    boundaryMode: 'off',
    ...overrides,
  })
}

function accepted(decision: RouteDecision): Extract<RouteDecision, { ok: true }> {
  if (!decision.ok) throw new Error(`expected a route, got: ${decision.reason}`)
  return decision
}

function refused(decision: RouteDecision): Extract<RouteDecision, { ok: false }> {
  if (decision.ok) throw new Error(`expected no route, got ${decision.route.id}`)
  return decision
}

function stat(strategy: string, runs: number, acceptRate: number, medianWallMs: number, meanCostUsd?: number): StrategyStats {
  const base = { strategy, runs, accepted: Math.round(runs * acceptRate), acceptRate, medianWallMs }
  return meanCostUsd === undefined ? base : { ...base, meanCostUsd }
}

function lookup(stats: readonly StrategyStats[]): CalibrationLookup & { readonly calls: string[] } {
  const calls: string[] = []
  return {
    calls,
    stats(workload, hardwareKey) {
      calls.push(`${workload}@${hardwareKey}`)
      return stats
    },
  }
}

describe('parseRoutingConfig', () => {
  test('defaults to off with no routes', () => {
    expect(parseRoutingConfig({}, {})).toEqual({ mode: 'off', routes: [] })
  })

  test('env XERXES_ROUTING wins over the file', () => {
    expect(parseRoutingConfig({ routing: 'rules' }, { XERXES_ROUTING: 'learned' }).mode).toBe('learned')
    expect(parseRoutingConfig({ routing: 'learned' }, { XERXES_ROUTING: 'off' }).mode).toBe('off')
    expect(parseRoutingConfig({ routing: 'rules' }, {}).mode).toBe('rules')
  })

  test('parses routes into the camel-case shape and keeps trust optional', () => {
    const config = parseRoutingConfig({
      routing: 'rules',
      routing_routes: [
        { id: 'a', profile: 'p', model: 'm', trust: 'local', tier: 1, input_usd_per_mtok: 0, output_usd_per_mtok: 0, context_limit: 4096 },
        { id: 'b', profile: 'q', model: 'n' },
      ],
    }, {})
    expect(config.routes).toEqual([
      { id: 'a', profile: 'p', model: 'm', trust: 'local', tier: 1, inputUsdPerMTok: 0, outputUsdPerMTok: 0, contextLimit: 4096 },
      { id: 'b', profile: 'q', model: 'n' },
    ])
    expect(config.routes[1]).not.toHaveProperty('trust')
  })

  function field(runtime: Record<string, unknown>, env: Record<string, string | undefined> = {}): string {
    try {
      parseRoutingConfig(runtime, env)
    } catch (error) {
      expect(error).toBeInstanceOf(ValidationError)
      return (error as ValidationError).field
    }
    throw new Error('expected a ValidationError')
  }

  test('bad values throw a ValidationError naming the key', () => {
    const route = { id: 'a', profile: 'p', model: 'm' }
    expect(field({ routing: 'sometimes' })).toBe('routing')
    expect(field({}, { XERXES_ROUTING: 'bogus' })).toBe('XERXES_ROUTING')
    expect(field({ routing_routes: 'nope' })).toBe('routing_routes')
    expect(field({ routing_routes: ['x'] })).toBe('routing_routes[0]')
    expect(field({ routing_routes: [{ ...route, id: '  ' }] })).toBe('routing_routes[0].id')
    expect(field({ routing_routes: [{ id: 'a', model: 'm' }] })).toBe('routing_routes[0].profile')
    expect(field({ routing_routes: [{ id: 'a', profile: 'p' }] })).toBe('routing_routes[0].model')
    expect(field({ routing_routes: [route, { ...route }] })).toBe('routing_routes[1].id')
    expect(field({ routing_routes: [{ ...route, tier: 4 }] })).toBe('routing_routes[0].tier')
    expect(field({ routing_routes: [{ ...route, tier: '2' }] })).toBe('routing_routes[0].tier')
    expect(field({ routing_routes: [{ ...route, trust: 'maybe' }] })).toBe('routing_routes[0].trust')
    expect(field({ routing_routes: [{ ...route, input_usd_per_mtok: -1 }] })).toBe('routing_routes[0].input_usd_per_mtok')
    expect(field({ routing_routes: [{ ...route, output_usd_per_mtok: 'free' }] })).toBe('routing_routes[0].output_usd_per_mtok')
    expect(field({ routing_routes: [{ ...route, context_limit: 0 }] })).toBe('routing_routes[0].context_limit')
    expect(field({ routing_routes: [{ ...route, context_limit: 1.5 }] })).toBe('routing_routes[0].context_limit')
    expect(field({ routing_routes: [{ ...route, colour: 'red' }] })).toBe('routing_routes[0].colour')
  })
})

describe('estimateCostUsd', () => {
  test('computes USD from both prices and never invents a missing one', () => {
    expect(estimateCostUsd(CLOUD_BIG, 1_000_000, 100_000)).toBeCloseTo(4.5)
    expect(estimateCostUsd(LOCAL, 1000, 1000)).toBeUndefined()
    expect(estimateCostUsd({ inputUsdPerMTok: 1 }, 1000, 1000)).toBeUndefined()
    expect(estimateCostUsd({ inputUsdPerMTok: 0, outputUsdPerMTok: 0 }, 1000, 1000)).toBe(0)
  })
})

describe('PolicyRouter: off', () => {
  test('returns the fixed route unchanged', () => {
    const decision = accepted(router({ mode: 'off' }).decide(REQ, undefined, CLOUD_BIG))
    expect(decision.route).toBe(CLOUD_BIG)
    expect(decision.source).toBe('fixed')
    expect(decision.excluded).toEqual([])
  })

  test('without a fixed route uses the first declared route, never reordering', () => {
    const decision = accepted(router({ mode: 'off', routes: [CLOUD_BIG, LOCAL] }).decide({ ...REQ, kind: 'tool_use' }))
    expect(decision.route.id).toBe('cloud-big')
    expect(decision.source).toBe('fixed')
  })

  test('ignores budget, tier and context in off mode', () => {
    const decision = accepted(router({ mode: 'off' }).decide({ ...REQ, minTier: 3, estimatedInputTokens: 500_000 }, { remainingUsd: 0, remainingCalls: 0 }, LOCAL))
    expect(decision.route).toBe(LOCAL)
  })

  test('no route at all is no_allowed_route', () => {
    const decision = refused(router({ mode: 'off', routes: [] }).decide(REQ))
    expect(decision.code).toBe('no_allowed_route')
  })

  test('the data boundary still refuses a cloud fixed route', () => {
    const decision = refused(router({ mode: 'off', boundaryMode: 'private' }).decide(REQ, undefined, CLOUD_BIG))
    expect(decision.excluded).toEqual([{ id: 'cloud-big', reason: expect.stringContaining('private') }])
    expect(accepted(router({ mode: 'off', boundaryMode: 'private' }).decide(REQ, undefined, LOCAL)).route).toBe(LOCAL)
    expect(refused(router({ mode: 'off', boundaryMode: 'hybrid' }).decide({ ...REQ, touchesPrivate: true }, undefined, CLOUD_MID)).excluded).toHaveLength(1)
    // boundary off: private-touching steps are not restricted.
    expect(accepted(router({ mode: 'off' }).decide({ ...REQ, touchesPrivate: true }, undefined, CLOUD_MID)).route).toBe(CLOUD_MID)
  })

  test('resolves a missing trust through trustOf and fails closed to cloud', () => {
    const routes: ConfiguredRoute[] = [
      { id: 'x', profile: 'unknown-profile', model: 'm' },
      { id: 'y', profile: 'loopback', model: 'm' },
    ]
    const trustOf = (profile: string) => (profile === 'loopback' ? ('local' as const) : undefined)
    const decision = accepted(router({ routes, trustOf, boundaryMode: 'private' }).decide(REQ))
    expect(decision.route.id).toBe('y')
    expect(decision.route.trust).toBe('local')
    expect(decision.excluded.map((item) => item.id)).toEqual(['x'])
  })

  test('rejects duplicate route ids', () => {
    expect(() => router({ routes: [LOCAL, { ...CLOUD_MID, id: LOCAL.id }] })).toThrow(ValidationError)
  })
})

describe('PolicyRouter: allowed-route filter', () => {
  test('private boundary excludes every cloud route with the reason', () => {
    const decision = accepted(router({ boundaryMode: 'private' }).decide({ ...REQ, kind: 'reasoning' }))
    expect(decision.route.id).toBe('local-small')
    expect(decision.excluded).toEqual([
      { id: 'cloud-mid', reason: expect.stringContaining('data boundary is private') },
      { id: 'cloud-big', reason: expect.stringContaining('data boundary is private') },
    ])
  })

  test('touchesPrivate under hybrid restricts to local; without it cloud is allowed', () => {
    const priv = accepted(router({ boundaryMode: 'hybrid' }).decide({ ...REQ, kind: 'reasoning', touchesPrivate: true }))
    expect(priv.route.id).toBe('local-small')
    expect(priv.excluded.map((item) => item.id)).toEqual(['cloud-mid', 'cloud-big'])
    expect(priv.excluded[0]?.reason).toContain('private data')
    const open = accepted(router({ boundaryMode: 'hybrid' }).decide({ ...REQ, kind: 'reasoning' }))
    expect(open.route.id).toBe('cloud-big')
  })

  test('boundary off does not restrict touchesPrivate steps', () => {
    const decision = accepted(router({ boundaryMode: 'off' }).decide({ ...REQ, kind: 'reasoning', touchesPrivate: true }))
    expect(decision.route.id).toBe('cloud-big')
  })

  test('cloud boundary mode allows cloud routes, but a step touching private data still stays local', () => {
    const open = accepted(router({ boundaryMode: 'cloud' }).decide({ ...REQ, kind: 'reasoning' }))
    expect(open.route.id).toBe('cloud-big')
    expect(open.excluded).toEqual([])
    const priv = accepted(router({ boundaryMode: 'cloud' }).decide({ ...REQ, kind: 'reasoning', touchesPrivate: true }))
    expect(priv.route.id).toBe('local-small')
  })

  test('no allowed route reports ok:false with all exclusions listed, never a disallowed route', () => {
    const decision = refused(router({ routes: [CLOUD_MID, CLOUD_BIG], boundaryMode: 'private' }).decide(REQ))
    expect(decision.code).toBe('no_allowed_route')
    expect(decision.excluded.map((item) => item.id)).toEqual(['cloud-mid', 'cloud-big'])
    expect(decision.reason).toContain('2')
    expect(renderRouteDecision(decision)).toContain('excluded cloud-big')
  })

  test('no configured routes is reported distinctly', () => {
    const decision = refused(router({ routes: [] }).decide(REQ))
    expect(decision.reason).toContain('no routes are configured')
  })

  test('placement local / cloud restricts', () => {
    const local = accepted(router().decide({ ...REQ, kind: 'reasoning', placement: 'local' }))
    expect(local.route.id).toBe('local-small')
    const cloud = accepted(router().decide({ ...REQ, kind: 'tool_use', placement: 'cloud' }))
    expect(cloud.route.trust).toBe('cloud')
    expect(cloud.excluded).toEqual([{ id: 'local-small', reason: expect.stringContaining('cloud route') }])
    const either = accepted(router().decide({ ...REQ, kind: 'tool_use', placement: 'either' }))
    expect(either.route.id).toBe('local-small')
  })

  test('context limit below input + output tokens excludes a route', () => {
    const decision = accepted(router().decide({ ...REQ, kind: 'tool_use', estimatedInputTokens: 7500, estimatedOutputTokens: 1000 }))
    expect(decision.route.id).not.toBe('local-small')
    expect(decision.excluded).toEqual([{ id: 'local-small', reason: expect.stringContaining('context limit 8000') }])
    // exactly at the limit fits
    expect(accepted(router().decide({ ...REQ, kind: 'tool_use', estimatedInputTokens: 7000, estimatedOutputTokens: 1000 })).route.id).toBe('local-small')
  })

  test('tier below minTier is excluded; unknown tier is excluded when a minimum is required', () => {
    const decision = accepted(router().decide({ ...REQ, kind: 'tool_use', minTier: 2 }))
    expect(decision.route.id).toBe('cloud-mid')
    expect(decision.excluded).toEqual([{ id: 'local-small', reason: expect.stringContaining('below the required tier 2') }])
    const { tier: _tier, ...noTier } = CLOUD_MID
    const unknown = refused(router({ routes: [noTier] }).decide({ ...REQ, minTier: 2 }))
    expect(unknown.excluded[0]?.reason).toContain('tier is unknown')
  })

  test('estimated cost above remainingUsd excludes a route', () => {
    // cloud-big: 1000 in * $3/M = $0.003 ; cloud-mid: $0.001
    const decision = accepted(router().decide({ ...REQ, kind: 'reasoning' }, { remainingUsd: 0.002 }))
    expect(decision.route.id).toBe('cloud-mid')
    expect(decision.excluded).toEqual([{ id: 'cloud-big', reason: expect.stringContaining('exceeds the remaining budget') }])
    const none = refused(router({ routes: [CLOUD_MID, CLOUD_BIG] }).decide(REQ, { remainingUsd: 0.0001 }))
    expect(none.excluded.map((item) => item.id)).toEqual(['cloud-mid', 'cloud-big'])
  })

  test('unknown price is not excluded by budget but the reason says so', () => {
    const unpriced: RouteCandidate = { id: 'unpriced', profile: 'p', model: 'm', trust: 'cloud', tier: 3 }
    const decision = accepted(router({ routes: [unpriced] }).decide({ ...REQ, kind: 'reasoning' }, { remainingUsd: 0 }))
    expect(decision.route.id).toBe('unpriced')
    expect(decision.excluded).toEqual([])
    expect(decision.reason).toContain('price unknown')
    expect(decision.estimatedCostUsd).toBeUndefined()
    // a free local route with no price stays allowed under a zero budget too
    expect(accepted(router().decide({ ...REQ, kind: 'tool_use' }, { remainingUsd: 0 })).route.id).toBe('local-small')
  })

  test('remainingCalls 0 excludes cloud routes only', () => {
    const decision = accepted(router().decide({ ...REQ, kind: 'reasoning' }, { remainingCalls: 0 }))
    expect(decision.route.id).toBe('local-small')
    expect(decision.excluded.map((item) => item.id)).toEqual(['cloud-mid', 'cloud-big'])
    expect(decision.excluded[0]?.reason).toContain('no cloud calls remain')
    expect(accepted(router().decide({ ...REQ, kind: 'reasoning' }, { remainingCalls: 1 })).route.id).toBe('cloud-big')
  })
})

describe('PolicyRouter: rules', () => {
  test('tool_use, summarize and analysis prefer a local route', () => {
    for (const kind of ['tool_use', 'summarize', 'analysis'] as const) {
      const decision = accepted(router().decide({ ...REQ, kind }))
      expect(decision.route.id).toBe('local-small')
      expect(decision.source).toBe('rules')
    }
  })

  test('local-first kinds fall back to the cheapest allowed route when no local fits', () => {
    const decision = accepted(router().decide({ ...REQ, kind: 'tool_use', minTier: 2 }))
    expect(decision.route.id).toBe('cloud-mid')
    expect(decision.reason).toContain('none is allowed')
  })

  test('reasoning, review and plan prefer the highest tier', () => {
    for (const kind of ['reasoning', 'review', 'plan'] as const) {
      expect(accepted(router().decide({ ...REQ, kind })).route.id).toBe('cloud-big')
    }
    // highest allowed tier when the best is excluded
    expect(accepted(router({ boundaryMode: 'private' }).decide({ ...REQ, kind: 'plan' })).route.id).toBe('local-small')
  })

  test('ties break on lower known cost, then declared order', () => {
    const cheap: RouteCandidate = { id: 'cheap', profile: 'p', model: 'm', trust: 'cloud', tier: 3, inputUsdPerMTok: 1, outputUsdPerMTok: 1 }
    const dear: RouteCandidate = { ...cheap, id: 'dear', inputUsdPerMTok: 9, outputUsdPerMTok: 9 }
    const unknown: RouteCandidate = { id: 'unknown', profile: 'p', model: 'm', trust: 'cloud', tier: 3 }
    expect(accepted(router({ routes: [dear, cheap] }).decide({ ...REQ, kind: 'review' })).route.id).toBe('cheap')
    expect(accepted(router({ routes: [unknown, dear] }).decide({ ...REQ, kind: 'review' })).route.id).toBe('dear')
    const twin: RouteCandidate = { ...unknown, id: 'twin' }
    expect(accepted(router({ routes: [twin, unknown] }).decide({ ...REQ, kind: 'review' })).route.id).toBe('twin')
    expect(accepted(router({ routes: [unknown, twin] }).decide({ ...REQ, kind: 'review' })).route.id).toBe('unknown')
  })

  test('kinds without a preference pick the lowest known cost, then declared order', () => {
    // a known price beats an unknown one (never invent a price for the local route)
    expect(accepted(router().decide({ ...REQ, kind: 'research' })).route.id).toBe('cloud-mid')
    expect(accepted(router({ routes: [CLOUD_BIG, CLOUD_MID] }).decide({ ...REQ, kind: 'research' })).route.id).toBe('cloud-mid')
    const a: RouteCandidate = { id: 'a', profile: 'p', model: 'm', trust: 'cloud' }
    const b: RouteCandidate = { ...a, id: 'b' }
    expect(accepted(router({ routes: [a, b] }).decide({ ...REQ, kind: 'research' })).route.id).toBe('a')
    expect(accepted(router({ routes: [b, a] }).decide({ ...REQ, kind: 'research' })).route.id).toBe('b')
  })

  test('is deterministic and does not mutate its inputs', () => {
    const r = router()
    const first = accepted(r.decide({ ...REQ, kind: 'reasoning' }))
    const second = accepted(r.decide({ ...REQ, kind: 'reasoning' }))
    expect(second.route).toEqual(first.route)
    expect(second.reason).toBe(first.reason)
  })

  test('reason is one sentence without newlines', () => {
    expect(accepted(router().decide(REQ)).reason).not.toContain('\n')
  })
})

describe('PolicyRouter: estimate and timing', () => {
  test('estimatedCostUsd only when both prices are known', () => {
    const priced = accepted(router().decide({ ...REQ, kind: 'reasoning', estimatedInputTokens: 100_000, estimatedOutputTokens: 10_000 }))
    expect(priced.route.id).toBe('cloud-big')
    expect(priced.estimatedCostUsd).toBeCloseTo(0.45)
    const local = accepted(router().decide({ ...REQ, kind: 'tool_use' }))
    expect(local).not.toHaveProperty('estimatedCostUsd')
    const half: RouteCandidate = { id: 'half', profile: 'p', model: 'm', trust: 'cloud', inputUsdPerMTok: 1 }
    expect(accepted(router({ routes: [half] }).decide(REQ)).estimatedCostUsd).toBeUndefined()
  })

  test('decisionMs is measured with the injected clock', () => {
    let t = 0
    const now = () => {
      t += 5
      return t
    }
    expect(accepted(router({ now }).decide(REQ)).decisionMs).toBe(5)
    expect(refused(router({ now, routes: [] }).decide(REQ)).decisionMs).toBe(5)
    expect(accepted(router({ now, mode: 'off' }).decide(REQ, undefined, LOCAL)).decisionMs).toBe(5)
  })

  test('default clock yields a non-negative decisionMs', () => {
    expect(accepted(router().decide(REQ)).decisionMs).toBeGreaterThanOrEqual(0)
  })
})

describe('PolicyRouter: learned', () => {
  const workload = 'csv-reconcile'
  const hardwareKey = 'm3-max-64'
  const req: TaskRequirements = { ...REQ, kind: 'reasoning', workload }

  test('picks the highest acceptRate, then lower cost, then lower wall time', () => {
    const stats = [
      stat('local-small', 5, 0.6, 900),
      stat('cloud-mid', 4, 0.9, 2000, 0.02),
      stat('cloud-big', 4, 0.9, 3000, 0.05),
    ]
    const calibration = lookup(stats)
    const decision = accepted(router({ mode: 'learned', calibration, hardwareKey }).decide(req))
    expect(decision.source).toBe('learned')
    expect(decision.route.id).toBe('cloud-mid')
    expect(decision.reason).toContain('90%')
    expect(calibration.calls).toEqual([`${workload}@${hardwareKey}`])

    const sameCost = [stat('cloud-mid', 3, 0.8, 4000, 0.01), stat('cloud-big', 3, 0.8, 1000, 0.01)]
    expect(accepted(router({ mode: 'learned', calibration: lookup(sameCost), hardwareKey }).decide(req)).route.id).toBe('cloud-big')

    const unknownCost = [stat('cloud-mid', 3, 0.8, 1000), stat('cloud-big', 3, 0.8, 5000, 0.5)]
    expect(accepted(router({ mode: 'learned', calibration: lookup(unknownCost), hardwareKey }).decide(req)).route.id).toBe('cloud-big')
  })

  test('only considers allowed routes, never one the filter excluded', () => {
    const stats = [stat('cloud-big', 50, 1, 100, 0.001), stat('cloud-mid', 50, 0.95, 100, 0.001), stat('local-small', 3, 0.2, 5000)]
    const decision = accepted(router({ mode: 'learned', calibration: lookup(stats), hardwareKey, boundaryMode: 'private' }).decide(req))
    expect(decision.route.id).toBe('local-small')
    expect(decision.source).toBe('learned')
    expect(decision.excluded.map((item) => item.id)).toEqual(['cloud-mid', 'cloud-big'])
  })

  test('falls back to rules with a reason when no allowed route has 3 runs', () => {
    const stats = [stat('cloud-big', 2, 1, 100), stat('cloud-mid', 2, 1, 100), stat('unknown-id', 99, 1, 1)]
    const decision = accepted(router({ mode: 'learned', calibration: lookup(stats), hardwareKey }).decide(req))
    expect(decision.source).toBe('rules')
    expect(decision.route.id).toBe('cloud-big')
    expect(decision.reason).toContain('not enough calibration data')
  })

  test('stats only for excluded routes also fall back to rules', () => {
    const stats = [stat('cloud-big', 50, 1, 100)]
    const decision = accepted(router({ mode: 'learned', calibration: lookup(stats), hardwareKey, boundaryMode: 'private' }).decide(req))
    expect(decision.source).toBe('rules')
    expect(decision.route.id).toBe('local-small')
  })

  test('falls back to rules without a workload, calibration or hardware key', () => {
    const stats = [stat('cloud-mid', 10, 1, 100)]
    const noWorkload = accepted(router({ mode: 'learned', calibration: lookup(stats), hardwareKey }).decide({ ...req, workload: undefined as never }))
    expect(noWorkload.source).toBe('rules')
    expect(accepted(router({ mode: 'learned', hardwareKey }).decide(req)).source).toBe('rules')
    expect(accepted(router({ mode: 'learned', calibration: lookup(stats) }).decide(req)).source).toBe('rules')
  })

  test('no allowed route stays ok:false in learned mode', () => {
    const stats = [stat('cloud-big', 50, 1, 100)]
    const decision = refused(router({ mode: 'learned', routes: [CLOUD_BIG], calibration: lookup(stats), hardwareKey, boundaryMode: 'private' }).decide(req))
    expect(decision.code).toBe('no_allowed_route')
  })
})

describe('renderRouteDecision', () => {
  test('one line for the choice plus one per exclusion', () => {
    const decision = router({ boundaryMode: 'private' }).decide({ ...REQ, kind: 'reasoning' })
    const lines = renderRouteDecision(decision).split('\n')
    expect(lines[0]).toContain('route local-small')
    expect(lines[0]).toContain('via rules')
    expect(lines.slice(1)).toEqual([
      expect.stringContaining('excluded cloud-mid'),
      expect.stringContaining('excluded cloud-big'),
    ])
  })

  test('shows the estimated cost when known and the failure for no route', () => {
    const priced = renderRouteDecision(router().decide({ ...REQ, kind: 'reasoning' }))
    expect(priced).toContain('est. $0.0030')
    const none = renderRouteDecision(router({ routes: [CLOUD_BIG], boundaryMode: 'private' }).decide(REQ))
    expect(none.startsWith('no allowed route:')).toBe(true)
  })
})
