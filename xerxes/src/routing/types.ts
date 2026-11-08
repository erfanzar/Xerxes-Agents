// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Shared vocabulary for M2 (routing, context preparation, escalation) and M3
 * (hybrid planning, shared execution state, calibration). Types only: every
 * slice codes against these so they can be built in parallel. Every feature
 * that uses them is OFF by default.
 */

import type { DataBoundaryMode, DataTrust } from '../security/boundaryTypes.js'

export type { DataBoundaryMode, DataTrust }

/** `off` keeps today's fixed model selection exactly. */
/** A learned mode, choosing by recorded outcomes, returns when outcomes are recorded (M3). */
export type RoutingMode = 'off' | 'rules'

/** What kind of work a step or call is; rules and calibration key on it. */
export type StepKind = 'plan' | 'tool_use' | 'analysis' | 'reasoning' | 'review' | 'summarize' | 'research'

/** Where a step may run. `either` lets the router choose among allowed routes. */
export type Placement = 'local' | 'cloud' | 'either'

/** Data class of an artifact or input. Private never reaches a cloud-trust worker. */
export type Sensitivity = 'private' | 'shareable'

/** One model route the router may pick: a provider profile + model. */
export interface RouteCandidate {
  readonly id: string
  readonly profile: string
  readonly model: string
  readonly trust: DataTrust
  /** USD per million tokens; absent means unknown (never invented). */
  readonly inputUsdPerMTok?: number
  readonly outputUsdPerMTok?: number
  readonly contextLimit?: number
  /** Relative capability the user or catalog assigned (1 = small local, 3 = frontier). */
  readonly tier?: 1 | 2 | 3
}

export interface TaskRequirements {
  readonly kind: StepKind
  /** The step's context includes private data (taint, private paths). */
  readonly touchesPrivate: boolean
  readonly estimatedInputTokens: number
  readonly estimatedOutputTokens?: number
  readonly needsTools: boolean
  readonly minTier?: 1 | 2 | 3
  readonly placement?: Placement
}

export interface BudgetState {
  readonly remainingUsd?: number
  readonly remainingCalls?: number
}

export interface RouteExclusion {
  readonly id: string
  readonly reason: string
}

export type RouteDecision =
  | {
    readonly ok: true
    readonly route: RouteCandidate
    readonly source: 'fixed' | 'rules'
    readonly reason: string
    readonly excluded: readonly RouteExclusion[]
    /** Time the decision itself took; routing overhead is counted, not hidden. */
    readonly decisionMs: number
    readonly estimatedCostUsd?: number
  }
  | {
    readonly ok: false
    readonly code: 'no_allowed_route'
    readonly reason: string
    readonly excluded: readonly RouteExclusion[]
    readonly decisionMs: number
  }

/** Aggregated outcomes of one strategy on one workload + hardware. */
export interface StrategyStats {
  readonly strategy: string
  readonly runs: number
  readonly accepted: number
  readonly acceptRate: number
  readonly medianWallMs: number
  /** Mean cost over runs that reported one; undefined when none did. */
  readonly meanCostUsd?: number
}

/** One evaluated outcome, recorded locally. Never contains task text. */
export interface CalibrationRecord {
  readonly at: string
  readonly workload: string
  readonly hardwareKey: string
  /** Route id, or a strategy name such as 'local-only' / 'local+advisor'. */
  readonly strategy: string
  readonly accepted: boolean
  readonly wallMs: number
  readonly costUsd?: number
  readonly retries?: number
}

/** Read side of the calibration store, consumed by the learned router. */
export interface CalibrationLookup {
  stats(workload: string, hardwareKey: string): readonly StrategyStats[]
}
