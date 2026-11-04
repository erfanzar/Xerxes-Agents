// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Per-turn model routing (M2.1) for the daemon turn runner.
 *
 * Off by default: with `runtime.routing` unset the factory returns undefined
 * and the turn keeps its configured model. When on, the policy router picks a
 * route among those the data boundary allows; a turn with no allowed route is
 * refused with the reason, never sent to a disallowed model.
 */

import type { ProviderProfile } from '../bridge/profiles.js'
import type { LlmClient } from '../llms/client.js'
import type { ProviderOverrides } from '../llms/providerRegistry.js'
import { parseRoutingConfig } from '../routing/config.js'
import { PolicyRouter, renderRouteDecision } from '../routing/router.js'
import type { CalibrationLookup, DataBoundaryMode, DataTrust, StepKind, TaskRequirements } from '../routing/types.js'
import { deriveDataTrust } from '../security/dataBoundary.js'

/** The provider half of a routed turn, the same shape `resolveSessionProvider` returns. */
export interface RoutedProvider {
  readonly llm: LlmClient
  readonly createLlmForModel?: (model: string) => LlmClient
  readonly providerOverrides?: ProviderOverrides
  readonly contextLimit?: number | undefined
  readonly maxOutputTokens?: (model: string) => number | undefined
}

export type TurnRoute =
  | {
    readonly ok: true
    readonly model: string
    readonly routeId: string
    /** Provider profile name and its data trust: the outbound gate judges this route, not the session default. */
    readonly profile: string
    readonly trust: DataTrust
    readonly provider: RoutedProvider
    readonly reason: string
  }
  | { readonly ok: false; readonly reason: string }

export interface TurnRoutingSession {
  readonly cwd: string
  readonly messages: readonly unknown[]
  readonly metadata: Record<string, unknown>
}

export interface TurnRouterOptions<S extends TurnRoutingSession> {
  readonly runtime: () => Record<string, unknown>
  readonly env?: Record<string, string | undefined>
  readonly profiles: { get(name: string): ProviderProfile | undefined }
  readonly boundaryMode: (session: S) => DataBoundaryMode
  /** Whether this session's context already holds private data (boundary taint). */
  readonly touchesPrivate?: (session: S) => boolean
  readonly buildProvider: (profile: ProviderProfile, model: string) => RoutedProvider
  readonly calibration?: CalibrationLookup
  readonly hardwareKey?: () => string | undefined
}

const KIND_HINTS: readonly (readonly [StepKind, RegExp])[] = [
  ['review', /\b(review|audit|critique|check (?:this|my))\b/i],
  ['plan', /\b(plan|design|architect|break (?:it|this) down)\b/i],
  ['research', /\b(research|search|look up|find sources|cite)\b/i],
  ['summarize', /\b(summari[sz]e|tl;?dr|recap)\b/i],
  ['reasoning', /\b(why|prove|derive|reason about|explain how)\b/i],
]

/** A coarse step kind from the user's words; anything else is analysis. */
export function turnKind(text: string): StepKind {
  for (const [kind, pattern] of KIND_HINTS) if (pattern.test(text)) return kind
  return 'analysis'
}

/** chars/4, the same estimate the prompt budget uses; bounded by what is cheap to measure. */
function estimateTokens(session: TurnRoutingSession, text: string): number {
  let chars = text.length
  for (const message of session.messages) {
    chars += typeof message === 'string' ? message.length : JSON.stringify(message ?? '').length
    if (chars > 4_000_000) break
  }
  return Math.ceil(chars / 4)
}

/**
 * Build the turn runner's `routeForSession`. Returns undefined when routing is
 * off, so the runner keeps today's selection byte for byte.
 */
export function createTurnRouter<S extends TurnRoutingSession>(
  options: TurnRouterOptions<S>,
): (session: S, model: string, text: string) => TurnRoute | undefined {
  return (session, _model, text) => {
    const routing = parseRoutingConfig(options.runtime(), options.env ?? process.env)
    if (routing.mode === 'off') return undefined
    const hardwareKey = options.hardwareKey?.()
    const router = new PolicyRouter({
      mode: routing.mode,
      routes: routing.routes,
      boundaryMode: options.boundaryMode(session),
      trustOf: profile => {
        const found = options.profiles.get(profile)
        return found ? deriveDataTrust(found) : undefined
      },
      ...(options.calibration ? { calibration: options.calibration } : {}),
      ...(hardwareKey ? { hardwareKey } : {}),
    })
    const kind = turnKind(text)
    const workload = typeof session.metadata.workload === 'string' ? session.metadata.workload : undefined
    const request: TaskRequirements = {
      kind,
      touchesPrivate: options.touchesPrivate?.(session) ?? false,
      estimatedInputTokens: estimateTokens(session, text),
      needsTools: true,
      ...(workload ? { workload } : {}),
    }
    const decision = router.decide(request)
    if (!decision.ok) return { ok: false, reason: renderRouteDecision(decision) }
    const profile = options.profiles.get(decision.route.profile)
    if (!profile) return { ok: false, reason: `route ${decision.route.id} names provider profile "${decision.route.profile}", which does not exist` }
    return {
      ok: true,
      model: decision.route.model,
      routeId: decision.route.id,
      profile: profile.name,
      trust: decision.route.trust,
      provider: options.buildProvider(profile, decision.route.model),
      reason: renderRouteDecision(decision),
    }
  }
}
