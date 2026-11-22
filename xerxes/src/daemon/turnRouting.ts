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
import { contextCalibrationRatio, estimateContextTokens } from '../context/windowUsage.js'
import type { LlmClient } from '../llms/client.js'
import type { ProviderOverrides } from '../llms/providerRegistry.js'
import { parseRoutingConfig } from '../routing/config.js'
import { PolicyRouter, renderRouteDecision } from '../routing/router.js'
import type { BudgetState, DataBoundaryMode, DataTrust, StepKind, TaskRequirements } from '../routing/types.js'
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
  readonly requestScaffold?: {
    readonly systemPrompt?: string | undefined
    readonly toolSchemas?: readonly Readonly<Record<string, unknown>>[] | undefined
  } | undefined
}

export interface TurnRouterOptions<S extends TurnRoutingSession> {
  readonly runtime: () => Record<string, unknown>
  readonly env?: Record<string, string | undefined>
  readonly profiles: { get(name: string): ProviderProfile | undefined }
  readonly boundaryMode: (session: S) => DataBoundaryMode
  /** Whether this session's context already holds private data (boundary taint). */
  readonly touchesPrivate?: (session: S) => boolean
  /** Optional remaining USD / cloud-call budget consulted during route selection. */
  readonly budgetForSession?: (session: S) => BudgetState | undefined
  /** Optional expected reply token reserve so context-limit exclusion includes output headroom. */
  readonly estimatedOutputTokens?: (session: S, model: string) => number | undefined
  readonly buildProvider: (profile: ProviderProfile, model: string) => RoutedProvider
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

function asContextMessage(message: unknown): Readonly<Record<string, unknown>> {
  if (typeof message === 'string') return { role: 'user', content: message }
  if (message !== null && typeof message === 'object' && !Array.isArray(message)) {
    return message as Readonly<Record<string, unknown>>
  }
  return { role: 'user', content: String(message ?? '') }
}

/**
 * Estimate input tokens using the shared context estimator so inline base64
 * screenshots count as vision tiles (~1.6K tokens) rather than raw base64 text,
 * and request scaffolding + provider calibration are honored when available.
 */
export function estimateTurnRoutingTokens(session: TurnRoutingSession, model: string, text: string): number {
  const normalizedMessages = [
    ...session.messages.map(asContextMessage),
    ...(text ? [{ role: 'user', content: text }] : []),
  ]
  const rawEstimate = estimateContextTokens(normalizedMessages, {
    model,
    ...(session.requestScaffold?.systemPrompt ? { systemPrompt: session.requestScaffold.systemPrompt } : {}),
    ...(session.requestScaffold?.toolSchemas?.length ? { toolSchemas: session.requestScaffold.toolSchemas } : {}),
  })
  return Math.ceil(rawEstimate * contextCalibrationRatio(session.metadata, model))
}

/**
 * Build the turn runner's `routeForSession`. Returns undefined when routing is
 * off, so the runner keeps today's selection byte for byte.
 */
export function createTurnRouter<S extends TurnRoutingSession>(
  options: TurnRouterOptions<S>,
): (session: S, model: string, text: string) => TurnRoute | undefined {
  return (session, model, text) => {
    const routing = parseRoutingConfig(options.runtime(), options.env ?? process.env)
    if (routing.mode === 'off') return undefined
    const router = new PolicyRouter({
      mode: routing.mode,
      routes: routing.routes,
      boundaryMode: options.boundaryMode(session),
      trustOf: profile => {
        const found = options.profiles.get(profile)
        return found ? deriveDataTrust(found) : undefined
      },
    })
    const kind = turnKind(text)
    const estimatedOutput = options.estimatedOutputTokens?.(session, model)
    const request: TaskRequirements = {
      kind,
      touchesPrivate: options.touchesPrivate?.(session) ?? false,
      estimatedInputTokens: estimateTurnRoutingTokens(session, model, text),
      ...(estimatedOutput === undefined ? {} : { estimatedOutputTokens: estimatedOutput }),
      needsTools: true,
    }
    const decision = router.decide(request, options.budgetForSession?.(session))
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
