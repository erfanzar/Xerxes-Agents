// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * PlanTool's hybrid placement (M3.1). Off by default (`runtime.hybrid_plans`,
 * XERXES_HYBRID_PLANS): with it off this returns undefined and PlanTool plans
 * and runs exactly as before. With it on, every generated step is placed
 * local / cloud / either, and a plan the data boundary cannot run safely is
 * refused before any agent starts.
 */

import type { ProviderProfile } from '../bridge/profiles.js'
import { analyzeSplit, assignPlacements, parseHybridPlansConfig, validateHybridPlan } from '../cortex/hybridPlan.js'
import type { ExecutionPlan } from '../cortex/planner.js'
import { LOCAL_PROFILE_NAME } from '../local/command.js'
import { parseRoutingConfig } from '../routing/config.js'
import { deriveDataTrust, parseDataBoundaryConfig } from '../security/dataBoundary.js'
import type { WorkflowHybridPlacement, WorkflowPlanStep } from '../tools/claudeTools/workflow.js'

export interface HybridPlacementOptions {
  readonly runtime: () => Record<string, unknown>
  readonly env?: Record<string, string | undefined>
  /** Workspace of the session calling PlanTool (for the project's data policy). */
  readonly cwd: () => string
  readonly profiles: { get(name: string): ProviderProfile | undefined }
}

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

/** The model local steps run on: the first local routing route, else the `bonsai-local` profile. */
function localRoute(
  runtime: Record<string, unknown>,
  env: Record<string, string | undefined>,
  profiles: HybridPlacementOptions['profiles'],
): { readonly providerProfile: string; readonly model: string } | undefined {
  for (const route of parseRoutingConfig(runtime, env).routes) {
    const profile = profiles.get(route.profile)
    if (!profile) continue
    if ((route.trust ?? deriveDataTrust(profile)) === 'local') return { providerProfile: route.profile, model: route.model }
  }
  const managed = profiles.get(LOCAL_PROFILE_NAME)
  return managed && deriveDataTrust(managed) === 'local' ? { providerProfile: managed.name, model: managed.model } : undefined
}

export function createHybridPlacement(
  options: HybridPlacementOptions,
): (objective: string, steps: readonly WorkflowPlanStep[]) => WorkflowHybridPlacement | undefined {
  return (objective, steps) => {
    const runtime = options.runtime()
    const env = options.env ?? process.env
    if (!parseHybridPlansConfig(runtime, env)) return undefined
    const boundaryMode = parseDataBoundaryConfig(runtime, env, { root: options.cwd() }).mode
    const plan: ExecutionPlan = {
      id: 'plantool',
      objective,
      complexity: 'medium',
      estimatedMinutes: 0,
      steps: steps.map(step => ({
        id: step.id,
        action: step.description,
        description: step.description,
        agentId: step.agent,
        arguments: {},
        dependencies: [...step.depends],
      })),
    }
    const refused = (refusal: string): WorkflowHybridPlacement => ({ placements: new Map(), notes: [], refusal })
    let placed: ReturnType<typeof assignPlacements>
    try {
      placed = assignPlacements(plan, { boundaryMode, inputSensitivity: () => 'shareable' })
    } catch (error) {
      return refused(`the plan could not be placed: ${errorText(error)}`)
    }
    const valid = validateHybridPlan(placed.plan, boundaryMode)
    if (!valid.ok) return refused(valid.errors.join('; '))
    const split = analyzeSplit(placed.plan)
    const localAgent = localRoute(runtime, env, options.profiles)
    const needsLocal = placed.plan.steps.some(step => step.placement === 'local')
    const refusal = boundaryMode !== 'off' && needsLocal && !localAgent
      ? 'the data boundary needs local steps to run on a local model, but none is configured (add a local entry to runtime.routing_routes or run `xerxes local setup`)'
      : undefined
    return {
      placements: new Map(placed.plan.steps.map(step => [step.id, { placement: step.placement, kind: step.kind }])),
      notes: [...placed.notes, split.reason],
      ...(localAgent ? { localAgent } : {}),
      ...(refusal ? { refusal } : {}),
    }
  }
}
