// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ValidationError } from '../core/errors.js'
import type { ToolExecutionContext } from '../executors/toolRegistry.js'
import { ToolRegistry } from '../executors/toolRegistry.js'
import type { JsonObject, ToolDefinition } from '../types/toolCalls.js'
import { appendContextDelta, contextDeltaFor } from './contextDeltas.js'
import { resolveInteractionMode, type InteractionMode } from './interactionModes.js'

export interface InteractionModeChange {
  readonly mode: InteractionMode
  readonly planMode: boolean
}

export interface InteractionModeToolHost {
  setMode(request: {
    readonly context: ToolExecutionContext
    readonly mode: InteractionMode
    readonly reason: string
  }): InteractionModeChange | Promise<InteractionModeChange>
}

export const INTERACTION_MODE_TOOL_DEFINITION: ToolDefinition = Object.freeze({
  type: 'function',
  function: {
    name: 'SetInteractionModeTool',
    description: 'Schedule code, researcher, plan, or objective mode for the next user turn.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        mode: {
          type: 'string',
          enum: ['code', 'researcher', 'plan', 'objective'],
          description: 'New interaction mode for the active session.',
        },
        reason: {
          type: 'string',
          description: 'Optional concise reason for the transition.',
        },
      },
      required: ['mode'],
    },
  },
})

/**
 * Plan-mode entry and exit, bound to the same live session as the mode tool.
 *
 * These names also exist in the Claude workflow tool set, where they only ever
 * flipped a WorkflowState object that no policy code reads — so the model was
 * told "Exited plan mode. Resuming normal execution." while the session stayed
 * in plan mode and kept refusing to write. Registering them here, against a
 * real host, means the workflow registration's already-registered guard skips
 * its inert copies; the WorkflowState pair survives only for embeddings that
 * never install a session host at all.
 */
const PLAN_MODE_TOOL_DEFINITIONS: Readonly<Record<string, { readonly definition: ToolDefinition; readonly mode: InteractionMode }>> =
  Object.freeze({
    EnterPlanModeTool: {
      mode: 'plan',
      definition: Object.freeze({
        type: 'function',
        function: {
          name: 'EnterPlanModeTool',
          description: 'Enter plan mode for the active session. Research and planning only; mutations are refused.',
          parameters: { type: 'object', additionalProperties: false, properties: {} },
        },
      }),
    },
    ExitPlanModeTool: {
      mode: 'code',
      definition: Object.freeze({
        type: 'function',
        function: {
          name: 'ExitPlanModeTool',
          description:
            'Leave plan mode and return the active session to code mode. Call this once the user has approved the plan.',
          parameters: { type: 'object', additionalProperties: false, properties: {} },
        },
      }),
    },
  })

const MODE_TOOL_CAPABILITIES = Object.freeze({
  concurrencySafe: false,
  defer: false,
  destructive: false,
  openWorld: false,
  readOnly: false,
})

/**
 * Apply a mode transition to the live session and describe it honestly.
 *
 * The host commits the change immediately, but the running turn keeps the tool
 * list it was built with, so the two facts have to be stated separately. The
 * old copy claimed the change was "scheduled for the next turn" while the
 * session had already moved — wrong in one direction — and then the enforced
 * policy really did lag by a turn — wrong in the other.
 */
async function applyMode(
  host: InteractionModeToolHost,
  context: ToolExecutionContext,
  mode: InteractionMode,
  reason: string,
): Promise<Record<string, unknown>> {
  assertMainAgentContext(context)
  const previousMode = resolveInteractionMode(context.metadata.interaction_mode)
  const changed = await host.setMode({ context, mode, reason })
  const delta = contextDeltaFor(previousMode, changed.mode, Date.now(), 'interaction-mode')
  if (delta) appendContextDelta(context.metadata, delta)
  // Recorded as PENDING, never written over `interaction_mode`. The session
  // itself has already moved — the host commits immediately and the daemon
  // announces it — but the running turn's policy is keyed off this metadata,
  // and the objective gate in particular re-reads it. Flipping it here makes a
  // turn start enforcing a mode it was not built with, mid-flight.
  context.metadata.pending_interaction_mode = changed.mode
  return {
    mode: changed.mode,
    plan_mode: changed.planMode,
    ...(reason ? { reason } : {}),
    message: `Interaction mode is now ${changed.mode}.${reason ? ` Reason: ${reason}` : ''}`,
    guidance: `The session is in ${changed.mode} mode from now on. This turn keeps the tool list it started with, `
      + `so finish the current turn with the tools you already have; ${changed.mode} mode's tool policy applies from `
      + 'the next turn.',
  }
}

/** Register the model-facing mode tools against a host-owned live session. */
export function registerInteractionModeTool(
  registry: ToolRegistry,
  host: InteractionModeToolHost,
  agentId = 'default',
): void {
  registry.replace(INTERACTION_MODE_TOOL_DEFINITION, async (inputs, context) => {
    const mode = requiredMode(inputs)
    return applyMode(host, context, mode, optionalString(inputs.reason))
  }, agentId, MODE_TOOL_CAPABILITIES)

  for (const { definition, mode } of Object.values(PLAN_MODE_TOOL_DEFINITIONS)) {
    registry.replace(
      definition,
      async (_inputs, context) => applyMode(host, context, mode, ''),
      agentId,
      MODE_TOOL_CAPABILITIES,
    )
  }
}

/** Fail closed when a host accidentally exposes this main-session control to a child. */
function assertMainAgentContext(context: ToolExecutionContext): void {
  const sessionKind = optionalString(context.metadata.session_kind).toLowerCase()
  const subagentId = optionalString(context.metadata.subagent_id)
  if (sessionKind === 'subagent' || subagentId) {
    throw new ValidationError(
      'context',
      'only the main agent may schedule interaction-mode transitions',
      sessionKind || subagentId,
    )
  }
}

function requiredMode(inputs: JsonObject): InteractionMode {
  const value = optionalString(inputs.mode)
  const mode = resolveInteractionMode(value)
  if (!value || mode === undefined) {
    throw new ValidationError('mode', 'must be code, researcher, plan, or objective', inputs.mode)
  }
  return mode
}

function optionalString(value: unknown): string {
  return typeof value === 'string' ? value.trim() : ''
}
