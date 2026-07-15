// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ValidationError } from '../core/errors.js'
import type { ToolExecutionContext } from '../executors/toolRegistry.js'
import { ToolRegistry } from '../executors/toolRegistry.js'
import type { JsonObject, ToolDefinition } from '../types/toolCalls.js'
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

/** Register the model-facing mode tool against a host-owned live session. */
export function registerInteractionModeTool(
  registry: ToolRegistry,
  host: InteractionModeToolHost,
  agentId = 'default',
): void {
  registry.replace(INTERACTION_MODE_TOOL_DEFINITION, async (inputs, context) => {
    const mode = requiredMode(inputs)
    const reason = optionalString(inputs.reason)
    const currentMode = resolveInteractionMode(context.metadata.interaction_mode)
    if (currentMode && currentMode !== 'code' && mode !== currentMode) {
      throw new ValidationError(
        'mode',
        `cannot be changed by the model while ${currentMode} mode is active; the user or session host must switch modes`,
        mode,
      )
    }
    const changed = await host.setMode({ context, mode, reason })
    context.metadata.pending_interaction_mode = changed.mode
    return {
      mode: changed.mode,
      plan_mode: changed.planMode,
      ...(reason ? { reason } : {}),
      message: `Interaction mode ${changed.mode} is scheduled for the next turn.${reason ? ` Reason: ${reason}` : ''}`,
      guidance: `Finish the current turn under its existing mode. ${changed.mode} mode and its enforced tool policy apply on the next user turn.`,
    }
  }, agentId)
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
