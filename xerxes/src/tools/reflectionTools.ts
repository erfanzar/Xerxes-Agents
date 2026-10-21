// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { ToolRegistry } from '../executors/toolRegistry.js'
import {
  ReflectionError, type ReflectionApplier, type ReflectionProposalStore,
} from '../runtime/reflection.js'
import type { ToolDefinition } from '../types/toolCalls.js'

export interface ReflectionToolsOptions {
  readonly store: ReflectionProposalStore
  readonly activeWorkspace: () => string
  /** Host-bound to the active workspace. Skill writes must use the linted skill_manage path. */
  readonly applier: ReflectionApplier
}

export const REFLECTION_TOOL_DEFINITIONS: readonly ToolDefinition[] = [
  {
    type: 'function', function: {
      name: 'reflection_list',
      description: 'List reviewable post-goal improvement proposals in this workspace, with full evidence and concrete proposed content. Read-only: nothing is applied. Inspect the exact id, rationale, evidence and content before asking the human to approve.',
      parameters: { type: 'object', additionalProperties: false, properties: {} },
    },
  },
  {
    type: 'function', function: {
      name: 'reflection_review',
      description: 'Review one exact proposal from reflection_list with a reason. Approval requires explicit human authority and the normal permission gate; automatic goal rounds cannot approve. Skill/agent edits go through host validation; plugin ideas are only acknowledged with a /plugin-creator hint, never written. Non-pending proposals cannot be applied again.',
      parameters: {
        type: 'object', additionalProperties: false, required: ['id', 'action', 'reason'], properties: {
          id: { type: 'string', minLength: 1, description: 'Exact proposal id copied from reflection_list.' },
          action: { type: 'string', enum: ['approve', 'reject'] },
          reason: { type: 'string', minLength: 1, description: 'Human review decision or concrete rejection reason, retained in the audit log.' },
        },
      },
    },
  },
]

/** Turn authority comes only from the host's turn metadata, never model-supplied arguments. */
export function registerReflectionTools(registry: ToolRegistry, options: ReflectionToolsOptions): void {
  for (const definition of REFLECTION_TOOL_DEFINITIONS) {
    const readOnly = definition.function.name === 'reflection_list'
    registry.register(definition, async (inputs, context) => {
      try {
        const workspace = options.activeWorkspace()
        if (readOnly) return { ok: true, proposals: await options.store.list(workspace) }
        const id = text(inputs.id, 'id')
        const reason = text(inputs.reason, 'reason')
        const action = text(inputs.action, 'action')
        if (action !== 'approve' && action !== 'reject') throw new ReflectionError('Reflection action must be approve or reject.', 'REFLECTION_INVALID_INPUT')
        if (action === 'approve' && context.metadata.goal_turn_human === false) {
          throw new ReflectionError('Approving a reflection requires a direct human turn and explicit human approval; an automatic goal round cannot approve its own improvement.', 'REFLECTION_HUMAN_APPROVAL_REQUIRED')
        }
        const proposal = action === 'approve'
          ? await options.store.approve(id, options.applier, { workspace, reason })
          : await options.store.reject(id, reason, workspace)
        return { ok: proposal.status !== 'failed', proposal }
      } catch (error) {
        if (!(error instanceof ReflectionError)) throw error
        return { ok: false, code: error.code, error: error.message }
      }
    }, 'default', {
      readOnly, destructive: !readOnly, concurrencySafe: readOnly, defer: true,
      openWorld: false, interruptBehavior: readOnly ? 'cancel' : 'block',
    })
  }
}

function text(value: unknown, field: string): string {
  if (typeof value !== 'string' || !value.trim()) throw new ReflectionError(`Reflection ${field} must be nonempty text.`, 'REFLECTION_INVALID_INPUT')
  return value.trim()
}
