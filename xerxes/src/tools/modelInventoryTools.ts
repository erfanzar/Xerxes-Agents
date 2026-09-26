// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import type { AgentDefinition } from '../agents/definitions.js'
import type { ToolRegistry } from '../executors/toolRegistry.js'
import type { JsonObject } from '../types/toolCalls.js'
export type ModelInventoryHost = (sessionId: string, params: JsonObject, signal?: AbortSignal) => Promise<unknown>
export function registerModelInventoryTool(registry: ToolRegistry, host: ModelInventoryHost): void {
  registry.register({ type: 'function', function: { name: 'list_available_models', description: 'Discover configured providers and actual model choices before delegating to an agent. Without provider_profile, lists configured profiles. With it, discovers models and runtime-supported reasoning levels, context and output capacities. Set include_usage with provider_profile to request profile-bound subscription usage; unavailable quota stays unknown. Use returned revision with subsequent page offsets; discovery does not change the conversation model. Skip it when the configured intelligence tiers fit the task. Use profile, model and reasoning names exactly as returned, never from memory.', parameters: { type: 'object', additionalProperties: false, properties: {
    include_usage: { type: 'boolean' },
    provider_profile: { type: 'string', maxLength: 512 }, query: { type: 'string', maxLength: 512, description: 'Case-insensitive substring filter: on model ids with provider_profile, otherwise on profile name, provider, and model.' },
    offset: { type: 'integer', minimum: 0, description: 'Omit or use 0 for a fresh inventory. For another page, use next_offset from the previous response.' }, limit: { type: 'integer', minimum: 1, maximum: 50, description: 'Page size.' }, revision: { type: 'string', maxLength: 512, description: 'Only needed when offset is greater than 0. Copy the previous response revision for the same provider_profile and query. Offset 0 always refreshes, ignoring this token.' },
  } } } }, async (args, context, signal) => {
    if (!context.sessionId?.trim()) throw new Error('Model inventory requires a session')
    signal?.throwIfAborted()
    return JSON.stringify(await host(context.sessionId, args, signal))
  }, 'default', { concurrencySafe: true, destructive: false, openWorld: false, readOnly: true, maxResultBytes: 64000 })
}

/** Only widen built-in catalogs; user-defined allow/exclude rules still apply. */
export function addModelInventoryToBuiltinAgents(definitions: Map<string, AgentDefinition>): void {
  for (const [name, definition] of definitions) {
    if (definition.source === 'built-in') definitions.set(name, { ...definition, tools: [...new Set([...definition.tools, 'list_available_models'])] })
  }
}
