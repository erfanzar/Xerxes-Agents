// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { ChatMessage } from '../types/messages.js'
import type { ToolDefinition } from '../types/toolCalls.js'
import { piCatalogModelCapabilities } from './piModelCatalog.js'

/**
 * pi-ai's deferred-tool split (utils/deferred-tools.js): tools announced by a
 * tool result's `added_tool_names` are deferred whenever the announcement
 * precedes the model's first call of them — their schema rides the provider's
 * native load item anchored at that result instead of the initial tools
 * array. A stale marker for an already-used tool is ignored, and with the
 * mode disabled everything is immediate.
 */
export interface DeferredToolSplit {
  readonly deferred: ReadonlyMap<string, ToolDefinition>
  readonly immediate: readonly ToolDefinition[]
}

export function splitDeferredTools(
  tools: readonly ToolDefinition[] | undefined,
  messages: readonly ChatMessage[],
  enabled: boolean,
): DeferredToolSplit {
  const uniqueTools = new Map<string, ToolDefinition>()
  for (const tool of tools ?? []) uniqueTools.set(tool.function.name, tool)
  if (!enabled) {
    return { deferred: new Map(), immediate: [...uniqueTools.values()] }
  }
  const deferredNames = new Set<string>()
  const usedNames = new Set<string>()
  for (const message of messages) {
    if (message.role === 'assistant') {
      for (const call of message.tool_calls ?? []) usedNames.add(call.function.name)
    } else if (message.role === 'tool') {
      for (const name of message.added_tool_names ?? []) {
        if (!usedNames.has(name)) deferredNames.add(name)
      }
    }
  }
  const immediate: ToolDefinition[] = []
  const deferred = new Map<string, ToolDefinition>()
  for (const [name, tool] of uniqueTools) {
    if (deferredNames.has(name)) deferred.set(name, tool)
    else immediate.push(tool)
  }
  return { deferred, immediate }
}

/** Provider-native deferred-tool serialization modes, mirroring pi-ai's four paths. */
export type DeferredToolsMode = 'additional-tools' | 'kimi' | 'tool-reference' | 'tool-search'

/**
 * Anthropic's native tool-reference support: pi-ai defaults it on for
 * first-party Claude models at version 4.5+ except Haiku, and honors an
 * explicit compat flag when the catalog carries one.
 */
export function anthropicSupportsToolReferences(model: string, provider: string): boolean {
  const compat = piCatalogModelCapabilities(model, provider)?.compat
  if (compat?.supportsToolReferences === true) return true
  if (compat?.supportsToolReferences === false) return false
  const id = model.includes('/') ? model.slice(model.indexOf('/') + 1) : model
  if (!id.startsWith('claude-')) return false
  if (id.includes('haiku')) return false
  const match = /claude-[a-z]+-(\d+)(?:-(\d+))?/.exec(id)
  if (!match) return false
  const major = Number(match[1])
  const minor = Number(match[2] ?? '0')
  return major > 4 || (major === 4 && minor >= 5)
}

/**
 * Responses-API deferred mode (pi-ai openai-responses.js): additional-tools
 * developer items win over the synthetic tool-search replay pair.
 */
export function responsesDeferredToolsMode(
  provider: string,
  model: string,
): Extract<DeferredToolsMode, 'additional-tools' | 'tool-search'> | undefined {
  const compat = piCatalogModelCapabilities(model, provider)?.compat
  if (compat?.supportsAdditionalTools === true) return 'additional-tools'
  if (compat?.supportsToolSearch === true) return 'tool-search'
  return undefined
}

/** Chat-completions deferred mode; Kimi's system-message-with-tools is the only one pi-ai ships. */
export function completionsDeferredToolsMode(
  provider: string,
  model: string,
): Extract<DeferredToolsMode, 'kimi'> | undefined {
  return piCatalogModelCapabilities(model, provider)?.compat?.deferredToolsMode === 'kimi'
    ? 'kimi'
    : undefined
}
