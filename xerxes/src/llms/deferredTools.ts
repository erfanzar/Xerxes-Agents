// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { ChatMessage } from '../types/messages.js'
import type { ToolDefinition } from '../types/toolCalls.js'
import { wireCapability } from './modelsDev.js'

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
 * Anthropic tool references, only where the provider reports support — no
 * model-name or version rule. Unreported models send their tools up front
 * (deferred loading is opt-in; this only trims tokens when it is on).
 */
export function anthropicSupportsToolReferences(model: string, provider: string): boolean {
  return wireCapability({ provider, model }).toolReferences === true
}

/** Responses-API deferred mode, only where reported: tool search (Codex `supports_search_tool`) or additional tools. */
export function responsesDeferredToolsMode(
  provider: string,
  model: string,
): Extract<DeferredToolsMode, 'additional-tools' | 'tool-search'> | undefined {
  const capability = wireCapability({ provider, model })
  if (capability.toolSearch === true) return 'tool-search'
  return capability.additionalTools === true ? 'additional-tools' : undefined
}

/** Chat-completions deferred mode: Kimi's, where Kimi reports `supports_dynamic_tools`. */
export function completionsDeferredToolsMode(
  provider: string,
  model: string,
): Extract<DeferredToolsMode, 'kimi'> | undefined {
  return wireCapability({ provider, model }).dynamicTools === true ? 'kimi' : undefined
}
