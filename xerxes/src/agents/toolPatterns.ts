// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { AgentDefinition } from './definitions.js'
import type { ToolDefinition } from '../types/toolCalls.js'

/**
 * Tool names in agent tool lists may end in `*` to admit a family of tools
 * whose exact names are only known at runtime. The one family that needs it
 * today is managed plugins: their tools are minted as `plugin_<name>` when a
 * user (or, with approval, the model) installs one, so a fixed list could
 * never name them in advance.
 */

/** Prefix of every tool a managed plugin contributes. */
export const PLUGIN_TOOL_PREFIX = 'plugin_'
/** Tool-list entry that admits every managed plugin tool. */
export const PLUGIN_TOOL_PATTERN = `${PLUGIN_TOOL_PREFIX}*`
/** The model-facing tool that installs, reloads and removes managed plugins. */
export const PLUGIN_MANAGE_TOOL_NAME = 'PluginManageTool'

/** Whether a tool name was minted for a managed plugin. */
export function isPluginToolName(name: string): boolean {
  return name.startsWith(PLUGIN_TOOL_PREFIX)
}

/** One list entry against one tool name: exact, or a trailing-`*` prefix. */
export function toolPatternMatches(pattern: string, name: string): boolean {
  if (pattern.length > 1 && pattern.endsWith('*')) return name.startsWith(pattern.slice(0, -1))
  return pattern === name
}

/** Whether any entry of a tool list admits the name. */
export function toolListAdmits(list: Iterable<string>, name: string): boolean {
  for (const pattern of list) if (toolPatternMatches(pattern, name)) return true
  return false
}

/**
 * Replace the plugin tools in a request's tool list with the current ones.
 * Returns `current` itself when the plugin set is unchanged, so a turn that
 * re-reads its tools every step keeps a byte-identical, cache-friendly request.
 */
export function withPluginTools(
  current: readonly ToolDefinition[] | undefined,
  plugins: readonly ToolDefinition[],
): readonly ToolDefinition[] | undefined {
  const byName = (left: ToolDefinition, right: ToolDefinition) => left.function.name.localeCompare(right.function.name)
  const existing = (current ?? []).filter(tool => isPluginToolName(tool.function.name)).sort(byName)
  const next = plugins.filter(tool => isPluginToolName(tool.function.name)).sort(byName)
  if (JSON.stringify(existing) === JSON.stringify(next)) return current
  return [...(current ?? []).filter(tool => !isPluginToolName(tool.function.name)), ...next]
}

/** Shipped profiles that may call plugin tools. Read-only reviewers and planners stay without them. */
const PLUGIN_TOOL_AGENTS = ['default', 'creator', 'plugin-creator', 'coder', 'tester', 'objective', 'general-purpose']
/** Shipped profiles that may also install and manage plugins (never delegated agents). */
const PLUGIN_MANAGER_AGENTS = new Set(['default', 'creator', 'plugin-creator'])

/**
 * Let shipped full-access profiles see managed plugin tools, and optionally
 * manage plugins. User-authored profiles opt in by listing `plugin_*`.
 */
export function addPluginToolsToBuiltinAgents(definitions: Map<string, AgentDefinition>, options: { readonly manage?: boolean } = {}): void {
  for (const name of PLUGIN_TOOL_AGENTS) {
    const definition = definitions.get(name)
    if (definition?.source !== 'built-in') continue
    const extra = [PLUGIN_TOOL_PATTERN, ...(options.manage && PLUGIN_MANAGER_AGENTS.has(name) ? [PLUGIN_MANAGE_TOOL_NAME] : [])]
    definitions.set(name, {
      ...definition,
      // An empty list already admits everything.
      tools: definition.tools.length ? [...new Set([...definition.tools, ...extra])] : definition.tools,
      allowedTools: definition.allowedTools === null ? null : [...new Set([...definition.allowedTools, ...extra])],
    })
  }
}
