// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { ToolExecutionContext, ToolRegistry } from '../executors/toolRegistry.js'
import type { PluginRegistry } from '../extensions/plugins.js'
import type { AgentPresetRoster } from '../agents/presets.js'
import type { JsonObject, ToolDefinition } from '../types/toolCalls.js'

export const AGENT_PRESET_INSPECT_TOOL_NAME = 'AgentPresetInspectTool'
export const AGENT_PRESET_TOOL_NAME = 'AgentPresetTool'
export const CREATOR_RUNTIME_TOOL_NAME = 'CreatorRuntimeTool'

const INSPECT_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: AGENT_PRESET_INSPECT_TOOL_NAME,
    description: 'List, read, or validate Xerxes agent presets without modifying them.',
    parameters: {
      type: 'object',
      properties: {
        action: { type: 'string', enum: ['list', 'read', 'validate'] },
        id: { type: 'string', description: 'Preset identifier for read or validate.' },
      },
      required: ['action'],
      additionalProperties: false,
    },
  },
}

const AUTHOR_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: AGENT_PRESET_TOOL_NAME,
    description: 'Copy, rewrite, remove, or choose the default Xerxes agent preset. Every mutation is policy-gated.',
    parameters: {
      type: 'object',
      properties: {
        action: { type: 'string', enum: ['copy', 'write', 'remove', 'set_default'] },
        from: { type: 'string', description: 'Source preset id for copy.' },
        id: { type: 'string', description: 'Target preset id.' },
        name: { type: 'string', description: 'Optional display name for a copied preset.' },
        content: { type: 'string', description: 'Complete version-1 agent.yaml content for write.' },
      },
      required: ['action', 'id'],
      additionalProperties: false,
    },
  },
}

const RUNTIME_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: CREATOR_RUNTIME_TOOL_NAME,
    description: 'Inspect the live Xerxes tool, plugin, or preset catalog before authoring an agent composition.',
    parameters: {
      type: 'object',
      properties: {
        catalog: { type: 'string', enum: ['tools', 'plugins', 'presets'] },
      },
      required: ['catalog'],
      additionalProperties: false,
    },
  },
}

export interface AgentPresetToolOptions {
  readonly agentId?: string
  readonly onChanged?: () => void | Promise<void>
  readonly pluginRegistry?: PluginRegistry
}

/** Register the self-inspection and authoring tools only Creator mode can see. */
export function registerAgentPresetTools(
  registry: ToolRegistry,
  roster: AgentPresetRoster,
  options: AgentPresetToolOptions = {},
): void {
  // Shared registration lets a duplicated Creator composition retain the same
  // capabilities. Every handler re-checks that the calling preset explicitly
  // declares the tool, so another agent cannot invoke a hidden schema by name.
  const agentId = options.agentId ?? 'default'
  registry.register(
    INSPECT_DEFINITION,
    (inputs, context) => {
      requireDeclaredCapability(roster, context, AGENT_PRESET_INSPECT_TOOL_NAME)
      return inspectPreset(roster, inputs)
    },
    agentId,
    {
      concurrencySafe: true,
      defer: false,
      destructive: false,
      interruptBehavior: 'cancel',
      maxResultBytes: 131_072,
      openWorld: false,
      readOnly: true,
    },
    'Read before writing. A broken preset may be read for repair but cannot be selected or copied.',
  )
  registry.register(
    AUTHOR_DEFINITION,
    async (inputs, context) => {
      requireDeclaredCapability(roster, context, AGENT_PRESET_TOOL_NAME)
      const result = authorPreset(roster, inputs)
      await options.onChanged?.()
      return result
    },
    agentId,
    {
      concurrencySafe: false,
      defer: false,
      destructive: true,
      interruptBehavior: 'cancel',
      maxResultBytes: 131_072,
      openWorld: false,
      readOnly: false,
    },
    'Copy a known-good preset before editing. Writes replace one user preset atomically only after strict schema validation.',
  )
  registry.register(
    RUNTIME_DEFINITION,
    (inputs, context) => {
      requireDeclaredCapability(roster, context, CREATOR_RUNTIME_TOOL_NAME)
      return runtimeCatalog(registry, roster, options.pluginRegistry, inputs, context.agentId ?? 'creator')
    },
    agentId,
    {
      concurrencySafe: true,
      defer: false,
      destructive: false,
      interruptBehavior: 'cancel',
      maxResultBytes: 131_072,
      openWorld: false,
      readOnly: true,
    },
    'Inspect installed names rather than inventing tools or plugins in a preset.',
  )
}

function inspectPreset(roster: AgentPresetRoster, inputs: JsonObject): unknown {
  const action = text(inputs.action)
  const id = text(inputs.id)
  if (action === 'list') return { presets: roster.list().map(publicPreset) }
  if (!id) throw new Error(`AgentPresetInspectTool ${action || 'action'} requires id`)
  if (action === 'read') {
    const preset = roster.read(id)
    return { ...publicPreset(preset), content: preset.content }
  }
  if (action === 'validate') return { valid: true, preset: publicPreset(roster.validate(id)) }
  throw new Error(`Unknown AgentPresetInspectTool action '${action}'`)
}

function authorPreset(roster: AgentPresetRoster, inputs: JsonObject): unknown {
  const action = text(inputs.action)
  const id = text(inputs.id)
  if (!id) throw new Error(`AgentPresetTool ${action || 'action'} requires id`)
  if (action === 'copy') {
    const from = text(inputs.from)
    if (!from) throw new Error('AgentPresetTool copy requires from')
    return { preset: publicPreset(roster.copy(from, id, optionalText(inputs.name))) }
  }
  if (action === 'write') {
    const content = typeof inputs.content === 'string' ? inputs.content : ''
    if (!content) throw new Error('AgentPresetTool write requires content')
    return { preset: publicPreset(roster.write(id, content)) }
  }
  if (action === 'remove') {
    roster.remove(id)
    return { removed: id }
  }
  if (action === 'set_default') return { preset: publicPreset(roster.setDefault(id)) }
  throw new Error(`Unknown AgentPresetTool action '${action}'`)
}

function runtimeCatalog(
  registry: ToolRegistry,
  roster: AgentPresetRoster,
  plugins: PluginRegistry | undefined,
  inputs: JsonObject,
  agentId: string,
): unknown {
  const catalog = text(inputs.catalog)
  if (catalog === 'tools') {
    return {
      tools: registry.definitions(agentId).map(definition => ({
        name: definition.function.name,
        description: definition.function.description ?? '',
      })).sort((left, right) => left.name.localeCompare(right.name)),
    }
  }
  if (catalog === 'plugins') {
    return {
      plugins: (plugins?.pluginNames ?? []).sort().map(name => {
        const plugin = plugins?.getPlugin(name)
        return {
          name,
          version: plugin?.meta.version ?? '',
          description: plugin?.meta.description ?? '',
          tools: [...(plugin?.tools.keys() ?? [])].sort(),
          hooks: [...(plugin?.hooks.keys() ?? [])].sort(),
          channels: [...(plugin?.channels.keys() ?? [])].sort(),
        }
      }),
    }
  }
  if (catalog === 'presets') return { presets: roster.list().map(publicPreset) }
  throw new Error(`Unknown CreatorRuntimeTool catalog '${catalog}'`)
}

function requireDeclaredCapability(
  roster: AgentPresetRoster,
  context: ToolExecutionContext,
  toolName: string,
): void {
  const agentId = context.agentId?.trim() ?? ''
  const projectRoot = typeof context.metadata?.project_root === 'string'
    ? context.metadata.project_root
    : roster.projectDirectory
  const definition = agentId ? roster.definition(agentId, projectRoot) : undefined
  if (!definition?.tools.includes(toolName)) {
    throw new Error(`Agent preset '${agentId || 'unknown'}' does not declare ${toolName}`)
  }
}

function publicPreset(preset: ReturnType<AgentPresetRoster['resolve']>): Record<string, unknown> {
  return {
    id: preset.id,
    name: preset.name,
    description: preset.description,
    trust: preset.trust,
    is_default: preset.isDefault,
    manageable: preset.manageable,
    ...(preset.path ? { path: preset.path } : {}),
    ...(preset.broken ? { broken: preset.broken } : {}),
  }
}

function text(value: unknown): string {
  return typeof value === 'string' ? value.trim() : ''
}

function optionalText(value: unknown): string | undefined {
  const valueText = text(value)
  return valueText || undefined
}
