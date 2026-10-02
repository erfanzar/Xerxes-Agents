// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import type { AgentDefinition } from '../agents/definitions.js'
import type { ToolRegistry } from '../executors/toolRegistry.js'
import type { LspManager } from '../lsp/manager.js'
import { ClaudeSearchTools, LSP_TOOL_DEFINITION } from './claudeTools/search.js'

export function registerConfiguredLspTool(registry: ToolRegistry, manager: LspManager, workspace: () => string): void {
  if (!manager.configured) return
  registry.replace({ ...LSP_TOOL_DEFINITION, function: { ...LSP_TOOL_DEFINITION.function,
    description: 'Query a configured language server for definition, references, hover, symbols, or diagnostics in a workspace file. file_path is required. line and character are zero-based UTF-16 positions. Diagnostics marked fresh:false are unconfirmed, not a clean bill of health. Unsupported file types or server capabilities return an error; use GrepTool or GlobTool instead. No language server is bundled.',
    parameters: { type: 'object', additionalProperties: false, properties: {
      action: { type: 'string', enum: ['definition', 'references', 'hover', 'symbols', 'diagnostics'] },
      file_path: { type: 'string', minLength: 1 }, line: { type: 'integer', minimum: 0, default: 0 }, character: { type: 'integer', minimum: 0, default: 0 },
    }, required: ['action', 'file_path'] },
  } }, (inputs, context, signal) => new ClaudeSearchTools({ lspAdapter: { execute: (request, abort) => manager.forWorkspace(workspace()).execute({ ...request, diagnosticsWaitMs: 1000 }, abort) } }).execute(inputs, context, signal))
}
/** The tool {@link addLspToolToBuiltinAgents} adds while a language server is configured. */
export const LSP_RUNTIME_TOOL_NAME = LSP_TOOL_DEFINITION.function.name

export function addLspToolToBuiltinAgents(definitions: Map<string, AgentDefinition>, manager: LspManager): void {
  if (!manager.configured) return
  for (const name of ['default', 'creator']) {
    const definition = definitions.get(name)
    if (definition?.source === 'built-in') definitions.set(name, { ...definition, tools: [...new Set([...definition.tools, LSP_RUNTIME_TOOL_NAME])] })
  }
}
