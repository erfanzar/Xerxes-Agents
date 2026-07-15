// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ClientError, ValidationError } from '../../core/errors.js'
import { ToolRegistry, type ToolExecutionContext } from '../../executors/toolRegistry.js'
import type { JsonObject, ToolDefinition } from '../../types/toolCalls.js'
import { optionalInteger, optionalString, requiredString } from '../inputs.js'
import { WorkspacePathResolver } from '../pathSafety.js'

/** Existing core tools already implement these legacy Claude search names. */
export const CLAUDE_SEARCH_DELEGATES = Object.freeze({
  GlobTool: 'GlobTool',
  GrepTool: 'GrepTool',
})

export interface LspRequest {
  readonly action: string
  readonly character: number
  readonly filePath: string
  readonly line: number
}

/** Host port for a language-server bridge. The runtime deliberately does not bundle language servers. */
export interface LspAdapter {
  execute(request: LspRequest, signal?: AbortSignal): Promise<unknown>
}

export interface ClaudeSearchToolsOptions {
  readonly lspAdapter?: LspAdapter
  readonly paths?: WorkspacePathResolver
}

export const LSP_TOOL_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: 'LSPTool',
    description: 'Run an LSP action through a host-provided language-server adapter.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        action: { type: 'string', description: 'LSP action, such as definition, references, hover, or diagnostics.' },
        file_path: { type: 'string', description: 'Optional workspace-relative source file path.' },
        line: { type: 'integer', minimum: 0, default: 0, description: 'Zero-based source line.' },
        character: { type: 'integer', minimum: 0, default: 0, description: 'Zero-based source character.' },
      },
      required: ['action'],
    },
  },
}

/** Register only LSPTool; GlobTool and GrepTool are reused from `fileTools.ts`. */
export function registerClaudeSearchTools(
  registry: ToolRegistry,
  options: ClaudeSearchToolsOptions = {},
  agentId = 'default',
): readonly ToolDefinition[] {
  const adapter = new ClaudeSearchTools(options)
  registry.replace(LSP_TOOL_DEFINITION, (inputs, context, signal) => adapter.execute(inputs, context, signal), agentId)
  return [LSP_TOOL_DEFINITION]
}

/** Claude-compatible LSP facade with workspace path normalization at the tool boundary. */
export class ClaudeSearchTools {
  constructor(private readonly options: ClaudeSearchToolsOptions) {}

  async execute(inputs: JsonObject, _context: ToolExecutionContext, signal?: AbortSignal): Promise<unknown> {
    const adapter = this.options.lspAdapter
    if (adapter === undefined) {
      throw new ClientError('lsp', 'no LspAdapter is attached; configure an IDE or language-server bridge')
    }
    const action = requiredString(inputs, 'action')
    const requestedPath = optionalString(inputs, 'file_path') ?? ''
    const filePath = requestedPath && this.options.paths !== undefined
      ? await this.options.paths.resolve(requestedPath)
      : requestedPath
    const line = nonnegativeInteger(optionalInteger(inputs, 'line', 0), 'line')
    const character = nonnegativeInteger(optionalInteger(inputs, 'character', 0), 'character')
    return adapter.execute({ action, filePath, line, character }, signal)
  }
}

function nonnegativeInteger(value: number, name: string): number {
  if (value < 0) throw new ValidationError(name, 'must be a non-negative integer', value)
  return value
}
