// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { FunctionExecutionError, ValidationError } from '../core/errors.js'
import { validateToolArguments } from '../runtime/argumentValidation.js'
import type { JsonObject, ToolCall, ToolDefinition } from '../types/toolCalls.js'

export interface ToolExecutionContext {
  readonly agentId?: string
  readonly metadata: Record<string, unknown>
  readonly sessionId?: string
}

export type ToolHandler = (
  inputs: JsonObject,
  context: ToolExecutionContext,
  signal?: AbortSignal,
) => Promise<unknown> | unknown

export interface ToolExecutor {
  execute(call: ToolCall, context: ToolExecutionContext, signal?: AbortSignal): Promise<string>
}

interface RegisteredTool {
  readonly agentId: string
  readonly definition: ToolDefinition
  readonly handler: ToolHandler
}

/** Function registry that preserves Xerxes' current-agent-first lookup semantics. */
export class ToolRegistry implements ToolExecutor {
  private readonly entries = new Map<string, RegisteredTool[]>()

  register(definition: ToolDefinition, handler: ToolHandler, agentId = 'default'): void {
    const name = definition.function.name
    if (!name) {
      throw new ValidationError('tool.name', 'must not be empty')
    }
    const tools = this.entries.get(name) ?? []
    tools.push({ definition, handler, agentId })
    this.entries.set(name, tools)
  }

  /** Replace the tool registered for one agent without disturbing other agent-specific variants. */
  replace(definition: ToolDefinition, handler: ToolHandler, agentId = 'default'): void {
    const name = definition.function.name
    if (!name) {
      throw new ValidationError('tool.name', 'must not be empty')
    }
    const tools = [...(this.entries.get(name) ?? [])]
    const index = tools.findIndex(entry => entry.agentId === agentId)
    if (index >= 0) {
      tools[index] = { definition, handler, agentId }
    } else {
      tools.push({ definition, handler, agentId })
    }
    this.entries.set(name, tools)
  }

  /** Remove one agent-specific tool implementation. Returns whether an entry was removed. */
  unregister(name: string, agentId = 'default'): boolean {
    const entries = this.entries.get(name)
    if (entries === undefined) return false
    const remaining = entries.filter(entry => entry.agentId !== agentId)
    if (remaining.length === entries.length) return false
    if (remaining.length) this.entries.set(name, remaining)
    else this.entries.delete(name)
    return true
  }

  definitions(agentId?: string): ToolDefinition[] {
    return [...this.entries.values()]
      .map(entries => this.pick(entries, agentId)?.definition)
      .filter((definition): definition is ToolDefinition => definition !== undefined)
  }

  get(name: string, agentId?: string): ToolHandler | undefined {
    return this.pick(this.entries.get(name) ?? [], agentId)?.handler
  }

  async execute(call: ToolCall, context: ToolExecutionContext, signal?: AbortSignal): Promise<string> {
    if (signal?.aborted) {
      throw new FunctionExecutionError(call.function.name, 'cancelled before execution')
    }
    const registered = this.pick(this.entries.get(call.function.name) ?? [], context.agentId)
    if (!registered) {
      throw new FunctionExecutionError(call.function.name, 'is not registered')
    }
    const validation = validateToolArguments(
      call.function.name,
      call.function.arguments,
      registered.definition.function.parameters,
    )
    if (!validation.ok) {
      throw new FunctionExecutionError(call.function.name, validation.error)
    }
    try {
      return serializeToolResult(await registered.handler(call.function.arguments, context, signal))
    } catch (error) {
      if (error instanceof FunctionExecutionError) {
        throw error
      }
      throw new FunctionExecutionError(call.function.name, errorMessage(error), error)
    }
  }

  private pick(entries: readonly RegisteredTool[], agentId?: string): RegisteredTool | undefined {
    if (agentId) {
      const agentTool = entries.find(entry => entry.agentId === agentId)
      if (agentTool) {
        return agentTool
      }
    }
    // Fall back only to the shared default registration; a variant registered for one
    // agent must never be silently callable by (or visible to) any other agent.
    return entries.find(entry => entry.agentId === 'default')
  }
}

export function serializeToolResult(value: unknown): string {
  if (typeof value === 'string') {
    return value
  }
  if (value === undefined) {
    return ''
  }
  try {
    return JSON.stringify(value)
  } catch (error) {
    throw new ValidationError('tool_result', 'must be JSON serializable', value, { cause: errorMessage(error) })
  }
}

export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}
