// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { FunctionExecutionError, ValidationError } from '../core/errors.js'
import { isReadOnlyInvocation } from '../security/shellAnalysis.js'
import { validateToolArguments } from '../runtime/argumentValidation.js'
import type { ChatMessage } from '../types/messages.js'
import type { JsonObject, ToolCall, ToolDefinition } from '../types/toolCalls.js'

export interface ToolExecutionContext {
  readonly agentId?: string
  readonly metadata: Record<string, unknown>
  readonly sessionId?: string
}

/**
 * Everything a host needs to know about a tool that is not part of the provider
 * wire shape. It deliberately does not live on ToolDefinition: that object is
 * serialized straight into request payloads, so any field added there would be
 * shipped to the model on every single request.
 */
export interface ToolCapabilities {
  /** Two invocations may overlap without corrupting each other's state. */
  readonly concurrencySafe: boolean
  /** The schema may be withheld from the request until ToolSearchTool loads it. */
  readonly defer: boolean
  /**
   * What a mid-turn interrupt does to an in-flight call.
   *
   * `cancel` aborts it. `block` lets it finish: killing a write half-way leaves
   * the workspace in a state neither the user nor the model asked for, which is
   * worse than waiting out the call's own timeout. Only tools that are bounded
   * by construction may declare `block`.
   */
  readonly interruptBehavior: 'block' | 'cancel'
  /** Can destroy work that the tool itself cannot restore. */
  readonly destructive: boolean
  /** Byte ceiling a host may impose on this tool's serialized result. */
  readonly maxResultBytes: number
  /** Reaches beyond the workspace: network, other processes, other agents, the user. */
  readonly openWorld: boolean
  /** Never mutates host state, so re-running it is always free of side effects. */
  readonly readOnly: boolean
}

/** Conservative ceiling for a tool that never declared one. */
export const DEFAULT_MAX_TOOL_RESULT_BYTES = 32_768

/**
 * Fail-closed defaults. An undeclared tool is assumed to mutate, to be unsafe to
 * run twice at once, to touch the outside world, and to be deferrable — so that
 * forgetting to declare a record costs context or an extra prompt, never a
 * silently auto-approved destructive call.
 */
export const DEFAULT_TOOL_CAPABILITIES: ToolCapabilities = Object.freeze({
  concurrencySafe: false,
  defer: true,
  interruptBehavior: 'cancel',
  destructive: true,
  maxResultBytes: DEFAULT_MAX_TOOL_RESULT_BYTES,
  openWorld: true,
  readOnly: false,
})

/**
 * Tools the agent cannot work without: hiding these behind a search round-trip
 * would make the very first turn of every session cost an extra request. They
 * are keyed by name because most of them are registered from files that do not
 * yet declare a capability record; a declaration always wins over this seed.
 */
export const ALWAYS_LOADED_TOOL_NAMES: ReadonlySet<string> = Object.freeze(new Set([
  'AppendFile',
  'FileEditTool',
  'GlobTool',
  'GrepTool',
  'ListDir',
  'ReadFile',
  'TodoWriteTool',
  'ToolSearchTool',
  'WriteFile',
  'exec_command',
]))

/**
 * Marker key emitted per loaded match by ToolSearchTool. The set of live schemas
 * is recovered by scanning the transcript for it, so it must stay a literal key
 * in the serialized result — see {@link revealedToolNames}.
 */
export const TOOL_SEARCH_LOADED_KEY = 'loaded_tool'

const TOOL_SEARCH_LOADED_PATTERN = new RegExp(`"${TOOL_SEARCH_LOADED_KEY}"\\s*:\\s*"([^"\\\\]{1,120})"`, 'g')
const MAX_DEFERRED_DESCRIPTION_CHARACTERS = 160

/** One deferred tool as advertised to the model: enough to search, not enough to call. */
export interface DeferredToolSummary {
  readonly description: string
  readonly name: string
}

/** Merge a declaration over the fail-closed defaults, applying the always-loaded seed by name. */
export function resolveToolCapabilities(
  name: string,
  declared?: Partial<ToolCapabilities>,
): ToolCapabilities {
  const deferDefault = !ALWAYS_LOADED_TOOL_NAMES.has(name)
  return Object.freeze({
    concurrencySafe: declared?.concurrencySafe ?? DEFAULT_TOOL_CAPABILITIES.concurrencySafe,
    interruptBehavior: declared?.interruptBehavior ?? DEFAULT_TOOL_CAPABILITIES.interruptBehavior,
    defer: declared?.defer ?? deferDefault,
    destructive: declared?.destructive ?? DEFAULT_TOOL_CAPABILITIES.destructive,
    maxResultBytes: declared?.maxResultBytes ?? DEFAULT_TOOL_CAPABILITIES.maxResultBytes,
    openWorld: declared?.openWorld ?? DEFAULT_TOOL_CAPABILITIES.openWorld,
    readOnly: declared?.readOnly ?? DEFAULT_TOOL_CAPABILITIES.readOnly,
  })
}

/**
 * Recover the set of tool schemas a transcript has already loaded.
 *
 * This is derived, never stored: a per-session mutable "currently live tools"
 * set would silently disagree with the messages actually sent after compaction,
 * resume, or a rewound turn, and the model would be told about tools whose
 * schemas are no longer in its context (or denied ones that are).
 */
export function revealedToolNames(messages: readonly ChatMessage[]): ReadonlySet<string> {
  const names = new Set<string>()
  for (const message of messages) {
    // Only a successful tool result can have loaded a schema; an errored search
    // returned nothing, and user text must not be able to conjure tools.
    if (message.role !== 'tool' || message.is_error === true) continue
    // matchAll clones the regex, so the shared /g literal keeps no lastIndex state.
    for (const match of message.content.matchAll(TOOL_SEARCH_LOADED_PATTERN)) {
      const name = match[1]
      if (name) names.add(name)
    }
  }
  return names
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
  readonly capabilities: ToolCapabilities
  readonly declaredCapabilities: boolean
  readonly definition: ToolDefinition
  /** Usage policy shipped with this tool; rendered only while its schema is visible. */
  readonly guidance?: string
  readonly handler: ToolHandler
}

/** One tool's usage-policy section, in the provider tool-list order of a request. */
export interface ToolGuidanceSegment {
  readonly name: string
  readonly text: string
}

export interface ToolRegistryOptions {
  /**
   * Opt-in deferred schema loading. Off by default: turning it on changes the
   * tool list of every request, which is the repo owner's call, not a library
   * default. While off, definitionsForTranscript() sends everything, as today.
   */
  readonly deferredToolLoading?: boolean
  /**
   * Called when register() repeats an existing (name, agentId) pair. The first
   * handler wins lookup, so a duplicate silently shadows the new handler; the
   * warning keeps that observable. Use replace() for intentional overrides.
   */
  readonly onDuplicateRegistration?: (name: string, agentId: string) => void
}

/** Function registry that preserves Xerxes' current-agent-first lookup semantics. */
export class ToolRegistry implements ToolExecutor {
  readonly deferredToolLoading: boolean
  private readonly entries = new Map<string, RegisteredTool[]>()
  private readonly onDuplicateRegistration: (name: string, agentId: string) => void

  constructor(options: ToolRegistryOptions = {}) {
    this.deferredToolLoading = options.deferredToolLoading ?? false
    this.onDuplicateRegistration = options.onDuplicateRegistration ?? ((name, agentId) => {
      console.warn(
        `ToolRegistry: duplicate registration of "${name}" for agent "${agentId}" is shadowed by the first handler; use replace() for intentional overrides`,
      )
    })
  }

  register(
    definition: ToolDefinition,
    handler: ToolHandler,
    agentId = 'default',
    capabilities?: Partial<ToolCapabilities>,
    guidance?: string,
  ): void {
    const name = definition.function.name
    if (!name) {
      throw new ValidationError('tool.name', 'must not be empty')
    }
    const tools = this.entries.get(name) ?? []
    if (tools.some(entry => entry.agentId === agentId)) {
      this.onDuplicateRegistration(name, agentId)
    }
    tools.push(makeRegistered(definition, handler, agentId, capabilities, guidance))
    this.entries.set(name, tools)
  }

  /** Replace the tool registered for one agent without disturbing other agent-specific variants. */
  replace(
    definition: ToolDefinition,
    handler: ToolHandler,
    agentId = 'default',
    capabilities?: Partial<ToolCapabilities>,
    guidance?: string,
  ): void {
    const name = definition.function.name
    if (!name) {
      throw new ValidationError('tool.name', 'must not be empty')
    }
    const tools = [...(this.entries.get(name) ?? [])]
    const registered = makeRegistered(definition, handler, agentId, capabilities, guidance)
    const index = tools.findIndex(entry => entry.agentId === agentId)
    if (index >= 0) {
      tools[index] = registered
    } else {
      tools.push(registered)
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

  /**
   * Capabilities for the tool this agent would actually reach, resolved through
   * the same agent-first lookup as get()/definitions(). Unknown or undeclared
   * tools resolve to the fail-closed defaults rather than throwing, so a caller
   * gating on capabilities can never be tricked into the permissive branch.
   */
  capabilities(
    name: string,
    agentId?: string,
    /**
     * Optional call arguments. A tool whose safety genuinely depends on its
     * input — `exec_command` above all — can refine one axis per invocation
     * instead of forcing the whole registry to answer by name. Refinement may
     * only ever be consulted through {@link refineCapabilities}, which is
     * allowed to widen `concurrencySafe` and nothing else.
     */
    args?: Readonly<Record<string, unknown>>,
  ): ToolCapabilities {
    const declared = this.pick(this.entries.get(name) ?? [], agentId)?.capabilities
      ?? resolveToolCapabilities(name)
    return args === undefined ? declared : refineCapabilities(name, declared, args)
  }

  /** Whether the registration itself supplied a record, as opposed to inheriting defaults. */
  hasDeclaredCapabilities(name: string, agentId?: string): boolean {
    return this.pick(this.entries.get(name) ?? [], agentId)?.declaredCapabilities ?? false
  }

  /** Deferred tools as name plus a one-line description: the index the model searches. */
  deferredCatalog(agentId?: string): readonly DeferredToolSummary[] {
    return [...this.entries.values()]
      .map(entries => this.pick(entries, agentId))
      .filter((entry): entry is RegisteredTool => entry !== undefined && entry.capabilities.defer)
      .map(entry => Object.freeze({
        name: entry.definition.function.name,
        description: oneLine(entry.definition.function.description),
      }))
      .sort((left, right) => left.name.localeCompare(right.name))
  }

  /**
   * Schemas to send with a request built from these messages.
   *
   * With deferred loading off this is exactly definitions(). With it on, the
   * result is the always-loaded core plus whatever prior ToolSearchTool results
   * are still present in the transcript, which is why it takes the messages
   * instead of consulting session state: after a compaction drops those results
   * the schemas drop with them, and the model is never told about a tool whose
   * schema is no longer in front of it.
   */
  definitionsForTranscript(messages: readonly ChatMessage[], agentId?: string): ToolDefinition[] {
    if (!this.deferredToolLoading) {
      return this.definitions(agentId)
    }
    const revealed = revealedToolNames(messages)
    return [...this.entries.values()]
      .map(entries => this.pick(entries, agentId))
      .filter((entry): entry is RegisteredTool =>
        entry !== undefined && (!entry.capabilities.defer || revealed.has(entry.definition.function.name)))
      .map(entry => entry.definition)
  }

  /**
   * Usage-policy sections for exactly the tools a request will expose.
   *
   * Callers pass the names of the definitions actually being sent (after agent,
   * mode, and resumed-subagent filtering), so guidance can never outlive its
   * schema: hiding or deferring a tool removes both from the request together.
   * Order follows the caller's list — provider tool order, not registration
   * order — which keeps the rendered block byte-stable for an unchanged surface.
   */
  guidanceForTools(
    names: readonly string[],
    agentId?: string,
  ): readonly ToolGuidanceSegment[] {
    const segments: ToolGuidanceSegment[] = []
    for (const name of names) {
      const text = this.pick(this.entries.get(name) ?? [], agentId)?.guidance?.trim()
      if (text) segments.push({ name, text })
    }
    return segments
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
    // The validator accepted the call on the strength of its repaired payload, so the
    // handler has to receive that one; forwarding the raw arguments would hand a tool
    // back the very `"0"`/`"true"` strings the type check just converted away.
    const inputs = validation.coerced ?? call.function.arguments
    try {
      return serializeToolResult(await registered.handler(inputs, context, signal))
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

function makeRegistered(
  definition: ToolDefinition,
  handler: ToolHandler,
  agentId: string,
  capabilities?: Partial<ToolCapabilities>,
  guidance?: string,
): RegisteredTool {
  return {
    agentId,
    capabilities: resolveToolCapabilities(definition.function.name, capabilities),
    declaredCapabilities: capabilities !== undefined,
    definition,
    ...(guidance === undefined ? {} : { guidance }),
    handler,
  }
}

/**
 * Render tool guidance as one stable prompt block.
 *
 * Empty in, empty out — a surface with no guided tools contributes nothing to
 * the system prompt, and an unchanged surface renders byte-identically, so the
 * provider's prefix cache survives.
 */
export function renderToolGuidance(segments: readonly ToolGuidanceSegment[]): string {
  return segments
    .map(segment => `[Tool usage: ${segment.name}]\n${segment.text.trim()}`)
    .join('\n\n')
}

function oneLine(description: string): string {
  const first = description.split('\n', 1)[0]?.trim() ?? ''
  return first.length > MAX_DEFERRED_DESCRIPTION_CHARACTERS
    ? first.slice(0, MAX_DEFERRED_DESCRIPTION_CHARACTERS - 1).trimEnd() + '…'
    : first
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

/**
 * Widen one axis for a call whose declared record is deliberately conservative.
 *
 * Strictly one direction and strictly one axis: a refinement may only mark an
 * invocation concurrency-safe, never grant it read-only status, downgrade
 * `destructive`, or change how it is permissioned. Concurrency is a scheduling
 * decision the loop makes after permission has already been resolved, so
 * widening it cannot admit a call that would otherwise have been refused.
 */
function refineCapabilities(
  name: string,
  declared: ToolCapabilities,
  args: Readonly<Record<string, unknown>>,
): ToolCapabilities {
  if (declared.concurrencySafe || !SHELL_TOOL_NAMES.has(name)) return declared
  const command = typeof args.cmd === 'string' ? args.cmd : undefined
  if (!command) return declared
  const argv = Array.isArray(args.args)
    ? args.args.filter((value): value is string => typeof value === 'string')
    : []
  // The same analyzer the permission gate already trusts for this tool.
  return isReadOnlyInvocation(command, argv)
    ? { ...declared, concurrencySafe: true }
    : declared
}

/** Tools whose arguments are a shell invocation the read-only analyzer understands. */
const SHELL_TOOL_NAMES: ReadonlySet<string> = new Set(['exec_command'])
