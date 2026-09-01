// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ProviderError, ValidationError } from '../core/errors.js'
import type { ToolDefinition } from '../types/toolCalls.js'
import { isJsonObject, type JsonObject } from '../types/toolCalls.js'

/**
 * OpenAI grammar formats pi-ai knows how to serialize. Lark grammars are
 * preferred over regexes because OpenAI's custom-tool grammar constraint is
 * Lark-native; regex is the fallback variant a tool author may ship instead.
 */
export type GrammarFormat = 'openai_lark' | 'openai_regex'

export interface GrammarConstrainedSampling {
  readonly type: 'grammar'
  readonly variants: Partial<Record<GrammarFormat, string>>
}

const GRAMMAR_FORMAT_WIRE: Readonly<Record<GrammarFormat, 'lark' | 'regex'>> = {
  openai_lark: 'lark',
  openai_regex: 'regex',
}

export interface ResolvedGrammar {
  readonly definition: string
  /** The single required string property the raw grammar output lands in. */
  readonly inputProperty: string
  readonly syntax: 'lark' | 'regex'
}

/**
 * Resolve a tool's grammar variant when grammar tools are supported.
 *
 * Returns undefined for tools without grammar constrained sampling or when the
 * model's compat flags say grammar tools are unsupported — the caller then
 * serializes the tool as an ordinary function tool, exactly like pi-ai's
 * fallback. Throws for a tool that opted into grammar but shipped neither
 * variant or a schema without exactly one required string property: sending it
 * as a function tool would let the model call it with arbitrary JSON the
 * executor cannot parse as grammar output.
 */
export function resolveGrammar(
  tool: ToolDefinition,
  supportsGrammarTools: boolean,
): ResolvedGrammar | undefined {
  const sampling = tool.constrainedSampling
  if (sampling === false || sampling === undefined || sampling.type !== 'grammar') return undefined
  if (!supportsGrammarTools) return undefined
  const variants = sampling.variants
  const [format, definition] = variants.openai_lark
    ? ['openai_lark' as const, variants.openai_lark]
    : variants.openai_regex
      ? ['openai_regex' as const, variants.openai_regex]
      : [undefined, undefined]
  if (format === undefined || definition === undefined) {
    throw new ValidationError(
      'constrained_sampling',
      `tool "${tool.function.name}" cannot use grammar constrained sampling: no supported grammar variant was provided`,
      { tool: tool.function.name },
    )
  }
  return {
    definition,
    inputProperty: inferGrammarInputProperty(tool),
    syntax: GRAMMAR_FORMAT_WIRE[format],
  }
}

/**
 * The grammar schema contract (pi-ai `inferGrammarInputProperty`): the
 * parameters object must declare exactly one required string property, which
 * becomes the property the raw grammar-constrained text is reported under.
 */
export function inferGrammarInputProperty(tool: ToolDefinition): string {
  const parameters = tool.function.parameters
  const required = Array.isArray(parameters.required)
    ? parameters.required.filter((entry): entry is string => typeof entry === 'string')
    : []
  if (parameters.type !== 'object' || required.length !== 1) {
    throw new ValidationError(
      'constrained_sampling',
      `tool "${tool.function.name}" grammar schema must declare exactly one required string property`,
      { tool: tool.function.name },
    )
  }
  const [property] = required
  if (property === undefined) {
    throw new ValidationError('constrained_sampling', 'missing grammar input property', {})
  }
  const properties = isJsonObject(parameters.properties) ? parameters.properties : {}
  const schema = properties[property]
  if (!isJsonObject(schema) || schema.type !== 'string') {
    throw new ValidationError(
      'constrained_sampling',
      `tool "${tool.function.name}" grammar input property "${property}" must be a string`,
      { tool: tool.function.name },
    )
  }
  return property
}

/**
 * Map grammar tool name → input property, computed once per request and
 * threaded through streaming and replay, mirroring pi-ai's
 * `createGrammarToolInputProperties`.
 */
export function createGrammarToolInputProperties(
  tools: readonly ToolDefinition[] | undefined,
  supportsGrammarTools: boolean,
): ReadonlyMap<string, string> {
  const properties = new Map<string, string>()
  if (!tools) return properties
  for (const tool of tools) {
    const grammar = resolveGrammar(tool, supportsGrammarTools)
    if (grammar) properties.set(tool.function.name, grammar.inputProperty)
  }
  return properties
}

/**
 * Read the raw grammar output of a replayed custom tool call.
 *
 * pi-ai throws when the stored arguments lost the string property: replaying
 * it as a JSON function call would corrupt the conversation silently, so the
 * failure is loud at the request boundary instead.
 */
export function grammarToolInput(
  toolName: string,
  property: string,
  args: JsonObject,
): string {
  const value = args[property]
  if (typeof value !== 'string') {
    throw new ValidationError(
      'tool_call',
      `grammar tool call "${toolName}" requires argument "${property}" to be a string`,
      args,
    )
  }
  return value
}

/**
 * Streaming accumulator state for one custom tool call. The model streams raw
 * grammar-constrained text; consumers of the neutral delta stream expect the
 * argument text of a growing JSON object, so the raw text is re-wrapped as the
 * incremental JSON `{"<property>":"<text>"}` like pi-ai's
 * `appendGrammarToolInputJsonDelta`.
 */
export interface GrammarInputAccumulator {
  closed: boolean
  input: string
  started: boolean
}

export function createGrammarInputAccumulator(): GrammarInputAccumulator {
  return { closed: false, input: '', started: false }
}

/**
 * Append raw custom-tool text and return the corresponding JSON-argument
 * delta. Enforces pi-ai's monotonicity contract: the provider's `done` event
 * carries the authoritative full text, which must extend the accumulated
 * prefix; anything else is a wire protocol violation.
 */
export function appendGrammarInput(
  accumulator: GrammarInputAccumulator,
  text: string,
  close: boolean,
  property: string,
): string {
  if (accumulator.closed) {
    throw new ProviderError(
      'grammar',
      `grammar tool input for property "${property}" changed after it was closed`,
    )
  }
  const nextInput = close ? text : accumulator.input + text
  if (!nextInput.startsWith(accumulator.input)) {
    throw new ProviderError(
      'grammar',
      `grammar tool input for property "${property}" changed non-monotonically`,
    )
  }
  const appended = nextInput.slice(accumulator.input.length)
  accumulator.input = nextInput
  accumulator.closed = close
  let delta = ''
  if (!accumulator.started) {
    delta += `{"${property}":"`
    accumulator.started = true
  }
  delta += JSON.stringify(appended).slice(1, -1)
  if (close) delta += '"}'
  return delta
}
