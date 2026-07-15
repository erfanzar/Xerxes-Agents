// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import {
  functionToJson,
  type FunctionToJsonOptions,
} from '../../core/utils.js'
import type { JsonSchema, ToolDefinition } from '../../types/toolCalls.js'

/** A callable retained by a Cortex tool without imposing an execution transport. */
export type CortexToolFunction = Function

export interface CortexToolOptions<F extends CortexToolFunction = CortexToolFunction> {
  /** Generate a schema from explicit callable metadata when parameters are omitted. */
  readonly autoGenerateSchema?: boolean
  readonly description: string
  readonly function: F
  readonly name: string
  /** Explicit OpenAI-compatible parameter schema. */
  readonly parameters?: JsonSchema
  /** Native metadata used when a schema must be generated for a TypeScript callable. */
  readonly schemaOptions?: FunctionToJsonOptions
}

export interface CortexToolFromFunctionOptions {
  readonly description?: string
  readonly name?: string
  readonly schemaOptions?: FunctionToJsonOptions
}

const EMPTY_PARAMETERS: JsonSchema = Object.freeze({
  type: 'object',
  properties: Object.freeze({}),
  required: Object.freeze([]),
})

/**
 * Wrap a callable with its model-facing name, description, and JSON Schema.
 *
 * TypeScript does not retain Python-style annotations or docstrings at
 * runtime. Automatic generation consequently draws only on the explicit
 * callable metadata registered through `defineCallableSchema` or supplied as
 * `schemaOptions`; it never parses source code or invokes another runtime.
 */
export class CortexTool<F extends CortexToolFunction = CortexToolFunction> {
  readonly autoGenerateSchema: boolean
  readonly description: string
  readonly function: F
  readonly name: string
  readonly parameters: JsonSchema
  private readonly schemaOptions: FunctionToJsonOptions | undefined

  constructor(options: CortexToolOptions<F>) {
    if (typeof options.function !== 'function') throw new TypeError('function must be callable')
    if (typeof options.name !== 'string') throw new TypeError('name must be a string')
    if (typeof options.description !== 'string') throw new TypeError('description must be a string')
    this.function = options.function
    this.name = options.name
    this.description = options.description
    this.parameters = options.parameters ?? Object.freeze({})
    this.autoGenerateSchema = options.autoGenerateSchema ?? true
    this.schemaOptions = options.schemaOptions
  }

  /** Return the canonical OpenAI-style function descriptor for this tool. */
  toFunctionJson(): ToolDefinition {
    if (this.autoGenerateSchema && Object.keys(this.parameters).length === 0) {
      const generated = functionToJson(this.function, this.schemaOptions)
      return {
        type: 'function',
        function: {
          name: this.name,
          description: this.description,
          parameters: generated.function.parameters,
        },
      }
    }
    return {
      type: 'function',
      function: {
        name: this.name,
        description: this.description,
        parameters: Object.keys(this.parameters).length === 0 ? EMPTY_PARAMETERS : this.parameters,
      },
    }
  }

  /** Create a tool from a plain callable and optional native schema metadata. */
  static fromFunction<F extends CortexToolFunction>(
    functionValue: F,
    options: CortexToolFromFunctionOptions = {},
  ): CortexTool<F> {
    if (typeof functionValue !== 'function') throw new TypeError('function must be callable')
    const generated = functionToJson(functionValue, options.schemaOptions)
    return new CortexTool({
      function: functionValue,
      name: options.name ?? generated.function.name,
      description: options.description ?? generated.function.description,
      ...(options.schemaOptions === undefined ? {} : { schemaOptions: options.schemaOptions }),
    })
  }
}
