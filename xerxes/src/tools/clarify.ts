// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ToolRegistry, type ToolExecutionContext } from '../executors/toolRegistry.js'
import type { JsonObject, ToolDefinition } from '../types/toolCalls.js'
import { optionalBoolean, optionalStringArray, requiredString } from './inputs.js'

export interface ClarifyResult {
  readonly answer: string
  readonly answered: boolean
  readonly selectedIndex?: number
  readonly skipped: boolean
}

/** UX boundary for clarification prompts; a TUI, channel, or test supplies it. */
export interface InteractiveAsker {
  ask(
    question: string,
    options: readonly string[],
    allowFreeform: boolean,
    context?: ToolExecutionContext,
  ): ClarifyResult | Promise<ClarifyResult>
}

/** Deterministic clarification responder suitable for tests and non-interactive automation. */
export class StaticAsker implements InteractiveAsker {
  constructor(
    private readonly options: { readonly answer?: string; readonly index?: number; readonly skip?: boolean } = {},
  ) {}

  ask(_question: string, choices: readonly string[], _allowFreeform: boolean): ClarifyResult {
    if (this.options.skip) return { answered: false, answer: '', skipped: true }
    const index = this.options.index
    if (index !== undefined && index >= 0 && index < choices.length) {
      return { answered: true, answer: choices[index] ?? '', selectedIndex: index, skipped: false }
    }
    return { answered: true, answer: this.options.answer ?? '', skipped: false }
  }
}

/** Ask for a structured clarification without fabricating a response when no UI is installed. */
export async function clarify(
  options: {
    readonly allowFreeform?: boolean
    readonly asker?: InteractiveAsker
    readonly choices?: readonly string[]
    readonly question: string
  },
  context?: ToolExecutionContext,
): Promise<JsonObject> {
  const question = options.question.trim()
  if (!question) return { ok: false, error: 'question required' }
  const choices = [...(options.choices ?? [])]
  const allowFreeform = options.allowFreeform ?? true
  if (!choices.length && !allowFreeform) {
    return { ok: false, error: 'either options must be supplied or allow_freeform=true' }
  }
  if (!options.asker) return { ok: true, answered: false, needs_ui: true }
  const result = await options.asker.ask(question, choices, allowFreeform, context)
  const output: JsonObject = {
    ok: true,
    answered: result.answered,
    answer: result.answer,
    selected_index: result.selectedIndex ?? null,
    skipped: result.skipped,
  }
  return output
}

export const CLARIFY_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: 'clarify',
    description: 'Request a structured user clarification through a configured interactive UI.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        question: { type: 'string' },
        options: { type: 'array', items: { type: 'string' } },
        allow_freeform: { type: 'boolean', default: true },
      },
      required: ['question'],
    },
  },
}

export interface ClarifyToolOptions {
  readonly asker?: InteractiveAsker
}

/** Register clarify with an explicitly supplied interaction adapter. */
export function registerClarifyTool(registry: ToolRegistry, options: ClarifyToolOptions = {}): void {
  registry.register(CLARIFY_DEFINITION, (inputs, context) => {
    const asker = options.asker
    return clarify({
      question: requiredString(inputs, 'question'),
      choices: optionalStringArray(inputs, 'options'),
      allowFreeform: optionalBoolean(inputs, 'allow_freeform', true),
      ...(asker === undefined ? {} : { asker }),
    }, context)
  })
}
