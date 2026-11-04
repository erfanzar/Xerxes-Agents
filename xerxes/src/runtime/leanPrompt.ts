// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ValidationError } from '../core/errors.js'
import type { JsonSchema, ToolDefinition } from '../types/toolCalls.js'
import { protectedRunExclusion } from './protectedRun.js'

/** Environment variable that selects the prompt profile. */
export const PROMPT_PROFILE_ENV = 'XERXES_PROMPT_PROFILE'

/** `lean` sends a terse system prompt and a small tool set; `full` is today's behavior. */
export type LeanPromptSetting = 'lean' | 'full'

/** Tools a small local model needs for file and data work. Others are left out of the request. */
export const LEAN_CORE_TOOL_NAMES: readonly string[] = Object.freeze([
  'ReadFile',
  'GlobTool',
  'GrepTool',
  'ListDir',
  'WriteFile',
  'FileEditTool',
  'exec_command',
  'CSVProcessor',
  'XlsxWriter',
  'XlsxReader',
  'AskUserQuestionTool',
  'AskAdvisor',
])

const MAX_TOOL_DESCRIPTION_CHARS = 200
const MAX_SCHEMA_DESCRIPTION_CHARS = 80
const MAX_PROJECT_INSTRUCTION_CHARS = 1500

function normalizeSetting(value: unknown, field: string): LeanPromptSetting {
  if (typeof value !== 'string') throw new ValidationError(field, "must be 'lean' or 'full'", value)
  const normalized = value.trim().toLowerCase()
  if (normalized === 'lean' || normalized === 'full') return normalized
  throw new ValidationError(field, "must be 'lean' or 'full'", value)
}

/**
 * Resolve the prompt profile. Precedence: the provider profile's own field,
 * then `XERXES_PROMPT_PROFILE`, then `runtime.prompt_profile`, then `full`.
 * An unreadable value is a ValidationError, never a silent `full`.
 */
export function parsePromptProfileConfig(
  runtime: Record<string, unknown>,
  env: Record<string, string | undefined>,
  profilePromptProfile?: LeanPromptSetting,
): LeanPromptSetting {
  if (profilePromptProfile !== undefined) return normalizeSetting(profilePromptProfile, 'prompt_profile')
  const fromEnv = env[PROMPT_PROFILE_ENV]?.trim()
  if (fromEnv) return normalizeSetting(fromEnv, PROMPT_PROFILE_ENV)
  const fromFile = runtime['prompt_profile']
  if (fromFile === undefined || fromFile === null) return 'full'
  return normalizeSetting(fromFile, 'prompt_profile')
}

export interface LeanToolOptions {
  /** A protected run never gets shell, MCP, plugin or computer tools, even when listed in `extra`. */
  readonly protected?: boolean
  /** Additional tool names to keep on top of the core set. */
  readonly extra?: readonly string[]
}

/**
 * Keep the core tools that are present in `definitions`, with each description
 * cut to its first sentence and each nested schema description cut short.
 * Names, types, required lists and enums are not changed.
 */
export function selectLeanTools<T extends ToolDefinition>(
  definitions: readonly T[],
  options: LeanToolOptions = {},
): ToolDefinition[] {
  const wanted = new Set<string>([...LEAN_CORE_TOOL_NAMES, ...(options.extra ?? [])])
  const seen = new Set<string>()
  const selected: ToolDefinition[] = []
  for (const definition of definitions) {
    const name = definition.function.name
    if (!wanted.has(name) || seen.has(name)) continue
    if (options.protected === true && protectedRunExclusion(name) !== undefined) continue
    seen.add(name)
    selected.push({
      ...definition,
      function: {
        ...definition.function,
        description: firstSentence(definition.function.description, MAX_TOOL_DESCRIPTION_CHARS),
        parameters: shrinkSchema(definition.function.parameters),
      },
    })
  }
  return selected
}

export interface LeanSystemPromptInput {
  readonly cwd: string
  readonly model: string
  /** Date text as it should appear in the prompt, for example 2026-11-02. */
  readonly date: string
  readonly toolNames: readonly string[]
  readonly projectInstructions?: string
}

/** A terse system prompt for small local models. */
export function buildLeanSystemPrompt(input: LeanSystemPromptInput): string {
  const has = (name: string): boolean => input.toolNames.includes(name)
  const lines: string[] = [
    'You are Xerxes, an assistant for code and data work on the user\'s computer.',
    `Model: ${input.model}. Date: ${input.date}. Working directory: ${input.cwd}`,
    '',
    'Rules:',
    '- Read a file before you edit it. Make the smallest change that does the task.',
    '- Input files are read-only unless the user asks you to change them. Write results to a new file.',
    '- Do not guess file contents or paths. Use the tools to look.',
    '- Check your result by running a check (a test, a validator, or reading the output back) before you say it is done.',
    '- Never say you checked or validated something you did not run. If you could not check, say so.',
  ]
  if (has('exec_command')) {
    lines.push('- exec_command runs one program without a shell: cmd is the program, args is the list of arguments.')
  }
  if (has('XlsxWriter')) {
    lines.push('- Write .xlsx files with XlsxWriter. WriteFile would store plain text, not a workbook.')
  }
  if (has('AskUserQuestionTool')) {
    lines.push('- If the request is unclear in a way that changes the result, ask with AskUserQuestionTool.')
  }
  if (has('AskAdvisor')) {
    lines.push('- If you are stuck after a real attempt, call AskAdvisor once with the problem. Do not call it otherwise.')
  }
  const instructions = input.projectInstructions?.trim()
  if (instructions) {
    lines.push('', 'Project instructions:', clipText(instructions, MAX_PROJECT_INSTRUCTION_CHARS))
  }
  return lines.join('\n')
}

export interface LeanPromptSize {
  readonly chars: number
  /** Same chars / 4 heuristic as `context/promptBudget.ts`. */
  readonly estimatedTokens: number
}

/** Size of the system prompt plus the serialized tool schemas that ride with every request. */
export function leanPromptSize(prompt: string, tools: readonly ToolDefinition[]): LeanPromptSize {
  const chars = prompt.length + JSON.stringify(tools).length
  return { chars, estimatedTokens: Math.ceil(chars / 4) }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function shrinkSchema(schema: JsonSchema): JsonSchema {
  const shrunk: Record<string, unknown> = {}
  for (const [key, value] of Object.entries(schema)) {
    shrunk[key] = key === 'description' && typeof value === 'string'
      ? clipText(value.replace(/\s+/gu, ' ').trim(), MAX_SCHEMA_DESCRIPTION_CHARS)
      : shrinkValue(value)
  }
  return shrunk
}

function shrinkValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(shrinkValue)
  if (isRecord(value)) return shrinkSchema(value)
  return value
}

/** Cut `text` to at most `max` characters, at a word boundary when one is near. */
function clipText(text: string, max: number): string {
  if (text.length <= max) return text
  const cut = text.slice(0, max - 1)
  const space = cut.lastIndexOf(' ')
  return `${(space > max * 0.6 ? cut.slice(0, space) : cut).trimEnd()}…`
}

/** First sentence of `text`; `e.g.` and `i.e.` do not end a sentence. */
function firstSentence(text: string, max: number): string {
  const flat = text.replace(/\s+/gu, ' ').trim()
  const end = /[.!?](?=\s+[A-Z`"'(\[]|\s*$)/gu
  for (let match = end.exec(flat); match !== null; match = end.exec(flat)) {
    const sentence = flat.slice(0, match.index + 1)
    if (/(?:\be\.g|\bi\.e|\bvs)\.$/u.test(sentence)) continue
    return clipText(sentence, max)
  }
  return clipText(flat, max)
}
