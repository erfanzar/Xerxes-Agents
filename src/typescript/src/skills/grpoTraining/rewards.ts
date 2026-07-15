// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { GrpoCompletion, GrpoCompletionMessage, GrpoRewardInput, GrpoRewardProgram } from './types.js'

/** Return the first XML tag body from a model completion, or an empty string when absent. */
export function extractXmlTag(text: string, tag: string): string {
  const content = requireText(text, 'text')
  const normalizedTag = requireTag(tag)
  const expression = new RegExp(`<${normalizedTag}>([\\s\\S]*?)<\\/${normalizedTag}>`)
  return expression.exec(content)?.[1]?.trim() ?? ''
}

/** Return the first `<answer>` value from a completion. */
export function extractAnswer(text: string): string {
  return extractXmlTag(text, 'answer')
}

/** Recover text from the string, message, or message-list completion forms accepted by GRPO. */
export function completionContent(completion: GrpoCompletion): string {
  if (typeof completion === 'string') return completion
  if (isCompletionMessageArray(completion)) {
    const first = completion[0]
    if (first === undefined || typeof first.content !== 'string') {
      throw new TypeError('completion message arrays must contain a first text content item')
    }
    return first.content
  }
  if (typeof completion.content !== 'string') throw new TypeError('completion messages must contain text content')
  return completion.content
}

/** Return 2.0 for an extracted answer that exactly matches the corresponding GSM8K target. */
export function correctnessReward(
  completions: readonly GrpoCompletion[],
  answers: readonly (string | null | undefined)[],
): readonly number[] {
  return completions.slice(0, answers.length).map((completion, index) => {
    const target = answers[index]
    return typeof target === 'string' && extractAnswer(completionContent(completion)) === target ? 2 : 0
  })
}

/** Return 0.5 when a response uses the complete reasoning/answer structure. */
export function formatReward(completions: readonly GrpoCompletion[]): readonly number[] {
  const expression = /<reasoning>.*?<\/reasoning>\s*<answer>.*?<\/answer>/s
  return completions.map(completion => expression.test(completionContent(completion)) ? 0.5 : 0)
}

/** Reward each required tag and penalize non-whitespace after the final answer close tag. */
export function incrementalFormatReward(completions: readonly GrpoCompletion[]): readonly number[] {
  return completions.map(completion => {
    const text = completionContent(completion)
    let score = 0
    if (text.includes('<reasoning>')) score += 0.125
    if (text.includes('</reasoning>')) score += 0.125
    if (text.includes('<answer>')) score += 0.125
    if (text.includes('</answer>')) score += 0.125
    const closing = text.lastIndexOf('</answer>')
    if (closing !== -1) score -= text.slice(closing + '</answer>'.length).trim().length * 0.001
    return score
  })
}

/** The three reward programs configured by the bundled GRPO training template. */
export const BASIC_GRPO_REWARD_PROGRAMS: readonly GrpoRewardProgram[] = [
  {
    id: 'incremental-format',
    description: 'Rewards each reasoning/answer tag and penalizes trailing text after </answer>.',
    evaluate: input => incrementalFormatReward(input.completions),
  },
  {
    id: 'format',
    description: 'Rewards complete <reasoning> followed by <answer> structure.',
    evaluate: input => formatReward(input.completions),
  },
  {
    id: 'correctness',
    description: 'Rewards an extracted <answer> that exactly equals the GSM8K target.',
    evaluate: input => correctnessReward(input.completions, input.answers ?? []),
  },
]

/** Evaluate every bundled reward program and retain the program identifier beside its scores. */
export function evaluateBasicGrpoRewards(input: GrpoRewardInput): Readonly<Record<string, readonly number[]>> {
  return Object.fromEntries(BASIC_GRPO_REWARD_PROGRAMS.map(program => [program.id, program.evaluate(input)]))
}

function requireText(value: string, name: string): string {
  if (typeof value !== 'string') throw new TypeError(`${name} must be text`)
  return value
}

function requireTag(value: string): string {
  if (!/^[A-Za-z][A-Za-z0-9_-]*$/.test(value)) throw new TypeError('tag must be an XML-style tag name')
  return value
}

function isCompletionMessageArray(value: GrpoCompletion): value is readonly GrpoCompletionMessage[] {
  return Array.isArray(value)
}
