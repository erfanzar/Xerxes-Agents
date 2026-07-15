// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { GrpoDatasetExample, Gsm8kDatasetPort, Gsm8kSourceExample } from './types.js'

/** The delimiter used by GSM8K to separate working from its final answer. */
export const GSM8K_ANSWER_DELIMITER = '####'

/** The system message used by the reference GRPO template. */
export const GRPO_SYSTEM_PROMPT = `Respond in the following format:
<reasoning>
[Your step-by-step thinking]
</reasoning>
<answer>
[Final answer]
</answer>`

/** Extract the final answer segment used by the correctness reward, if present. */
export function extractGsm8kAnswer(value: string): string | null {
  if (typeof value !== 'string') throw new TypeError('answer must be text')
  const source = value
  const sections = source.split(GSM8K_ANSWER_DELIMITER)
  return sections.length < 2 ? null : (sections[1] ?? '').trim()
}

/** Convert one GSM8K record to the native prompt/answer shape consumed by GRPO hosts. */
export function mapGsm8kExample(example: Gsm8kSourceExample): GrpoDatasetExample {
  return {
    answer: extractGsm8kAnswer(example.answer),
    prompt: [
      { role: 'system', content: GRPO_SYSTEM_PROMPT },
      { role: 'user', content: requireText(example.question, 'question') },
    ],
  }
}

/** Convert a GSM8K split without importing Hugging Face or any Python dataset library. */
export function mapGsm8kExamples(examples: readonly Gsm8kSourceExample[]): readonly GrpoDatasetExample[] {
  if (!Array.isArray(examples)) throw new TypeError('examples must be an array')
  return examples.map(mapGsm8kExample)
}

/** Load and normalize a GSM8K split through a caller-owned dataset adapter. */
export async function loadGsm8kTrainingDataset(
  dataset: Gsm8kDatasetPort,
  split = 'train',
): Promise<readonly GrpoDatasetExample[]> {
  const normalizedSplit = requireText(split, 'split')
  const source = await dataset.loadGsm8k({ config: 'main', split: normalizedSplit })
  return mapGsm8kExamples(source)
}

/** Parse JSONL GSM8K records for the safe Bun CLI/template path. */
export function parseGsm8kJsonl(value: string): readonly Gsm8kSourceExample[] {
  if (typeof value !== 'string') throw new TypeError('JSONL input must be text')
  const records: Gsm8kSourceExample[] = []
  for (const [index, line] of value.split(/\r?\n/).entries()) {
    if (!line.trim()) continue
    let parsed: unknown
    try {
      parsed = JSON.parse(line)
    } catch (error) {
      throw new SyntaxError(`invalid GSM8K JSONL at line ${index + 1}: ${errorMessage(error)}`)
    }
    if (!isRecord(parsed) || typeof parsed.question !== 'string' || typeof parsed.answer !== 'string') {
      throw new TypeError(`GSM8K JSONL line ${index + 1} must contain string question and answer fields`)
    }
    records.push({ question: parsed.question, answer: parsed.answer })
  }
  return records
}

function requireText(value: string, name: string): string {
  if (typeof value !== 'string' || !value.trim()) throw new TypeError(`${name} must be a non-empty string`)
  return value.trim()
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}
