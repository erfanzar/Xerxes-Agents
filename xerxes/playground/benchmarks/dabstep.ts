// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { join } from 'node:path'

import {
  BenchmarkDatasetError,
  type BenchmarkAdapter,
  type BenchmarkTask,
  type BenchmarkTaskFile,
  type LoadResult,
} from './types.js'
import {
  applySelection,
  listFilesRecursive,
  optionalText,
  parseJsonl,
  pathExists,
  readDatasetText,
  requiredText,
  verdict,
} from './util.js'

const ID = 'dabstep'

export const DABSTEP_EXPECTED_LAYOUT: readonly string[] = [
  'tasks/dev.jsonl (one JSON object per line: task_id, question, answer, optional guidelines and level)',
  'context/ (optional shared data files, copied into every task working directory as context/...)',
]

/** DABstep: data-analysis questions with a single short answer. Graded by normalized answer match. */
export function createDabstepAdapter(): BenchmarkAdapter {
  return {
    id: ID,
    async load(datasetDir, opts): Promise<LoadResult> {
      const rel = 'tasks/dev.jsonl'
      const text = await readDatasetText(ID, datasetDir, rel, DABSTEP_EXPECTED_LAYOUT)
      const records = parseJsonl(ID, datasetDir, rel, text, DABSTEP_EXPECTED_LAYOUT)
      const fail = (problem: string): BenchmarkDatasetError =>
        new BenchmarkDatasetError(ID, datasetDir, problem, DABSTEP_EXPECTED_LAYOUT)

      const files: BenchmarkTaskFile[] = []
      const contextDir = join(datasetDir, 'context')
      if (await pathExists(contextDir)) {
        for (const file of await listFilesRecursive(contextDir)) {
          files.push({ path: `context/${file}`, sourcePath: join(contextDir, file) })
        }
      }

      const tasks: BenchmarkTask[] = []
      const excluded: { id: string; reason: string }[] = []
      for (const record of records) {
        const rawId = requiredText(record, 'task_id', fail)
        const id = `dabstep-${rawId}`
        const question = requiredText(record, 'question', fail)
        const answerValue = record.answer
        const expected = typeof answerValue === 'number' && Number.isFinite(answerValue)
          ? String(answerValue)
          : optionalText(record, 'answer')
        if (expected === undefined) {
          excluded.push({ id, reason: 'no public answer to grade against' })
          continue
        }
        const guidelines = optionalText(record, 'guidelines')
        const level = optionalText(record, 'level')
        const prompt = [
          question,
          guidelines === undefined ? '' : `\nGuidelines: ${guidelines}`,
          '\nThe data files are in ./context. End your reply with one line: FINAL ANSWER: <answer>',
        ].join('')
        tasks.push({
          id,
          benchmark: ID,
          prompt,
          files,
          grader: { kind: 'answer', expected },
          tags: level === undefined ? ['data-analysis'] : ['data-analysis', `level-${level}`],
        })
      }
      return applySelection({ tasks, excluded }, opts)
    },

    async grade(task, run) {
      if (task.grader.kind !== 'answer') throw new Error(`${task.id}: dabstep adapter cannot grade a ${task.grader.kind} task`)
      const actual = extractFinalAnswer(run.finalText)
      const ok = dabstepAnswersMatch(task.grader.expected, actual)
      return verdict(ok, ok ? 1 : 0, ok ? `answer matched (${actual})` : `expected ${task.grader.expected}, got ${actual || '(no answer)'}`)
    },
  }
}

/** The text after the last `FINAL ANSWER:` marker, else the last non-empty line. */
export function extractFinalAnswer(text: string): string {
  const matches = [...text.matchAll(/final\s+answer\s*[:\-]\s*(.+)/gi)]
  const last = matches.at(-1)?.[1]
  if (last !== undefined) return last.trim()
  const lines = text.split('\n').map(line => line.trim()).filter(Boolean)
  return lines.at(-1) ?? ''
}

function clean(value: string): string {
  return value
    .trim()
    .replace(/^[*_`"'\s]+|[*_`"'\s]+$/g, '')
    .replace(/\.$/, '')
    .replace(/\s+/g, ' ')
    .toLowerCase()
}

const THOUSANDS = /^-?\d{1,3}(,\d{3})+(\.\d+)?$/

function parseNumber(value: string): number | undefined {
  const stripped = value.replace(/[$€£%\s]/g, '').replace(THOUSANDS.test(value.replace(/[$€£%\s]/g, '')) ? /,/g : /$^/, '')
  if (!/^-?\d+(\.\d+)?$/.test(stripped)) return undefined
  const parsed = Number(stripped)
  return Number.isFinite(parsed) ? parsed : undefined
}

function decimals(value: string): number {
  const stripped = value.replace(/[$€£%\s,]/g, '')
  const dot = stripped.indexOf('.')
  return dot < 0 ? 0 : stripped.length - dot - 1
}

/**
 * Normalized comparison: case/quote/trailing-period insensitive; numbers compare after removing
 * currency, percent and thousands separators and rounding the answer to the expected precision;
 * comma lists compare as sets.
 */
export function dabstepAnswersMatch(expectedRaw: string, actualRaw: string): boolean {
  const expected = clean(expectedRaw)
  const actual = clean(actualRaw)
  if (!expected || !actual) return false
  const expectedNumber = parseNumber(expected)
  if (expectedNumber !== undefined) {
    const actualNumber = parseNumber(actual)
    if (actualNumber === undefined) return false
    const places = decimals(expected)
    const scale = 10 ** places
    return Math.round(actualNumber * scale) / scale === Math.round(expectedNumber * scale) / scale
  }
  if (expected.includes(',')) {
    const toSet = (value: string): string[] => value.split(',').map(item => clean(item)).filter(Boolean).sort()
    const left = toSet(expected)
    const right = toSet(actual)
    return left.length === right.length && left.every((item, index) => item === right[index])
  }
  return expected === actual
}
