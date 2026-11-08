// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { readFile, stat } from 'node:fs/promises'
import { join } from 'node:path'

import {
  BenchmarkDatasetError,
  type BenchmarkAdapter,
  type BenchmarkTask,
  type BenchmarkTaskFile,
  type LoadResult,
  type ReviewerDecision,
} from './types.js'
import {
  applySelection,
  isMissingError,
  isRecord,
  isStringArray,
  listFilesRecursive,
  optionalText,
  parseJsonl,
  pathExists,
  readDatasetText,
  requiredText,
  safeJoin,
} from './util.js'

const ID = 'gdpval'

export const GDPVAL_EXPECTED_LAYOUT: readonly string[] = [
  'tasks.jsonl (one JSON object per line: task_id, sector, occupation, prompt, optional reference_files: string[] relative to the dataset directory)',
  'reviews.json (optional: { "<task_id>": { accepted: boolean, score?: number, reviewer?: string, notes?: string } })',
]

/**
 * GDPval: professional deliverables judged by people. The adapter never accepts a task itself:
 * it produces a review packet, and `accepted` comes only from a reviewer decision in reviews.json.
 */
export function createGdpvalAdapter(): BenchmarkAdapter {
  return {
    id: ID,
    async load(datasetDir, opts): Promise<LoadResult> {
      const rel = 'tasks.jsonl'
      const text = await readDatasetText(ID, datasetDir, rel, GDPVAL_EXPECTED_LAYOUT)
      const records = parseJsonl(ID, datasetDir, rel, text, GDPVAL_EXPECTED_LAYOUT)
      const fail = (problem: string): BenchmarkDatasetError =>
        new BenchmarkDatasetError(ID, datasetDir, problem, GDPVAL_EXPECTED_LAYOUT)
      const decisions = await loadDecisions(datasetDir, fail)

      const tasks: BenchmarkTask[] = []
      const excluded: { id: string; reason: string }[] = []
      for (const record of records) {
        const rawId = requiredText(record, 'task_id', fail)
        const id = `gdpval-${rawId}`
        const sector = optionalText(record, 'sector') ?? 'unknown'
        const occupation = optionalText(record, 'occupation') ?? 'unknown'
        const prompt = requiredText(record, 'prompt', fail)
        const references = record.reference_files
        if (references !== undefined && !isStringArray(references)) throw fail(`task ${rawId}: reference_files must be an array of strings`)

        const files: BenchmarkTaskFile[] = []
        let missing: string | undefined
        for (const reference of references ?? []) {
          const sourcePath = safeJoin(datasetDir, reference)
          if (!(await pathExists(sourcePath))) {
            missing = reference
            break
          }
          files.push({ path: `reference/${reference.split('/').at(-1) ?? reference}`, sourcePath })
        }
        if (missing !== undefined) {
          excluded.push({ id, reason: `reference file not found: ${missing}` })
          continue
        }
        const decision = decisions.get(rawId)
        tasks.push({
          id,
          benchmark: ID,
          prompt: `${prompt}\n\nReference files (if any) are in ./reference. Save every deliverable in ./deliverables.`,
          files,
          grader: {
            kind: 'review',
            taskId: rawId,
            sector,
            occupation,
            ...(decision === undefined ? {} : { decision }),
          },
          tags: ['judgment', sector, occupation],
        })
      }
      return applySelection({ tasks, excluded }, opts)
    },

    async grade(task, run) {
      if (task.grader.kind !== 'review') throw new Error(`${task.id}: gdpval adapter cannot grade a ${task.grader.kind} task`)
      const grader = task.grader
      const packet = await reviewPacket(task, grader.sector, grader.occupation, run.workdir, run.finalText)
      const decision = grader.decision
      if (decision === undefined) {
        return {
          accepted: false,
          score: 0,
          details: 'needs human review: no reviewer decision recorded for this task',
          status: 'needs_review',
          disclosureFailure: false,
          reviewPacket: packet,
        }
      }
      return {
        accepted: decision.accepted,
        score: decision.score ?? (decision.accepted ? 1 : 0),
        details: `reviewer${decision.reviewer === undefined ? '' : ` ${decision.reviewer}`}: ${decision.accepted ? 'accepted' : 'rejected'}${decision.notes === undefined ? '' : ` (${decision.notes})`}`,
        status: decision.accepted ? 'accepted' : 'rejected',
        disclosureFailure: false,
        reviewPacket: packet,
      }
    },
  }
}

async function loadDecisions(
  datasetDir: string,
  fail: (problem: string) => BenchmarkDatasetError,
): Promise<Map<string, ReviewerDecision>> {
  const decisions = new Map<string, ReviewerDecision>()
  let text: string
  try {
    text = await readFile(join(datasetDir, 'reviews.json'), 'utf8')
  } catch (error) {
    if (isMissingError(error)) return decisions
    throw error
  }
  let parsed: unknown
  try {
    parsed = JSON.parse(text)
  } catch (error) {
    throw fail(`reviews.json is not valid JSON (${error instanceof Error ? error.message : String(error)})`)
  }
  if (!isRecord(parsed)) throw fail('reviews.json must be an object keyed by task id')
  for (const [taskId, value] of Object.entries(parsed)) {
    if (!isRecord(value) || typeof value.accepted !== 'boolean') throw fail(`reviews.json entry ${taskId} needs a boolean "accepted"`)
    const score = value.score
    if (score !== undefined && (typeof score !== 'number' || !Number.isFinite(score) || score < 0 || score > 1)) {
      throw fail(`reviews.json entry ${taskId}: score must be a number from 0 to 1`)
    }
    const reviewer = optionalText(value, 'reviewer')
    const notes = optionalText(value, 'notes')
    decisions.set(taskId, {
      accepted: value.accepted,
      ...(typeof score === 'number' ? { score } : {}),
      ...(reviewer === undefined ? {} : { reviewer }),
      ...(notes === undefined ? {} : { notes }),
    })
  }
  return decisions
}

async function reviewPacket(
  task: BenchmarkTask,
  sector: string,
  occupation: string,
  workdir: string,
  finalText: string,
): Promise<string> {
  const deliverables: string[] = []
  const root = join(workdir, 'deliverables')
  if (await pathExists(root)) {
    for (const file of await listFilesRecursive(root)) {
      const size = (await stat(join(root, file))).size
      deliverables.push(`- deliverables/${file} (${size} bytes)`)
    }
  }
  const reply = finalText.length > 4_000 ? `${finalText.slice(0, 4_000)}\n[... ${finalText.length - 4_000} more characters]` : finalText
  return [
    `# Review packet: ${task.id}`,
    '',
    `Sector: ${sector}`,
    `Occupation: ${occupation}`,
    '',
    '## Task',
    task.prompt,
    '',
    '## Deliverables',
    deliverables.length === 0 ? '_none were written_' : deliverables.join('\n'),
    '',
    '## Final reply',
    reply || '_empty_',
  ].join('\n')
}
