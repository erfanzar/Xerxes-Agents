// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { readFile } from 'node:fs/promises'
import { join } from 'node:path'

import {
  BenchmarkDatasetError,
  type BenchmarkAdapter,
  type BenchmarkTask,
  type LoadResult,
} from './types.js'
import {
  applySelection,
  isMissingError,
  isRecord,
  optionalText,
  pathExists,
  readDatasetText,
  requiredText,
  safeJoin,
  verdict,
} from './util.js'
import {
  columnLetters,
  compareCells,
  readWorkbookCells,
  type SheetCell,
  type WorkbookCells,
} from './xlsxRead.js'

const ID = 'spreadsheetbench'

export const SPREADSHEETBENCH_EXPECTED_LAYOUT: readonly string[] = [
  'dataset.json (JSON array of { id, instruction, answer_position, optional spreadsheet_path, instruction_type, answer_sheet })',
  '<spreadsheet_path or spreadsheet/<id>>/1_<id>_input.xlsx (copied into the working directory as input.xlsx)',
  '<spreadsheet_path or spreadsheet/<id>>/1_<id>_answer.xlsx (the expected workbook, never shown to the model)',
]

/** SpreadsheetBench: edit a workbook; graded by comparing the answer cells with the expected workbook. */
export function createSpreadsheetBenchAdapter(): BenchmarkAdapter {
  return {
    id: ID,
    async load(datasetDir, opts): Promise<LoadResult> {
      const text = await readDatasetText(ID, datasetDir, 'dataset.json', SPREADSHEETBENCH_EXPECTED_LAYOUT)
      const fail = (problem: string): BenchmarkDatasetError =>
        new BenchmarkDatasetError(ID, datasetDir, problem, SPREADSHEETBENCH_EXPECTED_LAYOUT)
      let parsed: unknown
      try {
        parsed = JSON.parse(text)
      } catch (error) {
        throw fail(`dataset.json is not valid JSON (${error instanceof Error ? error.message : String(error)})`)
      }
      if (!Array.isArray(parsed)) throw fail('dataset.json must be a JSON array')

      const tasks: BenchmarkTask[] = []
      const excluded: { id: string; reason: string }[] = []
      for (const record of parsed) {
        if (!isRecord(record)) throw fail('dataset.json entries must be objects')
        const rawId = requiredText(record, 'id', fail)
        const id = `spreadsheetbench-${rawId}`
        const instruction = requiredText(record, 'instruction', fail)
        const position = requiredText(record, 'answer_position', fail)
        const folder = optionalText(record, 'spreadsheet_path') ?? `spreadsheet/${rawId}`
        let dir: string
        try {
          dir = safeJoin(datasetDir, folder)
        } catch (error) {
          throw fail(`entry ${rawId} has an unsafe spreadsheet_path (${error instanceof Error ? error.message : String(error)})`)
        }
        const inputPath = join(dir, `1_${rawId}_input.xlsx`)
        const answerPath = join(dir, `1_${rawId}_answer.xlsx`)
        if (!(await pathExists(inputPath)) || !(await pathExists(answerPath))) {
          excluded.push({ id, reason: `missing 1_${rawId}_input.xlsx or 1_${rawId}_answer.xlsx` })
          continue
        }
        const type = optionalText(record, 'instruction_type')
        tasks.push({
          id,
          benchmark: ID,
          prompt: [
            instruction,
            `\nThe workbook is input.xlsx in the working directory. The result belongs in cells ${position}.`,
            'Save the finished workbook as output.xlsx in the working directory.',
          ].join('\n'),
          files: [{ path: 'input.xlsx', sourcePath: inputPath }],
          grader: {
            kind: 'spreadsheet',
            expectedPath: answerPath,
            outputFile: 'output.xlsx',
            answerPosition: position,
          },
          tags: type === undefined ? ['spreadsheet'] : ['spreadsheet', type],
        })
      }
      return applySelection({ tasks, excluded }, opts)
    },

    async grade(task, run) {
      if (task.grader.kind !== 'spreadsheet') {
        throw new Error(`${task.id}: spreadsheetbench adapter cannot grade a ${task.grader.kind} task`)
      }
      const grader = task.grader
      const expected = readWorkbookCells(new Uint8Array(await readFile(grader.expectedPath)))
      const actual = await readProduced(run.workdir, grader.outputFile, expected.sheets[0]?.name ?? 'Sheet1')
      if (actual === undefined) return verdict(false, 0, `no ${grader.outputFile} (or output.csv) was produced`)
      const comparison = compareCells(expected, actual, grader.answerPosition)
      if (comparison.total === 0) return verdict(false, 0, 'the answer position covers no cells')
      const accepted = comparison.matched === comparison.total
      const score = comparison.matched / comparison.total
      const details = accepted
        ? `all ${comparison.total} answer cells match`
        : `${comparison.matched}/${comparison.total} cells match; ${comparison.mismatches.join('; ')}`
      return verdict(accepted, score, details)
    },
  }
}

async function readProduced(workdir: string, outputFile: string, csvSheetName: string): Promise<WorkbookCells | undefined> {
  try {
    return readWorkbookCells(new Uint8Array(await readFile(safeJoin(workdir, outputFile))))
  } catch (error) {
    if (!isMissingError(error)) throw error
  }
  try {
    return readCsvCells(await readFile(safeJoin(workdir, 'output.csv'), 'utf8'), csvSheetName)
  } catch (error) {
    if (isMissingError(error)) return undefined
    throw error
  }
}

/** A CSV as a one-sheet workbook; numeric-looking fields become numbers. */
export function readCsvCells(text: string, sheetName: string): WorkbookCells {
  const cells = new Map<string, SheetCell>()
  parseCsvRows(text).forEach((row, rowIndex) => {
    row.forEach((value, columnIndex) => {
      if (value === '') return
      const trimmed = value.trim()
      const number = trimmed === '' ? Number.NaN : Number(trimmed)
      cells.set(`${columnLetters(columnIndex + 1)}${rowIndex + 1}`, Number.isFinite(number) ? number : value)
    })
  })
  return { sheets: [{ cells, name: sheetName }] }
}

export function parseCsvRows(text: string): string[][] {
  const rows: string[][] = []
  let row: string[] = []
  let field = ''
  let quoted = false
  const source = text.replace(/^\uFEFF/, '')
  for (let index = 0; index < source.length; index += 1) {
    const character = source[index]
    if (quoted) {
      if (character === '"') {
        if (source[index + 1] === '"') {
          field += '"'
          index += 1
        } else {
          quoted = false
        }
      } else {
        field += character
      }
    } else if (character === '"') {
      quoted = true
    } else if (character === ',') {
      row.push(field)
      field = ''
    } else if (character === '\n' || character === '\r') {
      if (character === '\r' && source[index + 1] === '\n') index += 1
      row.push(field)
      rows.push(row)
      row = []
      field = ''
    } else {
      field += character
    }
  }
  if (field !== '' || row.length > 0) {
    row.push(field)
    rows.push(row)
  }
  return rows
}
