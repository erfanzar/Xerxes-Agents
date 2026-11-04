// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** Benchmarks the comparison matrix can run. Developer-only; nothing here is user-facing. */
export type BenchmarkId = 'terminal-bench' | 'gdpval' | 'spreadsheetbench' | 'dabstep' | 'synthetic-private'

export const BENCHMARK_IDS: readonly BenchmarkId[] = [
  'terminal-bench',
  'gdpval',
  'spreadsheetbench',
  'dabstep',
  'synthetic-private',
]

/** A file placed in the task's fresh working directory before the run. */
export type BenchmarkTaskFile =
  | { readonly path: string; readonly contents: string }
  | { readonly path: string; readonly sourcePath: string }

/** A human reviewer's verdict for one judgment-graded task. */
export interface ReviewerDecision {
  readonly accepted: boolean
  readonly score?: number
  readonly reviewer?: string
  readonly notes?: string
}

/** A declarative, TS-native check for a Terminal Bench task (never executes dataset code). */
export type NativeCheck =
  | { readonly kind: 'file_exists'; readonly path: string }
  | { readonly kind: 'file_contains'; readonly path: string; readonly text: string }
  | { readonly kind: 'file_equals'; readonly path: string; readonly text: string }
  | { readonly kind: 'answer_contains'; readonly text: string }

export interface SyntheticWorkbookCheck {
  readonly file: string
  readonly sheet: string
  readonly cells: readonly { readonly ref: string; readonly value: string | number }[]
}

/** How a task is graded. Each adapter only understands its own kinds. */
export type BenchmarkGrader =
  | { readonly kind: 'answer'; readonly expected: string }
  | {
    readonly kind: 'spreadsheet'
    readonly expectedPath: string
    readonly outputFile: string
    readonly answerPosition?: string
  }
  | {
    readonly kind: 'review'
    readonly taskId: string
    readonly sector: string
    readonly occupation: string
    readonly decision?: ReviewerDecision
  }
  | { readonly kind: 'native-checks'; readonly checks: readonly NativeCheck[] }
  | {
    readonly kind: 'synthetic'
    /** Unique strings planted in private files; none may appear in outbound text. */
    readonly canaries: readonly string[]
    /** Substrings the final reply must contain (case and whitespace insensitive). */
    readonly replyIncludes?: readonly string[]
    readonly workbook?: SyntheticWorkbookCheck
    /** A file the run must have written, containing this text (case insensitive). */
    readonly fileIncludes?: { readonly file: string; readonly text: string }
  }

export interface BenchmarkTask {
  readonly id: string
  readonly benchmark: BenchmarkId
  readonly prompt: string
  readonly files: readonly BenchmarkTaskFile[]
  readonly grader: BenchmarkGrader
  readonly tags: readonly string[]
}

export interface LoadOptions {
  readonly limit?: number
  readonly ids?: readonly string[]
}

export interface LoadResult {
  readonly tasks: readonly BenchmarkTask[]
  readonly excluded: readonly { readonly id: string; readonly reason: string }[]
}

export interface GradeRun {
  readonly workdir: string
  readonly finalText: string
  /** Everything the run reported sending to an external model/service; absent when not measured. */
  readonly outboundTexts?: readonly string[]
}

export type GradeStatus = 'accepted' | 'rejected' | 'needs_review'

export interface GradeResult {
  readonly accepted: boolean
  /** 0..1 */
  readonly score: number
  readonly details: string
  readonly status: GradeStatus
  /** True when a planted private canary appeared in outbound text. Always makes `accepted` false. */
  readonly disclosureFailure: boolean
  /** Markdown handed to a human for judgment-graded tasks. */
  readonly reviewPacket?: string
}

export interface BenchmarkAdapter {
  readonly id: BenchmarkId
  /** Reads a LOCALLY downloaded dataset directory. Never fetches anything. */
  load(datasetDir: string, opts: LoadOptions): Promise<LoadResult>
  grade(task: BenchmarkTask, run: GradeRun): Promise<GradeResult>
}

/** The dataset directory does not have the layout the adapter expects. */
export class BenchmarkDatasetError extends Error {
  readonly adapter: BenchmarkId
  readonly datasetDir: string
  readonly expected: readonly string[]

  constructor(adapter: BenchmarkId, datasetDir: string, problem: string, expected: readonly string[]) {
    super(`${adapter}: ${problem} (dataset ${datasetDir}). Expected: ${expected.join('; ')}`)
    this.name = 'BenchmarkDatasetError'
    this.adapter = adapter
    this.datasetDir = datasetDir
    this.expected = expected
  }
}
